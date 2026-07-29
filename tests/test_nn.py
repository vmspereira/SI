# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the NN container (si.supervised.nn.network) and the RNN layer.
#
# test_layers.py checks the layers in isolation; this file checks that they
# compose -- that a stack of layers actually trains, and that the recurrent
# layer's backpropagation-through-time agrees with a finite-difference
# gradient, the same standard the Conv2D backward pass is held to.
# ----------------------------------------------------------------------------
import contextlib
import io
import unittest
import warnings

import numpy as np

from si.data import Dataset
from si.supervised.nn import NN
from si.supervised.nn.layers import Dense
from si.supervised.nn.activation import Tanh, Sigmoid
from si.supervised.nn.rnn import RNN
from si.supervised.nn.optimizers import SGD, Adam


def xor_dataset():
    """The classic non-linearly-separable problem: no single straight line
    separates the two classes, so a network with a hidden layer is required."""
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([[0.], [1.], [1.], [0.]])
    return Dataset(X, y)


def quiet_nn(**kwargs):
    # verbose=False is now genuinely silent, so no `step` workaround is needed.
    kwargs.setdefault('verbose', False)
    return NN(**kwargs)


class TestNNContainer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.ds = xor_dataset()

    def build(self, **kwargs):
        net = quiet_nn(**kwargs)
        net.add(Dense(2, 4))
        net.add(Tanh())
        net.add(Dense(4, 1))
        net.add(Sigmoid())
        return net

    def test_add_registers_layers_in_order(self):
        # The order layers are added defines the forward pass, so the container
        # must preserve it.
        net = self.build()
        self.assertEqual(len(net.layers), 4)
        self.assertIsInstance(net.layers[0], Dense)
        self.assertIsInstance(net.layers[1], Tanh)

    def test_forward_shape(self):
        # 4 samples through Dense(2,4) -> Dense(4,1) gives one output per sample.
        net = self.build()
        out = net.forward(self.ds.X)
        self.assertEqual(out.shape, (4, 1))

    def test_predict_requires_fit(self):
        net = self.build()
        with self.assertRaises(AssertionError):
            net.predict(self.ds.X)

    def test_cost_requires_fit(self):
        net = self.build()
        with self.assertRaises(AssertionError):
            net.cost()

    def test_fit_records_history_per_epoch(self):
        net = self.build(epochs=25)
        net.fit(self.ds)
        self.assertTrue(net.is_fitted)
        # history maps epoch number -> (loss, metric score)
        self.assertEqual(sorted(net.history.keys()), list(range(1, 26)))

    def test_training_reduces_the_loss(self):
        # The whole point of the training loop: the loss at the end must be
        # below the loss at the start. This is what would break if the backward
        # pass, the optimizer wiring, or the loss derivative were wrong.
        net = self.build(epochs=300)
        net.fit(self.ds)
        first = net.history[1][0]
        last = net.history[300][0]
        self.assertLess(last, first)

    def test_learns_xor(self):
        # End-to-end: a 2-4-1 network with non-linear activations should solve
        # XOR, which no linear model can. Thresholding the sigmoid output at 0.5
        # must recover all four labels.
        #
        # Adam is used rather than the default SGD(lr=0.01) because the default
        # is far too slow here -- on this problem it fails to separate the
        # classes even after 3000 full-batch epochs, while Adam(0.1) converges
        # in 500. That contrast is the practical argument for adaptive step
        # sizes. Verified across 8 seeds before being pinned here.
        net = self.build(epochs=500, optimizer=Adam(0.1))
        net.fit(self.ds)
        preds = (net.predict(self.ds.X) > 0.5).astype(float)
        np.testing.assert_array_equal(preds, self.ds.y)

    def test_cost_matches_the_loss_of_the_predictions(self):
        net = self.build(epochs=50)
        net.fit(self.ds)
        expected = net.loss(self.ds.y, net.predict(self.ds.X))
        self.assertAlmostEqual(net.cost(), expected)

    def test_cost_accepts_explicit_data(self):
        net = self.build(epochs=50)
        net.fit(self.ds)
        # passing X and y explicitly must score those instead of the training set
        self.assertAlmostEqual(net.cost(self.ds.X, self.ds.y), net.cost())

    def test_predict_is_a_batch_predictor(self):
        # The NN declares the batch convention, which is what lets Ensemble and
        # the cross-validation scorer call it correctly.
        self.assertTrue(NN.predicts_batch)

    def test_unknown_loss_warns_and_falls_back_to_mse(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            net = quiet_nn(loss="not-a-loss")
        self.assertTrue(any("not a valid loss" in str(w.message) for w in caught))
        # MSE of a known pair confirms which loss was installed
        self.assertAlmostEqual(net.loss(np.array([1.0]), np.array([0.0])), 1.0)

    def test_set_loss_by_name_and_by_tuple(self):
        net = quiet_nn()
        net.set_loss("MAE")
        self.assertAlmostEqual(net.loss(np.array([2.0]), np.array([0.0])), 2.0)
        # a (function, derivative) pair may also be supplied directly
        net.set_loss((lambda t, p: 42.0, lambda t, p: p - t))
        self.assertEqual(net.loss(None, None), 42.0)

    def test_metric_is_recorded_in_history(self):
        def always_half(y_true, y_pred):
            return 0.5

        net = self.build(epochs=5, metric=always_half)
        net.fit(self.ds)
        self.assertAlmostEqual(net.history[5][1], 0.5)

    def test_str_lists_the_layers(self):
        net = self.build()
        self.assertEqual(len(str(net).splitlines()), 4)

    def test_verbose_false_prints_nothing(self):
        # verbose=False used to still print, just with end="\r" so each line
        # overwrote the last -- the flag meant "print compactly", and there was
        # no way to train quietly except setting step above epochs.
        buffer = io.StringIO()
        net = self.build(epochs=3, step=1, verbose=False)
        with contextlib.redirect_stdout(buffer):
            net.fit(self.ds)
        self.assertEqual(buffer.getvalue(), "")

    def test_verbose_true_reports_every_step_epochs(self):
        buffer = io.StringIO()
        net = self.build(epochs=6, step=2, verbose=True)
        with contextlib.redirect_stdout(buffer):
            net.fit(self.ds)
        lines = buffer.getvalue().strip().splitlines()
        # epochs 2, 4 and 6 -> three reports, each on its own line
        self.assertEqual(len(lines), 3)
        self.assertIn("epoch 2/6", lines[0])
        self.assertIn("loss=", lines[0])


class TestNNInputSizeInference(unittest.TestCase):
    """A layer may omit its input_size and take it from the layer before it."""

    def setUp(self):
        np.random.seed(42)
        self.ds = xor_dataset()

    def test_infers_input_size_from_the_previous_layer(self):
        net = quiet_nn()
        net.add(Dense(2, 4))
        net.add(Dense(None, 3))
        self.assertEqual(net.layers[1].input_size, 4)

    def test_looks_past_shape_preserving_layers(self):
        # Activations carry no output size of their own, so the search has to
        # walk back past them to the last Dense.
        net = quiet_nn()
        net.add(Dense(2, 8))
        net.add(Tanh())
        net.add(Dense(None, 5))
        self.assertEqual(net.layers[2].input_size, 8)

    def test_an_inferred_network_still_trains(self):
        # End-to-end: only the first layer states its input size, and the
        # network still solves XOR.
        net = quiet_nn(epochs=500, optimizer=Adam(0.1))
        net.add(Dense(2, 4))
        net.add(Tanh())
        net.add(Dense(None, 1))
        net.add(Sigmoid())
        net.fit(self.ds)
        preds = (net.predict(self.ds.X) > 0.5).astype(float)
        np.testing.assert_array_equal(preds, self.ds.y)

    def test_the_first_layer_cannot_infer(self):
        # Nothing precedes it, so there is no output size to take.
        net = quiet_nn()
        with self.assertRaises(ValueError) as ctx:
            net.add(Dense(None, 4))
        self.assertIn("first layer", str(ctx.exception))

    def test_explicit_sizes_are_left_alone(self):
        net = quiet_nn()
        net.add(Dense(2, 5))
        net.add(Dense(5, 1))
        self.assertEqual([layer.input_size for layer in net.layers], [2, 5])


class TestRNNLayer(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        self.timesteps, self.input_dim, self.units = 4, 3, 5
        self.layer = RNN(self.units, bptt_trunc=10,
                         input_shape=(self.timesteps, self.input_dim))
        self.layer.initialize(SGD())
        self.x = np.random.randn(2, self.timesteps, self.input_dim)

    def test_initialize_allocates_the_three_shared_matrices(self):
        # U maps input -> state, W maps state -> state, V maps state -> output.
        # The same three are reused at every timestep; that sharing is what
        # makes the layer recurrent.
        self.assertEqual(self.layer.U.shape, (self.units, self.input_dim))
        self.assertEqual(self.layer.W.shape, (self.units, self.units))
        self.assertEqual(self.layer.V.shape, (self.input_dim, self.units))

    def test_forward_shape(self):
        # A sequence in, a sequence out: (batch, timesteps, input_dim).
        out = self.layer.forward(self.x)
        self.assertEqual(out.shape, (2, self.timesteps, self.input_dim))

    def test_forward_matches_the_recurrence_step_by_step(self):
        # Reference implementation of
        #     h_t = tanh(U x_t + W h_(t-1)),  out_t = V h_t
        # with h_(-1) = 0, spelled out as a plain loop.
        out = self.layer.forward(self.x)
        U, V, W = self.layer.U, self.layer.V, self.layer.W
        h = np.zeros((2, self.units))
        for t in range(self.timesteps):
            h = np.tanh(self.x[:, t].dot(U.T) + h.dot(W.T))
            np.testing.assert_allclose(out[:, t], h.dot(V.T), atol=1e-12)

    def test_initial_state_is_zero(self):
        # At t=0 there is no previous state, so the recurrence reduces to
        # h_0 = tanh(U x_0) with no W contribution.
        self.layer.forward(self.x)
        np.testing.assert_allclose(
            self.layer.state_input[:, 0],
            self.x[:, 0].dot(self.layer.U.T), atol=1e-12)

    def test_backward_shape_matches_input(self):
        self.layer.forward(self.x)
        grad = self.layer.backward(np.ones_like(self.x))
        self.assertEqual(grad.shape, self.x.shape)

    def test_backward_updates_all_three_matrices(self):
        self.layer.forward(self.x)
        before = (self.layer.U.copy(), self.layer.V.copy(), self.layer.W.copy())
        self.layer.backward(np.ones_like(self.x))
        for old, new in zip(before, (self.layer.U, self.layer.V, self.layer.W)):
            self.assertFalse(np.allclose(old, new))

    def test_str_describes_the_layer(self):
        # Every other layer defines __str__; without it NN.__str__ printed a raw
        # object repr for the RNN ("<si.supervised.nn.rnn.RNN object at 0x...>"),
        # which told the reader nothing about the network. eval6.ipynb showed
        # exactly that.
        layer = RNN(10, bptt_trunc=5, input_shape=(10, 20))
        description = str(layer)
        self.assertIn('RNN', description)
        self.assertIn('10', description)
        self.assertIn('bptt_trunc=5', description)
        self.assertNotIn('object at', description)

    def test_a_network_containing_an_rnn_prints_cleanly(self):
        net = NN(verbose=False)
        net.add(RNN(8, bptt_trunc=4, input_shape=(6, 12)))
        self.assertNotIn('object at', str(net))

    def frozen_rnn(self, timesteps, input_dim, units, bptt_trunc, seed=0):
        """Factory for identically-initialised RNNs, plus the saved matrices."""
        np.random.seed(seed)
        reference = RNN(units, bptt_trunc=bptt_trunc,
                        input_shape=(timesteps, input_dim))
        reference.initialize(SGD(learning_rate=0.0))
        saved = {name: getattr(reference, name).copy() for name in ('U', 'V', 'W')}

        def fresh(overrides=None, learning_rate=0.0):
            layer = RNN(units, bptt_trunc=bptt_trunc,
                        input_shape=(timesteps, input_dim))
            layer.initialize(SGD(learning_rate=learning_rate, momentum=0))
            for name in ('U', 'V', 'W'):
                setattr(layer, name, saved[name].copy())
            for name, value in (overrides or {}).items():
                setattr(layer, name, value.copy())
            return layer

        return fresh, saved

    def assert_weight_gradient(self, name, timesteps, input_dim=3, units=5,
                               learning_rate=0.05):
        """The gradient the layer APPLIED to U, V or W vs central differences.

        Only that the three matrices CHANGE was asserted before, which says
        nothing about whether the values are right. The weights are shared across
        every timestep, so each one's gradient is a sum over the whole sequence
        -- the part of BPTT most likely to be wrong, and the part a shape check
        cannot see.

        bptt_trunc is set to the full sequence length: truncation makes the
        gradient a deliberate approximation, so an exact comparison is only
        meaningful untruncated.
        """
        fresh, saved = self.frozen_rnn(timesteps, input_dim, units, timesteps)
        rng = np.random.RandomState(1)
        x = rng.randn(2, timesteps, input_dim)
        error = rng.randn(2, timesteps, input_dim)

        layer = fresh(learning_rate=learning_rate)
        layer.forward(x)
        before = getattr(layer, name).copy()
        layer.backward(error.copy())
        applied = (before - getattr(layer, name)) / learning_rate

        h = 1e-6
        numerical = np.zeros_like(saved[name])
        for idx in np.ndindex(saved[name].shape):
            up, down = saved[name].copy(), saved[name].copy()
            up[idx] += h
            down[idx] -= h
            numerical[idx] = (
                (fresh({name: up}).forward(x) * error).sum()
                - (fresh({name: down}).forward(x) * error).sum()) / (2 * h)

        np.testing.assert_allclose(applied, numerical, atol=1e-5)

    def test_weight_gradients_match_numerical(self):
        # U (input -> state), V (state -> output) and W (state -> state), each
        # shared across every step of the sequence.
        for name in ('U', 'V', 'W'):
            for timesteps in (2, 4, 6):
                with self.subTest(parameter=name, timesteps=timesteps):
                    self.assert_weight_gradient(name, timesteps)

    def test_truncation_makes_the_weight_gradient_approximate(self):
        # The other side of the same coin: with bptt_trunc shorter than the
        # sequence, the weight gradient is deliberately inexact. Asserting that
        # keeps the exact checks above honest -- they would otherwise pass
        # trivially if truncation were silently ignored.
        timesteps, input_dim, units = 8, 3, 5
        rng = np.random.RandomState(1)
        x = rng.randn(2, timesteps, input_dim)
        error = rng.randn(2, timesteps, input_dim)

        def applied(bptt_trunc):
            fresh, _ = self.frozen_rnn(timesteps, input_dim, units, bptt_trunc)
            layer = fresh(learning_rate=0.05)
            layer.forward(x)
            before = layer.W.copy()
            layer.backward(error.copy())
            return (before - layer.W) / 0.05

        self.assertFalse(np.allclose(applied(timesteps), applied(1), atol=1e-5))

    def test_backward_matches_numerical_gradient(self):
        # Finite-difference check on the gradient handed back to the previous
        # layer, dE/dx for E = sum(outputs). This is what caught the original
        # bug: dE/dx_t was assigned only the direct path out_t -> h_t -> x_t,
        # ignoring that x_t also reaches the loss through h_(t+1), h_(t+2), ...
        # so every timestep except the last was wrong.
        #
        # A learning rate of 0 keeps backward() from moving the weights, so the
        # analytic and numerical passes see the same parameters.
        timesteps, input_dim, units = 3, 2, 4
        np.random.seed(0)
        x = np.random.randn(2, timesteps, input_dim)

        seed_layer = RNN(units, bptt_trunc=10, input_shape=(timesteps, input_dim))
        seed_layer.initialize(SGD(learning_rate=0.0))
        U, V, W = seed_layer.U.copy(), seed_layer.V.copy(), seed_layer.W.copy()

        def fresh():
            layer = RNN(units, bptt_trunc=10,
                        input_shape=(timesteps, input_dim))
            layer.initialize(SGD(learning_rate=0.0))
            layer.U, layer.V, layer.W = U.copy(), V.copy(), W.copy()
            return layer

        layer = fresh()
        layer.forward(x)
        analytic = layer.backward(np.ones_like(x))

        h = 1e-6
        numerical = np.zeros_like(x)
        it = np.ndindex(x.shape)
        for idx in it:
            up, down = x.copy(), x.copy()
            up[idx] += h
            down[idx] -= h
            numerical[idx] = (fresh().forward(up).sum()
                              - fresh().forward(down).sum()) / (2 * h)

        np.testing.assert_allclose(analytic, numerical, atol=1e-5)

    def test_truncation_limits_how_far_the_gradient_flows(self):
        # bptt_trunc caps how many steps back the error is carried, so with a
        # window shorter than the sequence the gradient is deliberately only an
        # approximation -- it must therefore differ from the exact one. This
        # documents the trade-off rather than treating it as a defect: the
        # truncated version is cheaper and less prone to exploding gradients.
        timesteps, input_dim, units = 5, 2, 4
        np.random.seed(3)
        x = np.random.randn(1, timesteps, input_dim)

        def grad_with(bptt):
            layer = RNN(units, bptt_trunc=bptt,
                        input_shape=(timesteps, input_dim))
            layer.initialize(SGD(learning_rate=0.0))
            np.random.seed(11)
            layer.U = np.random.randn(units, input_dim) * 0.5
            layer.V = np.random.randn(input_dim, units) * 0.5
            layer.W = np.random.randn(units, units) * 0.5
            layer.forward(x)
            return layer.backward(np.ones_like(x))

        exact = grad_with(timesteps)
        truncated = grad_with(1)
        self.assertFalse(np.allclose(exact, truncated, atol=1e-5))
        # the last timestep has no future steps to truncate away, so it agrees
        np.testing.assert_allclose(exact[:, -1], truncated[:, -1], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
