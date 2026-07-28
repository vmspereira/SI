# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the optimizers (si.supervised.nn.optimizers).
#
# An optimizer answers one question: given a parameter and the gradient of the
# loss w.r.t. it, how should the parameter change? Each one below is a different
# recipe, so each is pinned to the arithmetic of its own update rule, and all of
# them are checked against the one contract they share -- that a layer can call
# `update(w, grad)` and end up closer to a minimum.
#
# TestAdam lives here too (moved out of test_layers.py, which covers layers).
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.supervised.nn.layers import Dense
from si.supervised.nn.optimizers import (
    Optimizer,
    SGD,
    Adam,
    NesterovAcceleratedGradient,
    Adagrad,
    Adadelta,
    RMSprop,
)


def all_optimizers():
    """A fresh instance of every optimizer, with a usable learning rate."""
    return [
        SGD(learning_rate=0.1),
        SGD(learning_rate=0.1, momentum=0.9),
        Adam(learning_rate=0.1),
        NesterovAcceleratedGradient(learning_rate=0.1, momentum=0.5),
        Adagrad(learning_rate=0.1),
        Adadelta(),
        RMSprop(learning_rate=0.1),
    ]


def descend_quadratic(opt, start=0.0, target=3.0, steps=3000):
    """Minimise f(w) = (w - target)^2, whose gradient is 2*(w - target)."""
    w = np.array([float(start)])
    for _ in range(steps):
        w = opt.update(w, 2.0 * (w - target))
    return w[0]


class TestOptimizerContract(unittest.TestCase):
    """Every optimizer must be usable interchangeably by any layer."""

    def test_all_subclass_the_optimizer_interface(self):
        for opt in all_optimizers():
            self.assertIsInstance(opt, Optimizer)

    def test_all_can_drive_a_dense_layer(self):
        # Regression guard. NesterovAcceleratedGradient used to fail here with
        #   ValueError: operands could not be broadcast together
        #               with shapes (3,2) (0,)
        # because it read its velocity buffer before allocating it, and because
        # its update() expected a callable gradient FUNCTION while every layer
        # passes a gradient ARRAY. It was the only optimizer in the module that
        # no layer could actually use.
        for opt in all_optimizers():
            with self.subTest(optimizer=type(opt).__name__):
                np.random.seed(0)
                layer = Dense(3, 2)
                layer.initialize(opt)
                x = np.random.rand(4, 3)
                layer.forward(x)
                before = layer.weights.copy()
                layer.backward(np.ones((4, 2)))
                self.assertFalse(np.allclose(before, layer.weights),
                                 "weights did not move")

    def test_all_step_against_the_gradient(self):
        # A positive gradient must decrease the parameter, for every recipe.
        for opt in all_optimizers():
            with self.subTest(optimizer=type(opt).__name__):
                w = opt.update(np.array([0.0]), np.array([1.0]))
                self.assertLess(w[0], 0.0)

    def test_all_minimize_a_quadratic(self):
        # The end-to-end contract: whatever the recipe, iterating it should
        # reach the minimum of a simple convex function.
        for opt in all_optimizers():
            with self.subTest(optimizer=type(opt).__name__):
                # Adadelta has no learning rate and takes very small steps, so
                # it needs a longer run to cover the same distance.
                steps = 60000 if isinstance(opt, Adadelta) else 3000
                found = descend_quadratic(opt, steps=steps)
                self.assertAlmostEqual(found, 3.0, places=2)

    def test_all_preserve_parameter_shape(self):
        for opt in all_optimizers():
            with self.subTest(optimizer=type(opt).__name__):
                w = np.zeros((2, 3))
                out = opt.update(w, np.ones((2, 3)))
                self.assertEqual(out.shape, (2, 3))


class TestSGD(unittest.TestCase):
    def test_plain_gradient_descent_without_momentum(self):
        # momentum=0 leaves the textbook rule: w <- w - lr * g
        opt = SGD(learning_rate=0.1)
        w = opt.update(np.array([1.0]), np.array([2.0]))
        self.assertAlmostEqual(w[0], 1.0 - 0.1 * 2.0)

    def test_momentum_blends_the_previous_velocity(self):
        # This SGD smooths with (1 - momentum) weighting on the new gradient:
        #   v <- momentum * v + (1 - momentum) * g
        # so the first step of a constant gradient is only lr * (1-m) * g, and
        # the steps grow towards lr * g as v warms up to g.
        opt = SGD(learning_rate=0.1, momentum=0.9)
        w = np.array([0.0])
        first = w[0] - opt.update(w, np.array([1.0]))[0]
        self.assertAlmostEqual(first, 0.1 * 0.1)
        w = np.array([0.0])
        opt2 = SGD(learning_rate=0.1, momentum=0.9)
        steps = []
        for _ in range(4):
            prev = w[0]
            w = opt2.update(w, np.array([1.0]))
            steps.append(prev - w[0])
        self.assertTrue(all(b > a for a, b in zip(steps, steps[1:])),
                        f"steps should grow towards lr: {steps}")


class TestNesterovAcceleratedGradient(unittest.TestCase):
    """The reformulated update, stepping with a momentum-corrected gradient:

        v <- momentum * v + g
        w <- w - lr * (g + momentum * v)
    """

    def test_first_two_steps_match_the_update_rule(self):
        opt = NesterovAcceleratedGradient(learning_rate=0.1, momentum=0.5)
        w = np.array([0.0])
        w = opt.update(w, np.array([1.0]))
        # v = 0.5*0 + 1 = 1 ;  w = 0 - 0.1*(1 + 0.5*1) = -0.15
        self.assertAlmostEqual(opt.w_updt[0], 1.0)
        self.assertAlmostEqual(w[0], -0.15)
        w = opt.update(w, np.array([1.0]))
        # v = 0.5*1 + 1 = 1.5 ; w = -0.15 - 0.1*(1 + 0.5*1.5) = -0.325
        self.assertAlmostEqual(opt.w_updt[0], 1.5)
        self.assertAlmostEqual(w[0], -0.325)

    def test_momentum_zero_reduces_to_gradient_descent(self):
        # With no momentum the look-ahead correction vanishes and the rule must
        # collapse to w <- w - lr * g.
        opt = NesterovAcceleratedGradient(learning_rate=0.1, momentum=0.0)
        w = opt.update(np.array([0.0]), np.array([2.0]))
        self.assertAlmostEqual(w[0], -0.2)

    def test_velocity_is_allocated_on_first_use(self):
        # The buffer starts as None and is shaped from the parameter, which is
        # what the empty-array initialisation got wrong.
        opt = NesterovAcceleratedGradient()
        self.assertIsNone(opt.w_updt)
        opt.update(np.zeros((2, 3)), np.ones((2, 3)))
        self.assertEqual(opt.w_updt.shape, (2, 3))

    def test_accepts_a_gradient_array_not_a_callable(self):
        # The old signature was update(w, grad_func) and called grad_func(...),
        # so passing the array every layer passes raised a TypeError/ValueError.
        opt = NesterovAcceleratedGradient(learning_rate=0.1)
        out = opt.update(np.zeros((2, 2)), np.ones((2, 2)))
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(np.all(out < 0))


class TestAdagrad(unittest.TestCase):
    def test_step_shrinks_as_the_inverse_square_root_of_time(self):
        # Adagrad accumulates the SUM of squared gradients, which only grows, so
        # the effective learning rate decays monotonically. For a constant
        # gradient g=1 the accumulator is exactly t after t steps, making the
        # step lr/sqrt(t) -- the property that makes Adagrad good for sparse
        # features and prone to stalling on long runs.
        opt = Adagrad(learning_rate=0.1)
        w = np.array([0.0])
        for t in range(1, 6):
            prev = w[0]
            w = opt.update(w, np.array([1.0]))
            self.assertAlmostEqual(prev - w[0], 0.1 / np.sqrt(t), places=6)

    def test_accumulator_grows_monotonically(self):
        opt = Adagrad(learning_rate=0.1)
        w = np.array([0.0])
        seen = []
        for _ in range(4):
            w = opt.update(w, np.array([2.0]))
            seen.append(opt.G[0])
        self.assertTrue(all(b > a for a, b in zip(seen, seen[1:])))


class TestRMSprop(unittest.TestCase):
    def test_first_step_uses_the_decayed_average(self):
        # RMSprop replaces Adagrad's ever-growing sum with a moving average, so
        # it does not stall. After one step of g=1 the average is (1-rho)=0.1,
        # making the step lr/sqrt(0.1).
        opt = RMSprop(learning_rate=0.1, rho=0.9)
        w = np.array([0.0])
        step = w[0] - opt.update(w, np.array([1.0]))[0]
        self.assertAlmostEqual(step, 0.1 / np.sqrt(0.1), places=5)

    def test_steps_settle_towards_the_learning_rate(self):
        # For a constant gradient the moving average converges to g^2 = 1, so
        # the step converges down to lr -- unlike Adagrad, which keeps shrinking.
        opt = RMSprop(learning_rate=0.1, rho=0.9)
        w = np.array([0.0])
        steps = []
        for _ in range(200):
            prev = w[0]
            w = opt.update(w, np.array([1.0]))
            steps.append(prev - w[0])
        self.assertTrue(all(b < a for a, b in zip(steps[:20], steps[1:21])))
        self.assertAlmostEqual(steps[-1], 0.1, places=3)


class TestAdadelta(unittest.TestCase):
    def test_needs_no_learning_rate(self):
        # Adadelta's step is a ratio of two running averages, so it carries its
        # own units and takes no learning-rate argument at all.
        opt = Adadelta()
        self.assertFalse(hasattr(opt, 'learning_rate'))

    def test_first_step_matches_the_ratio_of_running_averages(self):
        # On the first update E[dw^2] is still 0, so
        #   step = sqrt(eps) / sqrt((1-rho)*g^2 + eps) * g
        opt = Adadelta(rho=0.95, eps=1e-6)
        w = np.array([0.0])
        step = w[0] - opt.update(w, np.array([1.0]))[0]
        expected = np.sqrt(1e-6) / np.sqrt(0.05 * 1.0 + 1e-6)
        self.assertAlmostEqual(step, expected, places=8)

    def test_tracks_both_running_averages(self):
        opt = Adadelta()
        opt.update(np.array([0.0]), np.array([1.0]))
        # the gradient average is live after one step, and the update average
        # starts accumulating from the step just taken
        self.assertGreater(opt.E_grad[0], 0.0)
        self.assertGreater(opt.E_w_updt[0], 0.0)


class TestAdam(unittest.TestCase):
    """Pins down Adam's bias correction.

    Adam keeps two exponential moving averages, both initialised at zero:

        m_t = b1*m_(t-1) + (1-b1)*g      v_t = b2*v_(t-1) + (1-b2)*g^2

    Starting at zero biases them low, badly so on the first few steps: with the
    default b1=0.9 the raw m_1 is only 10% of the gradient. Dividing by
    (1 - b^t) removes exactly that bias, and the correction fades as t grows.

    The observable consequence, and what these tests check, is that the FIRST
    step has magnitude `learning_rate` no matter how large the gradient is:

        m_hat = (1-b1)g / (1-b1) = g        v_hat = (1-b2)g^2 / (1-b2) = g^2
        step  = lr * g / (sqrt(g^2) + eps) ~= lr

    Drop the correction and the same step becomes lr*(1-b1)*g/sqrt((1-b2)*g^2)
    = lr*3.16 -- a silent 3x overshoot on every early step, which is the bug
    these tests guard.
    """

    def test_first_step_size_is_the_learning_rate(self):
        opt = Adam(learning_rate=0.01)
        w = opt.update(np.array([0.0]), np.array([1.0]))
        # Without bias correction this would be 0.0316, so the tolerance here
        # is what makes the test discriminating.
        self.assertAlmostEqual(-w[0], 0.01, places=6)

    def test_step_size_is_gradient_scale_invariant(self):
        # m_hat/sqrt(v_hat) ~= g/|g| = sign(g), so the gradient magnitude
        # cancels out and only the learning rate sets the step length. This is
        # the property that makes Adam insensitive to gradient scaling.
        for grad in (1e-3, 1.0, 1e3):
            opt = Adam(learning_rate=0.01)
            w = opt.update(np.array([0.0]), np.array([grad]))
            self.assertAlmostEqual(-w[0], 0.01, places=5,
                                   msg=f"step changed at gradient {grad}")

    def test_sign_follows_the_gradient(self):
        # Descent direction: a positive gradient must decrease w.
        self.assertLess(Adam(0.01).update(np.array([0.0]), np.array([5.0]))[0], 0)
        self.assertGreater(Adam(0.01).update(np.array([0.0]), np.array([-5.0]))[0], 0)

    def test_constant_gradient_gives_constant_steps(self):
        # For a constant gradient the corrected moments are exactly g and g^2
        # at EVERY t (the (1-b^t) factors cancel the partial EMA sums), so 50
        # steps move w by exactly 50*lr. A t-dependent error in the correction
        # would make the steps drift instead.
        opt = Adam(learning_rate=0.01)
        w = np.array([0.0])
        for _ in range(50):
            w = opt.update(w, np.array([1.0]))
        self.assertAlmostEqual(w[0], -0.5, places=6)
        # the time step must track the number of updates, since the correction
        # factor (1 - b**t) depends on it
        self.assertEqual(opt.t, 50)

    def test_minimizes_a_quadratic(self):
        # End-to-end sanity: descend f(w) = (w-3)^2, whose gradient is
        # 2*(w-3), and land on the minimum at w=3.
        opt = Adam(learning_rate=0.1)
        w = np.array([0.0])
        for _ in range(500):
            w = opt.update(w, 2.0 * (w - 3.0))
        self.assertAlmostEqual(w[0], 3.0, places=4)

    def test_updates_arrays_elementwise(self):
        # Weight matrices are updated as a whole: each entry gets its own
        # adaptive step, and the returned array keeps the parameter's shape.
        opt = Adam(learning_rate=0.01)
        w = np.zeros((2, 3))
        grad = np.array([[1.0, -2.0, 100.0], [-0.5, 3.0, -1e4]])
        out = opt.update(w, grad)
        self.assertEqual(out.shape, (2, 3))
        # first step: every entry moves by lr against its gradient's sign
        np.testing.assert_allclose(out, -0.01 * np.sign(grad), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
