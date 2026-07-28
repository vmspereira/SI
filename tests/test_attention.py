# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for scaled dot-product self-attention.
#
# The correctness evidence is the finite-difference gradient check, as with
# Conv2D and BatchNorm. Two attention-specific properties get their own tests:
# that the attention weights really are a distribution over positions, and that
# the causal mask really does prevent a position from seeing the future -- the
# latter being the property a language model's whole premise rests on.
#
# Every gradient check uses a RANDOM incoming error rather than all-ones.
# Constant errors cancel terms in softmax-like backward passes, which is how a
# BatchNorm test in this suite once passed with a gradient term deleted.
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.supervised.nn.attention import (
    SelfAttention,
    softmax,
    softmax_backward,
)
from si.supervised.nn.optimizers import SGD


def frozen_pair(d_model, d_k=None, causal=False, seed=0):
    """A factory returning identically-initialised layers with a frozen
    optimizer, so repeated forward passes are comparable."""
    np.random.seed(seed)
    reference = SelfAttention(d_model, d_k=d_k, causal=causal)
    reference.initialize(SGD(learning_rate=0.0))
    saved = [(p.weights.copy(), p.bias.copy()) for p in
             (reference.W_q, reference.W_k, reference.W_v, reference.W_o)]

    def fresh():
        layer = SelfAttention(d_model, d_k=d_k, causal=causal)
        layer.initialize(SGD(learning_rate=0.0))
        for projection, (w, b) in zip(
                (layer.W_q, layer.W_k, layer.W_v, layer.W_o), saved):
            projection.weights, projection.bias = w.copy(), b.copy()
        return layer

    return fresh


class TestSoftmaxHelpers(unittest.TestCase):
    def test_softmax_is_a_distribution_over_the_last_axis(self):
        z = np.random.RandomState(0).randn(2, 3, 4)
        p = softmax(z)
        np.testing.assert_allclose(p.sum(axis=-1), np.ones((2, 3)))
        self.assertTrue(np.all(p > 0))

    def test_softmax_does_not_overflow(self):
        p = softmax(np.array([[1000.0, 999.0, -1000.0]]))
        self.assertTrue(np.all(np.isfinite(p)))
        np.testing.assert_allclose(p.sum(axis=-1), [1.0])

    def test_softmax_is_shift_invariant(self):
        z = np.array([[1.0, -2.0, 0.5]])
        np.testing.assert_allclose(softmax(z), softmax(z + 100.0), atol=1e-12)

    def test_softmax_backward_matches_numerical_gradient(self):
        # The full-Jacobian form, p * (dp - sum(dp * p)). The off-diagonal terms
        # are what the SoftMax ACTIVATION omits, so this is the piece attention
        # could not have borrowed from there.
        rng = np.random.RandomState(0)
        z = rng.randn(2, 5)
        error = rng.randn(2, 5)
        analytic = softmax_backward(softmax(z), error)
        h = 1e-6
        numerical = np.zeros_like(z)
        for idx in np.ndindex(z.shape):
            up, down = z.copy(), z.copy()
            up[idx] += h
            down[idx] -= h
            numerical[idx] = ((softmax(up) * error).sum()
                              - (softmax(down) * error).sum()) / (2 * h)
        np.testing.assert_allclose(analytic, numerical, atol=1e-6)

    def test_softmax_backward_differs_from_the_diagonal_only_form(self):
        # Guards the choice above: if the diagonal approximation happened to
        # agree, the distinction would not be worth making. It does not.
        p = softmax(np.array([[1.0, 2.0, 3.0]]))
        error = np.array([[0.5, -1.0, 2.0]])
        diagonal_only = p * (1 - p) * error
        self.assertFalse(np.allclose(softmax_backward(p, error), diagonal_only))


class TestSelfAttentionShapes(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.layer = SelfAttention(6)
        self.layer.initialize(SGD())
        self.x = np.random.randn(2, 4, 6)

    def test_output_has_the_input_shape(self):
        self.assertEqual(self.layer.forward(self.x).shape, (2, 4, 6))

    def test_d_k_may_differ_from_d_model(self):
        layer = SelfAttention(6, d_k=3)
        layer.initialize(SGD())
        # the internal width shrinks but the output projection restores d_model
        self.assertEqual(layer.forward(self.x).shape, (2, 4, 6))

    def test_attention_weights_are_a_distribution_over_positions(self):
        self.layer.forward(self.x)
        # one (seq_len, seq_len) matrix per example, every row summing to 1
        self.assertEqual(self.layer.weights.shape, (2, 4, 4))
        np.testing.assert_allclose(self.layer.weights.sum(axis=-1),
                                   np.ones((2, 4)))

    def test_backward_returns_the_input_shape(self):
        out = self.layer.forward(self.x)
        self.assertEqual(self.layer.backward(np.random.randn(*out.shape)).shape,
                         self.x.shape)

    def test_backward_updates_every_projection(self):
        before = [p.weights.copy() for p in
                  (self.layer.W_q, self.layer.W_k, self.layer.W_v, self.layer.W_o)]
        out = self.layer.forward(self.x)
        self.layer.backward(np.random.randn(*out.shape))
        after = [p.weights for p in
                 (self.layer.W_q, self.layer.W_k, self.layer.W_v, self.layer.W_o)]
        for name, old, new in zip('qkvo', before, after):
            with self.subTest(projection=name):
                self.assertFalse(np.allclose(old, new))

    def test_wrong_rank_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.layer.forward(np.random.randn(4, 6))

    def test_wrong_d_model_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.layer.forward(np.random.randn(2, 4, 7))
        self.assertIn('d_model', str(ctx.exception))

    def test_invalid_sizes_are_rejected(self):
        for kwargs in ({'d_model': 0}, {'d_model': 4, 'd_k': 0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SelfAttention(**kwargs)

    def test_large_inputs_do_not_overflow(self):
        out = self.layer.forward(np.random.randn(1, 4, 6) * 500)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.isfinite(self.layer.weights).all())


class TestSelfAttentionGradient(unittest.TestCase):
    def analytic_vs_numerical(self, shape, d_k=None, causal=False, seed=0):
        fresh = frozen_pair(shape[2], d_k=d_k, causal=causal, seed=seed)
        rng = np.random.RandomState(seed + 100)
        x = rng.randn(*shape)
        error = rng.randn(*shape)

        layer = fresh()
        layer.forward(x)
        analytic = layer.backward(error.copy())

        h = 1e-6
        numerical = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            up, down = x.copy(), x.copy()
            up[idx] += h
            down[idx] -= h
            numerical[idx] = ((fresh().forward(up) * error).sum()
                              - (fresh().forward(down) * error).sum()) / (2 * h)
        return analytic, numerical

    def test_gradient_matches_numerical(self):
        # Q, K and V are three projections of the SAME input, so its gradient is
        # the sum of three paths -- plus a fourth through the softmax. Dropping
        # any one of them still produces plausible-looking numbers, which is why
        # this check matters more than a shape assertion.
        for causal in (False, True):
            for shape, d_k in [((2, 4, 6), None), ((1, 6, 4), 3), ((3, 3, 5), 8)]:
                with self.subTest(causal=causal, shape=shape, d_k=d_k):
                    analytic, numerical = self.analytic_vs_numerical(
                        shape, d_k=d_k, causal=causal)
                    np.testing.assert_allclose(analytic, numerical, atol=1e-5)


class TestCausalMasking(unittest.TestCase):
    """The property a language model depends on: no peeking ahead."""

    def setUp(self):
        np.random.seed(1)
        self.x = np.random.randn(1, 5, 6)

    def test_future_tokens_cannot_change_earlier_outputs(self):
        # Perturbing the LAST token must leave every earlier position identical.
        # This is the test that actually proves the mask works -- checking that
        # some weights are zero would only show the mask was applied somewhere.
        layer = SelfAttention(6, causal=True)
        layer.initialize(SGD(learning_rate=0.0))
        base = layer.forward(self.x)
        disturbed = self.x.copy()
        disturbed[0, -1, :] += 100.0
        after = layer.forward(disturbed)
        for position in range(self.x.shape[1] - 1):
            with self.subTest(position=position):
                np.testing.assert_allclose(base[0, position], after[0, position],
                                           atol=1e-10)

    def test_without_the_mask_the_future_does_leak(self):
        # Control. If this passed too, the test above would prove nothing.
        layer = SelfAttention(6, causal=False)
        layer.initialize(SGD(learning_rate=0.0))
        base = layer.forward(self.x)
        disturbed = self.x.copy()
        disturbed[0, -1, :] += 100.0
        after = layer.forward(disturbed)
        self.assertFalse(np.allclose(base[0, 0], after[0, 0], atol=1e-10))

    def test_forbidden_weights_are_exactly_zero(self):
        layer = SelfAttention(6, causal=True)
        layer.initialize(SGD())
        layer.forward(self.x)
        # strictly above the diagonal = the future
        np.testing.assert_allclose(np.triu(layer.weights[0], k=1), 0.0)
        # and the allowed part still forms a distribution
        np.testing.assert_allclose(layer.weights.sum(axis=-1), np.ones((1, 5)))

    def test_the_first_position_attends_only_to_itself(self):
        layer = SelfAttention(6, causal=True)
        layer.initialize(SGD())
        layer.forward(self.x)
        self.assertAlmostEqual(layer.weights[0, 0, 0], 1.0)

    def test_masking_produces_no_nan_in_the_backward_pass(self):
        # The mask writes -inf into the scores, so a careless backward pass can
        # produce 0 * inf = NaN.
        layer = SelfAttention(6, causal=True)
        layer.initialize(SGD())
        out = layer.forward(self.x)
        grad = layer.backward(np.random.randn(*out.shape))
        self.assertTrue(np.isfinite(grad).all())


if __name__ == "__main__":
    unittest.main()
