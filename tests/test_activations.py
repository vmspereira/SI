# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the activation functions (si.supervised.nn.activation).
#
# An activation is a parameter-free layer: forward applies f element-wise, and
# backward multiplies the incoming error by f'(z) (the chain rule). So each one
# is checked three ways -- f at hand-computed points, f' at those same points,
# and f' against a central-difference approximation of f, which catches a wrong
# constant or a sign flip that a spot value might miss.
#
# SoftMax is the exception and is handled separately: it is not element-wise,
# and its `prime` deliberately returns only the diagonal of the Jacobian.
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.supervised.nn.activation import (
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Affine,
    Identity,
    ELU,
    Exponential,
    SELU,
    HardSigmoid,
    SoftPlus,
    SoftMax,
    Activation,
    ActivationFunction,
    functions,
)


# Element-wise activations paired with points where f is smooth, so a
# finite-difference check is valid. Kinks are avoided on purpose: ReLU and
# LeakyReLU are not differentiable at 0, and HardSigmoid is not at +/-2.5.
SMOOTH_POINTS = np.array([[-2.0, -0.7, 0.4, 1.3, 2.1]])

ELEMENTWISE = [
    (Sigmoid(), SMOOTH_POINTS),
    (ReLU(), SMOOTH_POINTS),
    (LeakyReLU(), SMOOTH_POINTS),
    (LeakyReLU(alpha=0.01), SMOOTH_POINTS),
    (Tanh(), SMOOTH_POINTS),
    (Affine(), SMOOTH_POINTS),
    (Affine(slope=2, intercept=1), SMOOTH_POINTS),
    (Identity(), SMOOTH_POINTS),
    (ELU(), SMOOTH_POINTS),
    (ELU(alpha=0.5), SMOOTH_POINTS),
    (Exponential(), SMOOTH_POINTS),
    (SELU(), SMOOTH_POINTS),
    (HardSigmoid(), SMOOTH_POINTS),
    (SoftPlus(), SMOOTH_POINTS),
]


class TestElementwiseActivationContract(unittest.TestCase):
    """Properties every element-wise activation must satisfy."""

    def test_forward_preserves_shape(self):
        for act, x in ELEMENTWISE:
            with self.subTest(activation=str(act)):
                self.assertEqual(act.forward(x).shape, x.shape)

    def test_backward_is_prime_times_the_incoming_error(self):
        # dE/dz = dE/da * f'(z), element-wise.
        for act, x in ELEMENTWISE:
            with self.subTest(activation=str(act)):
                act.forward(x)
                error = np.linspace(0.5, 2.0, x.size).reshape(x.shape)
                np.testing.assert_allclose(act.backward(error),
                                           act.prime(x) * error, rtol=1e-12)

    def test_prime_matches_numerical_derivative(self):
        # Central differences on f. This is the check that would catch a wrong
        # constant, a missing alpha, or a flipped sign in any `prime`.
        h = 1e-6
        for act, x in ELEMENTWISE:
            with self.subTest(activation=str(act)):
                numeric = (act.fn(x + h) - act.fn(x - h)) / (2 * h)
                np.testing.assert_allclose(act.prime(x), numeric,
                                           rtol=1e-5, atol=1e-7)

    def test_str_is_informative(self):
        for act, _ in ELEMENTWISE:
            with self.subTest(activation=type(act).__name__):
                self.assertTrue(str(act))
                self.assertNotEqual(str(act), "Activation")

    def test_call_promotes_a_1d_input_to_a_row(self):
        # __call__ reshapes a flat vector to (1, n) so downstream code can rely
        # on a 2-D (batch, features) layout.
        act = Sigmoid()
        out = act(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(out.shape, (1, 3))


class TestSecondDerivatives(unittest.TestCase):
    """Every element-wise activation exposes prime2, and none of it was tested.

    prime2 is unused by the library's own training loop (first-order optimizers
    only need prime), which is exactly why it could drift unnoticed. Verified
    here against a finite difference of `prime`, since f'' = d/dx f'.
    """

    def test_prime2_matches_the_derivative_of_prime(self):
        h = 1e-4
        for act, x in ELEMENTWISE:
            if not hasattr(act, 'prime2'):
                continue
            with self.subTest(activation=str(act)):
                numeric = (act.prime(x + h) - act.prime(x - h)) / (2 * h)
                np.testing.assert_allclose(act.prime2(x), numeric,
                                           rtol=1e-3, atol=1e-3)

    def test_piecewise_linear_activations_have_zero_curvature(self):
        # ReLU, LeakyReLU, Affine, Identity and HardSigmoid are straight lines
        # either side of their kinks, so away from those kinks f'' is exactly 0.
        for act in (ReLU(), LeakyReLU(), Affine(slope=3), Identity(),
                    HardSigmoid()):
            with self.subTest(activation=str(act)):
                np.testing.assert_allclose(act.prime2(SMOOTH_POINTS),
                                           np.zeros_like(SMOOTH_POINTS))

    def test_exponential_is_its_own_second_derivative(self):
        act = Exponential()
        x = np.array([[-1.0, 0.0, 1.5]])
        np.testing.assert_allclose(act.prime2(x), np.exp(x))

    def test_softplus_second_derivative_is_the_sigmoid_derivative(self):
        # d/dz sigma(z) = sigma(z)(1 - sigma(z))
        act = SoftPlus()
        x = np.array([[-1.0, 0.0, 2.0]])
        sigma = Sigmoid().fn(x)
        np.testing.assert_allclose(act.prime2(x), sigma * (1 - sigma))

    def test_softmax_has_no_second_derivative(self):
        # SoftMax is not element-wise and only defines prime (its Jacobian
        # diagonal), so it deliberately has no prime2 to check.
        self.assertFalse(hasattr(SoftMax(), 'prime2'))


class TestActivationValues(unittest.TestCase):
    """Hand-computed values, one per activation."""

    def test_leaky_relu_leaks_on_the_negative_side(self):
        # The whole point over ReLU: negatives keep a small non-zero slope, so
        # a unit cannot go permanently dead.
        act = LeakyReLU(alpha=0.3)
        np.testing.assert_allclose(act.fn(np.array([[-2.0, 3.0]])),
                                   [[-0.6, 3.0]])
        np.testing.assert_allclose(act.prime(np.array([[-2.0, 3.0]])),
                                   [[0.3, 1.0]])

    def test_leaky_relu_does_not_mutate_its_input(self):
        # fn() copies before scaling; without the copy the caller's array would
        # be modified in place.
        act = LeakyReLU(alpha=0.3)
        x = np.array([[-2.0, 3.0]])
        act.fn(x)
        np.testing.assert_allclose(x, [[-2.0, 3.0]])

    def test_affine_is_slope_times_z_plus_intercept(self):
        act = Affine(slope=2, intercept=1)
        np.testing.assert_allclose(act.fn(np.array([[0.0, 3.0]])), [[1.0, 7.0]])
        np.testing.assert_allclose(act.prime(np.array([[0.0, 3.0]])), [[2.0, 2.0]])

    def test_identity_is_affine_with_slope_one(self):
        act = Identity()
        x = np.array([[-1.0, 0.0, 2.5]])
        np.testing.assert_allclose(act.fn(x), x)
        np.testing.assert_allclose(act.prime(x), np.ones_like(x))
        self.assertIsInstance(act, Affine)

    def test_elu_is_linear_above_zero_and_saturates_below(self):
        act = ELU(alpha=1.0)
        np.testing.assert_allclose(act.fn(np.array([[2.0]])), [[2.0]])
        np.testing.assert_allclose(act.fn(np.array([[-1.0]])),
                                   [[np.exp(-1.0) - 1.0]])
        # saturates towards -alpha as z -> -inf, so outputs stay bounded below
        self.assertAlmostEqual(act.fn(np.array([[-50.0]]))[0, 0], -1.0, places=10)

    def test_exponential_is_its_own_derivative(self):
        act = Exponential()
        x = np.array([[0.0, 1.0, -2.0]])
        np.testing.assert_allclose(act.fn(x), np.exp(x))
        np.testing.assert_allclose(act.prime(x), act.fn(x))

    def test_selu_is_a_scaled_elu(self):
        act = SELU()
        x = np.array([[-1.5, 0.5, 2.0]])
        inner = ELU(alpha=act.alpha)
        np.testing.assert_allclose(act.fn(x), act.scale * inner.fn(x))
        # above zero the slope is just the scale constant
        np.testing.assert_allclose(act.prime(np.array([[2.0]])), [[act.scale]])

    def test_hard_sigmoid_clips_to_the_unit_interval(self):
        # A piecewise-linear stand-in for the sigmoid: 0.2*z + 0.5, clipped.
        act = HardSigmoid()
        np.testing.assert_allclose(act.fn(np.array([[0.0]])), [[0.5]])
        np.testing.assert_allclose(act.fn(np.array([[100.0], [-100.0]])),
                                   [[1.0], [0.0]])
        # slope 0.2 inside the linear band, 0 outside it
        np.testing.assert_allclose(act.prime(np.array([[0.0, 100.0, -100.0]])),
                                   [[0.2, 0.0, 0.0]])

    def test_softplus_derivative_is_the_sigmoid(self):
        # d/dz log(1 + e^z) = sigma(z) exactly -- a neat identity worth pinning.
        act = SoftPlus()
        x = np.array([[-1.0, 0.0, 2.0]])
        np.testing.assert_allclose(act.fn(np.array([[0.0]])), [[np.log(2.0)]])
        np.testing.assert_allclose(act.prime(x), Sigmoid().fn(x))

    def test_softplus_is_always_positive(self):
        act = SoftPlus()
        self.assertTrue(np.all(act.fn(np.array([[-10.0, 0.0, 10.0]])) > 0))

    def test_sigmoid_and_tanh_fixed_points(self):
        np.testing.assert_allclose(Sigmoid().fn(np.array([[0.0]])), [[0.5]])
        np.testing.assert_allclose(Tanh().fn(np.array([[0.0]])), [[0.0]])

    def test_relu_zeroes_negatives(self):
        act = ReLU()
        np.testing.assert_allclose(act.fn(np.array([[-3.0, 0.0, 4.0]])),
                                   [[0.0, 0.0, 4.0]])


class TestSoftMax(unittest.TestCase):
    """SoftMax is not element-wise: it normalises across the class axis."""

    def test_outputs_are_a_probability_distribution(self):
        act = SoftMax()
        out = act.fn(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(out.sum(axis=-1), [1.0, 1.0])
        self.assertTrue(np.all(out > 0))
        self.assertTrue(np.all(out < 1))

    def test_uniform_logits_give_a_uniform_distribution(self):
        out = SoftMax().fn(np.array([[5.0, 5.0, 5.0, 5.0]]))
        np.testing.assert_allclose(out, [[0.25] * 4])

    def test_invariant_to_shifting_all_logits(self):
        # softmax(z + c) == softmax(z): the constant cancels in the ratio, which
        # is exactly what licenses the max-subtraction used for stability.
        act = SoftMax()
        z = np.array([[1.0, -2.0, 0.5]])
        np.testing.assert_allclose(act.fn(z), act.fn(z + 100.0), atol=1e-12)

    def test_large_logits_do_not_overflow(self):
        # Without subtracting the row max, exp(1000) would be inf and the whole
        # row would come back NaN.
        out = SoftMax().fn(np.array([[1000.0, 999.0, -1000.0]]))
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out.sum(), 1.0)

    def test_preserves_the_ranking_of_the_logits(self):
        z = np.array([[0.1, 3.0, -1.0, 2.0]])
        out = SoftMax().fn(z)
        np.testing.assert_array_equal(np.argsort(z), np.argsort(out))

    def test_prime_returns_only_the_jacobian_diagonal(self):
        # Documented simplification: the true Jacobian is
        #   dp_i/dz_j = p_i (delta_ij - p_j),
        # a full matrix, but the element-wise backward() of ActivationFunction
        # can only carry the diagonal p_i (1 - p_i). This test records that
        # choice rather than asserting a full-Jacobian gradient, so nobody
        # "fixes" a finite-difference mismatch that is expected.
        act = SoftMax()
        z = np.array([[1.0, 2.0, 3.0]])
        p = act.fn(z)
        np.testing.assert_allclose(act.prime(z), p * (1 - p))


class TestActivationRegistry(unittest.TestCase):
    def test_every_registered_name_yields_an_activation(self):
        for name, instance in functions.items():
            with self.subTest(name=name):
                self.assertIsInstance(instance, ActivationFunction)

    def test_lookup_by_name_delegates_forward_and_backward(self):
        layer = Activation('relu')
        x = np.array([[-1.0, 2.0]])
        np.testing.assert_allclose(layer.forward(x), [[0.0, 2.0]])
        grad = layer.backward(np.ones_like(x))
        self.assertEqual(grad.shape, x.shape)

    def test_str_reports_the_wrapped_activation(self):
        self.assertEqual(str(Activation('softmax')), "SoftMax")

    def test_unknown_name_is_rejected(self):
        with self.assertRaises(ValueError):
            Activation('not-an-activation')

    def test_non_string_is_rejected(self):
        with self.assertRaises(ValueError):
            Activation(ReLU())


if __name__ == "__main__":
    unittest.main()
