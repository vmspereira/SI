# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the scoring functions and loss functions in si.util.metrics.
#
# Two jobs here. First, pin each metric to a value worked out by hand, so the
# formulas cannot drift. Second, guard the numerical edge cases: a log of zero,
# an exponential of a large logit, and a zero-variance target all used to
# produce inf / -inf and silently poison everything computed from them.
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.util.metrics import (
    accuracy_score,
    multiclass_accuracy,
    confusion_matrix,
    mae,
    mae_prime,
    mse,
    mse_prime,
    rmse,
    cross_entropy,
    cross_entropy_prime,
    softmax_cross_entropy,
    softmax_cross_entropy_prime,
    r2_score,
)


class TestClassificationMetrics(unittest.TestCase):
    def test_accuracy_score(self):
        # 3 of 4 labels match -> 0.75
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.75)

    def test_accuracy_score_perfect_and_zero(self):
        y = np.array([1, 0, 1])
        self.assertAlmostEqual(accuracy_score(y, y), 1.0)
        self.assertAlmostEqual(accuracy_score(y, 1 - y), 0.0)

    def test_multiclass_accuracy_compares_argmax(self):
        # One-hot / probability rows are reduced with argmax before comparing,
        # so the winning class is what matters, not the exact probabilities.
        y_true = np.array([[1., 0., 0.], [0., 0., 1.]])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        self.assertAlmostEqual(multiclass_accuracy(y_true, y_pred), 0.5)

    def test_confusion_matrix_counts(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        # rows = truth, columns = prediction
        self.assertEqual(cm.loc[0, 0], 1)   # true 0 predicted 0
        self.assertEqual(cm.loc[0, 1], 1)   # true 0 predicted 1
        self.assertEqual(cm.loc[1, 1], 2)   # true 1 predicted 1

    def test_confusion_matrix_diagonal_holds_the_correct_predictions(self):
        # What a confusion matrix adds over plain accuracy: it separates the
        # kinds of mistake. Here every error is a false positive (true 0
        # predicted 1) and there are no false negatives.
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.loc[0, 1], 2)          # false positives
        self.assertEqual(cm.loc[1].get(0, 0), 0)   # no false negatives
        # the diagonal sums to the number of correct predictions
        self.assertEqual(cm.loc[0, 0] + cm.loc[1, 1], 3)

    def test_confusion_matrix_output_format(self):
        # The parameter is `output_format`; it used to be named `format`, which
        # shadowed the builtin inside the function body.
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        import pandas as pd

        self.assertIsInstance(confusion_matrix(y_true, y_pred), pd.DataFrame)
        self.assertIsInstance(
            confusion_matrix(y_true, y_pred, output_format='df'), pd.DataFrame)
        # any other value returns the crosstab unchanged
        raw = confusion_matrix(y_true, y_pred, output_format='raw')
        self.assertEqual(raw.shape, (2, 2))


class TestRegressionMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0])
        self.y_pred = np.array([1.5, 2.0, 2.0, 5.0])
        # errors: -0.5, 0, +1, -1  -> squared 0.25, 0, 1, 1

    def test_mse(self):
        self.assertAlmostEqual(mse(self.y_true, self.y_pred), 2.25 / 4)

    def test_mae(self):
        self.assertAlmostEqual(mae(self.y_true, self.y_pred), 2.5 / 4)

    def test_rmse_is_root_of_mse(self):
        self.assertAlmostEqual(rmse(self.y_true, self.y_pred),
                               np.sqrt(mse(self.y_true, self.y_pred)))

    def test_mse_prime_matches_numerical_gradient(self):
        # d/dy_pred of mean((y_true - y_pred)^2) checked by central differences,
        # which is the cheapest way to catch a wrong constant or a sign flip.
        h = 1e-6
        analytic = mse_prime(self.y_true, self.y_pred)
        for i in range(len(self.y_pred)):
            up, down = self.y_pred.copy(), self.y_pred.copy()
            up[i] += h
            down[i] -= h
            numeric = (mse(self.y_true, up) - mse(self.y_true, down)) / (2 * h)
            self.assertAlmostEqual(analytic[i], numeric, places=6)

    def test_mae_prime_is_the_scaled_sign(self):
        # The MAE derivative is a step function: it only carries the direction
        # of the error, never its size. It is 0 exactly where the error is 0.
        grad = mae_prime(self.y_true, self.y_pred)
        m = len(self.y_pred)
        # the errors (y_true - y_pred) are [-0.5, 0, +1, -1], so the derivative
        # is +1/m where the error is negative, -1/m where it is positive, and
        # exactly 0 where the prediction is spot on
        np.testing.assert_allclose(grad, np.array([1, 0, -1, 1]) / m)


class TestR2Score(unittest.TestCase):
    def test_known_value(self):
        # Same example (and value) as the reference implementation in
        # scikit-learn's documentation.
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.9486081370449679)

    def test_perfect_prediction_is_one(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r2_score(y, y), 1.0)

    def test_constant_target_does_not_return_minus_inf(self):
        # A constant y_true has zero variance, so SS_tot = 0. The unguarded
        # division returned -inf here, which then propagated into every score
        # computed from it. Convention: no variance to explain -> 0.0.
        y_true = np.array([2.0, 2.0, 2.0])
        score = r2_score(y_true, np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.isfinite(score))
        self.assertAlmostEqual(score, 0.0)

    def test_constant_target_predicted_exactly_is_one(self):
        y_true = np.array([2.0, 2.0, 2.0])
        self.assertAlmostEqual(r2_score(y_true, y_true), 1.0)

    def test_multioutput_scores_each_column(self):
        # Column 1 is constant AND predicted exactly -> 1.0 by the convention
        # above; column 0 is scored normally.
        y_true = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
        y_pred = np.array([[1.1, 2.0], [2.9, 2.0], [5.0, 2.0]])
        scores = r2_score(y_true, y_pred)
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(scores[1], 1.0)
        self.assertGreater(scores[0], 0.99)

    def test_single_output_returns_a_scalar(self):
        score = r2_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.1, 2.9]))
        self.assertIsInstance(score, float)


class TestCrossEntropy(unittest.TestCase):
    def test_known_value(self):
        y_true = np.array([[1., 0.], [0., 1.]])
        y_pred = np.array([[0.7, 0.3], [0.2, 0.8]])
        expected = -(np.log(0.7) + np.log(0.8)) / 2
        self.assertAlmostEqual(cross_entropy(y_true, y_pred), expected)

    def test_zero_probability_is_clipped_not_infinite(self):
        # A confidently wrong prediction puts probability 0 on the true class.
        # log(0) = -inf would make the loss inf and every gradient derived from
        # it NaN, so the probabilities are clipped first.
        loss = cross_entropy(np.array([[1., 0.]]), np.array([[0., 1.]]))
        self.assertTrue(np.isfinite(loss))
        # the cap is -log(eps) for the default eps=1e-15
        self.assertAlmostEqual(loss, -np.log(1e-15), places=6)

    def test_clip_bound_is_configurable(self):
        loss = cross_entropy(np.array([[1., 0.]]), np.array([[0., 1.]]), eps=1e-7)
        self.assertAlmostEqual(loss, -np.log(1e-7), places=6)

    def test_confident_and_correct_is_near_zero(self):
        loss = cross_entropy(np.array([[1., 0.]]), np.array([[1., 0.]]))
        self.assertLess(loss, 1e-10)

    def test_prime_shape_and_sign(self):
        y_true = np.array([[1., 0.]])
        y_pred = np.array([[0.6, 0.4]])
        grad = cross_entropy_prime(y_true, y_pred)
        self.assertEqual(grad.shape, y_pred.shape)
        # under-predicting the true class -> negative gradient on that entry,
        # so a gradient step pushes its probability up
        self.assertLess(grad[0, 0], 0)


class TestSoftmaxCrossEntropy(unittest.TestCase):
    """Argument order is (y_true, logits) and the result is a scalar mean.

    It used to be declared (logits, y_true) and to return one value per sample.
    Both were wrong for the only consumer that matters: `NN.fit` calls
    `self.loss(y_true, y_pred)` and stores the result as a scalar, so
    `NN(loss="softmax-cross-entropy")` passed its arguments backwards and could
    not train. Every other loss in this module already used (y_true, y_pred) and
    reduced to a scalar.
    """

    def test_matches_the_naive_formula_on_safe_logits(self):
        # Where the direct computation does not overflow, the stabilised
        # version must agree with it exactly -- the shift is an identity, not
        # an approximation.
        logits = np.array([[2., 1., 0.], [0., 1., 2.]])
        y_true = np.array([0, 2])
        naive = np.mean(-logits[np.arange(2), y_true]
                        + np.log(np.exp(logits).sum(axis=-1)))
        self.assertAlmostEqual(softmax_cross_entropy(y_true, logits), naive)

    def test_returns_a_scalar_mean(self):
        logits = np.array([[2., 1., 0.], [0., 1., 2.]])
        loss = softmax_cross_entropy(np.array([0, 2]), logits)
        self.assertIsInstance(loss, float)
        self.assertEqual(np.ndim(loss), 0)

    def test_large_logits_do_not_overflow(self):
        # exp(1000) is inf in float64. Subtracting the row maximum keeps every
        # exponent <= 0, so the loss stays finite. The true class here IS the
        # largest logit, so the loss should be ~0.
        loss = softmax_cross_entropy(np.array([0]), np.array([[1000., 1.]]))
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_large_logits_wrong_class_is_large_but_finite(self):
        loss = softmax_cross_entropy(np.array([1]), np.array([[1000., 1.]]))
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(loss, 999.0, places=3)

    def test_prime_does_not_overflow_and_sums_to_zero(self):
        # softmax probabilities sum to 1 and the one-hot target sums to 1, so
        # the gradient w.r.t. the logits must sum to 0 across the class axis.
        grad = softmax_cross_entropy_prime(np.array([1]),
                                           np.array([[1000., 1., -1000.]]))
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertAlmostEqual(grad.sum(), 0.0, places=10)

    def test_prime_matches_naive_formula_on_safe_logits(self):
        logits = np.array([[2., 1., 0.], [0., 1., 2.]])
        y_true = np.array([0, 2])
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        onehot = np.zeros_like(logits)
        onehot[np.arange(2), y_true] = 1
        np.testing.assert_allclose(
            softmax_cross_entropy_prime(y_true, logits),
            (-onehot + softmax) / logits.shape[0])

    def test_prime_matches_numerical_gradient(self):
        # Central differences on the scalar loss. The 1/rows factor in the
        # derivative is exactly what makes it the gradient of the MEAN, so this
        # would fail if the two reductions disagreed.
        logits = np.array([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])
        y_true = np.array([2, 0])
        analytic = softmax_cross_entropy_prime(y_true, logits)
        h = 1e-6
        for idx in np.ndindex(logits.shape):
            up, down = logits.copy(), logits.copy()
            up[idx] += h
            down[idx] -= h
            numeric = (softmax_cross_entropy(y_true, up)
                       - softmax_cross_entropy(y_true, down)) / (2 * h)
            self.assertAlmostEqual(analytic[idx], numeric, places=8)


class TestSoftmaxCrossEntropyOnSequences(unittest.TestCase):
    """A language model scores (batch, seq_len, vocab) with one label per
    position. Everything in front of the class axis is flattened away, so the
    same loss serves both a plain classifier and a sequence model."""

    def setUp(self):
        np.random.seed(0)
        self.logits = np.random.randn(2, 4, 5)
        self.y_true = np.random.randint(0, 5, (2, 4))

    def test_scores_every_position(self):
        loss = softmax_cross_entropy(self.y_true, self.logits)
        self.assertTrue(np.isfinite(loss))
        # equals the mean over the same positions scored flat
        flat = softmax_cross_entropy(self.y_true.reshape(-1),
                                     self.logits.reshape(-1, 5))
        self.assertAlmostEqual(loss, flat)

    def test_gradient_has_the_shape_of_the_logits(self):
        grad = softmax_cross_entropy_prime(self.y_true, self.logits)
        self.assertEqual(grad.shape, self.logits.shape)

    def test_gradient_matches_numerical(self):
        analytic = softmax_cross_entropy_prime(self.y_true, self.logits)
        h = 1e-6
        numerical = np.zeros_like(self.logits)
        for idx in np.ndindex(self.logits.shape):
            up, down = self.logits.copy(), self.logits.copy()
            up[idx] += h
            down[idx] -= h
            numerical[idx] = (softmax_cross_entropy(self.y_true, up)
                              - softmax_cross_entropy(self.y_true, down)) / (2 * h)
        np.testing.assert_allclose(analytic, numerical, atol=1e-8)

    def test_one_label_per_position_is_required(self):
        with self.assertRaises(ValueError):
            softmax_cross_entropy(np.array([0, 1, 2]),
                                  np.array([[1., 2.], [3., 4.]]))


if __name__ == "__main__":
    unittest.main()
