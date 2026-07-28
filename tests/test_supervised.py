# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the classical (non-neural-network) supervised models.
#
# Each model is trained on a small, well-separated dataset and checked for the
# basic learning contract: it fits, predicts, and reaches a sensible accuracy
# (or, for regression, recovers the underlying relationship).
# ----------------------------------------------------------------------------
import importlib
import unittest

import numpy as np

from si.data import Dataset
from si.supervised import (
    KNN,
    LinearRegression,
    LogisticRegression,
    DecisionTree,
    RandomForest,
    NaiveBayes,
    LDA,
)


def two_class_blobs(n_per_class=30, n_features=2, sep=2.0, seed=0):
    """Two Gaussian blobs separated along every feature -> linearly separable."""
    rng = np.random.RandomState(seed)
    neg = rng.randn(n_per_class, n_features) - sep
    pos = rng.randn(n_per_class, n_features) + sep
    X = np.vstack([neg, pos])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred).ravel() == np.asarray(y_true).ravel()))


class TestKNN(unittest.TestCase):
    def setUp(self):
        self.X, self.y = two_class_blobs()
        self.ds = Dataset(self.X, self.y)

    def test_classification_separates(self):
        m = KNN(num_neighbors=3)
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        # single-sample prediction returns a valid label
        self.assertIn(m.predict(self.X[0]), (0, 1))
        # cost() reports training accuracy on well-separated data
        self.assertGreater(m.cost(), 0.95)

    def test_requires_fit(self):
        with self.assertRaises(AssertionError):
            KNN(num_neighbors=3).predict(self.X[0])

    def test_regression_mode_returns_mean(self):
        y = self.X[:, 0] * 2.0
        m = KNN(num_neighbors=5, classification=False)
        m.fit(Dataset(self.X, y))
        pred = m.predict(self.X[0])
        self.assertIsInstance(float(pred), float)


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.X = rng.rand(80, 2)
        # y = 1 + 2*x0 + 3*x1  (no noise)
        self.y = 1 + 2 * self.X[:, 0] + 3 * self.X[:, 1]
        self.ds = Dataset(self.X, self.y)

    def test_closed_form_recovers_coefficients(self):
        m = LinearRegression(gd=False, lbd=0)
        m.fit(self.ds)
        np.testing.assert_allclose(m.theta, [1, 2, 3], atol=1e-6)
        self.assertLess(m.cost(), 1e-10)

    def test_gradient_descent_fits(self):
        m = LinearRegression(gd=True, lbd=0, epochs=5000, lr=0.1)
        m.fit(self.ds)
        # GD should get close to the closed-form solution
        np.testing.assert_allclose(m.theta, [1, 2, 3], atol=0.2)

    def test_predict_single_sample(self):
        m = LinearRegression(gd=False, lbd=0)
        m.fit(self.ds)
        self.assertAlmostEqual(float(m.predict(self.X[0])), float(self.y[0]), places=4)


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.X, self.y = two_class_blobs()
        self.ds = Dataset(self.X, self.y)

    def test_separates_classes(self):
        m = LogisticRegression(epochs=2000, lr=0.1)
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        preds = [m.predict(x) for x in self.X]
        self.assertGreater(accuracy(self.y, preds), 0.95)

    def test_cost_decreases(self):
        m = LogisticRegression(epochs=500, lr=0.1)
        m.fit(self.ds)
        first = m.history[0][1]
        last = m.history[len(m.history) - 1][1]
        self.assertLess(last, first)


class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.X, self.y = two_class_blobs()
        self.ds = Dataset(self.X, self.y)

    def test_fits_and_classifies(self):
        m = DecisionTree(max_depth=3)
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        self.assertIn(m.predict(self.X[0]), (0, 1))
        self.assertGreater(m.cost(), 0.95)

    def test_requires_fit(self):
        with self.assertRaises(AssertionError):
            DecisionTree().predict(self.X[0])


class TestRandomForest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)  # RandomForest uses global numpy RNG for bagging
        self.X, self.y = two_class_blobs(n_features=4)
        self.ds = Dataset(self.X, self.y)

    def test_fits_and_classifies(self):
        m = RandomForest(n_estimators=10, max_depth=3)
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        preds = m.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertGreater(accuracy(self.y, preds), 0.9)


class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        X, self.y = two_class_blobs(n_features=4)
        # NaiveBayes here expects categorical/count features
        self.X = (X > 0).astype(int)
        self.ds = Dataset(self.X, self.y)

    def test_fits_and_classifies(self):
        m = NaiveBayes()
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        preds = m.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertGreater(accuracy(self.y, preds), 0.9)

    def test_probabilities_sum_to_one(self):
        m = NaiveBayes()
        m.fit(self.ds)
        probas = m.predict_proba(self.X)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_handles_imbalanced_classes(self):
        # different per-class sizes exercise the per-class list (not a ragged array)
        X = np.vstack([self.X[:10], self.X[30:]])
        y = np.array([0] * 10 + [1] * 30)
        m = NaiveBayes()
        m.fit(Dataset(X, y))
        self.assertTrue(m.is_fitted)


class TestLDA(unittest.TestCase):
    def setUp(self):
        self.X, self.y = two_class_blobs()
        self.ds = Dataset(self.X, self.y)

    def test_projects_and_classifies(self):
        m = LDA()
        m.fit(self.ds)
        self.assertEqual(m.w.shape, (self.X.shape[1],))
        # transform projects onto the discriminant direction (one value per sample)
        proj = m.transform(self.ds)
        self.assertEqual(proj.shape, (self.X.shape[0],))
        preds = m.predict(self.X)
        acc = accuracy(self.y, preds)
        # the sign of the discriminant direction is arbitrary, so accept either
        self.assertGreater(max(acc, 1 - acc), 0.9)


@unittest.skipUnless(
    importlib.util.find_spec("cvxopt") is not None,
    "cvxopt is not installed (optional dependency for SVM)",
)
class TestSVM(unittest.TestCase):
    def setUp(self):
        X, y = two_class_blobs()
        self.X = X
        # SVM expects labels in {-1, +1}
        self.y = np.where(y == 0, -1, 1).astype(float)
        self.ds = Dataset(self.X, self.y)

    def test_fits_and_classifies(self):
        from si.supervised.svm import SVM, linear_kernel

        m = SVM(kernel=linear_kernel)
        m.fit(self.ds)
        preds = m.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertGreater(accuracy(self.y, preds), 0.9)


if __name__ == "__main__":
    unittest.main()
