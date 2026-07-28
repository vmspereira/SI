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
    Ensemble,
    majority,
    average,
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
        # Two well-separated Gaussian blobs: nearest-neighbour voting should be
        # almost perfect because each point's neighbours are nearly always in
        # its own blob.
        self.X, self.y = two_class_blobs()
        self.ds = Dataset(self.X, self.y)

    def test_classification_separates(self):
        # Usage pattern: construct -> fit(dataset) -> predict(sample). With k=3
        # neighbours on linearly separable blobs the model should classify the
        # training data with >95% accuracy; cost() reports that training
        # accuracy.
        m = KNN(num_neighbors=3)
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        # single-sample prediction returns a valid label
        self.assertIn(m.predict(self.X[0]), (0, 1))
        # cost() reports training accuracy on well-separated data
        self.assertGreater(m.cost(), 0.95)

    def test_requires_fit(self):
        # Calling predict() before fit() must fail loudly rather than return
        # garbage: there is no stored training set to find neighbours in.
        with self.assertRaises(AssertionError):
            KNN(num_neighbors=3).predict(self.X[0])

    def test_regression_mode_returns_mean(self):
        # With classification=False, KNN averages the targets of the k nearest
        # neighbours instead of voting, so it returns a continuous value. Here
        # targets are a linear function of feature 0; we only check the output
        # is a real-valued scalar (regression output), not a class label.
        y = self.X[:, 0] * 2.0
        m = KNN(num_neighbors=5, classification=False)
        m.fit(Dataset(self.X, y))
        pred = m.predict(self.X[0])
        self.assertIsInstance(float(pred), float)


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        # Build a perfectly linear, NOISE-FREE target so the true parameters are
        # known exactly. Because there is no noise, a correct solver can recover
        # the coefficients precisely, which makes the tests below sharp.
        rng = np.random.RandomState(1)
        self.X = rng.rand(80, 2)
        # y = 1 + 2*x0 + 3*x1  (no noise); theta = [intercept, w0, w1] = [1, 2, 3]
        self.y = 1 + 2 * self.X[:, 0] + 3 * self.X[:, 1]
        self.ds = Dataset(self.X, self.y)

    def test_closed_form_recovers_coefficients(self):
        # The normal-equation (closed-form, gd=False, no regularization lbd=0)
        # solution is exact on noise-free linear data: it must recover the true
        # coefficients [intercept=1, 2, 3] to machine precision and drive the
        # training cost essentially to zero.
        m = LinearRegression(gd=False, lbd=0)
        m.fit(self.ds)
        np.testing.assert_allclose(m.theta, [1, 2, 3], atol=1e-6)
        self.assertLess(m.cost(), 1e-10)

    def test_gradient_descent_fits(self):
        # The iterative gradient-descent solver (gd=True) optimizes the same
        # objective. After enough epochs it should converge close to the exact
        # closed-form coefficients, but only approximately (looser atol=0.2)
        # since GD stops short of the analytic optimum.
        m = LinearRegression(gd=True, lbd=0, epochs=5000, lr=0.1)
        m.fit(self.ds)
        # GD should get close to the closed-form solution
        np.testing.assert_allclose(m.theta, [1, 2, 3], atol=0.2)

    def test_predict_single_sample(self):
        # End-to-end usage check: after a closed-form fit on noise-free data,
        # predicting on a training point must reproduce its target almost
        # exactly.
        m = LinearRegression(gd=False, lbd=0)
        m.fit(self.ds)
        self.assertAlmostEqual(float(m.predict(self.X[0])), float(self.y[0]), places=4)


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.X, self.y = two_class_blobs()
        self.ds = Dataset(self.X, self.y)

    def test_separates_classes(self):
        # On linearly separable blobs, gradient-descent logistic regression
        # should learn a decision boundary that classifies the training set with
        # >95% accuracy.
        m = LogisticRegression(epochs=2000, lr=0.1)
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        preds = [m.predict(x) for x in self.X]
        self.assertGreater(accuracy(self.y, preds), 0.95)

    def test_cost_decreases(self):
        # Sanity check that training is actually optimizing: the loss recorded at
        # the last epoch must be lower than at the first. If the cost did not
        # fall, the gradient sign or update step would be wrong.
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
        # A depth-3 tree easily splits two well-separated blobs, so training
        # accuracy (cost()) should exceed 95%.
        m = DecisionTree(max_depth=3)
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        self.assertIn(m.predict(self.X[0]), (0, 1))
        self.assertGreater(m.cost(), 0.95)

    def test_requires_fit(self):
        # Predicting before fit() must raise: there is no tree to traverse yet.
        with self.assertRaises(AssertionError):
            DecisionTree().predict(self.X[0])

    def test_handles_degenerate_split(self):
        # Edge case: all feature rows are identical, so no split can separate the
        # classes. A correct implementation must fall back to a leaf (majority
        # vote) rather than crash trying to find an impossible split.
        # identical feature rows -> no valid split exists; must not crash
        X = np.ones((6, 3))
        y = np.array([0, 0, 0, 1, 1, 1])
        m = DecisionTree(max_depth=3)
        m.fit(Dataset(X, y))
        self.assertIn(m.predict(X[0]), (0, 1))


class TestDecisionTreeCriteria(unittest.TestCase):
    """gini and entropy are interchangeable impurity measures."""

    def setUp(self):
        self.X, self.y = two_class_blobs()
        self.ds = Dataset(self.X, self.y)

    def test_defaults_to_gini(self):
        self.assertEqual(DecisionTree().criterion, 'gini')

    def test_both_criteria_train_and_classify(self):
        for criterion in ('gini', 'entropy'):
            with self.subTest(criterion=criterion):
                m = DecisionTree(criterion=criterion)
                m.fit(self.ds)
                self.assertTrue(m.is_fitted)
                preds = [m.predict(row) for row in self.X]
                self.assertGreater(accuracy(self.y, preds), 0.9)

    def test_unknown_criterion_is_rejected_at_construction(self):
        # Caught in __init__ rather than surfacing as a confusing failure part
        # way through fit().
        with self.assertRaises(ValueError):
            DecisionTree(criterion='chi-squared')

    def test_impurity_measures_agree_on_purity_and_disagree_on_scale(self):
        from si.supervised.dt import gini, shannon_entropy

        pure = np.array([1.0, 0.0])
        # both are exactly 0 for a pure node -- nothing left to split
        self.assertAlmostEqual(gini(pure), 0.0)
        self.assertAlmostEqual(shannon_entropy(pure), 0.0)

        # both peak on a balanced node, but at different maxima:
        # gini at 1 - 1/k, entropy at log2(k)
        for k in (2, 4):
            balanced = np.full(k, 1.0 / k)
            self.assertAlmostEqual(gini(balanced), 1 - 1.0 / k)
            self.assertAlmostEqual(shannon_entropy(balanced), np.log2(k))

    def test_shannon_entropy_matches_the_label_based_entropy(self):
        # Same quantity, computed from probabilities rather than raw labels so it
        # mirrors gini's interface and works for non-integer labels.
        from si.supervised.dt import entropy, shannon_entropy

        labels = np.array([0, 1, 1, 1])
        np.testing.assert_allclose(shannon_entropy(np.array([0.25, 0.75])),
                                   entropy(labels))

    def test_entropy_criterion_handles_string_labels(self):
        # node_probs works off self.classes, so the probability-based entropy
        # copes with labels np.bincount could not accept.
        y = np.where(self.y == 0, 'cat', 'dog')
        m = DecisionTree(criterion='entropy')
        m.fit(Dataset(self.X, y))
        self.assertIn(m.predict(self.X[0]), ('cat', 'dog'))

    def test_calc_impurity_actually_uses_the_selected_criterion(self):
        # Without this, a calc_impurity that ignored self.criterion would still
        # pass every other test in this class, since gini alone classifies these
        # blobs perfectly.
        from si.supervised.dt import gini, shannon_entropy

        labels = np.array([0, 0, 0, 1])
        expected_probs = np.array([0.75, 0.25])

        for criterion, measure in (('gini', gini), ('entropy', shannon_entropy)):
            with self.subTest(criterion=criterion):
                m = DecisionTree(criterion=criterion)
                m.classes = np.array([0, 1])
                self.assertAlmostEqual(m.calc_impurity(labels),
                                       measure(expected_probs))
        # and the two measures genuinely differ on this node, so the assertions
        # above cannot both hold by coincidence
        self.assertNotAlmostEqual(gini(expected_probs),
                                  shannon_entropy(expected_probs))


class TestTreePredictionsAreLabels(unittest.TestCase):
    """predict must return a class LABEL, not a position in `classes`.

    argmax over a leaf's class-distribution vector gives an index into
    `self.classes`. Returning that index directly only looks correct when the
    labels happen to be 0..k-1: with labels {1, 2} or {'cat', 'dog'} the index
    never equals the label, so cost() compared indices against labels and
    reported 0% accuracy on perfectly separable data. RandomForest inherited
    the same fault, and additionally tallied votes with np.bincount, which
    indexes by label VALUE.
    """

    # separable blobs relabelled several ways; 0/1 is the only set where the
    # old index-returning behaviour coincided with the labels
    LABEL_SETS = {
        'zero_based': (0, 1),
        'one_based': (1, 2),
        'sparse_integers': (5, 9),
        'strings': ('cat', 'dog'),
    }

    def setUp(self):
        self.X, base = two_class_blobs(sep=3.0)
        self.base = base

    def relabel(self, low, high):
        return np.where(self.base == 0, low, high)

    def test_decision_tree_predicts_labels(self):
        for name, (low, high) in self.LABEL_SETS.items():
            with self.subTest(labels=name):
                y = self.relabel(low, high)
                m = DecisionTree()
                m.fit(Dataset(self.X, y))
                self.assertIn(m.predict(self.X[0]), (low, high))

    def test_decision_tree_cost_is_correct_for_any_label_set(self):
        # The observable consequence of the bug: this used to be 0.000 for
        # every label set except the zero-based one.
        for name, (low, high) in self.LABEL_SETS.items():
            with self.subTest(labels=name):
                m = DecisionTree()
                m.fit(Dataset(self.X, self.relabel(low, high)))
                self.assertGreater(m.cost(), 0.9)

    def test_random_forest_predicts_labels(self):
        for name, (low, high) in self.LABEL_SETS.items():
            with self.subTest(labels=name):
                y = self.relabel(low, high)
                m = RandomForest(n_estimators=5)
                m.fit(Dataset(self.X, y))
                preds = m.predict(self.X)
                self.assertEqual(len(preds), len(y))
                self.assertTrue(set(np.unique(preds)) <= {low, high})

    def test_random_forest_cost_is_correct_for_any_label_set(self):
        for name, (low, high) in self.LABEL_SETS.items():
            with self.subTest(labels=name):
                m = RandomForest(n_estimators=5)
                m.fit(Dataset(self.X, self.relabel(low, high)))
                self.assertGreater(m.cost(), 0.9)

    def test_random_forest_takes_a_majority_vote_over_labels(self):
        # Deterministic check of the combination step, independent of training:
        # three trees voting cat/cat/dog must yield cat.
        from si.supervised.ensemble import majority

        self.assertEqual(majority(['cat', 'cat', 'dog']), 'cat')
        self.assertEqual(majority([9, 5, 9]), 9)

    def test_cross_validation_scores_trees_correctly_with_any_labels(self):
        # predict_all routes through DecisionTree.predict, so the bug also made
        # every cross-validated score 0 for non-zero-based labels.
        from si.util.cv import CrossValidationScore

        for name, (low, high) in self.LABEL_SETS.items():
            with self.subTest(labels=name):
                cv = CrossValidationScore(
                    DecisionTree(), Dataset(self.X, self.relabel(low, high)),
                    score=accuracy, cv=3, random_state=0)
                self.assertGreater(np.mean(cv.run()[1]), 0.8)


class TestRandomForest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)  # RandomForest uses global numpy RNG for bagging
        self.X, self.y = two_class_blobs(n_features=4)
        self.ds = Dataset(self.X, self.y)

    def test_fits_and_classifies(self):
        # A forest of 10 trees (each trained on a bootstrap sample) aggregates
        # by majority vote. predict() accepts the whole matrix and returns one
        # label per row; on separable data accuracy should be >90%.
        m = RandomForest(n_estimators=10, max_depth=3)
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        preds = m.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertGreater(accuracy(self.y, preds), 0.9)

    def test_cost_reports_accuracy(self):
        # cost() should report the ensemble's training accuracy, consistent with
        # the direct accuracy() computed above.
        m = RandomForest(n_estimators=10, max_depth=3)
        m.fit(self.ds)
        self.assertGreater(m.cost(), 0.9)


class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        # This NaiveBayes implementation models categorical/count features, so
        # the continuous blob features are binarized (>0 -> 1, else 0) into 0/1
        # indicators before fitting.
        X, self.y = two_class_blobs(n_features=4)
        # NaiveBayes here expects categorical/count features
        self.X = (X > 0).astype(int)
        self.ds = Dataset(self.X, self.y)

    def test_fits_and_classifies(self):
        # With informative binary features and separable classes, the Naive
        # Bayes posterior picks the right class >90% of the time.
        m = NaiveBayes()
        m.fit(self.ds)
        self.assertTrue(m.is_fitted)
        preds = m.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertGreater(accuracy(self.y, preds), 0.9)

    def test_probabilities_sum_to_one(self):
        # predict_proba returns a posterior distribution over classes, so every
        # row must sum to 1.0. A violation would mean the normalization step is
        # wrong.
        m = NaiveBayes()
        m.fit(self.ds)
        probas = m.predict_proba(self.X)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_handles_imbalanced_classes(self):
        # Regression guard: when classes have different sample counts, the
        # per-class feature groups have different lengths. The implementation
        # must keep them in a Python list (not stack them into one NumPy array,
        # which would either raise or silently build a ragged/object array).
        # different per-class sizes exercise the per-class list (not a ragged array)
        X = np.vstack([self.X[:10], self.X[30:]])
        y = np.array([0] * 10 + [1] * 30)
        m = NaiveBayes()
        m.fit(Dataset(X, y))
        self.assertTrue(m.is_fitted)

    def test_cost_is_training_accuracy(self):
        # cost() must run (NaiveBayes.predict is a batch predictor, so it can be
        # called directly on X -- the apply_along_axis bridge for single-sample
        # predictors would break its broadcasting) and equal the training
        # accuracy on the fitted data.
        m = NaiveBayes()
        m.fit(self.ds)
        self.assertAlmostEqual(m.cost(), accuracy(self.y, m.predict(self.X)))


class TestLDA(unittest.TestCase):
    def setUp(self):
        self.X, self.y = two_class_blobs()
        self.ds = Dataset(self.X, self.y)

    def test_projects_and_classifies(self):
        # Linear Discriminant Analysis learns a single projection direction w
        # (one weight per feature) that best separates the two classes.
        m = LDA()
        m.fit(self.ds)
        self.assertEqual(m.w.shape, (self.X.shape[1],))
        # transform projects onto the discriminant direction (one value per sample)
        # transform() projects each sample onto w, collapsing it to one scalar,
        # so the result is 1-D with one value per sample.
        proj = m.transform(self.ds)
        self.assertEqual(proj.shape, (self.X.shape[0],))
        preds = m.predict(self.X)
        # Classification thresholds the projection at the midpoint between the
        # two class means; with that threshold the label orientation is fixed
        # and accuracy on separable blobs should exceed 90%.
        # with a proper midpoint threshold the label orientation is fixed
        self.assertGreater(accuracy(self.y, preds), 0.9)


@unittest.skipUnless(
    importlib.util.find_spec("cvxopt") is not None,
    "cvxopt is not installed (optional dependency for SVM)",
)
class TestSVM(unittest.TestCase):
    def setUp(self):
        # The SVM hinge-loss / dual formulation is defined for labels in
        # {-1, +1} (the margin term y*(w.x+b) only makes sense with signed
        # labels), so the {0, 1} blob labels are remapped to {-1, +1}.
        X, y = two_class_blobs()
        self.X = X
        # SVM expects labels in {-1, +1}
        self.y = np.where(y == 0, -1, 1).astype(float)
        self.ds = Dataset(self.X, self.y)

    def test_fits_and_classifies(self):
        # A linear-kernel SVM should find a separating hyperplane on the
        # linearly separable blobs and classify the training set with >90%
        # accuracy. (Skipped entirely if the optional cvxopt QP solver is
        # missing - see the class decorator.)
        from si.supervised.svm import SVM, linear_kernel

        m = SVM(kernel=linear_kernel)
        m.fit(self.ds)
        preds = m.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertGreater(accuracy(self.y, preds), 0.9)


class TestEnsemble(unittest.TestCase):
    """Ensemble must combine members regardless of their predict convention.

    The members of this library disagree about what `predict` takes: some want
    a single 1-D sample, some want a 2-D batch (see `Model.predicts_batch`).
    Ensemble is handed a single sample by its own `cost`, so it has to bridge
    that gap for every member.
    """

    def setUp(self):
        X, self.y = two_class_blobs(n_features=4)
        self.X = X
        # NaiveBayes here models categorical/count features, so it gets the
        # binarized copy of the same blobs.
        self.Xbin = (X > 0).astype(int)

    def test_batch_members(self):
        # Regression guard: NaiveBayes and LDA both predict batches. Handing
        # them the raw 1-D sample made them iterate over features instead of
        # samples, which used to raise
        #   ValueError: operands could not be broadcast together
        #               with shapes (2,0,2) (0,)
        m = Ensemble([NaiveBayes(), LDA()], accuracy)
        m.fit(Dataset(self.Xbin, self.y))
        self.assertIn(m.predict(self.Xbin[0]), (0, 1))
        self.assertGreater(m.cost(), 0.9)

    def test_single_sample_members(self):
        # The convention Ensemble always supported: members that take one
        # sample and return one scalar.
        m = Ensemble([KNN(3), DecisionTree()], accuracy)
        m.fit(Dataset(self.X, self.y))
        self.assertIn(m.predict(self.X[0]), (0, 1))
        self.assertGreater(m.cost(), 0.9)

    def test_mixed_members(self):
        # The case that motivated the fix: both conventions in one ensemble.
        # KNN is single-sample, LDA and RandomForest are batch predictors.
        m = Ensemble([KNN(3), LDA(), RandomForest(n_estimators=5)], accuracy)
        m.fit(Dataset(self.X, self.y))
        self.assertIn(m.predict(self.X[0]), (0, 1))
        self.assertGreater(m.cost(), 0.9)

    def test_requires_fit(self):
        m = Ensemble([KNN(3)], accuracy)
        with self.assertRaises(AssertionError):
            m.predict(self.X[0])

    def test_average_vote_for_regression(self):
        # With `average` as the decision function the ensemble output is the
        # mean of the members' predictions, so two regressors fit on an exactly
        # linear target must recover it. lbd=0 switches off the default L2
        # penalty (lbd=1), which would otherwise shrink the coefficients and
        # leave a residual error of ~8e-4.
        y = self.X[:, 0] * 2.0 + 1.0
        m = Ensemble([LinearRegression(lbd=0), LinearRegression(lbd=0)],
                     lambda t, p: float(np.mean((t - p) ** 2)),
                     fvote=average)
        m.fit(Dataset(self.X, y))
        self.assertLess(m.cost(), 1e-20)

    def test_average_vote_is_the_mean_of_members(self):
        # Two members fit differently (closed form vs gradient descent) so they
        # genuinely disagree; the ensemble output must be exactly their mean.
        y = self.X[:, 0] * 2.0 + 1.0
        exact = LinearRegression(lbd=0)
        approx = LinearRegression(lbd=0, gd=True, epochs=50)
        ds = Dataset(self.X, y)
        exact.fit(ds)
        approx.fit(ds)
        # fitted=True: the members are already trained, so skip refitting them
        m = Ensemble([exact, approx], accuracy, fvote=average, fitted=True)
        sample = self.X[0]
        self.assertAlmostEqual(
            m.predict(sample),
            (exact.predict(sample) + approx.predict(sample)) / 2,
        )

    def test_majority_breaks_ties_deterministically(self):
        # A 2-2 split has no modal value, so the tie-break has to be defined:
        # the smallest label wins. Relying on `set` iteration order instead
        # would make the vote an implementation detail.
        self.assertEqual(majority([0, 0, 1, 1]), 0)
        self.assertEqual(majority([1, 1, 0, 0]), 0)
        self.assertEqual(majority([2, 2, 2, 5]), 2)


if __name__ == "__main__":
    unittest.main()
