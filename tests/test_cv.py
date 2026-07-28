# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the model-selection utilities in si.util.cv.
#
# The point of cross-validation is that a sample is never scored by a model
# that trained on it. These tests check the resampling itself -- that the folds
# really are disjoint and really do cover the data -- and that neither
# CrossValidationScore nor GridSearchCV lets state leak between rounds.
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.data import Dataset
from si.util.cv import CrossValidationScore, GridSearchCV
from si.util.metrics import accuracy_score
from si.supervised import KNN, LogisticRegression, NaiveBayes, LDA


def blobs(n_per_class=25, n_features=3, sep=2.0, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.randn(n_per_class, n_features) - sep,
                   rng.randn(n_per_class, n_features) + sep])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


class TestKFoldResampling(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs()
        self.ds = Dataset(self.X, self.y)

    def test_folds_are_disjoint_and_cover_every_sample(self):
        # THE defining property of k-fold: the test folds partition the data.
        # Each sample is tested exactly once, so the k test scores are computed
        # on non-overlapping data. Repeated random splitting -- the old
        # behaviour of this class -- cannot promise this: a sample could land in
        # the test set several times, or never.
        cv = CrossValidationScore(KNN(3), self.ds, score=accuracy_score,
                                  cv=5, random_state=1)
        cv.run()
        self.assertEqual(len(cv.ds), 5)
        test_rows = np.vstack([test.X for _, test in cv.ds])
        # no sample missing, no sample counted twice
        self.assertEqual(test_rows.shape[0], self.X.shape[0])
        self.assertEqual(len(np.unique(test_rows, axis=0)), self.X.shape[0])

    def test_each_round_trains_on_everything_else(self):
        cv = CrossValidationScore(KNN(3), self.ds, score=accuracy_score,
                                  cv=5, random_state=1)
        cv.run()
        for train, test in cv.ds:
            self.assertEqual(train.X.shape[0] + test.X.shape[0],
                             self.X.shape[0])

    def test_uneven_fold_sizes_lose_no_samples(self):
        # 50 samples into 3 folds does not divide evenly; the remainder is
        # spread over the first folds rather than dropped.
        cv = CrossValidationScore(KNN(3), self.ds, score=accuracy_score,
                                  cv=3, random_state=1)
        cv.run()
        sizes = sorted(test.X.shape[0] for _, test in cv.ds)
        self.assertEqual(sum(sizes), self.X.shape[0])
        # sizes differ by at most one
        self.assertLessEqual(sizes[-1] - sizes[0], 1)

    def test_random_state_makes_folds_reproducible(self):
        a = CrossValidationScore(KNN(3), self.ds, score=accuracy_score,
                                 cv=4, random_state=7)
        b = CrossValidationScore(KNN(3), self.ds, score=accuracy_score,
                                 cv=4, random_state=7)
        np.testing.assert_allclose(a.run()[1], b.run()[1])

    def test_rejects_impossible_fold_counts(self):
        # cv=1 leaves nothing to test on; cv > n_samples cannot make that many
        # non-empty folds.
        with self.assertRaises(ValueError):
            CrossValidationScore(KNN(3), self.ds, cv=1).run()
        with self.assertRaises(ValueError):
            CrossValidationScore(KNN(3), self.ds,
                                 cv=self.X.shape[0] + 1).run()

    def test_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError):
            CrossValidationScore(KNN(3), self.ds, strategy='bootstrap')


class TestHoldoutStrategy(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs()
        self.ds = Dataset(self.X, self.y)

    def test_holdout_honours_the_split_fraction(self):
        # The legacy strategy: `cv` independent random partitions sized by
        # `split`. Kept because the eval notebooks pass split= and expect it.
        cv = CrossValidationScore(KNN(3), self.ds, score=accuracy_score,
                                  cv=3, split=0.8, strategy='holdout')
        cv.run()
        self.assertEqual(len(cv.ds), 3)
        for train, test in cv.ds:
            self.assertEqual(train.X.shape[0], int(0.8 * self.X.shape[0]))
            self.assertEqual(test.X.shape[0],
                             self.X.shape[0] - int(0.8 * self.X.shape[0]))


class TestCrossValidationScoring(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs()
        self.ds = Dataset(self.X, self.y)

    def test_scores_are_returned_per_fold(self):
        cv = CrossValidationScore(KNN(3), self.ds, score=accuracy_score,
                                  cv=4, random_state=0)
        train_scores, test_scores = cv.run()
        self.assertEqual(len(train_scores), 4)
        self.assertEqual(len(test_scores), 4)
        # separable blobs -> KNN should be almost perfect on held-out folds
        self.assertGreater(np.mean(test_scores), 0.9)

    def test_falls_back_to_model_cost_without_a_score_function(self):
        # score=None means "use the model's own cost()", which for
        # LogisticRegression is the log-loss, so lower is better and positive.
        cv = CrossValidationScore(LogisticRegression(), self.ds,
                                  cv=3, random_state=0)
        train_scores, test_scores = cv.run()
        self.assertEqual(len(test_scores), 3)
        self.assertTrue(all(np.isfinite(s) and s > 0 for s in test_scores))

    def test_works_with_batch_predicting_models(self):
        # Regression guard. Scoring used to go through
        #   np.ma.apply_along_axis(model.predict, axis=0, arr=X.T)
        # which hands ONE sample to predict. NaiveBayes and LDA predict whole
        # batches, so they received a 1-D array and iterated over features,
        # raising a broadcasting error. predict_all now dispatches on the
        # model's declared convention.
        nb = CrossValidationScore(NaiveBayes(), Dataset((self.X > 0).astype(int), self.y),
                                  score=accuracy_score, cv=3, random_state=0)
        self.assertEqual(len(nb.run()[1]), 3)
        lda = CrossValidationScore(LDA(), self.ds, score=accuracy_score,
                                   cv=3, random_state=0)
        scores = lda.run()[1]
        self.assertEqual(len(scores), 3)
        self.assertGreater(np.mean(scores), 0.9)

    def test_does_not_refit_the_callers_model(self):
        # Each fold trains a copy, so the instance handed in is left alone --
        # otherwise it would silently end up holding the last fold's parameters.
        model = KNN(3)
        CrossValidationScore(model, self.ds, score=accuracy_score,
                             cv=3, random_state=0).run()
        self.assertFalse(model.is_fitted)

    def test_toDataframe_needs_a_run_first(self):
        cv = CrossValidationScore(KNN(3), self.ds, score=accuracy_score, cv=3)
        with self.assertRaises(AssertionError):
            cv.toDataframe()
        cv.run()
        df = cv.toDataframe()
        self.assertEqual(list(df.columns), ['Train Scores', 'Test Scores'])
        self.assertEqual(len(df), 3)


class TestGridSearchCV(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs()
        self.ds = Dataset(self.X, self.y)

    def test_enumerates_the_whole_grid(self):
        gs = GridSearchCV(LogisticRegression(), self.ds,
                          {'epochs': [50, 100], 'lr': [0.01, 0.1]},
                          cv=3, random_state=0)
        results = gs.run()
        # 2 values x 2 values = 4 grid points
        self.assertEqual(len(results), 4)
        self.assertEqual({conf for conf, _ in results},
                         {(50, 0.01), (50, 0.1), (100, 0.01), (100, 0.1)})

    def test_does_not_mutate_the_callers_model(self):
        # The grid used to be applied with setattr to ONE shared instance, so
        # after run() the caller's model was left configured as the last grid
        # point and already fitted -- and each grid point was scored by a model
        # still carrying the previous point's state.
        model = LogisticRegression()
        original_epochs = model.epochs
        GridSearchCV(model, self.ds, {'epochs': [50, 100]},
                     cv=3, random_state=0).run()
        self.assertEqual(model.epochs, original_epochs)
        self.assertFalse(model.is_fitted)

    def test_grid_point_scores_are_order_independent(self):
        # With per-point copies, a grid point's score depends only on its own
        # configuration, so reversing the traversal order must not change it.
        forward = GridSearchCV(LogisticRegression(), self.ds,
                               {'epochs': [50, 100]}, cv=3, random_state=0).run()
        reverse = GridSearchCV(LogisticRegression(), self.ds,
                               {'epochs': [100, 50]}, cv=3, random_state=0).run()
        forward_by_conf = {conf: scores for conf, scores in forward}
        reverse_by_conf = {conf: scores for conf, scores in reverse}
        for conf in forward_by_conf:
            np.testing.assert_allclose(forward_by_conf[conf][1],
                                       reverse_by_conf[conf][1])

    def test_rejects_parameters_the_model_does_not_have(self):
        with self.assertRaises(ValueError):
            GridSearchCV(LogisticRegression(), self.ds, {'not_a_param': [1, 2]})

    def test_toDataframe_reports_a_row_per_grid_point(self):
        gs = GridSearchCV(LogisticRegression(), self.ds, {'epochs': [50, 100]},
                          cv=3, random_state=0)
        gs.run()
        df = gs.toDataframe()
        self.assertEqual(len(df), 2)
        self.assertIn('epochs', df.columns)
        # one train and one test column per fold
        for i in (1, 2, 3):
            self.assertIn(f'CV_{i} train', df.columns)
            self.assertIn(f'CV_{i} test', df.columns)


if __name__ == "__main__":
    unittest.main()
