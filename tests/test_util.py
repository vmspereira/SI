# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the helpers in si.util.util.
#
# Small functions, but several are load-bearing: add_intercept is what makes the
# bias term work in the linear models, minibatch drives the neural-network
# training loop, and predict_all is what lets any scorer call a model without
# knowing which predict convention it follows.
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.data import Dataset
from si.util.util import (
    label_gen,
    l1_distance,
    l2_distance,
    train_test_split,
    predict_all,
    get_random_subsets,
    add_intercept,
    sigmoid,
    to_categorical,
    minibatch,
)
from si.supervised import KNN, LDA


class TestLabelGen(unittest.TestCase):
    def test_generates_the_requested_number_of_distinct_labels(self):
        labels = label_gen(5)
        self.assertEqual(len(labels), 5)
        self.assertEqual(len(set(labels)), 5)

    def test_labels_start_like_spreadsheet_columns(self):
        self.assertEqual(label_gen(3), ['A', 'B', 'C'])

    def test_rolls_over_to_two_character_labels(self):
        # The alphabet here omits Y (reserved for dependent variables), so it is
        # 25 letters long and the 26th label is the first two-character one.
        labels = label_gen(27)
        self.assertEqual(len(set(labels)), 27)
        self.assertEqual(len(labels[-1]), 2)

    def test_has_a_docstring(self):
        # The import used to sit above the docstring, which demoted it to a
        # no-op string expression and left __doc__ as None.
        self.assertIsNotNone(label_gen.__doc__)
        self.assertIn("distinct labels", label_gen.__doc__)


class TestDistances(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0.0, 0.0])
        self.y = np.array([[3.0, 4.0], [1.0, 1.0]])

    def test_l1_is_the_sum_of_absolute_differences(self):
        np.testing.assert_allclose(l1_distance(self.x, self.y), [7.0, 2.0])

    def test_l2_returns_squared_euclidean_distance(self):
        # Note this returns the SQUARED distance (no square root). That is fine
        # for KNN, which only needs the ordering, and the ordering is unchanged
        # by the monotone sqrt.
        np.testing.assert_allclose(l2_distance(self.x, self.y), [25.0, 2.0])

    def test_distance_to_itself_is_zero(self):
        p = np.array([2.0, -3.0])
        self.assertAlmostEqual(l1_distance(p, p.reshape(1, -1))[0], 0.0)
        self.assertAlmostEqual(l2_distance(p, p.reshape(1, -1))[0], 0.0)


class TestAddIntercept(unittest.TestCase):
    def test_prepends_a_column_of_ones(self):
        # This is what gives the linear models their bias term: with a leading
        # 1 in every row, theta[0] becomes the intercept and the whole model is
        # a single dot product.
        X = np.array([[2.0, 3.0], [4.0, 5.0]])
        out = add_intercept(X)
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_allclose(out[:, 0], [1.0, 1.0])
        np.testing.assert_allclose(out[:, 1:], X)


class TestSigmoid(unittest.TestCase):
    def test_maps_zero_to_one_half(self):
        self.assertAlmostEqual(sigmoid(np.array([0.0]))[0], 0.5)

    def test_is_bounded_and_monotone(self):
        z = np.array([-50.0, -1.0, 0.0, 1.0, 50.0])
        out = sigmoid(z)
        self.assertTrue(np.all((out >= 0) & (out <= 1)))
        self.assertTrue(np.all(np.diff(out) > 0))

    def test_is_symmetric_about_the_origin(self):
        # sigma(-z) = 1 - sigma(z)
        z = np.array([0.3, 1.7, 4.0])
        np.testing.assert_allclose(sigmoid(-z), 1 - sigmoid(z))


class TestToCategorical(unittest.TestCase):
    def test_one_hot_encodes_a_label_vector(self):
        out = to_categorical(np.array([0, 2, 1]))
        np.testing.assert_allclose(out, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    def test_infers_the_class_count_from_the_maximum_label(self):
        self.assertEqual(to_categorical(np.array([0, 3])).shape, (2, 4))

    def test_explicit_num_classes_widens_the_encoding(self):
        # Needed when encoding a subset (e.g. one batch) where the highest class
        # may not appear -- inference from max(y) would silently produce too few
        # columns.
        out = to_categorical(np.array([0, 1]), num_classes=5)
        self.assertEqual(out.shape, (2, 5))
        self.assertTrue(np.all(out[:, 2:] == 0))

    def test_every_row_sums_to_one(self):
        out = to_categorical(np.array([0, 1, 2, 2, 1]))
        np.testing.assert_allclose(out.sum(axis=1), np.ones(5))

    def test_squeezes_a_trailing_axis_of_size_one(self):
        # (n, 1) input is treated as n labels, not as n rows of one feature.
        out = to_categorical(np.array([[0], [1], [2]]))
        self.assertEqual(out.shape, (3, 3))

    def test_dtype_is_configurable(self):
        self.assertEqual(to_categorical(np.array([0, 1]), dtype='int64').dtype,
                         np.dtype('int64'))


class TestMinibatch(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(20, dtype=float).reshape(10, 2)
        self.y = np.arange(10, dtype=float)

    def test_yields_whole_batches_of_the_requested_size(self):
        batches = list(minibatch(self.X, self.y, batchsize=5, shuffle=False))
        self.assertEqual(len(batches), 2)
        for xb, yb in batches:
            self.assertEqual(xb.shape, (5, 2))
            self.assertEqual(yb.shape, (5,))

    def test_batchsize_larger_than_the_data_gives_one_full_batch(self):
        # batch_size is clamped to the sample count, which is why the NN's
        # default batch_size of 128 works on a 4-sample problem like XOR.
        batches = list(minibatch(self.X, self.y, batchsize=100, shuffle=False))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape, (10, 2))

    def test_without_shuffling_batches_are_in_order(self):
        xb, yb = next(iter(minibatch(self.X, self.y, batchsize=3, shuffle=False)))
        np.testing.assert_allclose(yb, [0, 1, 2])

    def test_shuffling_keeps_x_and_y_aligned(self):
        # The critical property: rows may be reordered, but each X row must stay
        # with its own label. Here y[i] indexes X, so X[:, 0] == 2*y.
        np.random.seed(0)
        for xb, yb in minibatch(self.X, self.y, batchsize=5, shuffle=True):
            np.testing.assert_allclose(xb[:, 0], 2 * yb)

    def test_drops_a_trailing_partial_batch(self):
        # 10 samples in batches of 4 yields 2 full batches; the remaining 2
        # samples are skipped rather than forming a short batch.
        batches = list(minibatch(self.X, self.y, batchsize=4, shuffle=False))
        self.assertEqual(len(batches), 2)

    def test_works_without_labels(self):
        batches = list(minibatch(self.X, batchsize=5, shuffle=False))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape, (5, 2))

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(AssertionError):
            list(minibatch(self.X, self.y[:3]))


class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset(np.arange(40, dtype=float).reshape(20, 2),
                          np.arange(20, dtype=float))

    def test_split_sizes_follow_the_fraction(self):
        train, test = train_test_split(self.ds, split=0.75)
        self.assertEqual(train.X.shape[0], 15)
        self.assertEqual(test.X.shape[0], 5)

    def test_the_two_halves_partition_the_data(self):
        train, test = train_test_split(self.ds, split=0.7)
        combined = np.concatenate([train.y, test.y])
        np.testing.assert_array_equal(np.sort(combined), np.sort(self.ds.y))

    def test_rows_keep_their_own_labels(self):
        # X[i, 0] == 2 * y[i] in this dataset, so a shuffle that decoupled X
        # from y would show up immediately.
        train, test = train_test_split(self.ds, split=0.6)
        for part in (train, test):
            np.testing.assert_allclose(part.X[:, 0], 2 * part.y)


class TestPredictAll(unittest.TestCase):
    """predict_all hides the inconsistent predict conventions from callers."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = np.vstack([rng.randn(15, 3) - 2, rng.randn(15, 3) + 2])
        self.y = np.array([0] * 15 + [1] * 15)
        self.ds = Dataset(self.X, self.y)

    def test_handles_a_single_sample_predictor(self):
        model = KNN(3)
        model.fit(self.ds)
        preds = predict_all(model, self.X)
        self.assertEqual(preds.shape[0], self.X.shape[0])

    def test_handles_a_batch_predictor(self):
        model = LDA()
        model.fit(self.ds)
        preds = predict_all(model, self.X)
        self.assertEqual(preds.shape[0], self.X.shape[0])

    def test_both_conventions_agree_with_calling_the_model_directly(self):
        knn = KNN(3)
        knn.fit(self.ds)
        np.testing.assert_allclose(
            predict_all(knn, self.X),
            [knn.predict(row) for row in self.X])
        lda = LDA()
        lda.fit(self.ds)
        np.testing.assert_allclose(predict_all(lda, self.X), lda.predict(self.X))

    def test_defaults_to_the_single_sample_convention(self):
        # A duck-typed model that never declares predicts_batch is treated as a
        # single-sample predictor, the more common case in this library.
        class Custom:
            def predict(self, x):
                return float(np.sum(x))

        preds = predict_all(Custom(), np.ones((4, 3)))
        np.testing.assert_allclose(preds, [3.0, 3.0, 3.0, 3.0])


class TestGetRandomSubsets(unittest.TestCase):
    def test_returns_the_requested_number_of_subsets(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.arange(20, dtype=float)
        subsets = get_random_subsets(X, y, n_subsets=4)
        self.assertEqual(len(subsets), 4)
        for xs, ys in subsets:
            self.assertEqual(xs.shape[0], ys.shape[0])

    def test_with_replacement_draws_full_size_subsets(self):
        # This is the bootstrap sampling behind RandomForest's bagging: each
        # tree sees a resample of the same size as the original data.
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.arange(20, dtype=float)
        for xs, _ in get_random_subsets(X, y, n_subsets=3, replacements=True):
            self.assertEqual(xs.shape[0], 20)

    def test_without_replacement_uses_half_the_samples(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.arange(20, dtype=float)
        for xs, _ in get_random_subsets(X, y, n_subsets=3, replacements=False):
            self.assertEqual(xs.shape[0], 10)

    def test_subsets_keep_rows_with_their_labels(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.arange(20, dtype=float)
        for xs, ys in get_random_subsets(X, y, n_subsets=3):
            np.testing.assert_allclose(xs[:, 0], 2 * ys)


if __name__ == "__main__":
    unittest.main()
