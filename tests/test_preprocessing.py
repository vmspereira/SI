# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the data preprocessing transformers (scaling, feature selection,
# label/one-hot encoding).
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.data import Dataset
from si.data.scale import StandardScaler
from si.data.feature_selection import VarianceThreshold, SelectKBest, f_classif
from si.data.encoder import LabelEncoder, OneHotEncoder


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.rand(50, 3) * 10 + 5
        self.ds = Dataset(self.X)

    def test_standardizes_to_zero_mean_unit_variance(self):
        out = StandardScaler().fit_transform(self.ds)
        np.testing.assert_allclose(out.X.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(out.X.var(axis=0), 1, atol=1e-9)

    def test_inverse_transform_recovers_original(self):
        scaler = StandardScaler()
        out = scaler.fit_transform(self.ds)
        recovered = scaler.inverse_transform(out)
        np.testing.assert_allclose(recovered.X, self.X)

    def test_transform_does_not_mutate_by_default(self):
        scaler = StandardScaler()
        scaler.fit(self.ds)
        _ = scaler.transform(self.ds)  # inline defaults to False
        np.testing.assert_array_equal(self.ds.X, self.X)


class TestVarianceThreshold(unittest.TestCase):
    def test_drops_zero_variance_feature(self):
        rng = np.random.RandomState(0)
        X = np.hstack([rng.rand(40, 3), np.ones((40, 1))])  # last column constant
        ds = Dataset(X, xnames=["a", "b", "c", "const"])
        out = VarianceThreshold(threshold=0).fit_transform(ds)
        self.assertEqual(out.X.shape, (40, 3))
        self.assertEqual(out._xnames, ["a", "b", "c"])

    def test_threshold_filters_low_variance(self):
        rng = np.random.RandomState(1)
        X = np.column_stack([rng.rand(100) * 100, rng.rand(100) * 0.001])
        ds = Dataset(X, xnames=["high", "low"])
        out = VarianceThreshold(threshold=1.0).fit_transform(ds)
        self.assertEqual(out._xnames, ["high"])


class TestSelectKBest(unittest.TestCase):
    def test_selects_k_features(self):
        rng = np.random.RandomState(0)
        informative = rng.rand(60, 3) * 10
        noise = rng.rand(60, 2)
        X = np.hstack([informative, noise])
        y = (informative[:, 0] > informative[:, 0].mean()).astype(int)
        ds = Dataset(X, y, xnames=[f"f{i}" for i in range(5)])
        out = SelectKBest(k=2, score_func=f_classif).fit_transform(ds)
        self.assertEqual(out.X.shape, (60, 2))
        self.assertEqual(len(out._xnames), 2)

    def test_keeps_most_informative_feature(self):
        rng = np.random.RandomState(0)
        # feature 0 perfectly tracks the label; the rest are noise
        y = np.array([0] * 30 + [1] * 30)
        signal = np.where(y == 0, -5.0, 5.0) + rng.randn(60) * 0.1
        X = np.column_stack([signal, rng.rand(60, 3)])
        ds = Dataset(X, y, xnames=["signal", "n1", "n2", "n3"])
        out = SelectKBest(k=1, score_func=f_classif).fit_transform(ds)
        self.assertEqual(out._xnames, ["signal"])


class TestLabelEncoder(unittest.TestCase):
    def test_encodes_string_labels_to_indices(self):
        X = np.zeros((4, 2))
        y = np.array(["cat", "dog", "cat", "bird"])
        ds = Dataset(X, y)
        le = LabelEncoder()
        out = le.fit_transform(ds)
        # classes are sorted: bird=0, cat=1, dog=2
        self.assertEqual(le.classes.tolist(), ["bird", "cat", "dog"])
        np.testing.assert_array_equal(out.y, [1, 2, 1, 0])


class TestOneHotEncoder(unittest.TestCase):
    def test_one_hot_encoding(self):
        X = np.zeros((4, 2))
        y = np.array([0, 1, 2, 1])
        ds = Dataset(X, y)
        out = OneHotEncoder().transform(ds)
        self.assertEqual(out.y.shape, (4, 3))
        np.testing.assert_array_equal(out.y[0], [1, 0, 0])
        np.testing.assert_array_equal(out.y[2], [0, 0, 1])
        # every row is a valid one-hot vector
        np.testing.assert_array_equal(out.y.sum(axis=1), np.ones(4))


if __name__ == "__main__":
    unittest.main()
