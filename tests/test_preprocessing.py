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
        # Data deliberately off-centre (scaled 10x, shifted +5) so that
        # standardization has visible work to do - raw means/variances are far
        # from 0/1.
        rng = np.random.RandomState(0)
        self.X = rng.rand(50, 3) * 10 + 5
        self.ds = Dataset(self.X)

    def test_standardizes_to_zero_mean_unit_variance(self):
        # StandardScaler subtracts each column's mean and divides by its std.
        # By definition the transformed columns must then have mean 0 and
        # variance 1 - the core property of standardization.
        out = StandardScaler().fit_transform(self.ds)
        np.testing.assert_allclose(out.X.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(out.X.var(axis=0), 1, atol=1e-9)

    def test_inverse_transform_recovers_original(self):
        # The scaling is an invertible affine map (multiply by std, add mean),
        # so inverse_transform must reconstruct the original data exactly.
        scaler = StandardScaler()
        out = scaler.fit_transform(self.ds)
        recovered = scaler.inverse_transform(out)
        np.testing.assert_allclose(recovered.X, self.X)

    def test_transform_does_not_mutate_by_default(self):
        # transform() should return a NEW dataset and leave the input untouched
        # (inline=False by default). Mutating the caller's data in place would be
        # a surprising side effect, so the original X must be unchanged.
        scaler = StandardScaler()
        scaler.fit(self.ds)
        _ = scaler.transform(self.ds)  # inline defaults to False
        np.testing.assert_array_equal(self.ds.X, self.X)


class TestVarianceThreshold(unittest.TestCase):
    def test_drops_zero_variance_feature(self):
        # A constant column carries no information. With threshold=0 the selector
        # must drop exactly that column, keeping the other three - and it must
        # keep the surviving feature NAMES aligned with the kept columns.
        rng = np.random.RandomState(0)
        X = np.hstack([rng.rand(40, 3), np.ones((40, 1))])  # last column constant
        ds = Dataset(X, xnames=["a", "b", "c", "const"])
        out = VarianceThreshold(threshold=0).fit_transform(ds)
        self.assertEqual(out.X.shape, (40, 3))
        self.assertEqual(out._xnames, ["a", "b", "c"])

    def test_threshold_filters_low_variance(self):
        # The threshold is on variance, not just "is it constant". The "high"
        # column (spread over ~0..100) has large variance; the "low" column
        # (~0..0.001) has tiny variance. A threshold of 1.0 keeps only "high".
        rng = np.random.RandomState(1)
        X = np.column_stack([rng.rand(100) * 100, rng.rand(100) * 0.001])
        ds = Dataset(X, xnames=["high", "low"])
        out = VarianceThreshold(threshold=1.0).fit_transform(ds)
        self.assertEqual(out._xnames, ["high"])


class TestSelectKBest(unittest.TestCase):
    def test_selects_k_features(self):
        # SelectKBest scores every feature with score_func (here the ANOVA
        # F-test, f_classif) and keeps the top k. This test checks the shape
        # contract: requesting k=2 yields exactly 2 columns and 2 matching names.
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
        # Correctness of the RANKING: feature "signal" almost perfectly
        # separates the two classes (means at -5 and +5 with tiny noise) while
        # the rest are pure noise. The F-test must rank "signal" highest, so
        # k=1 has to select exactly that column.
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
        # LabelEncoder maps string labels to integer indices. Classes are stored
        # in sorted order, so the index of each class is deterministic
        # (bird=0, cat=1, dog=2) regardless of the order they appear in y, and
        # the encoded y must reflect that mapping.
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
        # OneHotEncoder expands an integer label into a row with a single 1 in
        # the column for that class. With 3 distinct classes the label matrix
        # becomes (4, 3): label 0 -> [1,0,0], label 2 -> [0,0,1], and every row
        # must sum to exactly 1 (one hot bit set, the rest zero).
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
