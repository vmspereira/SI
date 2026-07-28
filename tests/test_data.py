# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the degenerate and edge-case paths through si.data.
#
# test_preprocessing.py covers the happy path of each transformer. This file
# covers the inputs that used to produce silently wrong answers: a constant
# feature, a NaN score, a negative label, a mistyped column name, a
# non-numeric target. None of these paths had any coverage, which is why the
# bugs survived several passes of review.
# ----------------------------------------------------------------------------
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from si.data import Dataset, summary
from si.data.scale import StandardScaler
from si.data.encoder import LabelEncoder, OneHotEncoder
from si.data.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regress,
)


def informative_and_constant():
    """3 informative-ish features plus one constant column.

    The constant column is what makes the ANOVA degenerate: it scores nan.
    """
    rng = np.random.RandomState(0)
    X = np.hstack([rng.randn(30, 1) * 5, rng.randn(30, 1) * 0.1, rng.randn(30, 1)])
    y = np.array([0] * 15 + [1] * 15)
    X[:15, 0] += 5                       # make column 0 clearly predictive
    X = np.hstack([X, np.ones((30, 1))])  # column 3: constant
    return X, y


class TestStandardScalerConstantFeature(unittest.TestCase):
    def setUp(self):
        # column 1 never varies, so its variance is 0
        self.X = np.array([[1., 5.], [2., 5.], [3., 5.]])
        self.ds = Dataset(self.X, np.array([0, 1, 0]))

    def test_constant_feature_does_not_become_nan(self):
        # (x - mean) / sqrt(0) divided by zero and produced NaN for the whole
        # column, which then silently propagated into any model trained on it.
        out = StandardScaler().fit_transform(self.ds)
        self.assertTrue(np.isfinite(out.X).all())

    def test_constant_feature_becomes_zeros(self):
        # Centring alone already maps a constant feature to 0; there is no
        # spread left to rescale, so a scale of 1 is the natural choice.
        out = StandardScaler().fit_transform(self.ds)
        np.testing.assert_allclose(out.X[:, 1], [0., 0., 0.])

    def test_other_features_are_still_standardized(self):
        out = StandardScaler().fit_transform(self.ds)
        self.assertAlmostEqual(out.X[:, 0].mean(), 0.0)
        self.assertAlmostEqual(out.X[:, 0].std(), 1.0)

    def test_round_trip_is_exact_even_with_a_constant_feature(self):
        scaler = StandardScaler()
        out = scaler.fit_transform(self.ds)
        back = scaler.inverse_transform(out)
        np.testing.assert_allclose(back.X, self.X)

    def test_transform_before_fit_is_rejected(self):
        with self.assertRaises(AssertionError):
            StandardScaler().transform(self.ds)

    def test_inverse_transform_before_fit_is_rejected(self):
        with self.assertRaises(AssertionError):
            StandardScaler().inverse_transform(self.ds)


class TestSelectKBestDegenerateScores(unittest.TestCase):
    def setUp(self):
        self.X, self.y = informative_and_constant()
        self.ds = Dataset(self.X, self.y)

    def test_a_constant_feature_is_never_selected_as_the_best(self):
        # np.argsort puts nan LAST -- exactly where the highest scores are -- so
        # `[-k:]` used to select the nan-scoring constant column first, throwing
        # away the genuinely predictive feature. Treating nan as -inf sends it to
        # the bottom of the ranking instead.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F, _ = f_classif(self.ds)
            out = SelectKBest(k=1).fit_transform(self.ds)
        # sanity: the fixture really does produce a nan score and a clear winner
        self.assertTrue(np.isnan(F[3]))
        self.assertEqual(int(np.nanargmax(F)), 0)
        # and the selected column is the informative one, not the constant
        np.testing.assert_allclose(out.X[:, 0], self.X[:, 0])

    def test_ranking_still_works_when_no_score_is_nan(self):
        X, y = self.X[:, :3], self.y
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = SelectKBest(k=2).fit_transform(Dataset(X, y))
        self.assertEqual(out.X.shape[1], 2)

    def test_k_is_validated(self):
        # k=0 used to keep EVERY feature, because argsort()[-0:] is argsort()[0:]
        # -- the whole array rather than an empty slice. Exactly backwards.
        for bad in (0, -1, self.X.shape[1] + 1):
            with self.subTest(k=bad):
                with self.assertRaises(ValueError):
                    SelectKBest(k=bad).fit(self.ds)

    def test_k_equal_to_all_features_is_allowed(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = SelectKBest(k=self.X.shape[1]).fit_transform(self.ds)
        self.assertEqual(out.X.shape[1], self.X.shape[1])

    def test_transform_before_fit_is_rejected(self):
        with self.assertRaises(AssertionError):
            SelectKBest(k=1).transform(self.ds)

    def test_selected_names_track_the_selected_columns(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = SelectKBest(k=2).fit_transform(self.ds)
        self.assertEqual(len(out._xnames), 2)
        self.assertTrue(set(out._xnames) <= set(self.ds._xnames))


class TestVarianceThreshold(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1., 5.], [2., 5.], [3., 5.]])
        self.ds = Dataset(self.X, np.array([0, 1, 0]))

    def test_negative_threshold_is_rejected(self):
        # It used to warn and then keep the negative value, so `var > threshold`
        # was true for everything: the transformer became a no-op that retained
        # the very zero-variance features it exists to remove.
        with self.assertRaises(ValueError):
            VarianceThreshold(threshold=-1)

    def test_zero_threshold_drops_the_constant_feature(self):
        out = VarianceThreshold().fit_transform(self.ds)
        self.assertEqual(out.X.shape[1], 1)
        np.testing.assert_allclose(out.X[:, 0], self.X[:, 0])

    def test_transform_before_fit_is_rejected(self):
        with self.assertRaises(AssertionError):
            VarianceThreshold().transform(self.ds)


class TestFRegress(unittest.TestCase):
    def test_perfect_correlation_stays_finite(self):
        # F = r^2 / (1 - r^2) * dof divides by zero when r^2 == 1, returning inf,
        # which then contaminates any later arithmetic.
        X = np.arange(20, dtype=float).reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F, p = f_regress(Dataset(X, X[:, 0] * 2 + 1))
        self.assertTrue(np.isfinite(F).all())

    def test_a_perfect_feature_still_outranks_a_noisy_one(self):
        # The cap must not disturb the ranking.
        rng = np.random.RandomState(0)
        target = np.arange(30, dtype=float)
        X = np.column_stack([target, rng.randn(30)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F, _ = f_regress(Dataset(X, target))
        self.assertGreater(F[0], F[1])


class TestOneHotEncoder(unittest.TestCase):
    def test_negative_labels_are_rejected(self):
        # np.eye(n)[y] reads a negative index as counting from the END, so label
        # -1 was silently encoded as the LAST class instead of raising.
        with self.assertRaises(ValueError):
            OneHotEncoder().transform(Dataset(np.zeros((3, 2)), np.array([0, -1, 1])))

    def test_non_integer_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            OneHotEncoder().transform(
                Dataset(np.zeros((2, 2)), np.array(['a', 'b'])))

    def test_float_valued_integers_are_accepted(self):
        # 0.0/1.0 is a common shape for labels and is unambiguous; indexing with
        # raw floats would otherwise raise a confusing TypeError.
        out = OneHotEncoder().transform(
            Dataset(np.zeros((2, 2)), np.array([0., 1.])))
        np.testing.assert_allclose(out.y, [[1., 0.], [0., 1.]])

    def test_fit_fixes_the_width_for_later_batches(self):
        # Without fit, the width comes from the maximum label in whatever data
        # is passed, so a batch missing the top class encodes too narrowly and
        # silently disagrees with the training data.
        encoder = OneHotEncoder()
        encoder.fit(Dataset(np.zeros((3, 2)), np.array([0, 1, 2])))
        out = encoder.transform(Dataset(np.zeros((2, 2)), np.array([0, 1])))
        self.assertEqual(out.y.shape, (2, 3))

    def test_a_label_beyond_the_fitted_width_is_rejected(self):
        encoder = OneHotEncoder()
        encoder.fit(Dataset(np.zeros((2, 2)), np.array([0, 1])))
        with self.assertRaises(ValueError):
            encoder.transform(Dataset(np.zeros((1, 2)), np.array([5])))

    def test_without_fit_the_width_comes_from_the_data(self):
        # Backwards-compatible: transform alone still works.
        out = OneHotEncoder().transform(
            Dataset(np.zeros((3, 2)), np.array([0, 1, 2])))
        self.assertEqual(out.y.shape, (3, 3))


class TestLabelEncoderUnseenLabels(unittest.TestCase):
    def setUp(self):
        self.encoder = LabelEncoder()
        self.encoder.fit(Dataset(np.zeros((3, 2)), np.array(['a', 'b', 'a'])))

    def test_unseen_label_reports_what_went_wrong(self):
        # This used to be a bare "KeyError: 'z'", which says nothing about the
        # cause -- typically a class present only in the test split.
        with self.assertRaises(ValueError) as ctx:
            self.encoder.transform(Dataset(np.zeros((2, 2)), np.array(['a', 'z'])))
        message = str(ctx.exception)
        self.assertIn('z', message)
        self.assertIn('not seen during fit', message)

    def test_known_labels_still_encode(self):
        out = self.encoder.transform(
            Dataset(np.zeros((2, 2)), np.array(['b', 'a'])))
        np.testing.assert_array_equal(out.y, [1, 0])

    def test_transform_before_fit_is_rejected(self):
        with self.assertRaises(AssertionError):
            LabelEncoder().transform(Dataset(np.zeros((1, 2)), np.array(['a'])))


class TestDatasetValidation(unittest.TestCase):
    def test_one_dimensional_x_is_rejected_with_a_useful_message(self):
        # Used to raise "IndexError: tuple index out of range" from X.shape[1].
        with self.assertRaises(ValueError) as ctx:
            Dataset(np.array([1., 2., 3.]))
        self.assertIn('2-D', str(ctx.exception))

    def test_xnames_length_must_match_the_columns(self):
        # Constructing succeeded and the mismatch only surfaced much later, as an
        # opaque pandas shape error inside toDataframe.
        with self.assertRaises(ValueError):
            Dataset(np.zeros((2, 3)), np.array([0, 1]), xnames=['only_one'])

    def test_y_length_must_match_the_samples(self):
        with self.assertRaises(ValueError):
            Dataset(np.zeros((2, 3)), np.array([0, 1, 2]))

    def test_valid_construction_still_works(self):
        ds = Dataset(np.zeros((2, 3)), np.array([0, 1]), xnames=['a', 'b', 'c'])
        self.assertEqual(ds.getNumFeatures(), 3)
        self.assertEqual(len(ds), 2)
        self.assertTrue(ds.hasLabel())

    def test_names_are_generated_when_omitted(self):
        ds = Dataset(np.zeros((2, 3)))
        self.assertEqual(len(ds._xnames), 3)
        self.assertFalse(ds.hasLabel())


class TestDatasetFromDataframe(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'target': [0, 1, 0]})

    def test_mistyped_ylabel_is_rejected(self):
        # It used to fall through to the UNLABELED branch, so a typo produced a
        # dataset with no target and the target column folded in among the
        # features -- the worst outcome for someone who asked for a label.
        with self.assertRaises(ValueError) as ctx:
            Dataset.from_dataframe(self.df, ylabel='targett')
        self.assertIn('targett', str(ctx.exception))

    def test_correct_ylabel_splits_features_from_target(self):
        ds = Dataset.from_dataframe(self.df, ylabel='target')
        self.assertTrue(ds.hasLabel())
        self.assertEqual(ds.X.shape[1], 2)
        self.assertEqual(ds._xnames, ['a', 'b'])
        np.testing.assert_array_equal(ds.y, [0, 1, 0])

    def test_no_ylabel_gives_an_unlabeled_dataset(self):
        ds = Dataset.from_dataframe(self.df)
        self.assertFalse(ds.hasLabel())
        self.assertEqual(ds.X.shape[1], 3)


class TestDatasetOutputWithNonNumericTargets(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1., 2.], [3., 4.]])
        self.ds = Dataset(self.X, np.array(['cat', 'dog']))

    def test_toDataframe_keeps_feature_columns_numeric(self):
        # np.hstack((X, y)) forces one dtype, so string labels used to upcast
        # every numeric feature to text.
        df = self.ds.toDataframe()
        self.assertEqual(df.shape, (2, 3))
        for name in self.ds._xnames:
            self.assertTrue(np.issubdtype(df[name].dtype, np.number), name)

    def test_summary_still_describes_the_numeric_features(self):
        # The upcast above made summary report NaN for EVERY column. Only the
        # non-numeric target should be NaN.
        stats = summary(self.ds)
        first = self.ds._xnames[0]
        self.assertAlmostEqual(stats[first]['mean'], 2.0)
        self.assertAlmostEqual(stats[first]['min'], 1.0)
        self.assertAlmostEqual(stats[first]['max'], 3.0)
        self.assertTrue(np.isnan(stats[self.ds._yname]['mean']))

    def test_summary_dict_output(self):
        stats = summary(self.ds, output_format='dict')
        self.assertIsInstance(stats, dict)
        self.assertEqual(set(stats), set(self.ds._xnames) | {self.ds._yname})

    def test_summary_of_numeric_data(self):
        ds = Dataset(np.array([[1., 10.], [3., 30.]]), np.array([0., 1.]))
        stats = summary(ds, output_format='dict')
        self.assertAlmostEqual(stats[ds._xnames[1]]['mean'], 20.0)
        self.assertAlmostEqual(stats[ds._yname]['max'], 1.0)

    def test_writeDataset_handles_string_labels(self):
        # np.savetxt's default '%.18e' raised a TypeError on the text array that
        # hstack produced.
        path = os.path.join(tempfile.mkdtemp(), 'out.csv')
        self.ds.writeDataset(path)
        with open(path) as handle:
            lines = handle.read().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn('cat', lines[0])
        self.assertIn('dog', lines[1])

    def test_repr_html_hook_is_named_for_ipython(self):
        # Defined as __repr_html__, it was never called: IPython looks up the
        # single-underscore name, so datasets rendered as a plain <object ...>.
        self.assertTrue(hasattr(Dataset, '_repr_html_'))
        self.assertIn('<table', self.ds._repr_html_())


if __name__ == "__main__":
    unittest.main()
