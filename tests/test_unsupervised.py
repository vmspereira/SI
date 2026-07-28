# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the unsupervised models (KMeans, PCA).
# ----------------------------------------------------------------------------
import warnings
import unittest

import numpy as np

from si.data import Dataset
from si.unsupervised.kmeans import KMeans
from si.unsupervised.pca import PCA


def three_blobs(n=20, seed=0):
    # Three Gaussian clusters with widely spaced centres (0,0), (8,8), (0,8).
    # The large gaps make the ground-truth clustering unambiguous, so KMeans
    # has a clearly "right" answer to find.
    rng = np.random.RandomState(seed)
    X = np.vstack([
        rng.randn(n, 2) + [0, 0],
        rng.randn(n, 2) + [8, 8],
        rng.randn(n, 2) + [0, 8],
    ])
    return X


class TestKMeans(unittest.TestCase):
    def test_clusters_separated_blobs(self):
        # Usage: KMeans(k).fit_transform(dataset) returns the learned centroids
        # and the cluster index assigned to each sample. We don't check exact
        # cluster identities (labels are arbitrary/permutable) but the structural
        # contract: every sample gets exactly one label in [0, k), and each
        # centroid lives in the same feature space as the data (2-D here).
        X = three_blobs()
        ds = Dataset(X)
        with warnings.catch_warnings():
            # empty clusters can momentarily produce mean-of-empty warnings
            warnings.simplefilter("ignore")
            km = KMeans(k=3, max_iterations=100)
            centroids, idxs = km.fit_transform(ds)
        # one assignment per sample, all within [0, k)
        self.assertEqual(len(idxs), X.shape[0])
        self.assertTrue(set(np.asarray(idxs).tolist()).issubset({0, 1, 2}))
        self.assertEqual(np.asarray(centroids).shape[1], X.shape[1])

    def test_get_closest_centroid(self):
        # Unit-tests the core assignment step in isolation by hand-placing two
        # centroids: a point near the origin must map to centroid 0 and a point
        # near (10,10) to centroid 1. This pins down the nearest-centroid
        # (minimum Euclidean distance) logic independent of the full fit loop.
        km = KMeans(k=2)
        km.centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        self.assertEqual(km.get_closest_centroid(np.array([0.5, 0.5])), 0)
        self.assertEqual(km.get_closest_centroid(np.array([9.0, 9.0])), 1)


class TestPCA(unittest.TestCase):
    def setUp(self):
        # 5-D data where feature 0 is inflated 20x so it carries the bulk of the
        # variance. PCA should therefore line its first principal component up
        # with that feature, giving the tests a known dominant direction.
        rng = np.random.RandomState(2)
        X = rng.randn(100, 5)
        X[:, 0] *= 20  # make the first feature dominate the variance
        self.ds = Dataset(X)

    def test_transform_shape_svd(self):
        # fit_transform projects the 100x5 data down to n_components columns.
        # With the SVD backend the result must be (100, 2) and strictly REAL -
        # SVD avoids the complex eigenvalues that an unsymmetric eig call can
        # spuriously produce.
        p = PCA(n_components=2, svd=True, scale_data=False)
        reduced = p.fit_transform(self.ds)
        self.assertEqual(reduced.shape, (100, 2))
        self.assertFalse(np.iscomplexobj(reduced))

    def test_transform_shape_eig(self):
        # The eigen-decomposition backend must produce the same kind of output:
        # requesting 3 components yields a (100, 3) projection.
        p = PCA(n_components=3, svd=False, scale_data=False)
        reduced = p.fit_transform(self.ds)
        self.assertEqual(reduced.shape, (100, 3))

    def test_variance_explained_sums_to_100(self):
        # variance_explained() returns the percentage of total variance each
        # retained component captures; the percentages must sum to 100%.
        p = PCA(n_components=2, svd=True, scale_data=False)
        p.fit_transform(self.ds)
        ve = p.variance_explained()
        self.assertAlmostEqual(sum(ve), 100.0, places=4)

    def test_dominant_component_captured(self):
        # Because feature 0 was scaled 20x, the first principal component should
        # absorb the overwhelming majority of the variance (>80%). Both backends
        # (SVD and eig) must agree on this, confirming they implement the same
        # decomposition.
        # both backends should agree the first PC dominates this data
        for svd in (True, False):
            p = PCA(n_components=2, svd=svd, scale_data=False)
            p.fit_transform(self.ds)
            ve = p.variance_explained()
            self.assertGreater(ve[0], 80.0, msg=f"svd={svd}")


if __name__ == "__main__":
    unittest.main()


class TestKMeansSeeding(unittest.TestCase):
    """The centroid draw used to sample WITH replacement.

    Generator.choice defaults to replace=True, so the same row could be drawn as
    several centroids. Two identical centroids are equidistant from every point,
    argmin always picks the first, and the duplicate therefore receives NO
    points -- whose mean is NaN, behind nothing louder than a RuntimeWarning.
    One NaN centroid makes every later distance NaN and the clustering
    collapses. Measured before the fix: KMeans(k=4) on 12 points returned NaN
    centroids in 17 of 40 runs.
    """

    def test_initial_centroids_are_distinct(self):
        X = np.arange(24, dtype=float).reshape(12, 2)
        for seed in range(20):
            with self.subTest(seed=seed):
                km = KMeans(k=4, random_state=seed)
                centroids = km.init_centroids(Dataset(X, None))
                self.assertEqual(len(np.unique(centroids, axis=0)), 4)

    def test_no_nan_centroids_over_many_runs(self):
        # The observable consequence of the duplicate seeds.
        rng = np.random.RandomState(0)
        for run in range(50):
            with self.subTest(run=run):
                X = rng.rand(12, 2)
                centroids, _ = KMeans(k=4, random_state=run).fit_transform(
                    Dataset(X, None))
                self.assertFalse(np.isnan(centroids).any())

    def test_an_empty_cluster_keeps_its_previous_centroid(self):
        # A cluster can lose all its points as the assignment shifts, and the mean
        # of an empty selection is NaN. It should hold its position instead.
        #
        # Forced through the public API by seeding two IDENTICAL centroids: they
        # are equidistant from every point, argmin always picks the first, so the
        # second is guaranteed to be starved.
        class DuplicateSeeded(KMeans):
            def init_centroids(self, dataset):
                self.centroids = np.array([[0., 0.], [0., 0.], [10., 10.]])
                return self.centroids

        X = np.array([[0., 0.], [0.1, 0.], [5., 5.], [5.1, 5.],
                      [10., 10.], [10.1, 10.]])
        centroids, idxs = DuplicateSeeded(k=3).fit_transform(Dataset(X, None))
        # The point of the fix: no NaN anywhere, and every point still labelled.
        self.assertFalse(np.isnan(centroids).any())
        self.assertTrue(np.isfinite(centroids).all())
        self.assertEqual(len(idxs), len(X))
        # Holding the centroid still (rather than NaN-ing it) leaves it a
        # candidate, and here it recovers: once cluster 0's centroid moves away
        # to the mean of its members, the two points near the origin are closer
        # to the held-still centroid and the starved cluster fills up again.
        self.assertGreater((idxs == 1).sum(), 0)

    def test_a_starved_cluster_would_have_been_nan_before(self):
        # Pins the exact mechanism: the mean of an empty selection is NaN, which
        # is what the guarded update replaces.
        empty = np.empty((0, 2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertTrue(np.isnan(empty.mean(axis=0)).all())

    def test_random_state_makes_a_run_reproducible(self):
        X = three_blobs()
        a = KMeans(k=3, random_state=42).fit_transform(Dataset(X, None))
        b = KMeans(k=3, random_state=42).fit_transform(Dataset(X, None))
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_different_seeds_can_differ(self):
        # K-means only finds a local optimum, so the seed genuinely matters --
        # which is why reproducibility needed to be controllable at all.
        X = np.random.RandomState(3).rand(40, 2)
        a = KMeans(k=4, random_state=0).fit_transform(Dataset(X, None))[0]
        b = KMeans(k=4, random_state=99).fit_transform(Dataset(X, None))[0]
        self.assertFalse(np.allclose(np.sort(a, axis=0), np.sort(b, axis=0)))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            KMeans(k=0)
        with self.assertRaises(ValueError):
            # the loop ran zero times, leaving `idxs` unbound -> UnboundLocalError
            KMeans(k=2, max_iterations=0)

    def test_k_larger_than_the_sample_count_is_rejected(self):
        # Sampling k distinct rows from fewer than k rows is impossible.
        with self.assertRaises(ValueError):
            KMeans(k=10, random_state=0).fit_transform(
                Dataset(np.random.rand(4, 2), None))

    def test_every_run_produces_a_valid_partition(self):
        # What K-means guarantees unconditionally: every point gets exactly one
        # label, and the centroids are finite. It does NOT guarantee finding the
        # true clusters -- see the next test.
        X = three_blobs()
        for seed in range(10):
            with self.subTest(seed=seed):
                centroids, idxs = KMeans(k=3, random_state=seed).fit_transform(
                    Dataset(X, None))
                self.assertEqual(len(idxs), len(X))
                self.assertEqual(centroids.shape, (3, 2))
                self.assertTrue(np.isfinite(centroids).all())
                self.assertTrue(set(np.unique(idxs)) <= {0, 1, 2})

    def test_the_best_of_several_seeds_recovers_the_blobs(self):
        # K-means only reaches a LOCAL optimum, so the outcome is genuinely
        # seed-dependent: on these three blobs 10 of 15 seeds recover them and
        # the rest split one blob in two. Asserting a fixed seed finds the truth
        # would pin an accident, so this follows the practice the class docstring
        # recommends -- run it several times and keep the best result.
        X = three_blobs()
        recovered = []
        for seed in range(10):
            _, idxs = KMeans(k=3, random_state=seed).fit_transform(
                Dataset(X, None))
            sizes = sorted(int((idxs == i).sum()) for i in range(3))
            recovered.append(sizes == [20, 20, 20])
        self.assertTrue(any(recovered),
                        "no seed recovered the three equal blobs")


class TestPCAUsesTrainingStatistics(unittest.TestCase):
    """transform used to re-derive its centring from the data it was given.

    That made the projection blind to any shift: X and X + 100 came out with
    IDENTICAL coordinates, so the transformer could not project held-out data at
    all (and leaked that data's statistics into the result).
    """

    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.randn(50, 4) @ np.diag([3, 2, 1, 0.5]) + 10
        self.ds = Dataset(self.X, None)

    def test_shifted_data_projects_differently(self):
        for scale_data in (True, False):
            with self.subTest(scale_data=scale_data):
                pca = PCA(n_components=2, svd=True, scale_data=scale_data)
                pca.fit(self.ds)
                base = pca.transform(self.ds)
                shifted = pca.transform(Dataset(self.X + 100.0, None))
                self.assertFalse(np.allclose(base, shifted))

    def test_projecting_the_training_data_is_unchanged_by_refits(self):
        pca = PCA(n_components=2, svd=True, scale_data=False)
        first = pca.fit_transform(self.ds)
        second = pca.transform(self.ds)
        np.testing.assert_allclose(first, second)

    def test_transform_before_fit_is_refused(self):
        with self.assertRaises(RuntimeError):
            PCA(n_components=2).transform(self.ds)

    def test_variance_explained_works_straight_after_fit(self):
        # The eigenpair ordering is a property of the fitted model, not of the
        # data being projected, but it used to be computed inside transform --
        # so variance_explained() raised AttributeError after a plain fit().
        pca = PCA(n_components=2)
        pca.fit(self.ds)
        explained = pca.variance_explained()
        self.assertAlmostEqual(sum(explained), 100.0, places=6)

    def test_eigen_decomposition_returns_real_values(self):
        # A real contract worth asserting: the eigenvalues and the projected
        # coordinates must be real.
        #
        # Note this does NOT discriminate between np.linalg.eig and eigh. The
        # covariance path was switched to eigh because a covariance matrix is
        # symmetric and eigh GUARANTEES real, orthonormal results, but measured
        # over 300 random covariance matrices eig never actually produced a
        # complex value either (worst imaginary part 0.0, worst deviation from
        # orthonormality 1.6e-14 against eigh's 2.2e-15). That switch is
        # therefore defensive and faster, not a fix for an observed failure --
        # and no test here would fail if it were reverted.
        pca = PCA(n_components=2, svd=False, scale_data=False)
        reduced = pca.fit_transform(self.ds)
        self.assertFalse(np.iscomplexobj(pca.e_vals))
        self.assertFalse(np.iscomplexobj(reduced))
        # eigh's guarantee, asserted directly
        identity = np.eye(pca.e_vecs.shape[0])
        np.testing.assert_allclose(pca.e_vecs.T @ pca.e_vecs, identity,
                                   atol=1e-10)

    def test_both_paths_agree_on_variance_explained(self):
        results = []
        for svd in (True, False):
            pca = PCA(n_components=2, svd=svd, scale_data=False)
            pca.fit(self.ds)
            results.append(np.round(pca.variance_explained(), 6))
        np.testing.assert_allclose(results[0], results[1])

    def test_more_features_than_samples(self):
        # svd returns min(n_features, n_samples) singular values but a full
        # (n_features, n_features) U, so the eigenvalue and eigenvector counts
        # disagreed and sorting silently dropped components.
        X = np.random.RandomState(1).randn(3, 6)
        pca = PCA(n_components=2, svd=True, scale_data=False)
        reduced = pca.fit_transform(Dataset(X, None))
        self.assertEqual(reduced.shape, (3, 2))
        self.assertEqual(len(pca.e_vals), pca.e_vecs.shape[1])
