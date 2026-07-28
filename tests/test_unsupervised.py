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
