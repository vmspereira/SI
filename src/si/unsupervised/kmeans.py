# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""KMeans module"""
# ---------------------------------------------------------------------------
import numpy as np
from ..util import l2_distance


class KMeans:

    def __init__(self, k: int, max_iterations=1000, distance=l2_distance) -> None:
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.distance = distance

    def fit(self, dataset):
        x = dataset.X
        self._min = np.min(x, axis=0)
        self._max = np.max(x, axis=0)

    def init_centroids(self, dataset):
        """Generates k centroids.
        There are many ways to achieve that...

        :param dataset: The dataset object
        """
        X = dataset.X
        rng = np.random.default_rng()
        centroids = rng.choice(X, self.k)
        return centroids

    def get_closest_centroid(self, x):
        """Identifies the index of the centroid closest to point x.
        :param x: a point (numpy.array)
        """
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index

    def transform(self, dataset):
        # generates initial centroids
        self.init_centroids(dataset)
        X = dataset.X

        # initialize stopping decision variables
        changed = True
        count = 0
        old_idxs = np.zeros(X.shape[0])

        # main cicle
        while changed and count < self.max_iterations:
            # array of indexes of nearest centroid
            idxs = np.apply_along_axis(self.get_closest_centroid,
                                       axis=0, arr=X.T)
            # compute the new centroids
            cent = [np.mean(X[idxs == i], axis=0) for i in range(self.k)]
            self.centroids = np.array(cent)
            # verify if the clustering has changed
            # and increment counting
            changed = np.any(old_idxs != idxs)
            old_idxs = idxs
            count += 1
        return self.centroids, idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
