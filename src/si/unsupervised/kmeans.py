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

    def __init__(self, 
                 k: int, 
                 max_iterations: int=1000, 
                 distance:callable=l2_distance
                 ) -> None:
        """ 
        KMeans algorithm.
        
        :param (int) k: number of clusters
        :param (int) max_iterations: Maximum number of iterations to run if the\
            algorithm does not converge. Default 1000
        :param (callable) distance: Distance function. Default euclidean distance.

        -----------------------------
        K-means groups object acording to their similarity by minimize the intra-class 
        variance.
        K-means is non-deterministic and depends on the choice of the initial
        centroids (seeds), the 'centers' of the initial clusters. There are 
        some approach such as kmean++ where the initial centroids are selected using a
        weighted probability distribution proportional to the square of distances to 
        the nearest random seeds.
        """
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
        Here, the centroids are points randomly selected 
        from the dataset.
        
        :param dataset: The dataset object
        """
        X = dataset.X
        rng = np.random.default_rng()
        self.centroids = rng.choice(X, self.k)
        return self.centroids

    def get_closest_centroid(self, x):
        """
        Identifies the index of the centroid closest to point x.
        
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
            # and increment the counter
            changed = np.any(old_idxs != idxs)
            old_idxs = idxs
            count += 1
        return self.centroids, idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
