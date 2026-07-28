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
                 max_iterations: int = 1000,
                 distance: callable = l2_distance
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

        The objective (what K-means tries to minimize)
        -----------------------------------------------
        Given k clusters C_1, ..., C_k with centroids mu_1, ..., mu_k, K-means
        searches for the assignment of points to clusters that minimizes the
        total within-cluster sum of squared distances (the "intra-cluster
        variance" or "inertia"):

            J = sum over clusters i  ( sum over points x in C_i  || x - mu_i ||^2 )

        Intuitively: we want every point to sit as close as possible to the
        centre of its own cluster, so clusters are tight and well separated.

        Lloyd's algorithm (the loop implemented in `transform`)
        -------------------------------------------------------
        Minimizing J exactly is NP-hard, so we use an iterative heuristic that
        alternates two cheap steps until the assignment stops changing:
            1. ASSIGN : attach each point to its nearest centroid.
            2. UPDATE : move each centroid to the mean of the points assigned
                        to it (the mean is the point that minimizes the squared
                        distance to a set of points, hence it lowers J).
        Each step can only decrease (or leave unchanged) J, so the algorithm
        always converges -- but only to a LOCAL minimum.

        Why is it non-deterministic / sensitive to initialization?
        ----------------------------------------------------------
        Because the final local minimum reached depends on where the centroids
        start. Different random seeds can produce different clusterings. There
        are some approach such as kmean++ where the initial centroids are
        selected using a weighted probability distribution proportional to the
        square of distances to the nearest random seeds, which tends to spread
        the seeds out and gives more reliable results. A common practice is to
        run K-means several times and keep the run with the lowest J.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.distance = distance

    def fit(self, dataset):
        # `fit` here just records the per-feature range (min/max) of the data.
        # It does not learn the centroids -- the clustering work happens in
        # `transform`. Storing the feature bounds is useful, for example, if
        # one wanted to seed centroids uniformly inside the data range.
        x = dataset.X
        self._min = np.min(x, axis=0)  # shape (n_features,)
        self._max = np.max(x, axis=0)  # shape (n_features,)

    def init_centroids(self, dataset):
        """Generates the k initial centroids (the "seeds").

        Recall that K-means only finds a LOCAL optimum, so this starting
        choice directly influences the final clustering.

        There are many ways to achieve that...
        Here, the centroids are points randomly selected
        from the dataset. Picking actual data points (rather than arbitrary
        coordinates) guarantees each seed lies inside the data cloud.

        :param dataset: The dataset object
        """
        X = dataset.X
        rng = np.random.default_rng()
        # Randomly draw k rows of X to use as the starting centroids.
        # `centroids` ends up with shape (k, n_features).
        self.centroids = rng.choice(X, self.k)
        return self.centroids

    def get_closest_centroid(self, x):
        """
        Identifies the index of the centroid closest to point x.

        This is the ASSIGN step applied to a single point: compute the
        distance from x to every centroid and return the index of the
        smallest one.

        :param x: a point (numpy.array)
        """
        # dist has one entry per centroid: the distance from x to each.
        dist = self.distance(x, self.centroids)
        # argmin gives the position (cluster label) of the nearest centroid.
        closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index

    def transform(self, dataset):
        # generates initial centroids
        self.init_centroids(dataset)
        X = dataset.X

        # initialize stopping decision variables
        changed = True              # did the assignment change last iteration?
        count = 0                   # iteration counter (safety cap)
        old_idxs = np.zeros(X.shape[0])  # previous cluster label of each point

        # main cicle -- Lloyd's algorithm: repeat ASSIGN + UPDATE until the
        # cluster labels stop changing (convergence) or we hit max_iterations.
        while changed and count < self.max_iterations:
            # ASSIGN step: label every point with its nearest centroid.
            # We pass X.T (features along axis 0) so apply_along_axis feeds one
            # data point at a time to get_closest_centroid. `idxs` is an array
            # of length n_samples holding each point's cluster index.
            idxs = np.apply_along_axis(self.get_closest_centroid,
                                       axis=0, arr=X.T)
            # UPDATE step: each new centroid is the mean of the points
            # currently assigned to it. X[idxs == i] selects the rows belonging
            # to cluster i; their column-wise mean is the new centroid.
            cent = [np.mean(X[idxs == i], axis=0) for i in range(self.k)]
            self.centroids = np.array(cent)
            # CONVERGENCE check: if no point switched clusters this round, the
            # assignment is stable and the loop will exit. Otherwise remember
            # the current labels and keep iterating.
            changed = np.any(old_idxs != idxs)
            old_idxs = idxs
            count += 1
        # Return the final centroids and the cluster label of each point.
        return self.centroids, idxs

    def fit_transform(self, dataset):
        # Convenience method: record the data range (fit) and then run the
        # clustering loop (transform) in a single call.
        self.fit(dataset)
        return self.transform(dataset)
