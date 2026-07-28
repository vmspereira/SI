# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""k-nearest neighbors module"""
# ---------------------------------------------------------------------------
from .model import Model
from si.util import l2_distance, accuracy_score, mse
import numpy as np


class KNN(Model):
    def __init__(self, num_neighbors: int, classification: bool = True):
        """
        k-nearest neighbors algorithm.

        “Tell me with whom you associate, and I will tell you who you are.”
            ― Johann Wolfgang von Goethe

        KNN is based on the notion that close data points are more likely to share
        a common label.

        :param (int) num_neighbors: Number of closest neighbors to consider in the inference.
        :param (bool) classification: If a classification or regression task.
            Default classification (True).

        """
        super().__init__()
        if num_neighbors < 1:
            # k=0 selected an empty neighbour set, and the majority vote then
            # failed with "ValueError: max() iterable argument is empty".
            raise ValueError(
                f"num_neighbors must be at least 1; got {num_neighbors}.")
        self.num_neighbors = num_neighbors
        self.classification = classification

    def fit(self, dataset):
        # KNN is a "lazy" learner: there is no model to estimate at fit time.
        # We simply memorise the training set; all the work happens at predict
        # time, when we look up the neighbours of the query point.
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):
        # Compute the (Euclidean / L2) distance from the query point x to every
        # training sample. l2_distance broadcasts x against the rows of
        # self.dataset.X, returning a 1D array of length n_samples.
        distances = l2_distance(x, self.dataset.X)
        # argsort gives the indices that would sort the distances ascending, so
        # the first entries are the closest training points.
        sorted_index = np.argsort(distances)
        # Keep only the indices of the k (= num_neighbors) nearest points.
        return sorted_index[:self.num_neighbors]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        # Find the k nearest training points and collect their target values.
        neighbors = self.get_neighbors(x)
        values = self.dataset.y[neighbors].tolist()
        if self.classification:
            # for classification we consider as label the modal one.
            # i.e. a majority vote among the k neighbours: the label that
            # appears most often (set(values) are the candidate labels,
            # values.count is the tie-breaking key).
            prediction = max(set(values), key=values.count)
        else:
            # for regression we consider the average of the k neighbor labels.
            # the prediction is the mean target of the k neighbours.
            prediction = sum(values) / len(values)
        return prediction

    def cost(self, X=None, y=None):
        """Accuracy for classification, mean squared error for regression.

        The metric used to follow the classification branch unconditionally, so a
        regressor built with classification=False was scored with accuracy_score
        -- comparing continuous predictions for exact equality with continuous
        targets. That is essentially always 0: a KNN regressor predicting
        0.5, 0.5, 1.5, 2.5 against targets 0, 1, 2, 3 reported cost() == 0.0
        while performing perfectly reasonably. The metric now follows the mode
        the model was built in.
        """
        # Default to the stored training data if no evaluation set is given.
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        # predict() expects one sample at a time. We transpose X so that
        # apply_along_axis (axis=0) feeds each original row (a sample) to
        # predict, producing one prediction per sample.
        y_pred = np.ma.apply_along_axis(self.predict,
                                        axis=0, arr=X.T)
        if self.classification:
            return accuracy_score(y, y_pred)
        # Regression: lower is better, unlike the accuracy above.
        return mse(y, y_pred)
