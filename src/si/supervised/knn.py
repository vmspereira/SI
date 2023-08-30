# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""k-nearest neighbors module"""
# ---------------------------------------------------------------------------
from .model import Model
from si.util import l2_distance, accuracy_score
import numpy as np


class KNN(Model):
    def __init__(self, num_neighbors:int, classification:bool=True):
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
        super(KNN).__init__()
        self.num_neighbors = num_neighbors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):
        distances = l2_distance(x, self.dataset.X)
        sorted_index = np.argsort(distances)
        return sorted_index[:self.num_neighbors]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        neighbors = self.get_neighbors(x)
        values = self.dataset.y[neighbors].tolist()
        if self.classification:
            # for classification we consider as label the modal one.
            prediction = max(set(values), key=values.count)
        else:
            # for regression we consider the average of the k neighbor labels.
            prediction = sum(values)/len(values)
        return prediction

    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = np.ma.apply_along_axis(self.predict,
                                        axis=0, arr=X.T)
        return accuracy_score(y, y_pred)
