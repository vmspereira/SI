# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Random Forrest module"""
# ---------------------------------------------------------------------------

import numpy as np
import math
from .model import Model
from si.data import Dataset
from .dt import DecisionTree
from si.util import accuracy_score, get_random_subsets

class RandomForest(Model):
    """Random Forest classifier. Uses a collection of decision trees that
    trains on random subsets of the data using a random subsets of the features.

    :param in n_estimators: The number of classification trees that are used.
    :param int max_features: The maximum number of features that the classification 
        trees are allowed to use.
    :param int min_samples_split: The minimum number of samples needed to make a split
        when building a tree.
    :param int max_depth: The maximum depth of a tree.
    """
    def __init__(self, n_estimators=100, 
                 max_features=None, 
                 min_samples_split=2,
                 max_depth=float("inf")):
        
        # Number of trees
        self.n_estimators = n_estimators
        # Maxmimum number of features per tree
        self.max_features = max_features
            
        self.min_samples_split = min_samples_split
        # Maximum depth for tree            
        self.max_depth = max_depth          
        
        # Initialize decision trees
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                DecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                    ))

    def fit(self, dataset):
        self.dataset = dataset
        X, y = dataset.getXy()
        n_features = np.shape(X)[1]
        # If max_features have not been defined => select it as
        # sqrt(n_features)
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        # Choose one random subset of the data for each tree
        subsets = get_random_subsets(X, y, self.n_estimators)

        for i in self.progressbar(range(self.n_estimators)):
            X_subset, y_subset = subsets[i]
            # Feature bagging (select random subsets of the features)
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            # Save the indices of the features for prediction
            self.trees[i].feature_indices = idx
            # Choose the features corresponding to the indices
            X_subset = X_subset[:, idx]
            # Fit the tree to the data
            self.trees[i].fit(Dataset(X_subset, y_subset))
        
        self.is_fitted = True
        
    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # Let each tree make a prediction on the data
        for i, tree in enumerate(self.trees):
            # Indices of the features that the tree has trained on
            idx = tree.feature_indices
            # Make a prediction based on those features
            prediction = tree.predict(X[:, idx])
            y_preds[:, i] = prediction
            
        y_pred = []
        # For each sample
        for sample_predictions in y_preds:
            # Select the most common class prediction
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred
    
    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        return accuracy_score(y, y_pred)
