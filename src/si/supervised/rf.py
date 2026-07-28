# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Random Forrest module"""
# ---------------------------------------------------------------------------
# A Random Forest is an *ensemble* of decision trees. A single tree has high
# variance: small changes in the data give very different trees, and a deep tree
# overfits. The forest reduces this variance by averaging many de-correlated
# trees, each trained on a slightly different view of the data. Predictions are
# combined by majority vote (classification).
#
# Two sources of randomness make the trees different from one another:
#   1. Bagging (Bootstrap AGGregatING): each tree is trained on a random subset
#      / bootstrap sample of the training rows.
#   2. Feature subsampling: at training each tree only sees a random subset of
#      the features (here, by default sqrt(n_features)). This stops one strong
#      feature from dominating every tree, further de-correlating them.
#
# Use a Random Forest when you want a strong, low-variance, low-tuning baseline
# that handles non-linear relationships and feature interactions out of the box.
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
        # sqrt(n_features) is the classic Random Forest default for
        # classification: enough features for useful splits, few enough to keep
        # the trees de-correlated.
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        # Choose one random subset of the data for each tree
        # (bagging: one bootstrap/random sample of the rows per tree).
        subsets = get_random_subsets(X, y, self.n_estimators)

        for i in range(self.n_estimators):
            X_subset, y_subset = subsets[i]
            # Feature bagging (select random subsets of the features)
            # pick max_features column indices at random for this tree, so each
            # tree learns from a different feature view.
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            # Save the indices of the features for prediction
            # we must apply the SAME columns at predict time, so remember them.
            self.trees[i].feature_indices = idx
            # Choose the features corresponding to the indices
            # restrict this tree's training data to its chosen columns.
            X_subset = X_subset[:, idx]
            # Fit the tree to the data
            self.trees[i].fit(Dataset(X_subset, y_subset))
        
        self.is_fitted = True
        
    def predict(self, X):
        # y_preds[s, t] = the vote of tree t for sample s.
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # Let each tree make a prediction on the data
        for i, tree in enumerate(self.trees):
            # Indices of the features that the tree has trained on
            # reuse exactly the columns this tree saw during fit.
            idx = tree.feature_indices
            # DecisionTree.predict works one sample at a time, so predict each
            # row of the (feature-bagged) input separately.
            X_subset = X[:, idx]
            y_preds[:, i] = [tree.predict(sample) for sample in X_subset]

        y_pred = []
        # For each sample
        # combine the trees by majority vote: bincount tallies the per-class
        # votes across trees and argmax picks the most-voted class.
        for sample_predictions in y_preds:
            # Select the most common class prediction
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred
    
    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        # predict() already takes the full 2D feature matrix
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
