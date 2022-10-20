# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Decision Tree module"""
# ---------------------------------------------------------------------------
from .model import Model
from ..util import accuracy_score
import numpy as np


class Node:
    """Implementation of a simple binary tree for DT classifier."""

    def __init__(self):
        self.right = None
        self.left = None
        # derived from splitting criteria
        self.column = None
        self.threshold = None
        # probability for object inside the Node to belong 
        # for each of the given classes
        self.probas = None
        # depth of the given node
        self.depth = None
        # if it is the root Node or not
        self.is_terminal = False


class DecisionTree(Model):

    def __init__(self, max_depth=3, min_samples_leaf=1, min_samples_split=2):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        # Decision tree itself
        self.Tree = None

    def node_probs(self, y):
        """
        Calculates probability of class in a given node
        """
        probas = []
        # for each unique label calculate the probability for it
        for one_class in self.classes:
            proba = y[y == one_class].shape[0] / y.shape[0]
            probas.append(proba)
        return np.asarray(probas)

    def gini(self, probas):
        """Calculates gini criterion"""
        return 1 - np.sum(probas**2)

    def calc_impurity(self, y):
        '''Wrapper for the impurity calculation. Calculates probas 
           first and then passses them
        to the Gini criterion.
        '''
        return self.gini(self.nodeProbas(y))

    def calc_best_split(self, X, y):
        '''Calculates the best possible split for the concrete node of the tree'''

        bestSplitCol = None
        bestThresh = None
        bestInfoGain = -999

        impurityBefore = self.calc_impurity(y)

        # for each column in X
        for col in range(X.shape[1]):
            x_col = X[:, col]

            # for each value in the column
            for x_i in x_col:
                threshold = x_i
                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]

                if y_right.shape[0] == 0 or y_left.shape[0] == 0:
                    continue

                # calculate impurity for the right and left nodes
                impurityRight = self.calc_impurity(y_right)
                impurityLeft = self.calc_impurity(y_left)

                # calculate information gain
                infoGain = impurityBefore
                infoGain -= (impurityLeft * y_left.shape[0] / y.shape[0]) + \
                    (impurityRight * y_right.shape[0] / y.shape[0])

                # is this infoGain better then all other?
                if infoGain > bestInfoGain:
                    bestSplitCol = col
                    bestThresh = threshold
                    bestInfoGain = infoGain

        # if we still didn't find the split
        if bestInfoGain == -999:
            return None, None, None, None, None, None

        # making the best split

        x_col = X[:, bestSplitCol]
        x_left, x_right = X[x_col <= bestThresh, :], X[x_col > bestThresh, :]
        y_left, y_right = y[x_col <= bestThresh], y[x_col > bestThresh]

        return bestSplitCol, bestThresh, x_left, y_left, x_right, y_right

    def build_dt(self, X, y, node):
        '''
        Recursively builds decision tree from the top to bottom
        '''
        # checking for the terminal conditions
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        if X.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        if np.unique(y).shape[0] == 1:
            node.is_terminal = True
            return

        # calculating current split
        splitCol, thresh, x_left, y_left, x_right, y_right = self.calc_best_split(X, y)

        if splitCol is None:
            node.is_terminal = True

        if x_left.shape[0] < self.min_samples_leaf or x_right.shape[0] < self.min_samples_leaf:
            node.is_terminal = True
            return

        node.column = splitCol
        node.threshold = thresh

        # creating left and right child nodes
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.probas = self.node_probs(y_left)

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.probas = self.node_probs(y_right)

        # splitting recursevely
        self.buildDT(x_right, y_right, node.right)
        self.buildDT(x_left, y_left, node.left)

    def fit(self, dataset):
        self.dataset = dataset
        X, y = dataset.getXy()
        # the dataset classes
        self.classes = np.unique(y)
        # root node creation
        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.probas = self.nodeProbas(y)
        self.build_dt(X, y, self.Tree)
        self.is_fitted = True

    def predict_sample(self, x, node):
        '''
        Passes one object through decision tree and return the probability of 
        it to belong to each class
        '''
        assert self.is_fitted, 'Model must be fit before predicting'
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.probas

        if x[node.column] > node.threshold:
            probas = self.predict_sample(x, node.right)
        else:
            probas = self.predict_sample(x, node.left)
        return probas

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        pred = np.argmax(self.predict_sample(x, self.Tree))
        return pred

    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = np.ma.apply_along_axis(self.predict,
                                        axis=0, arr=X.T)
        return accuracy_score(y, y_pred)
