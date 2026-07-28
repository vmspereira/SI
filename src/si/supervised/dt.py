# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Decision Tree module"""
# ---------------------------------------------------------------------------
# A Decision Tree classifier learns a hierarchy of yes/no questions of the form
# "is feature j <= threshold?". Each question splits the data in two; the goal
# is to choose splits that make the resulting groups as "pure" as possible
# (ideally each leaf contains a single class). Prediction then routes a sample
# down the tree until it reaches a leaf and returns that leaf's class
# distribution.
#
# Use a decision tree when you want an interpretable, non-linear model that
# needs little data preprocessing (no scaling required) and can capture feature
# interactions. Single trees overfit easily, which motivates ensembles such as
# Random Forests (see rf.py).
#
# "Impurity" measures how mixed the classes are in a node. Two common measures:
#   - Entropy:  H = -sum_c p_c * log2(p_c)   (from information theory; bits)
#   - Gini:     G = 1 - sum_c p_c^2          (prob. of misclassifying a random
#                                             element labelled by the node's
#                                             class distribution)
# Both are 0 for a pure node and maximal when classes are evenly mixed.
# A split is scored by its Information Gain: the impurity before the split minus
# the (sample-weighted) average impurity of the two children.
# ---------------------------------------------------------------------------
from .model import Model
from ..util import accuracy_score
import numpy as np


def entropy(y):
    """Calculates the entropy of a label vector y.

    Entropy H = -sum_c p_c * log2(p_c), where p_c is the fraction of samples
    of class c. It measures the average "surprise" / disorder of the labels:
    0 when all samples share one class (perfectly predictable) and maximal
    when the classes are uniformly distributed.
    """
    # bincount counts occurrences of each integer label -> per-class counts.
    hist = np.bincount(y)
    # Convert counts to probabilities p_c.
    ps = hist / len(y)
    # Sum only over classes that are actually present (p > 0) to avoid
    # log2(0) = -inf. Negated so the result is non-negative.
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def gini(probas):
    """Calculates the Gini impurity from a vector of class probabilities.

    Gini = 1 - sum_c p_c^2. It is the probability that two items drawn at
    random (with replacement) from the node have different labels, i.e. the
    chance of misclassifying a randomly labelled element. 0 for a pure node,
    largest when classes are balanced.
    """
    return 1 - np.sum(probas**2)


class Node:
    """Implementation of a simple binary tree for DT classifier.

    Each node either asks a question ("is feature `column` <= `threshold`?",
    routing samples to `left`/`right`) or is a leaf (`is_terminal`) that stores
    the class distribution `probas` used for prediction.
    """

    def __init__(self):
        # Child nodes reached when the split test is false (left) / true (right).
        self.right = None
        self.left = None
        # derived from splitting criteria
        # column = index of the feature this node splits on;
        # threshold = the value samples are compared against.
        self.column = None
        self.threshold = None
        # probability for object inside the Node to belong
        # for each of the given classes
        # (a vector of class frequencies for the samples that reached this node)
        self.probas = None
        # depth of the given node
        self.depth = None
        # if it is the root Node or not
        # a terminal (leaf) node makes no further split; it returns `probas`.
        self.is_terminal = False


class DecisionTree(Model):
    def __init__(
        self, max_depth: int = 3, min_samples_leaf: int = 1, min_samples_split: int = 2
    ) -> None:
        """Decision Tree classifier.

        The hyper-parameters below are pre-pruning / stopping rules that limit
        how big the tree can grow, trading bias for variance to curb overfitting.

        :param int max_depth: Maximum depth of the tree (longest root-to-leaf
            path). Shallower trees generalise better but may underfit.
        :param int min_samples_leaf: Minimum number of samples a child must
            contain for a split to be accepted.
        :param int min_samples_split: Minimum number of samples a node must
            contain to even be considered for splitting.
        """

        super().__init__()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        # Decision tree itself
        self.Tree = None

    def node_probs(self, y):
        """
        Calculates probability of class in a given node.

        Returns a vector aligned with self.classes giving the fraction of the
        node's samples belonging to each class. This vector is both stored on
        leaves (for prediction) and fed to the impurity measure.
        """
        probas = []
        # for each unique label calculate the probability for it
        for one_class in self.classes:
            # count of samples of this class / total samples in the node.
            proba = y[y == one_class].shape[0] / y.shape[0]
            probas.append(proba)
        return np.asarray(probas)

    def calc_impurity(self, y):
        """
        Wrapper for the impurity calculation. Calculates probabilities
        first and then passses them to the Gini criterion.

        The gini impurity measures the frequency at which any element
        of the dataset will be mislabelled when it is randomly labeled.

        The minimum value of the Gini Index is 0. This happens when the
        node is pure, this means that all the contained elements in the
        node are of one unique class. Therefore, this node will not be
        split again. Thus, the optimum split is chosen by the features
        with less Gini Index. Moreover, it gets the maximum value when
        the probability of the two classes are the same.

        """
        return gini(self.node_probs(y))

    # TODO: Entropy criterion #######################################

    def calc_best_split(self, X, y):
        """
        Calculates the best possible split for the concrete node of the tree.

        Greedy, exhaustive search: it tries every (feature, threshold) pair
        observed in the data and keeps the one with the highest Information
        Gain. Information Gain is

            IG = impurity(parent)
                 - [ (n_left / n) * impurity(left)
                     + (n_right / n) * impurity(right) ]

        i.e. how much the (sample-weighted) impurity drops after splitting.
        A larger drop means the split better separates the classes.
        """

        bestSplitCol = None
        bestThresh = None
        # Sentinel: any real Information Gain (>= 0 here) will beat it, so the
        # first valid split is always recorded.
        bestInfoGain = -999

        # Impurity of the parent node, computed once and reused for every split.
        impurityBefore = self.calc_impurity(y)

        # for each column in X
        for col in range(X.shape[1]):
            x_col = X[:, col]

            # for each value in the column
            # candidate thresholds are simply the observed feature values.
            for x_i in x_col:
                threshold = x_i
                # Partition the labels by the test "feature value > threshold".
                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]

                # Skip degenerate splits that send every sample to one side.
                if y_right.shape[0] == 0 or y_left.shape[0] == 0:
                    continue

                # calculate impurity for the right and left nodes
                impurityRight = self.calc_impurity(y_right)
                impurityLeft = self.calc_impurity(y_left)

                # calculate information gain
                # weight each child's impurity by its share of the samples.
                infoGain = impurityBefore
                infoGain -= (impurityLeft * y_left.shape[0] / y.shape[0]) + (
                    impurityRight * y_right.shape[0] / y.shape[0]
                )

                # is this infoGain better then all other?
                # keep the split that most reduces impurity.
                if infoGain > bestInfoGain:
                    bestSplitCol = col
                    bestThresh = threshold
                    bestInfoGain = infoGain

        # if we still didn't find the split
        # (no candidate ever improved on the sentinel -> no usable split).
        if bestInfoGain == -999:
            return None, None, None, None, None, None

        # making the best split
        # re-partition both X and y using the winning feature/threshold so the
        # caller can recurse on the two halves.

        x_col = X[:, bestSplitCol]
        x_left, x_right = X[x_col <= bestThresh, :], X[x_col > bestThresh, :]
        y_left, y_right = y[x_col <= bestThresh], y[x_col > bestThresh]

        return bestSplitCol, bestThresh, x_left, y_left, x_right, y_right

    def build_dt(self, X, y, node):
        """
        Recursively builds decision tree from the top to bottom.

        Starting at `node`, it picks the best split for (X, y), creates two
        child nodes, and recurses on each half. Recursion stops when a stopping
        rule fires, in which case the node becomes a leaf and keeps its stored
        class distribution as the prediction.
        """
        # checking for the terminal conditions
        # Stop 1: reached the maximum allowed depth.
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        # Stop 2: too few samples to justify another split.
        if X.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        # Stop 3: node is already pure (a single class) -> nothing to gain.
        if np.unique(y).shape[0] == 1:
            node.is_terminal = True
            return

        # calculating current split
        splitCol, thresh, x_left, y_left, x_right, y_right = self.calc_best_split(X, y)

        if splitCol is None:
            # no valid split was found (e.g. all rows identical): stop here
            # before dereferencing the (None) child splits below
            node.is_terminal = True
            return

        # Stop 4: an accepted split would create a child smaller than allowed.
        if (
            x_left.shape[0] < self.min_samples_leaf
            or x_right.shape[0] < self.min_samples_leaf
        ):
            node.is_terminal = True
            return

        # Record the winning test on this (now internal) node.
        node.column = splitCol
        node.threshold = thresh

        # creating left and right child nodes
        # each child is one level deeper and stores the class distribution of
        # the samples routed to it.
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.probas = self.node_probs(y_left)

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.probas = self.node_probs(y_right)

        # splitting recursevely
        # grow each subtree on its corresponding half of the data.
        self.build_dt(x_right, y_right, node.right)
        self.build_dt(x_left, y_left, node.left)

    def fit(self, dataset):
        self.dataset = dataset
        X, y = dataset.getXy()
        # the dataset classes
        # fix the global class ordering so every node's `probas` vector aligns.
        self.classes = np.unique(y)
        # root node creation
        # the root sits at depth 1 and starts with the full dataset's class mix.
        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.probas = self.node_probs(y)
        # grow the whole tree starting from the root.
        self.build_dt(X, y, self.Tree)
        self.is_fitted = True

    def predict_sample(self, x, node):
        """
        Passes one object through decision tree and return the probability of
        it to belong to each class.

        Routing rule: at each internal node, go right if the sample's feature
        value exceeds the threshold, otherwise go left, until a leaf is hit.
        """
        assert self.is_fitted, "Model must be fit before predicting"
        # if we have reached the terminal node of the tree
        # a leaf stores the answer: its class distribution.
        if node.is_terminal:
            return node.probas

        # Apply this node's split test and recurse into the matching child.
        if x[node.column] > node.threshold:
            probas = self.predict_sample(x, node.right)
        else:
            probas = self.predict_sample(x, node.left)
        return probas

    def predict(self, x):
        assert self.is_fitted, "Model must be fit before predicting"
        # The predicted class is the most probable one at the reached leaf.
        # argmax over the class-distribution vector returns the class index.
        pred = np.argmax(self.predict_sample(x, self.Tree))
        return pred

    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        # predict() works on a single sample; transpose so apply_along_axis
        # (axis=0) feeds each original row to it -> one prediction per sample.
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        return accuracy_score(y, y_pred)
