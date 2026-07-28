# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Logistic regression module"""
# ---------------------------------------------------------------------------
# Logistic Regression is a linear model for binary classification. It takes the
# linear score z = X . theta and squashes it through the sigmoid into a
# probability h = sigma(z) in (0, 1) that the sample belongs to class 1. We then
# threshold that probability to decide the class.
#
# It is trained by minimising the log-loss (a.k.a. binary cross-entropy):
#
#     J(theta) = -1/m * sum_i [ y_i log(h_i) + (1 - y_i) log(1 - h_i) ]
#
# This is the negative log-likelihood of the data under a Bernoulli model; it
# punishes confident wrong predictions heavily. Although there is no closed-form
# solution, J is convex, so gradient descent reliably finds the global minimum.
# A pleasingly simple fact: the gradient has the same shape as linear
# regression's,
#
#     grad = 1/m * X^T (h - y)
#
# We optionally add L2 regularization (lambda) to shrink the weights and reduce
# overfitting. The bias/intercept term (theta[0]) is deliberately NOT
# regularized — penalising it would just bias the decision boundary's offset.
#
# Use logistic regression as a fast, interpretable linear baseline whenever you
# want calibrated class probabilities.
# ---------------------------------------------------------------------------
from .model import Model
from si.util import sigmoid, add_intercept
import numpy as np


class LogisticRegression(Model):

    def __init__(self,
                 epochs: int = 10000,
                 lr: float = 0.1,
                 threshold: float = 0.5,
                 lbd: float = 1
                 ):
        """ Logistic regression model.

        :param bool gd: If True uses gradient descent (GD) to train the model
            otherwise uses closed form linear algebra. Default False.
        :param int epochs: Number of epochs for GD.
        :param float lr: Learning rate for GD. Default 0.1
        :param threshold: The decision threshold, a value in (0,1). Default 0.5
        :param float ldb: lambda for the regularization. Default 1.
        """
        super(LogisticRegression, self).__init__()
        self.theta = None
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.lbd = lbd

    def fit(self, dataset):
        X, y = dataset.getXy()

        # The log-loss -[y log h + (1-y) log(1-h)] is only defined for targets in
        # {0, 1}: those two terms are meant to select one branch each. Feed it
        # y=2 and both terms stay active with the wrong signs, and the cost comes
        # back as nan -- silently, with no error. String labels fail slightly
        # louder but no more clearly, as a UFuncTypeError from the arithmetic.
        # Check the encoding here so the message names the actual problem.
        labels = set(np.unique(y).tolist()) if y.dtype.kind not in 'US' else set(np.unique(y))
        if not labels <= {0, 1, 0.0, 1.0}:
            raise ValueError(
                "LogisticRegression requires labels in {0, 1} (its log-loss and "
                f"0.5-threshold decision rule assume them); got {sorted(labels, key=str)}. "
                "Encode the positive class as 1 and the negative as 0."
            )

        # Prepend a column of 1s so theta[0] acts as the bias/intercept term.
        X = add_intercept(X)

        self.X = X
        self.y = y

        self.train(X, y)
        self.is_fitted = True

    def train(self, X, y):
        n = X.shape[1]
        m = X.shape[0]
        # history records (weights, cost) at each epoch so the learning curve
        # can be inspected afterwards.
        self.history = {}
        # Start all weights (including the bias) at zero.
        self.theta = np.zeros(n)

        for epoch in range(self.epochs):
            # Linear score then sigmoid -> predicted probability of class 1.
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            # Log-loss gradient: 1/m * X^T (h - y). (h - y) is the per-sample
            # error; X^T projects it back onto each weight.
            gradient = np.dot(X.T, (h - y)) / y.size
            if self.lbd > 0:
                # L2 penalty gradient (lambda/m * theta), added to every weight
                # EXCEPT the bias theta[0] (note the [1:] slice).
                gradient[1:] = gradient[1:] + (self.lbd / m) * self.theta[1:]
            # Gradient-descent step: move the weights against the gradient.
            self.theta -= self.lr * gradient
            self.history[epoch] = [self.theta.copy(), self.cost()]

    def probability(self, x):
        """Return the model's estimated probability that x belongs to class 1."""
        assert self.is_fitted, 'Model must be fit before predicting'
        # Prepend the 1 to match the bias column added during fit.
        _x = np.hstack(([1], x))
        return sigmoid(np.dot(self.theta, _x))

    def predict(self, x):
        # Convert probability to a class label by comparing against `threshold`
        # (0.5 by default): >= threshold -> class 1, otherwise class 0.
        p = self.probability(x)
        res = 1 if p >= self.threshold else 0
        return res

    def cost(self, X=None, y=None, theta=None):
        """Log-loss (binary cross-entropy), optionally with L2 regularization."""
        # Fall back to the stored training data/weights if none are supplied.
        X = add_intercept(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta
        m = X.shape[0]

        h = sigmoid(np.dot(X, theta))
        # Per-sample log-loss: -[y log h + (1-y) log(1-h)]. The two terms switch
        # roles depending on the true label y (one term vanishes for each y).
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h))
        if self.lbd > 0:
            # L2 regularization term lambda/(2m) * ||theta||^2, summing over the
            # weights but NOT the bias theta[0] (hence theta[1:]).
            reg = np.dot(theta[1:], theta[1:]) * self.lbd / (2 * m)
            res = (np.sum(cost) / m) + reg
        else:
            res = np.sum(cost) / m
        return res
