# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Logistic regression module"""
# ---------------------------------------------------------------------------
from .model import Model
from ..util import sigmoid, add_intersect
import numpy as np


class LogisticRegression(Model):

    def __init__(self, epochs=10000, lr=0.1):
        """ Logistic regression model.

        :param bool gd: If True uses gradient descent (GD) to train the model\
            otherwise closed form lineal algebra. Default False.
        :param int epochs: Number of epochs for GD.
        :param float lr: Learning rate for GD.
        """
        super(LogisticRegression, self).__init__()
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self, dataset):
        X, y = dataset.getXy()
        X = add_intersect(X)
        
        self.X = X
        self.y = y
        
        self.train(X, y)
        self.is_fitted = True

    def train(self, X, y):
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            self.history[epoch] = [self.theta.copy(), self.cost()]

    def probability(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        _x = np.hstack(([1], x))
        return sigmoid(np.dot(self.theta, _x))

    def predict(self, x):
        p = self.probability(x)
        res = 1 if p >= 0.5 else 0
        return res

    def cost(self, X=None, y=None, theta=None):
        X = add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta

        h = sigmoid(np.dot(X, theta))
        cost = (-y * np.log(h) - (1-y) * np.log(1-h))
        res = np.sum(cost) / X.shape[0]
        return res


class LogisticRegressionReg(LogisticRegression):

    def __init__(self, epochs=1000, lr=0.1, lbd=1):
        """ Linear regression model with L2 regularization.

        :param bool gd: If True uses gradient descent (GD) to train the model\
            otherwise closed form lineal algebra. Default False.
        :param int epochs: Number of epochs for GD.
        :param float lr: Learning rate for GD.
        :param float ldb: lambda for the regularization.
        """
        super(LogisticRegressionReg, self).__init__(epochs=epochs, lr=lr)
        self.lbd = lbd

    def train(self, X, y):
        n = X.shape[1]
        m = X.shape[0]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            gradient[1:] = gradient[1:] + (self.lbd / m) * self.theta[1:]
            self.theta -= self.lr * gradient
            self.history[epoch] = [self.theta[:], self.cost()]

    def cost(self, X=None, y=None, theta=None):
        X = add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta

        m = X.shape[0]
        p = sigmoid(np.dot(X, theta))
        cost = (-y * np.log(p) - (1-y) * np.log(1-p))
        reg = np.dot(theta[1:], theta[1:]) * self.lbd / (2*m)
        res = (np.sum(cost) / m) + reg
        return res
