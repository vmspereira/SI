# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Linear Regression module"""
# ---------------------------------------------------------------------------
from .model import Model
from ..util import mse, add_intersect
import numpy as np


class LinearRegression(Model):

    def __init__(self, gd=False, epochs=1000, lr=0.001):
        """ Linear regression model.

        :param bool gd: If True uses gradient descent (GD) to train the model\
            otherwise closed form lineal algebra. Default False.
        :param int epochs: Number of epochs for GD.
        :param float lr: Learning rate for GD.
        :param float tolerance: The
        """
        super(LinearRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self, dataset):
        X, y = dataset.getXy()
        X = add_intersect(X)
        self.X = X
        self.y = y
        # Closed form or GD
        self.train_gd(X, y) if self.gd else self.train_closed(X, y)
        self.is_fitted = True

    def train_closed(self, X, y):
        """ Uses closed form linear algebra to fit the model.
        theta = inv(XT*X)*XT*y
        """
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def train_gd(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            grad = 1/m * (X.dot(self.theta)-y).dot(X)
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta.copy(), self.cost()]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        _x = np.hstack(([1], x))
        return np.dot(self.theta, _x)

    def cost(self, X=None, y=None, theta=None):
        X = add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta

        y_pred = np.dot(X, theta)
        return mse(y, y_pred)/2


class LinearRegressionReg(LinearRegression):

    def __init__(self, gd=False, epochs=1000, lr=0.001, lbd=1):
        """ Linear regression model with L2 regularization.

        :param bool gd: If True uses gradient descent (GD) to train the model\
            otherwise closed form lineal algebra. Default False.
        :param int epochs: Number of epochs for GD.
        :param float lr: Learning rate for GD.
        :param float ldb: lambda for the regularization.
        """
        super(LinearRegressionReg, self).__init__(gd=gd, epochs=epochs, lr=lr)
        self.lbd = lbd

    def train_closed(self, X, y):
        """ Uses closed form linear algebra to fit the model.
            theta = inv(XT*X+lbd*I)*XT*y
        """
        n = X.shape[1]
        identity = np.eye(n)
        identity[0, 0] = 0
        self.theta = np.linalg.inv(X.T.dot(X)+self.lbd*identity).dot(X.T).dot(y)
        self.is_fitted = True

    def train_gd(self, X, y):
        """ Uses gradient descent to fit the model."""
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        lbds = np.full(m, self.lbd)
        lbds[0] = 0
        for epoch in range(self.epochs):
            grad = (X.dot(self.theta)-y).dot(X)
            self.theta -= (self.lr/m) * (lbds+grad)
            self.history[epoch] = [self.theta.copy(), self.cost()]
