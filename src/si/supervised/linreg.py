# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Linear Regression module"""
# ---------------------------------------------------------------------------
from .model import Model
from si.util import mse, add_intersect
import numpy as np


class LinearRegression(Model):

    def __init__(self, epochs=1000, lr=0.001, ldb=1, gd=False):
        """ Linear regression model.

        :param int epochs: Number of epochs for GD.
        :param float lr: Learning rate for GD.
        :param float ldb: lambda for the regularization. 
            If non positive, L2 regularization is not applied.
        :param bool gd: If True uses gradient descent (GD) to train the model\
            otherwise uses closed form linear algebra. Default False. 
        """
        super(LinearRegression, self).__init__()
        self.theta = None
        self.epochs = epochs
        self.lr = lr
        self.lbd=ldb
        self.gd = gd

    def fit(self, dataset):
        X, y = dataset.getXy()
        X = add_intersect(X)
        self.X = X
        self.y = y
        # Closed form or GD
        self.train_gd(X, y) if self.gd else self.train_closed(X, y)
        self.is_fitted = True

    def train_closed(self, X, y):
        """ Uses closed form linear algebra to fit the model. Prefered to GD.
            
            theta = inv(XT*X)*XT*y
            -----------------------------------------------------------------
            
            Where does this formulation comes from?

            We can write a linear regression equation in the form
            
                Y = X W (see the add_intersect method)
            
            We want the predict values XW to be as close as possible to the
            real values Y, that is, we want to find W values that minimize the 
            distante between XW and Y, ie, minimize the MSE 1/2 (XW-Y)^2.
            
            In closed form mathematics, minimizing (maximizing) a differentiable function
            is to determine the zero of its derivative... so let us derive:
            
                    grad_W (1/2 (X W - Y)^2)                
                        Note: X^2 = XT * X
                
                =   grad_W (1/2 (X W - Y)T * (X W - Y))        
                        Note: (A+B)T = AT + BT and  (A*B)T = BT*AT
                
                =   grad_W (1/2 (WT XT X W - 2 WT XT Y - YT Y))
                        Note dx^2/dx = 2x ,  dxy/dx = y and dy/dx = 0   
                
                =   XT X W - XT Y

            Now that we have the W derivative, we need to know for which W the derivative is 0,
            that is, solve the equation:

                    XT X W - XT Y = 0
                <=> W = inv(XT X) * XT Y

        L2 Reg

            theta = inv(XT*X+lbd*I)*XT*y
        
            The difference to the LR closed for is the inclusion of the matrix 
        
                    | 0 0 0 ... 0 |
                    | 0 1 0 ... 0 |
            lbd  *  | 0 0 1 ... 0 |
                    |       ...
                    | 0 0 0 ... 1 |
        
            in the derivative resulting from adding the L2 regulation term.
            Note that the matrix is not an identity matrix as the first entry is 0.
            The regulatization is not applied to the intersect (bias) term. 
            You may, as exercice, derive the closed form.        
        """
        if self.lbd>0:
            n = X.shape[1]
            identity = np.eye(n)
            identity[0, 0] = 0
            self.theta = np.linalg.inv(X.T @ X + self.lbd*identity) @ X.T @ y
        else:
            self.theta = np.linalg.inv(X.T @ X) @ X.T @ y
        

    def train_gd(self, X, y):
        """ 
        The error between the predictions (XW) and the real values is
            E = XW-Y
        
        The cost funtion J is the Mean Square Error (MSE).
        whose gradient (impact of the weights in the error) is 
        
            dJ/dW = 1/m (X W - Y) X   
        
        At each iteration, the weights are updated considering a defined 
        learning rate (lr)

            W = W - lr * dJ/dW

        """
        m = X.shape[0]
        n = X.shape[1]
        
        # the history keeps track of the learning process
        self.history = {}
        
        # initialize the weights
        self.theta = np.zeros(n)
        
        # with L2
        if self.lbd>0:
            lbds = np.full(m, self.lbd)
            lbds[0] = 0

        # iterative GD
        for epoch in range(self.epochs):
            # the lambda value is not applied to the bias term.            
            
            # with L2
            if self.lbd>0:
                grad = (X.dot(self.theta)-y).dot(X)
                self.theta -= (self.lr/m) * (lbds*self.theta+grad)
            
            # without L2    
            else:
                grad = 1/m * (X @ self.theta - y) @ X
                self.theta -= self.lr * grad
            
            self.history[epoch] = [self.theta.copy(), self.cost()]



    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        _x = np.hstack(([1], x))
        return np.dot(self.theta, _x)

    def cost(self, X=None, y=None, theta=None):
        """ Uses MSE as cost function J
        """
        # uses the trained dataset and weights if not provided.
        X = add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta
        
        # computes the predictions
        y_pred = np.dot(X, theta)
        
        # calculates the MSE
        return mse(y, y_pred)/2
