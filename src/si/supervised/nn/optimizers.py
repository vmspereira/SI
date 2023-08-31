# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Optimizers module"""
# ---------------------------------------------------------------------------

import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Define how to update the learned parameters"""
    @abstractmethod
    def update(self, w, grad_wrt_w):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum"""
    def __init__(self, learning_rate=0.01, momentum=0):
        """Stochastic Gradient Descent with momentum.

        Args:
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
            momentum (int, optional):  The momentum to use when computing 
                the exponential moving average (EMA) of the model's weights. Defaults to 0.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = None

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt + (1 - self.momentum) * grad_wrt_w
        # Move against the gradient to minimize loss
        return w - self.learning_rate * self.w_updt


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8):
        """_summary_

        Args:
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            b1 (float, optional): The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
            b2 (float, optional): The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
            eps (_type_, optional): A small constant for numerical stability. Defaults to 1e-8.
        """
        
        self.learning_rate = learning_rate
        self.eps = eps
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return w - self.w_updt


class NesterovAcceleratedGradient(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.4):
        """Nesterov Accelerated Gradient.
        The Nesterov Accelerated Gradient method consists of a gradient descent step, 
        followed by something that looks a lot like a momentum term, but isn’t exactly 
        the same as that found in classical momentum.

        Args:
            learning_rate (float, optional): The learning rate. Defaults to 0.001.
            momentum (float, optional): _description_. Defaults to 0.4.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = np.array([])

    def update(self, w, grad_func):
        # Calculate the gradient of the loss a bit further down the slope from w
        approx_future_grad = np.clip(grad_func(w - self.momentum * self.w_updt), -1, 1)
        # Initialize on first update
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))

        self.w_updt = (
            self.momentum * self.w_updt + self.learning_rate * approx_future_grad
        )
        # Move against the gradient to minimize loss
        return w - self.w_updt


class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.01):
        """AMSGrad variant of this algorithm from the paper 
        "On the Convergence of Adam and beyond".

        Args:
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
        """
        self.learning_rate = learning_rate
        self.G = None  # Sum of squares of the gradients
        self.eps = 1e-8

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_wrt_w, 2)
        # Adaptive gradient with higher learning rate for sparse data
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)


class Adadelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6):
        """
        AdaDelta is a stochastic optimization technique that allows for 
        per-dimension learning rate method for SGD. 
        It is an extension of Adagrad that seeks to reduce its aggressive, 
        monotonically decreasing learning rate.

        Args:
            rho (float, optional): The decay rate. Defaults to 0.95.
            eps (_type_, optional): Small floating point value used to maintain numerical stability. 
                Defaults to 1e-6.
        """
        self.E_w_updt = None  # Running average of squared parameter updates
        self.E_grad = None  # Running average of the squared gradient of w
        self.w_updt = None  # Parameter update
        self.eps = eps
        self.rho = rho

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
            self.E_w_updt = np.zeros(np.shape(w))
            self.E_grad = np.zeros(np.shape(grad_wrt_w))

        # Update average of gradients at w
        self.E_grad = self.rho * self.E_grad + (1 - self.rho) * np.power(grad_wrt_w, 2)

        RMS_delta_w = np.sqrt(self.E_w_updt + self.eps)
        RMS_grad = np.sqrt(self.E_grad + self.eps)

        # Adaptive learning rate
        adaptive_lr = RMS_delta_w / RMS_grad

        # Calculate the update
        self.w_updt = adaptive_lr * grad_wrt_w

        # Update the running average of w updates
        self.E_w_updt = self.rho * self.E_w_updt + (1 - self.rho) * np.power(
            self.w_updt, 2
        )

        return w - self.w_updt


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.01, rho=0.9):
        """Root Mean Square propagation

        Args:
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
            rho (float, optional): Moving average parameter. Defaults to 0.9.
        """
        self.learning_rate = learning_rate
        self.Eg = None  # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_wrt_w))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.Eg + self.eps)
