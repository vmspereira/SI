# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Activation functions module"""
# ---------------------------------------------------------------------------

import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    def __init__(self):
        """Initialize the ActivationBase object"""
        super().__init__()

    def __call__(self, z):
        """Apply the activation function to an input"""
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        """Apply the activation function to an input"""
        raise NotImplementedError

    @abstractmethod
    def prime(self, x, **kwargs):
        """Compute the primeient of the activation function wrt the input"""
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self):
        """A logistic sigmoid activation function."""
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def prime(self, z):
        fn_x = self.fn(z)
        res = fn_x * (1 - fn_x)
        return res

    def prime2(self, x):
        """
        Evaluate the second derivative of the logistic sigmoid on the elements of `x`.
        """
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class ReLU(Activation):
    """A rectified linear activation function."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, X):
        return np.where(X > 0, X, 0)

    def prime(self, X):
        return np.where(X > 0, 1, 0)

    def prime2(self, x):
        return np.zeros_like(x)


class LeakyReLU(Activation):
    """
    'Leaky' version of a rectified linear unit (ReLU).
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "Leaky ReLU(alpha={})".format(self.alpha)

    def fn(self, z):
        _z = z.copy()
        _z[z < 0] = _z[z < 0] * self.alpha
        return _z

    def prime(self, x):
        out = np.ones_like(x)
        out[x < 0] *= self.alpha
        return out

    def prime2(self, x):
        return np.zeros_like(x)


class Tanh(Activation):
    def __init__(self):
        """A hyperbolic tangent activation function."""
        super().__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        return np.tanh(z)

    def prime(self, x):
        return 1 - np.tanh(x) ** 2

    def prime2(self, x):
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1 - tanh_x ** 2)


class Affine(Activation):
    def __init__(self, slope=1, intercept=0):
        """
        An affine activation function.
        """
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    def fn(self, z):
        return self.slope * z + self.intercept

    def prime(self, x):
        return self.slope * np.ones_like(x)

    def prime2(self, x):
        return np.zeros_like(x)


class Identity(Affine):
    def __init__(self):
        """
        Identity activation function.
        """
        super().__init__(slope=1, intercept=0)

    def __str__(self):
        return "Identity"


class ELU(Activation):
    def __init__(self, alpha=1.0):
        """
        An exponential linear unit (ELU).
        """
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "ELU(alpha={})".format(self.alpha)

    def fn(self, z):
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def prime(self, x):
        return np.where(x > 0, np.ones_like(x), self.alpha * np.exp(x))

    def prime2(self, x):
        return np.where(x >= 0, np.zeros_like(x), self.alpha * np.exp(x))


class Exponential(Activation):
    def __init__(self):
        """An exponential (base e) activation function"""
        super().__init__()

    def __str__(self):
        return "Exponential"

    def fn(self, z):
        return np.exp(z)

    def prime(self, x):
        return np.exp(x)

    def prime2(self, x):
        return np.exp(x)


class SELU(Activation):
    """
    A scaled exponential linear unit (SELU).
    """

    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.elu = ELU(alpha=self.alpha)
        super().__init__()

    def __str__(self):
        return "SELU"

    def fn(self, z):
        return self.scale * self.elu.fn(z)

    def prime(self, x):
        return np.where(
            x >= 0, np.ones_like(x) * self.scale, np.exp(x) * self.alpha * self.scale
        )

    def prime2(self, x):
        return np.where(x > 0, np.zeros_like(x), np.exp(x) * self.alpha * self.scale)


class HardSigmoid(Activation):
    def __init__(self):
        """
        A "hard" sigmoid activation function.
        """
        super().__init__()

    def __str__(self):
        return "Hard Sigmoid"

    def fn(self, z):
        return np.clip((0.2 * z) + 0.5, 0.0, 1.0)

    def prime(self, x):
        return np.where((x >= -2.5) & (x <= 2.5), 0.2, 0)

    def prime2(self, x):
        return np.zeros_like(x)


class SoftPlus(Activation):
    def __init__(self):
        """
        A softplus activation function.
        """
        super().__init__()

    def __str__(self):
        return "SoftPlus"

    def fn(self, z):
        return np.log(np.exp(z) + 1)

    def prime(self, x):
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)

    def prime2(self, x):
        exp_x = np.exp(x)
        return exp_x / ((exp_x + 1) ** 2)


class SoftMax(Activation):
    
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return "SoftMax"

    def fn(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div
        
    def prime(self, x):
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    
    
    

