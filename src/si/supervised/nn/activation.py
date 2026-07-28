# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Activation functions module"""
# ---------------------------------------------------------------------------

import numpy as np
from abc import abstractmethod
from .layers import Layer


class ActivationFunction(Layer):
    def __init__(self):
        """Activation layer.
        Activation "layers" allow NN to learn non linear functions,
        as would be the case if only dense layers were used.

        Why do we need non-linearities?
        ------------------------------------------------------------------
        A Dense layer computes an affine map y = W x + b. Stacking several
        affine maps still gives an affine map:

            W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)

        so a network made only of Dense layers can represent nothing more
        than a single linear regression, no matter how deep it is.
        Inserting a non-linear function f between layers breaks this
        collapse and lets the network approximate arbitrary (non-linear)
        functions.

        An activation is treated as a (parameter-free) layer:
          forward : a = f(z), where z is the pre-activation from the
                    previous layer (shape unchanged, applied element-wise).
          backward: by the chain rule dE/dz = dE/da * f'(z), so we just
                    multiply the incoming error by the derivative f'(z)
                    evaluated at the input we saw during the forward pass.
        """
        super().__init__()

    @abstractmethod
    def fn(self, z):
        """Apply the activation function to an input"""
        raise NotImplementedError

    @abstractmethod
    def prime(self, x, **kwargs):
        """Compute the derivative of the activation function wrt the input"""
        raise NotImplementedError

    def __call__(self, z):
        """Apply the activation function to an input"""
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    def initialize(self, optimizer):
        pass

    def forward(self, input_data, training=True):
        self.input = input_data

        # apply the activation function to the input
        self.output = self(self.input)
        return self.output

    def backward(self, output_error):
        # learning_rate is not used because there is no "learnable" parameters.
        # Only passed the error do the previous layer
        #
        # output_error is dE/da (gradient of the loss w.r.t. this layer's
        # output). The chain rule gives the gradient w.r.t. the input:
        #     dE/dz = dE/da * da/dz = output_error * f'(z)
        # The multiplication is element-wise because f is applied
        # element-wise, so prime(self.input) has the same shape as
        # output_error and no shape change occurs.
        return np.multiply(self.prime(self.input), output_error)

    def __str__(self):
        return "Activation"


class Sigmoid(ActivationFunction):
    def __init__(self):
        """A logistic sigmoid activation function."""
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, z):
        # sigma(z) = 1 / (1 + e^-z)
        # Squashes any real number into the open interval (0, 1), which is
        # why it is often used to model probabilities. Downside: it
        # saturates for large |z| (gradient ~ 0), causing vanishing
        # gradients in deep nets.
        return 1.0 / (1.0 + np.exp(-z))

    def prime(self, z):
        # Derivative has the elegant closed form:
        #   sigma'(z) = sigma(z) * (1 - sigma(z))
        # Note it is largest (0.25) at z=0 and tends to 0 as |z| grows.
        fn_x = self.fn(z)
        res = fn_x * (1 - fn_x)
        return res

    def prime2(self, x):
        """
        Evaluate the second derivative of the logistic sigmoid
        on the elements of `x`.

        sigma''(z) = sigma(z) * (1 - sigma(z)) * (1 - 2 sigma(z))
        (used by optimizers/analyses that need curvature information).
        """
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class ReLU(ActivationFunction):
    """A rectified linear activation function."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, X):
        # ReLU(x) = max(0, x): passes positives through unchanged and
        # clamps negatives to 0. Cheap to compute and does not saturate
        # for x > 0, which helps avoid vanishing gradients. The downside
        # is "dead" units: once a unit always outputs 0, its gradient is 0
        # and it can stop learning.
        return np.where(X > 0, X, 0)

    def prime(self, X):
        # Derivative is the step function: 1 for x > 0, 0 otherwise.
        # (The derivative at exactly x = 0 is undefined; here it is taken
        #  as 0 by the X > 0 test.)
        return np.where(X > 0, 1, 0)

    def prime2(self, x):
        # ReLU is piecewise linear, so its second derivative is 0
        # everywhere (ignoring the non-differentiable point at 0).
        return np.zeros_like(x)


class LeakyReLU(ActivationFunction):
    """
    'Leaky' version of a rectified linear unit (ReLU).

    Fixes ReLU's "dead unit" problem by letting a small, non-zero slope
    (alpha) leak through for negative inputs, so the gradient is never
    exactly 0:

        f(x) = x          if x > 0
        f(x) = alpha * x  if x <= 0
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "Leaky ReLU(alpha={})".format(self.alpha)

    def fn(self, z):
        # Copy so we do not mutate the caller's array, then scale only the
        # negative entries by alpha (positives are left unchanged).
        _z = z.copy()
        _z[z < 0] = _z[z < 0] * self.alpha
        return _z

    def prime(self, x):
        # Slope is 1 for positive inputs and alpha for negative inputs.
        out = np.ones_like(x)
        out[x < 0] *= self.alpha
        return out

    def prime2(self, x):
        # Piecewise linear, so the second derivative is 0.
        return np.zeros_like(x)


class Tanh(ActivationFunction):
    def __init__(self):
        """A hyperbolic tangent activation function."""
        super().__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        # tanh(z) = (e^z - e^-z) / (e^z + e^-z), squashing inputs into
        # (-1, 1). Like the sigmoid but zero-centred, which tends to make
        # optimization easier. It also saturates for large |z|.
        return np.tanh(z)

    def prime(self, x):
        # tanh'(z) = 1 - tanh(z)^2  (largest, = 1, at z = 0; ->0 as |z| grows)
        return 1 - np.tanh(x) ** 2

    def prime2(self, x):
        # Second derivative: tanh''(z) = -2 tanh(z) (1 - tanh(z)^2)
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1 - tanh_x ** 2)


class Affine(ActivationFunction):
    def __init__(self, slope=1, intercept=0):
        """
        An affine activation function.

        f(z) = slope * z + intercept. This is a *linear* activation (no
        non-linearity), useful mainly as an output activation for
        regression or as a building block (see Identity below).
        """
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    def fn(self, z):
        return self.slope * z + self.intercept

    def prime(self, x):
        # d/dz (slope*z + intercept) = slope (constant for all inputs).
        return self.slope * np.ones_like(x)

    def prime2(self, x):
        # Linear function => zero curvature.
        return np.zeros_like(x)


class Identity(Affine):
    def __init__(self):
        """
        Identity activation function.

        Special case of Affine with slope=1, intercept=0, so f(z) = z.
        Equivalent to applying no activation at all.
        """
        super().__init__(slope=1, intercept=0)

    def __str__(self):
        return "Identity"


class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):
        """
        An exponential linear unit (ELU).

        Like ReLU for positive inputs, but uses a smooth, saturating
        exponential curve for negatives so outputs can go slightly below
        zero. This pushes the mean activation closer to zero (similar
        benefit to Tanh) and avoids the hard "dead unit" of ReLU:

            f(z) = z                    if z > 0
            f(z) = alpha * (e^z - 1)    if z <= 0
        """
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "ELU(alpha={})".format(self.alpha)

    def fn(self, z):
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def prime(self, x):
        # Derivative: 1 for z > 0; alpha * e^z for z <= 0 (note that for
        # z <= 0, f(z) = alpha(e^z - 1) so f'(z) = alpha e^z = f(z)+alpha).
        return np.where(x > 0, np.ones_like(x), self.alpha * np.exp(x))

    def prime2(self, x):
        return np.where(x >= 0, np.zeros_like(x), self.alpha * np.exp(x))


class Exponential(ActivationFunction):
    def __init__(self):
        """An exponential (base e) activation function"""
        super().__init__()

    def __str__(self):
        return "Exponential"

    def fn(self, z):
        # f(z) = e^z. Its own derivative, so prime and prime2 are also e^z.
        return np.exp(z)

    def prime(self, x):
        return np.exp(x)

    def prime2(self, x):
        return np.exp(x)


class SELU(ActivationFunction):
    """
    A scaled exponential linear unit (SELU).

    SELU is ELU multiplied by a fixed scale, with alpha and scale set to
    specific magic constants. With those constants (and appropriate weight
    init) the activations "self-normalize": their mean and variance are
    driven towards 0 and 1 as signals flow through the network, removing
    the need for an explicit BatchNormalization layer.

        f(z) = scale * ELU_alpha(z)
    """

    def __init__(self):
        # These two constants are the fixed points derived in the original
        # SELU paper; they are what make the activations self-normalizing.
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


class HardSigmoid(ActivationFunction):
    def __init__(self):
        """
        A "hard" sigmoid activation function.

        A cheap, piecewise-linear approximation of the sigmoid that avoids
        the expensive exp(): a straight line 0.2*z + 0.5 clipped to [0, 1].
        """
        super().__init__()

    def __str__(self):
        return "Hard Sigmoid"

    def fn(self, z):
        return np.clip((0.2 * z) + 0.5, 0.0, 1.0)

    def prime(self, x):
        # Slope is 0.2 in the linear region (-2.5 <= z <= 2.5) and 0 in the
        # flat clipped regions outside it.
        return np.where((x >= -2.5) & (x <= 2.5), 0.2, 0)

    def prime2(self, x):
        return np.zeros_like(x)


class SoftPlus(ActivationFunction):
    def __init__(self):
        """
        A softplus activation function.

        f(z) = log(1 + e^z): a smooth (everywhere-differentiable)
        approximation of ReLU. Always positive and never exactly 0.
        """
        super().__init__()

    def __str__(self):
        return "SoftPlus"

    def fn(self, z):
        return np.log(np.exp(z) + 1)

    def prime(self, x):
        # Remarkably, the derivative of softplus is exactly the sigmoid:
        #   d/dz log(1 + e^z) = e^z / (1 + e^z) = sigma(z)
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)

    def prime2(self, x):
        # Second derivative = sigma(z)(1 - sigma(z)) = e^z / (1 + e^z)^2
        exp_x = np.exp(x)
        return exp_x / ((exp_x + 1) ** 2)


class SoftMax(ActivationFunction):
    """SoftMax activation.

    Turns a vector of raw scores (logits) into a probability distribution
    over classes: every output is in (0, 1) and the outputs along the last
    axis sum to 1. Typically the final layer of a multi-class classifier.

        softmax(z)_i = e^{z_i} / sum_j e^{z_j}
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SoftMax"

    def fn(self, z):
        # Subtract the row-wise max before exponentiating. This is a
        # numerical-stability trick: it leaves the result unchanged
        # (the constant cancels in the ratio) but keeps e^x from
        # overflowing for large logits. axis=-1 normalizes across the
        # class dimension, keepdims keeps the broadcast shape aligned.
        e_x = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def prime(self, x):
        # NOTE: the true softmax Jacobian is a full matrix
        #   d p_i / d z_j = p_i (delta_ij - p_j)
        # Here only the diagonal term p_i (1 - p_i) is returned (the
        # element-wise approximation), which is what the element-wise
        # backward() of ActivationFunction expects. In practice softmax is
        # usually paired with a cross-entropy loss whose combined gradient
        # simplifies to (p - y), sidestepping this Jacobian entirely.
        p = self.fn(x)
        return p * (1 - p)


functions = {
    'sigmoid': Sigmoid(),
    'relu': ReLU(),
    'softmax': SoftMax(),
    'softplus': SoftPlus(),
    'hardsigmoid': HardSigmoid(),
    'tanh': Tanh(),
    'selu': SELU(),
    'leakyrelu': LeakyReLU(),
    'affine': Affine(),
    'elu': ELU(),
    'exp': Exponential(),

}


class Activation(Layer):
    """Thin layer wrapper that looks up an activation by name.

    Lets you add an activation to a network with a string, e.g.
    ``nn.add(Activation('relu'))``, instead of importing the concrete
    class. It simply delegates forward/backward to the chosen
    ActivationFunction instance from the `functions` registry above.
    """

    def __init__(self, function):
        super().__init__()
        if isinstance(function, str) and function in functions:
            self.fun = functions[function]
        else:
            raise ValueError(f"The function name is not a string or is unknown."
                             f"possible values are {list(functions.keys())}"
                             )

    def initialize(self, optimizer):
        pass

    def forward(self, input):
        return self.fun.forward(input)

    def backward(self, output_error):
        return self.fun.backward(output_error)

    def __str__(self):
        return self.fun.__str__()
