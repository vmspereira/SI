from abc import ABC, abstractmethod
from copy import copy
import numpy as np

class Layer(ABC):
    def __init__(self):
        """Abstract class for layers.
        A layer is a funtion that takes an input and produces an output.
        The function may have learnable parameters, such as weights.
        """
        self.input = None
        self.output = None

    @abstractmethod
    def initialize(self, optimizer):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input):
        """Apply the layer 'function' to a given input
        returning an output.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error):
        """The backward method allows to measure how each input or parameters
        contributed to an error, such as, prediction errors.

        This is achieved using derivatives:
        dE/dX tells us how much X contibuted to the error E.

        Using the chain rule we can propagate errors across each layer
        or function:

        dE/dy = dE/dx * dx/dy
        dE/dz = dE/dx * dx/dy * dy/dz
        ...

        Knowing the contribution of a parameter to the final error, we
        can adjust the parameter. If w is a parameter whose contribution
        to the final error is dE_total/dw, the value of w is adjusted to
        w = w - lr * dE_total/dw.
        The learning rate (lr) controles the the learning speed...
        Note: learning too fast may lead to 'bad' learning, or not learning
        what you should.
        """
        raise NotImplementedError



class Dense(Layer):
    def __init__(self, input_size, output_size):
        """Fully Connected layer
        A dense layer is a set of linear functions wni * xni + ... + w0i * x0i + bi.
        The w and b are learnable parameters, that are usualy randomly initialized.

        :param input_size: the input size.
        :param output_size: the output size.
        """
        self.input_size = input_size
        self.output_size = output_size

    def initialize(self, optimizer):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_size, self.output_size) - 0.5
        # initialing biases
        self.bias = np.zeros((1, self.output_size))
        self.w_opt = copy(optimizer)
        self.b_opt = copy(optimizer)

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error):
        """Here is where the magic happens!

        Computes the dE/dW, dE/dB for a given output_error=dE/dY.

        Returns input_error=dE/dX to feed the previous layer.
        """
        # computes the weight error: dE/dW = X.T * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0)
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)

        # updates the parameters according to a defined optimizer
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.bias = self.b_opt.update(self.bias, bias_error)

        return input_error

    def __str__(self):
        return f"Dense {self.weights.shape}"

    def set_weights(self, weights, bias):
        """Sets the weights and bias of the
        layer.

        :params weights: A numpy array of weight values
        :params bias: A numpy array of bias values
        """
        self.weights = weights
        self.bias = bias


class Flatten(Layer):
    """A flatten layer,flattens all but the 1st dimention."""

    def forward(self, input):
        self.input_shape = input.shape
        # flattens all but the 1st dimention
        output = input.reshape(input.shape[0], -1)
        return output

    def initialize(self, optimizer):
        pass

    def backward(self, output_error):
        return output_error.reshape(self.input_shape)

    def __str__(self):
        return "Flatten"
    

class Reshape(Layer):
    def __init__(self, shape):
        """ Reshapes the input tensor into specified shape"""
        self.prev_shape = None
        self.shape = shape
    
    def initialize(self, optimizer):
        pass
    
    def forward(self, X):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], ) + self.shape)

    def backward(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def __str__(self):
        return "Flatten"


class Dropout(Layer):
    def __init__(self, prob=0.5):
        """A dropout layer.
        :param (float) prob: The dropout probability. Defaults to 0.5.
        """
        self.prob = prob

    def initialize(self, optimizer):
        pass

    def forward(self, input):
        self.mask = np.random.binomial(1, self.prob, size=input.shape) / self.prob
        out = input * self.mask
        return out.reshape(input.shape)

    def backward(self, output_error):
        dX = output_error * self.mask
        return dX

    def __str__(self):
        return f"DropOut {self.prob}"


class BatchNormalization(Layer):
    """Batch normalization.
    """
    def __init__(self, input_shape, momentum=0.99):
        self.momentum = momentum
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None
        self.input_shape = input_shape

    def initialize(self, optimizer):
        # Initialize the parameters
        self.gamma  = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        # parameter optimizers
        self.gamma_opt  = copy(optimizer)
        self.beta_opt = copy(optimizer)

    def forward(self, input):

        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = np.mean(input, axis=0)
            self.running_var = np.var(input, axis=0)

        mean = np.mean(input, axis=0)
        var = np.var(input, axis=0)
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
    
        # Statistics saved for backward pass
        self.X_centered = input - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta

        return output

    def backward(self, output_error):

        # Save parameters used during the forward pass
        gamma = self.gamma

        X_norm = self.X_centered * self.stddev_inv
        grad_gamma = np.sum(output_error * X_norm, axis=0)
        grad_beta = np.sum(output_error, axis=0)

        self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
        self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = output_error.shape[0]

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)
        output_error = (1 / batch_size) * gamma * self.stddev_inv * (
            batch_size * output_error
            - np.sum(output_error, axis=0)
            - self.X_centered * self.stddev_inv**2 * np.sum(output_error * self.X_centered, axis=0)
            )

        return output_error

    def __str__(self):
        return f"BatchNormalization"
