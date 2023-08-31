# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Neural Network module"""
# ---------------------------------------------------------------------------


import numpy as np
from abc import ABC, abstractmethod
from copy import copy
import warnings

from si.supervised.model import Model
from si.util import METRICS, minibatch
from .optimizers import SGD


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


class NN(Model):
    def __init__(
        self,
        epochs=1000,
        batch_size=128,
        optimizer=SGD(),
        verbose=True,
        loss="MSE",
        metric=None,
        step=100,
    ):
        """
        Neural Network model. The default loss function is the mean square error (MSE).
        A NN may be regarded as a sequence of layers, functions applied sequentialy one after the other.

        :param int epochs: Default number of epochs.
        :param int batch_size: Default minibach size
        :param Optimizer optimizer: The optimizer.
        :param bool verbose: If all loss (and quality metric) are to be outputed. Default True.
        :param str loss: The loss function. Default `MSE`.
        :param callable metric: The quality metric. Default None.
        :param int step: the verbose steps. Default 100.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.verbose = verbose
        self.layers = []

        if loss not in METRICS:
            warnings.warn(
                f"{loss} is not a valid loss."
                f"Available losses are {list(METRICS.keys())}."
                "Using MSE."
            )
            loss = "MSE"
        self.loss = METRICS[loss][0]
        self.loss_prime = METRICS[loss][1]

        self.metric = metric
        self.step = step
        self.is_fitted = False

    def add(self, layer):
        """Adds a layer to the network"""
        layer.initialize(self.optimizer)
        self.layers.append(layer)

    def set_loss(self, loss):
        """Changes the loss function.

        :param (str) loss: The loss function name.
        """
        if isinstance(loss,str):
            if loss in METRICS:
                self.loss = METRICS[loss][0]
                self.loss_prime = METRICS[loss][1]
            else:
                warnings.warn(
                    f"{loss} is not a valid loss."
                    f"Available losses are {list(METRICS.keys())}."
                )
        elif isinstance(loss, tuple):
            self.loss = loss[0]
            self.loss_prime = loss[1]


    def set_metric(self, metric):
        self.metric = metric

    def fit(self, dataset, **kwargs):
        
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        self.dataset = dataset
        X, y = dataset.getXy()
        
        self.history = dict()
        for epoch in range(1, epochs + 1):
            # lists to save the batch predicted and real values
            # to be later used to compute the epoch loss and
            # quality metrics
            x_ = []
            y_ = []

            for batch in minibatch(X, y, batch_size):
                output_batch, y_batch = batch
                # forward propagation
                # propagates values across all layers from
                # the input to the final output.
                for layer in self.layers:
                    output_batch = layer.forward(output_batch)

                # backward propagation (propagates errors)
                # Computes the derivatives to see how much each
                # parameter contributed to the total error and adjusts
                # the parameter acording to a defined learning rate
                error = self.loss_prime(y_batch, output_batch)
                for layer in reversed(self.layers):
                    error = layer.backward(error)

                x_.append(output_batch)
                y_.append(y_batch)

            # all the epoch outputs
            out_all = np.concatenate(x_)
            y_all = np.concatenate(y_)
            
            # compute the loss
            err = self.loss(y_all, out_all)

            # if a quality metric is defined
            if self.metric is not None:
                score = self.metric(y_all, out_all)
                score_s = f" {self.metric.__name__}={score}"
            else:
                score = 0
                score_s = ""
            # save into the history
            self.history[epoch] = (err, score)

            # verbosity
            if epoch % self.step == 0:
                s = f"epoch {epoch}/{epochs} loss={err}{score_s}"
                if self.verbose:
                    print(s)
                else:
                    print(s, end="\r")

        self.is_fitted = True

    def predict(self, input_data):
        assert self.is_fitted, "Model must be fit before predicting"
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, y=None):
        assert self.is_fitted, "Model must be fit before predicting"
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)

    def __str__(self) -> str:
        return "\n".join([str(layer) for layer in self.layers])


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
