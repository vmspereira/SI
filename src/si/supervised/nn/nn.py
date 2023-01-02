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
from si.supervised.model import Model
from si.util import mse, mse_prime

class Layer(ABC):
    def __init__(self):
        """ Abstract class for layers. 
        A layer is a funtion that takes an input and produces an output.
        The function may have learnable parameters, such as weights. 
        """
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        """ Apply the layer 'function' to a given input
        returning an output.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error, learning_rate):
        """ The backward method allows to measure how each input or parameters
            contributed to an error, such as, prediction errors. 
     
            This is achieved using derivatives: 
            dE/dX tells us how much X contibuted to the error E.

            Using the chain rule we can propagate errors across each layer or function:
            
            dE/dy = dE/dx * dx/dy
            dE/dz = dE/dx * dx/dy * dy/dz
            ...

            Knowing the contribution of a parameter to the final error, we can adjust the
            parameter. If w is a parameter whose contribution to the final error is dE_total/dw,
            the value of w is adjusted to w = w - lr * dE_total/dw.
            The learning rate (lr) controles the the learning speed... 
            Note: learning too fast may lead to 'bad' learning, or not learning what you should. 
        """
        raise NotImplementedError


class NN(Model):
    def __init__(self, epochs=1000, lr=0.1, verbose=True, metric = None):
        """Neural Network model. The default loss function is the mean square error (MSE).
        A NN may be regarded as a sequence of layers, functions applied sequentialy one after the other.
     
        :param int epochs: Number of epochs.
        :param float lr: The learning rate.
        """
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime
        self.metric = metric
        self.is_fitted = False

    def add(self, layer):
        """Adds a layer to the network"""
        self.layers.append(layer)

    def set_loss(self, loss, loss_prime):
        """Changes the loss function.

        :param loss: The loss function.
        :param loss_prime: The derivative of the loss function.
        """
        self.loss = loss
        self.loss_prime = loss_prime
        
    def set_metric(self, metric):
        self.metric = metric

    def fit(self, dataset):
        X, y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):

            output = X
            y_batch = y
        
            # forward propagation
            # propagates values across all layers from
            # the input to the final output.
            
            for layer in self.layers:
                output = layer.forward(output)
                
            # backward propagation (propagates errors)
            # Computes the derivatives to see how much each
            # parameter contributed to the total error and adjusts
            # the parameter acording to a defined learning rate

            error = self.loss_prime(y_batch, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)
            
            # calculates average error on all samples
            err = self.loss(y_batch, output)
            
            # if a quality metric is defined
            if self.metric is not None:
                score = self.metric(y, output)
                score_s = f" \t {self.metric.__name__}={score}"
            else:
                score = 0
                score_s = ""
            
            self.history[epoch] = (err, score)
            # verbosity
            s = f"epoch {epoch+1}/{self.epochs} \t loss={err}{score_s}"
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


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """Fully Connected layer
        A dense layer is a set of linear functions wni * xni + ... + w0i * x0i + bi.
        The w and b are learnable parameters, that are usualy randomly initialized.

        :param input_size: the input size.
        :param output_size: the output size.
        """
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(input_size, output_size) - 0.5

        # initialing biases
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        """ Here is where the magic happens!

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
        
        # updates the parameters accordind to a defined learning rate
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error

    def __str__(self):
        return "Dense"

    def set_weights(self, weights, bias):
        """ Sets the weights and bias of the 
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

    def backward(self, output_error, learning_rate):
        return output_error.reshape(self.input_shape)

    def __str__(self):
        return "Flatten"


class Dropout(Layer):

    def __init__(self, prob=0.5):
        """A dropout layer. 
        :param (float) prob: The dropout probability. Defaults to 0.5.
        """
        self.prob = prob

    def forward(self, input):
        self.mask = np.random.binomial(1, self.prob, size=input.shape) / self.prob
        out = input * self.mask
        return out.reshape(input.shape)

    def backward(self, output_error, learning_rate):
        dX = output_error * self.mask
        return dX
    
    def __str__(self):
        return f"DropOut {self.prob}"
