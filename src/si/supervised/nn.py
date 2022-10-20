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
from .model import Model
from si.util import mse, mse_prime
from si.util.im2col import pad2D, im2col, col2im

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
    def __init__(self, epochs=1000, lr=0.1, verbose=True, batchsize=None):
        """Neural Network model. The default loss function is the mean square error (MSE).
        A NN may be regarded as a sequence of layers, functions applied sequentialy one after the other.
     
        :param int epochs: Number of epochs.
        :param float lr: The learning rate.
        """
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.batchsize = batchsize
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

        self.is_fitted = False

    def add(self, layer):
        """Adds a layer to the network"""
        self.layers.append(layer)

    def useLoss(self, loss, loss_prime):
        """Changes the loss function.

        :param loss: The loss function.
        :param loss_prime: The derivative of the loss function.
        """
        self.loss = loss
        self.loss_prime = loss_prime

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

            # calculate average error on all samples
            err = self.loss(y_batch, output)
            self.history[epoch] = err
            
            # verbosity
            if self.verbose:
                print(f"epoch {epoch+1}/{self.epochs} error={err}")
            else:
                print(f"epoch {epoch+1}/{self.epochs} error={err}", end="\r")

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

        # initialing biases with 1 usualy performs better than random
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
        
        # computers the layer input error (the output error from the previous layer), 
        # dE/dX, to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)
        
        # updates the parameters accordind to a defined learning rate
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error

    def __str__(self):
        return "Dense"

    def setWeights(self, weights, bias):
        """ Sets the weights and bias of the 
        layer.

        :params weights: A numpy array of weight values
        :params bias: A numpy array of bias values
        """
        self.weights = weights
        self.bias = bias


class Activation(Layer):
    
    def __init__(self, activation):
        """Activation layer.
        Activation "layers" allow NN to learn non linear functions, as would be the
        case if only dense layers were used. 

        :param activation: An instance of si.util.activation.ActivationBase.
        """
        self.activation = activation

    def forward(self, input_data):
        self.input = input_data

        # apply the activation function to the input
        self.output = self.activation(self.input)
        
        return self.output

    def backward(self, output_error, learning_rate):
        # learning_rate is not used because there is no "learnable" parameters.
        # Only passed the error do the previous layer
        return np.multiply(self.activation.prime(self.input), output_error)

    def __str__(self):
        return "Activation"


class Flatten(Layer):

    def forward(self, input):
        self.input_shape = input.shape
        # flattens all but the 1st dimention
        output = input.reshape(input.shape[0], -1)
        return output

    def backward(self, output_error, learning_rate):
        return output_error.reshape(self.input_shape)

    def __str__(self):
        return "Flatten"


class Conv2D(Layer):
    def __init__(self, input_shape, kernel_shape, layer_depth, stride=1, padding=0):
        self.input_shape = input_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride
        self.padding = padding
        # weights
        self.weights = (
            np.random.rand(kernel_shape[0], kernel_shape[1], self.in_ch, self.out_ch)
            - 0.5
        )
        # bias
        self.bias = np.zeros((self.out_ch, 1))

    def forward(self, input):
        s = self.stride
        self.X_shape = input.shape
        _, p = pad2D(input, self.padding, self.weights.shape[:2], s)

        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input.shape

        # compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

        # convert X and W into the appropriate 2D matrices and take their product
        self.X_col, _ = im2col(input, self.weights.shape, p, s)
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        output_data = (
            (W_col @ self.X_col + self.bias)
            .reshape(out_ch, out_rows, out_cols, n_ex)
            .transpose(3, 1, 2, 0)
        )
        return output_data

    def backward(self, output_error, learning_rate):

        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding

        db = np.sum(output_error, axis=(0, 1, 2))
        db = db.reshape(out_ch,)

        dout_reshaped = output_error.transpose(1, 2, 3, 0).reshape(out_ch, -1)
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(self.weights.shape)

        W_reshape = self.weights.reshape(out_ch, -1)
        dX_col = W_reshape.T @ dout_reshaped
        input_error = col2im(dX_col, self.X_shape, self.weights.shape, p, self.stride)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return input_error

    def __str__(self):
        return "Conv2D"


class Pooling2D(Layer):
    def __init__(self, size=2, stride=1):
        self.size = size
        self.stride = stride

    def pool(self, X_col):
        raise NotImplementedError

    def dpool(self, dX_col, dout_col, pool_cache):
        raise NotImplementedError

    def forward(self, input):
        self.X_shape = input.shape
        n, h, w, d = input.shape

        h_out = (h - self.size) / self.stride + 1
        w_out = (w - self.size) / self.stride + 1

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception("Invalid output dimension!")

        h_out, w_out = int(h_out), int(w_out)

        X = input.transpose(0, 3, 1, 2)
        X = X.reshape(n * d, h, w, 1)

        self.X_col, _ = im2col(X, (self.size, self.size, d, d), 0, self.stride)

        out, self.max_idx = self.pool(self.X_col)
        out = out.reshape(d, h_out, w_out, n)
        out = out.transpose(3, 1, 2, 0)
        return out

    def backward(self, output_error, learning_rate):
        n, w, h, d = self.X_shape
        dX_col = np.zeros_like(self.X_col)
        dout_col = output_error.transpose(1, 2, 3, 0).ravel()

        dX = self.dpool(dX_col, dout_col, self.max_idx)

        dX = col2im(
            dX,
            (n * d, h, w, 1),
            (self.size, self.size, d, d),
            0,
            self.stride,
        )
        dX = dX.reshape(self.X_shape)

        return dX


class MaxPooling2D(Pooling2D):
    def pool(self, X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    def dpool(self, dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col

    def __str__(self):
        return "MaxPooling2D"
