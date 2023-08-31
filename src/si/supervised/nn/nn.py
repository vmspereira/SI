# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Neural Network module"""
# ---------------------------------------------------------------------------


import numpy as np
import warnings

from si.supervised.model import Model
from si.util import METRICS, minibatch
from .optimizers import SGD



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
        # TODO: add input size from previous layer output size
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

    def forward(self,input):
        """
        Forward propagation
        Propagates values across all layers from the input to the final output.
        Args:
            input (np.array): the input

        Returns:
            np.array: the output
        """
        output_batch = input
        for layer in self.layers:
                    output_batch = layer.forward(output_batch)
        return output_batch
        
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
                
                output_batch = self.forward(output_batch)
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
        output = self.forward(input_data)
        return output

    def cost(self, X=None, y=None):
        assert self.is_fitted, "Model must be fit before predicting"
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)

    def __str__(self) -> str:
        return "\n".join([str(layer) for layer in self.layers])


