# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""A simple RNN"""
# ---------------------------------------------------------------------------

import numpy as np
from copy import copy
import math
from .nn import Layer
from .activation import Tanh

class RNN(Layer):
    """A Vanilla Fully-Connected Recurrent Neural Network layer.

    Parameters:
    -----------
    n_units: int
        The number of hidden states in the layer.
    activation: string
        The name of the activation function which will be applied to the output of each state.
    bptt_trunc: int
        Decides how many time steps the gradient should be propagated backwards through states
        given the loss gradient for time step t.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """
    def __init__(self, n_units, activation=None, bptt_trunc=5, input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = Tanh() if activation is None else activation
        self.bptt_trunc = bptt_trunc
        self.W = None # Weight of the previous state
        self.V = None # Weight of the output
        self.U = None # Weight of the input

    def initialize(self, optimizer):
        timesteps, input_dim = self.input_shape
        # Initialize the weights
        limit = 1 / math.sqrt(input_dim)
        self.U  = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / math.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (input_dim, self.n_units))
        self.W  = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        # Weight optimizers
        self.U_opt  = copy(optimizer)
        self.V_opt = copy(optimizer)
        self.W_opt = copy(optimizer)

    def forward(self, input):
        self.layer_input = input
        batch_size, timesteps, input_dim = input.shape

        # Save these values for use in backprop.
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        # Set last time step to zero for calculation of the state_input at time step zero
        self.states[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            # Input to state_t is the current input and output of previous states
            self.state_input[:, t] = input[:, t].dot(self.U.T) + self.states[:, t-1].dot(self.W.T)
            self.states[:, t] = self.activation(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward(self, accum_grad):
        _, timesteps, _ = accum_grad.shape

        # Variables where we save the accumulated gradient w.r.t each parameter
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        # The gradient w.r.t the layer input.
        # Will be passed on to the previous layer in the network
        accum_grad_next = np.zeros_like(accum_grad)

        # Back Propagation Through Time
        for t in reversed(range(timesteps)):
            # Update gradient w.r.t V at time step t
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            # Calculate the gradient w.r.t the state input
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.gradient(self.state_input[:, t])
            # Gradient w.r.t the layer input
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            # Update gradient w.r.t W and U by backprop. from time step t for at most
            # self.bptt_trunc number of time steps
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_-1])
                # Calculate gradient w.r.t previous state
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.gradient(self.state_input[:, t_-1])

        # Update weights
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        return accum_grad_next

