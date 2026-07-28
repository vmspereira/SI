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
from .layers import Layer
from .activation import Tanh

class RNN(Layer):
    """A Vanilla Fully-Connected Recurrent Neural Network layer.

    :param int n_units: The number of hidden states in the layer.
    :param Activation activation: The activation function which will\
        be applied to the output of each state.
    :param int bptt_trunc: Decides how many time steps the gradient\
        should be propagated backwards through states given the loss gradient for time step t.
    :param tuple input_shape: The expected input shape of the layer. For dense layers a single
        digit specifying the number of features of the input. Must be specified if it is the
        first layer in the network.
    """
    def __init__(self, n_units, activation=None, bptt_trunc=5, input_shape=None):
        # An RNN processes a SEQUENCE one time step at a time, carrying a
        # hidden "state" vector h_t that summarizes everything seen so far.
        # At each step it mixes the new input with the previous state:
        #     state_t  = activation( U x_t + W h_(t-1) )
        #     output_t = V state_t
        # The SAME three weight matrices (U, V, W) are reused at every time
        # step (weight sharing across time) — that is what makes it
        # "recurrent" and lets it handle variable-length sequences.
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = Tanh() if activation is None else activation
        self.bptt_trunc = bptt_trunc

        self.W = None # Weight of the previous state (h_(t-1) -> state_t)
        self.V = None # Weight of the output (state_t -> output_t)
        self.U = None # Weight of the input (x_t -> state_t)

    def initialize(self, optimizer):
        timesteps, input_dim = self.input_shape
        # Initialize the weights.
        # The 1/sqrt(fan_in) scale keeps the initial activations from
        # exploding or vanishing as signals are combined.
        #   U : (n_units, input_dim)  maps an input vector to state space
        #   V : (input_dim, n_units)  maps a state back to output space
        #   W : (n_units, n_units)    maps the previous state to the next
        limit = 1 / math.sqrt(input_dim)
        self.U  = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / math.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (input_dim, self.n_units))
        self.W  = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        # Weight optimizers (one independent optimizer per shared matrix)
        self.U_opt  = copy(optimizer)
        self.V_opt = copy(optimizer)
        self.W_opt = copy(optimizer)

    def forward(self, input, training=True):
        # input shape: (batch_size, timesteps, input_dim) — a batch of
        # sequences. We unroll the recurrence over the time axis.
        self.layer_input = input
        batch_size, timesteps, input_dim = input.shape

        # Save these values for use in backprop.
        #   state_input : the pre-activation U x_t + W h_(t-1) at each step
        #   states      : the hidden states h_t (one extra slot so index -1
        #                 holds the initial h_(-1) = 0)
        #   outputs     : the per-step outputs V h_t
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        # Set last time step to zero for calculation of the state_input at time step zero
        # (states[:, -1] is the "previous state" used when t = 0).
        self.states[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            # Input to state_t is the current input and output of previous states.
            # state_input_t = U x_t + W h_(t-1)   (note .T because U,W are
            # stored as (out, in); the dot gives (batch, n_units)).
            self.state_input[:, t] = input[:, t].dot(self.U.T) + self.states[:, t-1].dot(self.W.T)
            # h_t = activation(state_input_t)  — the recurrent memory update
            self.states[:, t] = self.activation(self.state_input[:, t])
            # output_t = V h_t
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward(self, accum_grad):
        """Backpropagation Through Time (BPTT).

        Because the same U, V, W are reused at every step, each one's
        gradient is the SUM of its contributions across all time steps.
        And because h_t feeds into h_(t+1), the error at one step must also
        flow backwards through earlier steps. We therefore walk the
        timeline in reverse, and for each step t we unroll the recurrence
        back a few extra steps. `bptt_trunc` truncates how far back we go,
        which bounds the cost and tames exploding/vanishing gradients.
        """
        _, timesteps, _ = accum_grad.shape

        # Variables where we save the accumulated gradient w.r.t each parameter
        # (summed over time, since the weights are shared across all steps).
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        # The gradient w.r.t the layer input.
        # Will be passed on to the previous layer in the network
        accum_grad_next = np.zeros_like(accum_grad)

        # Back Propagation Through Time
        for t in reversed(range(timesteps)):
            # Update gradient w.r.t V at time step t (output_t = V h_t, so
            # dE/dV gets h_t weighted by the incoming output error).
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            # Calculate the gradient w.r.t the state input.
            # Chain rule: error at output_t flows back through V, then
            # through the activation derivative, to reach state_input_t.
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.prime(self.state_input[:, t])
            # Gradient w.r.t the layer input x_t (state_input_t = U x_t + ...).
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            # Update gradient w.r.t W and U by backprop. from time step t for at most
            # self.bptt_trunc number of time steps.
            # This inner loop walks backwards through the chain of hidden
            # states h_t -> h_(t-1) -> ... accumulating each step's share of
            # dE/dU and dE/dW, then carrying the error one step further back.
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                # dE/dU: state_input depends on the input x at step t_
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                # dE/dW: state_input depends on the previous hidden state h_(t_-1)
                grad_W += grad_wrt_state.T.dot(self.states[:, t_-1])
                # Calculate gradient w.r.t previous state: propagate through
                # W and the previous step's activation derivative (this is
                # where vanishing/exploding gradients come from in RNNs).
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.prime(self.state_input[:, t_-1])

        # Update weights with the time-summed gradients.
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        # Hand the input-error sequence back to the previous layer.
        return accum_grad_next

