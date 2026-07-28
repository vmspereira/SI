# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""NN Layers"""
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod
from copy import copy
import numpy as np


class Layer(ABC):
    def __init__(self):
        """Abstract class for layers.
        A layer is a function that takes an input and produces an output.
        The function may have learnable parameters, such as weights.
        """
        self.input = None
        self.output = None

    @abstractmethod
    def initialize(self, optimizer):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input, training=True):
        """Apply the layer 'function' to a given input
        returning an output.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error):
        """The backward method allows to measure how each input or parameters
        contributed to an error, such as, prediction errors.

        This is achieved using derivatives:
        dE/dX tells us how much X contributed to the error E.

        Using the chain rule we can propagate errors across each layer
        or function:

        dE/dy = dE/dx * dx/dy
        dE/dz = dE/dx * dx/dy * dy/dz
        ...

        Knowing the contribution of a parameter to the final error, we
        can adjust the parameter. If w is a parameter whose contribution
        to the final error is dE_total/dw, the value of w is adjusted to
        w = w - lr * dE_total/dw.
        The learning rate (lr) controls the learning speed...
        Note: learning too fast may lead to 'bad' learning, or not learning
        what you should.
        """
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """Fully Connected layer.

        A dense layer is a set of linear functions wni * xni + ... + w0i * x0i + bi.
        The w and b are learnable parameters, that are usually randomly initialized.

        :param input_size: the input size.
        :param output_size: the output size.
        """
        self.input_size = input_size
        self.output_size = output_size

    def initialize(self, optimizer):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        # Small, zero-centred random weights break symmetry (so units learn
        # different features) while keeping the initial signal small.
        # Shape: (input_size, output_size).
        self.weights = np.random.rand(self.input_size, self.output_size) - 0.5
        # initialing biases
        # Biases start at 0; shape (1, output_size) so they broadcast across
        # every row (sample) of a batch.
        self.bias = np.zeros((1, self.output_size))
        # Each learnable tensor gets its OWN optimizer copy, so that
        # per-parameter state (e.g. momentum buffers) stays independent.
        self.w_opt = copy(optimizer)
        self.b_opt = copy(optimizer)

    def forward(self, input_data, training=True):
        # Forward pass of the affine map  Y = X W + b
        #   X (input)   : (batch_size, input_size)
        #   W (weights) : (input_size, output_size)
        #   b (bias)    : (1, output_size)  -> broadcast over the batch
        #   Y (output)  : (batch_size, output_size)
        # We cache the input because the backward pass needs it to form
        # the weight gradient.
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error):
        """Here is where the magic happens!

        Computes the dE/dW, dE/dB for a given output_error=dE/dY.

        Returns input_error=dE/dX to feed the previous layer.

        Chain-rule derivation (Y = X W + b):
          - dE/dW = X^T (dE/dY)  because dY/dW = X
          - dE/db = sum over the batch of dE/dY  because dY/db = 1
          - dE/dX = (dE/dY) W^T  because dY/dX = W
        These are exactly the three lines below. dE/dX is the error this
        layer hands back to the layer before it (the chain continues).
        """
        # weight gradient: (input_size, batch) x (batch, output_size)
        #                  = (input_size, output_size), same shape as W
        # dE/dW = X.T * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # bias gradient: sum dE/dY over the batch axis -> (output_size,)
        # (b is shared across all samples, so its gradient accumulates)
        # dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0)
        # input gradient: (batch, output_size) x (output_size, input_size)
        #                 = (batch, input_size), same shape as the input X
        # dE/dX, passed on to the previous layer
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

        :params weights: A numpy array of shape (input_size, output_size)
        :params bias: A numpy array of shape (1, output_size)
        """
        # Shapes were previously accepted unchecked, so a mismatch only surfaced
        # much later as a broadcasting ValueError from inside forward -- far from
        # the assignment that caused it.
        expected_w = (self.input_size, self.output_size)
        expected_b = (1, self.output_size)
        if np.shape(weights) != expected_w:
            raise ValueError(
                f"weights must have shape {expected_w} for a "
                f"Dense({self.input_size}, {self.output_size}); "
                f"got {np.shape(weights)}."
            )
        if np.shape(bias) != expected_b:
            raise ValueError(
                f"bias must have shape {expected_b}; got {np.shape(bias)}."
            )
        self.weights = weights
        self.bias = bias


class Flatten(Layer):
    """A flatten layer.

       Flattens all but the 1st dimension.

       Used to bridge from multi-dimensional layers (e.g. the (batch, H, W,
       C) output of a convolution) into a Dense layer, which expects a 2D
       (batch, features) matrix. It has no learnable parameters and only
       reshapes data, so the backward pass just undoes the reshape.
    """

    def forward(self, input, training=True):
        # Remember the original shape so backward can restore it.
        self.input_shape = input.shape
        # flattens all but the 1st dimension
        # (batch, d1, d2, ...) -> (batch, d1*d2*...); -1 lets NumPy infer
        # the flattened feature length.
        output = input.reshape(input.shape[0], -1)
        return output

    def initialize(self, optimizer):
        pass

    def backward(self, output_error):
        # Reshape is a pure rearrangement of values: the gradient flows
        # straight through, reshaped back to the original input shape.
        return output_error.reshape(self.input_shape)

    def __str__(self):
        return "Flatten"


class Reshape(Layer):
    def __init__(self, shape):
        """ Reshapes the input tensor into specified shape.

        Like Flatten, but to an arbitrary target shape (per sample).
        `shape` is the desired shape of a single sample; the batch
        dimension is kept and prepended automatically. No learnable
        parameters.
        """
        self.prev_shape = None
        self.shape = shape

    def initialize(self, optimizer):
        pass

    def forward(self, X, training=True):
        # Cache the incoming shape, then reshape each sample to self.shape
        # while preserving the batch dimension X.shape[0].
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], ) + self.shape)

    def backward(self, accum_grad):
        # Reshape carries no gradient of its own; just restore the shape
        # the previous layer produced.
        return accum_grad.reshape(self.prev_shape)

    def __str__(self):
        return "Reshape"


class Dropout(Layer):
    def __init__(self, prob=0.5):
        """A dropout layer.

        Regularization technique: during TRAINING it randomly "drops"
        (zeroes) a fraction of the inputs on every forward pass. This stops
        the network from relying too heavily on any single unit and forces
        redundant, more robust representations (a cheap way to approximate
        averaging over many sub-networks). At INFERENCE nothing is dropped
        so the full network is used.

        This implementation uses INVERTED DROPOUT: see forward() for why we
        scale by 1/prob at training time.

        Note: here `prob` is the *keep* probability — each unit survives
        with probability `prob` and is dropped with probability 1 - prob.
        Many frameworks use the opposite convention (the drop rate), which is
        worth keeping in mind when porting a network: prob=0.2 here keeps only
        20% of the units, it does not drop 20% of them.

        :param (float) prob: The keep probability, in (0, 1]. Defaults to 0.5.
            prob=1.0 keeps everything, making the layer a no-op.
        """
        # Inverted dropout divides the mask by `prob`, so prob=0 is a division
        # by zero: it produced an all-NaN mask that then poisoned the forward
        # pass, the loss and every gradient, with only a RuntimeWarning. Reject
        # it up front instead. prob=0 is meaningless anyway -- it would drop
        # every unit and leave the layer with nothing to pass on.
        if not 0 < prob <= 1:
            raise ValueError(
                "Dropout `prob` is the KEEP probability and must be in (0, 1]; "
                "got {!r}. (Passing a drop RATE is the usual cause: to drop "
                "20% of units use prob=0.8, not prob=0.2.)".format(prob)
            )
        self.prob = prob

    def initialize(self, optimizer):
        pass

    def forward(self, input, training=True):
        if training:
            # Inverted dropout.
            # np.random.binomial(1, prob, ...) draws a 0/1 mask that is 1
            # ("keep") with probability `prob`. We immediately divide the
            # mask by `prob`, so surviving units are scaled UP by 1/prob.
            #
            # Why scale by 1/prob? We want the expected output to match the
            # un-dropped input so that the next layer sees the same average
            # magnitude in training and at inference:
            #     E[mask_i] = prob * (1/prob) + (1-prob) * 0 = 1
            # so E[out] = input. Doing the scaling here ("inverted") means
            # inference can be a plain pass-through with no extra work.
            self.mask = np.random.binomial(1, self.prob, size=input.shape) / self.prob
            out = input * self.mask
            return out.reshape(input.shape)
        else:
            # At inference: no dropping, no scaling. Because we already
            # compensated with 1/prob during training, the input passes
            # straight through unchanged.
            return input

    def backward(self, output_error):
        # Gradient flows only through the units that were kept, scaled by
        # the same 1/prob factor — i.e. multiply by the exact mask used in
        # the forward pass. Dropped units get zero gradient.
        dX = output_error * self.mask
        return dX

    def __str__(self):
        return f"DropOut {self.prob}"


class BatchNormalization(Layer):
    """Batch Normalization with Momentum.

       At each iteration or weigths update, the output distribution
       of a layer shifts (Internal Covariant Shift) making more difficult
       the training process.
       Batch Normalization normalizes the output of the previous output layer
       by subtracting the empirical mean over the batch divided by the empirical
       standard deviation, that is, it gives a Gaussian like look to the output
       distribution.
       BN also has a regularization effect. Indeed, BN introduces a certain level
       of noise into the sample mean and variance during the training process.
       Such a noise generation mechanism of BN regularizes the training process,
       helping the model to generalize.
       The level of noise depends on the batch size, and increases with it.
       When the batch size is small, the momentum helps increase the noise level
       by averaging the mean and variance of current mini-batch with
       the historical means and variances.

       [1] Yong, H., Huang, J., Meng, D., Hua, X., Zhang, L. (2020).
           Momentum Batch Normalization for Deep Learning with Small Batch Size.
           In: Vedaldi, A., Bischof, H., Brox, T., Frahm, JM. (eds) Computer Vision
           ECCV 2020. ECCV 2020. Lecture Notes in Computer Science(), vol 12357. Springer, Cham.
           https://doi.org/10.1007/978-3-030-58610-2_14
    """

    def __init__(self, input_shape, momentum=0.99, eps=1e-5):
        # momentum: weight given to the historical running statistics when
        #   blending in the current batch's mean/var (closer to 1 = slower,
        #   smoother updates).
        # eps: small constant added to the variance before taking the
        #   square root, to avoid division by zero / instability.
        #
        # eps was hardcoded at 0.01, a thousand times the conventional 1e-5, and
        # it sits INSIDE the square root: the scale applied is
        # 1/sqrt(var + eps). Once eps is comparable to a feature's variance it
        # dominates, and the layer stops normalising the very features that most
        # need it:
        #     variance 1.0    ->  0.5% under-scaled
        #     variance 0.01   -> 29.3% under-scaled
        #     variance 0.0001 -> 90.0% under-scaled
        # Now 1e-5 by default, and configurable, so it does what its own
        # docstring promises. Raise it if a batch is so small that the variance
        # estimate itself is unstable.
        self.momentum = momentum
        self.eps = eps
        # Running (exponential moving average) statistics, accumulated over
        # training batches and used at inference time. None until first use.
        self.running_mean = None
        self.running_var = None
        self.input_shape = input_shape

    def initialize(self, optimizer):
        # Initialize the parameters
        # gamma (scale) and beta (shift) are the two LEARNABLE parameters of
        # BatchNorm. They let the network undo or re-tune the normalization:
        # starting at gamma=1, beta=0 means "identity" at the first step.
        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        # parameter optimizers (one each, independent state)
        self.gamma_opt = copy(optimizer)
        self.beta_opt = copy(optimizer)

    def forward(self, input, training=True):

        # Inference requires statistics learned during TRAINING. Initialising
        # them here regardless meant that a forward(training=False) on an
        # untrained layer seeded the running mean/var from the inference batch
        # and then normalised that batch by its own statistics -- precisely the
        # leakage the running estimates exist to prevent, and silently: two test
        # samples came back as -0.995 and +0.995 as if they had been centred on
        # their own mean.
        if not training and self.running_mean is None:
            raise RuntimeError(
                "BatchNormalization has no running statistics yet: run at least "
                "one forward pass with training=True before evaluating with "
                "training=False. Otherwise inference would normalise each batch "
                "by its own mean and variance."
            )

        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = np.mean(input, axis=0)
            self.running_var = np.var(input, axis=0)

        # The train/inference branch is the heart of BatchNorm:
        if training:
            # TRAINING: normalize using THIS batch's own statistics
            # (computed across the batch axis, axis=0, per feature), and
            # fold those statistics into the running EMA for later use.
            mean = np.mean(input, axis=0)
            var = np.var(input, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # INFERENCE: a single sample has no meaningful "batch" stats,
            # and predictions must be deterministic, so we use the frozen
            # running mean/var accumulated during training instead.
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        # Normalization: X_norm = (x - mean) / sqrt(var + eps), giving each
        # feature ~zero mean and ~unit variance. Then scale & shift by the
        # learnable gamma/beta: output = gamma * X_norm + beta.
        self.X_centered = input - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta

        return output

    def backward(self, output_error):

        # Save parameters used during the forward pass
        gamma = self.gamma

        # Gradients of the two learnable parameters (sum over the batch,
        # since gamma/beta are shared across all samples):
        #   dE/dgamma = sum( dE/dY * X_norm )   (gamma multiplied X_norm)
        #   dE/dbeta  = sum( dE/dY )            (beta was just added)
        X_norm = self.X_centered * self.stddev_inv
        grad_gamma = np.sum(output_error * X_norm, axis=0)
        grad_beta = np.sum(output_error, axis=0)

        self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
        self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = output_error.shape[0]

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)
        # dE/dX is more involved than for Dense because the batch mean and
        # variance each depend on EVERY sample in the batch, so a sample's
        # gradient also flows back through the shared mean/var. The closed
        # form below (standard BatchNorm result) bundles the three paths:
        #   - direct path through X_norm           -> batch_size * output_error
        #   - through the shared mean               -> - sum(output_error)
        #   - through the shared variance           -> the X_centered term
        # all scaled by (1/batch_size) * gamma * stddev_inv.
        output_error = (1 / batch_size) * gamma * self.stddev_inv * (
            batch_size * output_error
            - np.sum(output_error, axis=0)
            - self.X_centered * self.stddev_inv**2 * np.sum(output_error * self.X_centered, axis=0)
            )

        return output_error

    def __str__(self):
        return "BatchNormalization"
