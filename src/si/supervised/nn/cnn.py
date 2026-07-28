# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Convolutional Layers

Convolution, pooling and padding layers built on top of the im2col trick (see
im2col.py). The recurring strategy here is: reshape the 4D image tensor into a
2D matrix, do the heavy lifting as a matrix multiply (forward) or a couple of
matmuls (backward), then reshape back. The forward/backward maths mirrors the
Dense layer in layers.py -- only the bookkeeping of shapes differs.

Tensor layout convention: every forward pass receives and returns NHWC
(n_examples, rows, cols, channels). im2col / col2im flip to NCHW internally,
which is why you will see a transpose right before / after every call.
"""
# ---------------------------------------------------------------------------

from .layers import Layer
from .im2col import pad2D, im2col, col2im
import numpy as np
from copy import copy

class Conv2D(Layer):
    def __init__(self, input_shape, kernel_shape, layer_depth, stride=1, padding=0):
        """2D convolutional layer.

        A conv layer learns `layer_depth` filters; each filter slides over the
        input and produces one output channel (a feature map). Conceptually it
        is a Dense layer with weight sharing: the same small kernel is reused at
        every spatial position, which is what makes CNNs efficient and
        translation-aware.

        :param input_shape: (in_rows, in_cols, in_ch) of a single example.
        :param kernel_shape: (fr, fc) spatial size of each filter.
        :param layer_depth: number of filters = number of output channels.
        :param stride: step (in pixels) between successive filter positions.
        :param padding: int / tuple / "same" zero-padding (see im2col.pad2D).
        """
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride
        self.padding = padding

    def initialize(self, optimizer=None):
        # weights: one (fr, fc, in_ch) kernel per output channel, drawn from a
        # 0-centred uniform [-0.5, 0.5) -> shape (fr, fc, in_ch, out_ch).
        self.weights = (
            np.random.rand(self.kernel_shape[0], self.kernel_shape[1], self.in_ch, self.out_ch)
            - 0.5
        )
        # bias: one scalar per output channel -> shape (out_ch, 1).
        self.bias = np.zeros((self.out_ch, 1))
        # independent optimizer state for weights and bias (they are updated
        # separately, just like in the Dense layer).
        self.w_opt = copy(optimizer)
        self.b_opt = copy(optimizer)

    def forward(self, input, training=True):
        """Convolution as a single matrix multiply.

        Reframe the conv: unroll the input into X_col (im2col) and flatten each
        filter into a row of W_col, then `W_col @ X_col` computes, in one shot,
        every filter applied at every spatial position of every example.
        """
        s = self.stride
        self.X_shape = input.shape  # cached for the backward pass
        # Resolve padding to a concrete 4-tuple (we only need p, not the
        # padded array -- im2col re-pads internally).
        _, p = pad2D(input, self.padding, self.weights.shape[:2], s)

        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input.shape

        # compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

        # convert X and W into the appropriate 2D matrices and take their product
        # X_col: (fr*fc*in_ch, out_rows*out_cols*n_ex) -- one column per patch.
        self.X_col, _ = im2col(input, self.weights.shape, p, s)
        # Flatten each filter into a row. transpose(3,2,0,1) reorders
        # (fr, fc, in_ch, out_ch) -> (out_ch, in_ch, fr, fc) so the flattened
        # row order (in_ch, fr, fc) matches X_col's row order.
        # W_col: (out_ch, in_ch*fr*fc) = (out_ch, fr*fc*in_ch).
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        # W_col @ X_col: (out_ch, out_rows*out_cols*n_ex); +bias broadcasts over
        # the columns. Then reshape to (out_ch, out_rows, out_cols, n_ex) and
        # transpose(3,1,2,0) back to NHWC (n_ex, out_rows, out_cols, out_ch).
        output_data = (
            (W_col @ self.X_col + self.bias)
            .reshape(out_ch, out_rows, out_cols, n_ex)
            .transpose(3, 1, 2, 0)
        )
        return output_data

    def backward(self, output_error):
        """Backprop through the convolution.

        Because the forward pass is just `Y = W_col @ X_col + b` (a Dense-style
        affine map), the three gradients are the same matmul rules as the Dense
        layer -- only reshaped to honour the 4D conv layouts:

            db    = sum of dY over batch + spatial positions   (one per filter)
            dW    = dY @ X_col.T      (then reshape to weight layout)
            dX_col= W_col.T @ dY      (then col2im back to an image)

        `output_error` is dE/dY in NHWC -> (n_ex, out_rows, out_cols, out_ch).
        We return dE/dX in NHWC so the previous layer receives a gradient shaped
        like its own output.
        """
        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding

        # Recover the (out_ch, N) layout used inside the forward pass. The
        # forward output was reshaped (out_ch, out_rows, out_cols, n_ex) and
        # transposed to NHWC, so we reverse exactly that here.
        # dout_reshaped: (out_ch, out_rows*out_cols*n_ex) -- matches Y's layout.
        dout_reshaped = output_error.transpose(3, 1, 2, 0).reshape(out_ch, -1)

        # gradient wrt bias: sum the error over the batch and spatial dims
        # (every output position adds the same bias, so the bias gradient is the
        # total error flowing through that channel). db: (out_ch, 1).
        db = np.sum(output_error, axis=(0, 1, 2)).reshape(out_ch, 1)

        # gradient wrt weights. X_col rows are ordered (in_ch, fr, fc), so the
        # product is ordered the same way and must be transposed back to the
        # (fr, fc, in_ch, out_ch) weight layout.
        # dW = dY @ X_col.T : (out_ch, fr*fc*in_ch); reshape splits the column
        # axis into (in_ch, fr, fc), then transpose(2,3,1,0) -> (fr,fc,in_ch,out_ch).
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)

        # gradient wrt input. Use the same weight reshaping as the forward pass
        # (transpose(3, 2, 0, 1)) so the (in_ch, fr, fc) ordering matches X_col.
        # W_col: (out_ch, fr*fc*in_ch); dX_col = W_col.T @ dY scatters the error
        # back onto each patch -> (fr*fc*in_ch, out_rows*out_cols*n_ex).
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)
        dX_col = W_col.T @ dout_reshaped
        # col2im returns the gradient in NCHW order; transpose back to the
        # NHWC layout used by every forward pass so the previous layer gets
        # a gradient shaped like its own output.
        # col2im out: (n_ex, in_ch, in_rows, in_cols) -> NHWC after transpose.
        input_error = col2im(dX_col, self.X_shape, self.weights.shape, p, self.stride)
        input_error = input_error.transpose(0, 2, 3, 1)

        # gradient-descent step: hand dW, db to the optimizers (as in Dense).
        self.weights = self.w_opt.update(self.weights, dW)
        self.bias = self.b_opt.update(self.bias, db)

        return input_error

    def __str__(self):
        return f"Conv2D {self.weights.shape}"


class Pooling2D(Layer):
    """Base class for 2D pooling (down-sampling) layers.

    Pooling shrinks each feature map by summarising small windows (e.g. 2x2)
    into a single value, which reduces resolution, adds a little translation
    invariance and cuts computation. It has no learnable parameters.

    We reuse im2col here too: each pooling window becomes a column, and the
    summary (max or mean) is taken *down each column*. Subclasses supply the
    summary (`pool`) and how to route the gradient back (`dpool`).
    """
    def __init__(self, size=2, stride=1):
        """:param size: side length of the (square) pooling window.
        :param stride: step between windows."""
        self.size = size
        self.stride = stride

    def pool(self, X_col):
        """Reduce each column of X_col to one value. Implemented by subclasses.
        Returns (out, cache) where cache is whatever backward needs."""
        raise NotImplementedError

    def dpool(self, dX_col, dout_col, pool_cache):
        """Scatter the per-window output gradient back into column form.
        Implemented by subclasses."""
        raise NotImplementedError

    def initialize(self, optimizer):
        pass

    def forward(self, input, training=True):
        self.X_shape = input.shape  # cached for backward
        n, h, w, d = input.shape  # NHWC

        # output spatial size from the standard down-sampling formula
        h_out = (h - self.size) / self.stride + 1
        w_out = (w - self.size) / self.stride + 1

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception("Invalid output dimension!")

        h_out, w_out = int(h_out), int(w_out)

        # Pooling acts on each channel independently, so fold the channel axis
        # into the batch and treat every channel as its own 1-channel image:
        # NHWC -> NCHW -> (n*d, h, w, 1). That way one im2col gives us columns
        # whose entries all belong to the same channel of the same example.
        X = input.transpose(0, 3, 1, 2)
        X = X.reshape(n * d, h, w, 1)

        # X_col: (size*size, h_out*w_out*n*d) -- one column per window, the
        # window's pixels stacked down the rows.
        self.X_col, _ = im2col(X, (self.size, self.size, d, d), 0, self.stride)

        # Reduce each column (max or mean). out: (h_out*w_out*n*d,).
        # max_idx caches which row won (max-pool) or None (avg-pool).
        out, self.max_idx = self.pool(self.X_col)
        # Unflatten and restore NHWC: (d, h_out, w_out, n) -> (n, h_out, w_out, d).
        out = out.reshape(d, h_out, w_out, n)
        out = out.transpose(3, 1, 2, 0)
        return out

    def backward(self, output_error):
        """Route the incoming gradient back to the pooled inputs.

        Pooling has no weights, so backward only reconstructs dE/dX. Each output
        value came from one window, so its gradient must be sent back into that
        window -- entirely to the max element (max-pool) or split evenly
        (avg-pool). dpool fills a column-form gradient, then col2im scatters it
        back to image coordinates (summing where windows overlapped).
        """
        n, w, h, d = self.X_shape
        # zero gradient buffer in column form, same shape as X_col.
        dX_col = np.zeros_like(self.X_col)
        # Flatten the incoming NHWC error to match the column ordering used in
        # forward: dout_col is one gradient per output position.
        dout_col = output_error.transpose(1, 2, 3, 0).ravel()

        # Place each output gradient into its window (subclass-specific routing).
        dX = self.dpool(dX_col, dout_col, self.max_idx)

        # Scatter columns back to the per-channel images, mirroring the forward
        # im2col on the (n*d, h, w, 1) reshaped input.
        dX = col2im(
            dX,
            (n * d, h, w, 1),
            (self.size, self.size, d, d),
            0,
            self.stride,
        )
        # Back to the original NHWC input shape.
        dX = dX.reshape(self.X_shape)

        return dX


class MaxPooling2D(Pooling2D):
    """Max pooling: each window is summarised by its largest value."""

    def pool(self, X_col):
        # argmax down each column = which pixel "won" each window. Cache it so
        # backward knows where to send the gradient.
        max_idx = np.argmax(X_col, axis=0)
        # gather the winning value from every column. out: (n_windows,).
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    def dpool(self, dX_col, dout_col, pool_cache):
        # Gradient routing for max: only the winning pixel of each window
        # receives the gradient; all other pixels in the window get 0 (they did
        # not affect the output). pool_cache holds the winners from forward.
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col

    def __str__(self):
        return "MaxPooling2D"


class AveragePooling2D(Pooling2D):
    """Average pooling: each window is summarised by its mean value."""

    def pool(self, X_col):
        # average pooling has no argmax to cache, return None as the index
        output = np.mean(X_col, axis=0)
        return output, None

    def dpool(self, dX_col, dout_col, pool_cache):
        # distribute each output gradient evenly across its pooling window
        dX_col[:, range(dout_col.size)] = (1. / dX_col.shape[0]) * dout_col
        return dX_col

    def __str__(self):
        return "AveragePooling2D"


class ConstantPadding2D(Layer):
    """Pads the spatial dimensions of an NHWC batch with a constant value.

    A trainable-parameter-free layer that simply borders each feature map with
    extra pixels (default value 0). Useful for controlling output sizes around
    convolutions. The backward pass just crops those added borders away, since
    the padded pixels are constants and have zero gradient.
    """

    def __init__(self, padding, padding_value=0):
        """:param padding: ((top, bottom), (left, right)); a bare int on either
            axis is broadened to a symmetric (x, x) pair.
        :param padding_value: constant used to fill the border (default 0)."""
        self.padding = padding
        # Normalise the row padding to a (top, bottom) tuple if given as a scalar.
        if not isinstance(padding[0], tuple):
            self.padding = ((padding[0], padding[0]), padding[1])
        # Normalise the col padding to a (left, right) tuple if given as a scalar.
        if not isinstance(padding[1], tuple):
            self.padding = (self.padding[0], (padding[1], padding[1]))
        self.padding_value = padding_value

    def forward(self, input, training=True):
        # remember the pre-pad shape so backward can crop back to it
        self.input_shape = input.shape
        # input is NHWC: pad the spatial dimensions (axes 1 and 2)
        output = np.pad(input,
            pad_width=((0,0), self.padding[0], self.padding[1], (0,0)),
            mode="constant",
            constant_values=self.padding_value)
        return output

    def backward(self, output_error):
        # The gradient only needs the region that came from the real input; the
        # padded border contributed nothing, so we slice it off (the inverse of
        # the forward np.pad) and hand back a gradient of the original size.
        pad_top, pad_left = self.padding[0][0], self.padding[1][0]
        height, width = self.input_shape[1], self.input_shape[2]
        output_error = output_error[:, pad_top:pad_top+height, pad_left:pad_left+width, :]
        return output_error

    def initialize(self, optimizer):
        pass

    def __str__(self):
        return "ConstantPadding2D"
