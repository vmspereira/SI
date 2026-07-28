# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Convolutional Layers"""
# ---------------------------------------------------------------------------

from .layers import Layer
from .im2col import pad2D, im2col, col2im
import numpy as np
from copy import copy

class Conv2D(Layer):
    def __init__(self, input_shape, kernel_shape, layer_depth, stride=1, padding=0):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride
        self.padding = padding
        
    def initialize(self, optimizer=None):
        # weights
        self.weights = (
            np.random.rand(self.kernel_shape[0], self.kernel_shape[1], self.in_ch, self.out_ch)
            - 0.5
        )
        # bias
        self.bias = np.zeros((self.out_ch, 1))
        self.w_opt = copy(optimizer)
        self.b_opt = copy(optimizer)

    def forward(self, input, training=True):
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

    def backward(self, output_error):

        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding

        # Recover the (out_ch, N) layout used inside the forward pass. The
        # forward output was reshaped (out_ch, out_rows, out_cols, n_ex) and
        # transposed to NHWC, so we reverse exactly that here.
        dout_reshaped = output_error.transpose(3, 1, 2, 0).reshape(out_ch, -1)

        # gradient wrt bias: sum the error over the batch and spatial dims
        db = np.sum(output_error, axis=(0, 1, 2)).reshape(out_ch, 1)

        # gradient wrt weights. X_col rows are ordered (in_ch, fr, fc), so the
        # product is ordered the same way and must be transposed back to the
        # (fr, fc, in_ch, out_ch) weight layout.
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)

        # gradient wrt input. Use the same weight reshaping as the forward pass
        # (transpose(3, 2, 0, 1)) so the (in_ch, fr, fc) ordering matches X_col.
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)
        dX_col = W_col.T @ dout_reshaped
        # col2im returns the gradient in NCHW order; transpose back to the
        # NHWC layout used by every forward pass so the previous layer gets
        # a gradient shaped like its own output.
        input_error = col2im(dX_col, self.X_shape, self.weights.shape, p, self.stride)
        input_error = input_error.transpose(0, 2, 3, 1)

        self.weights = self.w_opt.update(self.weights, dW)
        self.bias = self.b_opt.update(self.bias, db)

        return input_error

    def __str__(self):
        return f"Conv2D {self.weights.shape}"


class Pooling2D(Layer):
    def __init__(self, size=2, stride=1):
        self.size = size
        self.stride = stride

    def pool(self, X_col):
        raise NotImplementedError

    def dpool(self, dX_col, dout_col, pool_cache):
        raise NotImplementedError

    def initialize(self, optimizer):
        pass

    def forward(self, input, training=True):
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

    def backward(self, output_error):
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


class AveragePooling2D(Pooling2D):
    
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
    
    def __init__(self, padding, padding_value=0):
        self.padding = padding
        if not isinstance(padding[0], tuple):
            self.padding = ((padding[0], padding[0]), padding[1])
        if not isinstance(padding[1], tuple):
            self.padding = (self.padding[0], (padding[1], padding[1]))
        self.padding_value = padding_value

    def forward(self, input, training=True):
        self.input_shape = input.shape
        # input is NHWC: pad the spatial dimensions (axes 1 and 2)
        output = np.pad(input,
            pad_width=((0,0), self.padding[0], self.padding[1], (0,0)),
            mode="constant",
            constant_values=self.padding_value)
        return output

    def backward(self, output_error):
        pad_top, pad_left = self.padding[0][0], self.padding[1][0]
        height, width = self.input_shape[1], self.input_shape[2]
        output_error = output_error[:, pad_top:pad_top+height, pad_left:pad_left+width, :]
        return output_error

    def initialize(self, optimizer):
        pass

    def __str__(self):
        return "ConstantPadding2D"
