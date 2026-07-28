# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Image to column

This module implements the *im2col* trick, the key idea that turns a
convolution into a single big matrix multiplication.

Why bother? A naive convolution slides a small kernel over an image with
nested for-loops (over examples, output rows, output columns, channels).
That is correct but very slow in Python/NumPy. The im2col trick instead
*unrolls* every receptive field (the little patch the kernel sees at each
position) into one column of a matrix. Once the input is laid out that way,
the whole convolution becomes a single dense matmul `W_col @ X_col`, which
NumPy (backed by BLAS) executes extremely fast.

  forward : image  --im2col-->  X_col, then  W_col @ X_col  is the conv.
  backward: gradient in column form --col2im--> image-shaped gradient.

`col2im` is the exact inverse bookkeeping of `im2col`: it scatters each
column back to the pixels it came from, *summing* contributions where
receptive fields overlapped (because in the forward pass each pixel may feed
several output positions, so its gradient is the sum of all those paths).

Layout convention used throughout: tensors arrive in NHWC order
(n_examples, height/rows, width/cols, channels), the layout used by the rest
of the network. Internally we often transpose to NCHW (channels before the
spatial dims) because the fancy-indexing trick is easiest to express that way.
"""
# ---------------------------------------------------------------------------
import numpy as np


def calc_pad_dims_2D(X_shape, out_dim, kernel_shape, stride):
    """Compute the zero-padding needed to obtain a desired output size.

    Given the input shape, the *target* output spatial size, the kernel size
    and the stride, this solves for how many rows/cols of zeros to add on each
    side so that the convolution produces exactly `out_dim` outputs (this is
    how a "same" convolution keeps the spatial size unchanged).

    The convolution output-size formula is
        out = 1 + (in + 2*pad - kernel) / stride
    Re-arranging for `pad` gives the expressions below. Because integer
    division can leave the symmetric padding one short, we may add a single
    extra pixel on the right/bottom (asymmetric padding), which is exactly
    what `pr1, pr2` (top/bottom) and `pc1, pc2` (left/right) encode.

    :return: tuple (pr1, pr2, pc1, pc2) = (top, bottom, left, right) padding.
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(out_dim, tuple):
        raise ValueError("`out_dim` must be of type tuple")

    if not isinstance(kernel_shape, tuple):
        raise ValueError("`kernel_shape` must be of type tuple")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    # fr, fc = filter (kernel) rows and cols; NHWC input.
    fr, fc = kernel_shape
    out_rows, out_cols = out_dim
    n_ex, in_rows, in_cols, in_ch = X_shape

    # Solve the conv output-size formula for the (symmetric) padding on each
    # side. The total padding is stride*(out-1) + kernel - in; dividing by 2
    # splits it between the two sides.
    pr = int((stride * (out_rows - 1) + fr - in_rows) / 2)
    pc = int((stride * (out_cols - 1) + fc - in_cols) / 2)

    # Recompute the output size that this symmetric padding actually yields.
    # If integer rounding made it one short, we will fix it asymmetrically.
    out_rows1 = int(1 + (in_rows + 2 * pr - fr) / stride)
    out_cols1 = int(1 + (in_cols + 2 * pc - fc) / stride)

    # add asymmetric padding pixels to right / bottom
    # (one extra row/col there when symmetric padding fell a single unit short)
    pr1, pr2 = pr, pr
    if out_rows1 == out_rows - 1:
        pr1, pr2 = pr, pr + 1
    elif out_rows1 != out_rows:
        raise AssertionError

    pc1, pc2 = pc, pc
    if out_cols1 == out_cols - 1:
        pc1, pc2 = pc, pc + 1
    elif out_cols1 != out_cols:
        raise AssertionError

    if any(np.array([pr1, pr2, pc1, pc2]) < 0):
        raise ValueError(
            "Padding cannot be less than 0. Got: {}".format((pr1, pr2, pc1, pc2))
        )
    return (pr1, pr2, pc1, pc2)


def pad2D(X, pad, kernel_shape=None, stride=None):
    """Zero-pad the spatial dimensions of an NHWC batch of images.

    Accepts several convenient `pad` spellings and normalises them to the
    4-tuple (top, bottom, left, right):
        int p            -> (p, p, p, p)            same pad on all sides
        tuple (a, b)     -> (a, a, b, b)            a on rows, b on cols
        tuple of 4       -> used as-is
        "same"           -> compute the pad that preserves the input size

    :return: (X_pad, p) where X_pad is the padded array and p is the resolved
        4-tuple padding (handy for the backward pass / col2im).
    """
    p = pad
    if isinstance(p, int):
        p = (p, p, p, p)

    if isinstance(p, tuple):
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

        # np.pad only the H and W axes (axes 1 and 2); leave examples and
        # channels untouched. X: (n_ex, in_rows, in_cols, in_ch)
        # -> X_pad: (n_ex, in_rows + p[0] + p[1], in_cols + p[2] + p[3], in_ch)
        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # compute the correct padding dims for a 'same' convolution
    # (delegate to calc_pad_dims_2D, then recurse with the concrete 4-tuple)
    if p == "same" and kernel_shape and stride is not None:
        p = calc_pad_dims_2D(
            X.shape, X.shape[1:3], kernel_shape, stride)
        X_pad, p = pad2D(X, p)
    return X_pad, p


def _im2col_indices(X_shape, fr, fc, p, s):
    """Pre-compute the (channel, row, col) indices that build X_col.

    This is the heart of the im2col trick. Instead of copying patches with a
    Python loop, we build three index arrays (k, i, j) and let NumPy fancy
    indexing gather *every* receptive field in one shot:

        X_col = X_pad[:, k, i, j]   # X_pad is NCHW here

    The idea: a column of X_col corresponds to one (channel, kernel_row,
    kernel_col) of the kernel; a row of X_col corresponds to one output
    position. Each index array therefore has shape
        (fr*fc*n_in, out_rows*out_cols)
    and is built as the broadcast sum of a "within-patch offset" (subscript 0)
    and a "patch start position" (subscript 1):

        absolute_row = offset_within_kernel + stride * output_row_index
        i            = i0 (column vector)  +  i1 (row vector)

    NCHW input here: X_shape = (n_ex, n_in, in_rows, in_cols).
    """
    pr1, pr2, pc1, pc2 = p
    n_ex, n_in, in_rows, in_cols = X_shape

    # number of sliding-window positions along each spatial axis
    out_rows = (in_rows + pr1 + pr2 - fr) // s + 1
    out_cols = (in_cols + pc1 + pc2 - fc) // s + 1

    if any([out_rows <= 0, out_cols <= 0]):
        raise ValueError(
            "Dimension mismatch during convolution: "
            "out_rows = {}, out_cols = {}".format(out_rows, out_cols)
        )

    # --- "within-patch" row offsets (which kernel row each X_col row reads) ---
    # i0: 0,0,..(fc times)..,1,1,..,fr-1,...  -> shape (fr*fc,), then tiled per
    # channel to length (fr*fc*n_in,).
    i0 = np.repeat(np.arange(fr), fc)
    i0 = np.tile(i0, n_in)
    # --- "patch start" row positions (top row of each output cell, * stride) ---
    # i1: each output row repeated out_cols times -> shape (out_rows*out_cols,)
    i1 = s * np.repeat(np.arange(out_rows), out_cols)
    # within-patch col offsets and patch-start col positions, same idea
    j0 = np.tile(np.arange(fc), fr * n_in)        # shape (fr*fc*n_in,)
    j1 = s * np.tile(np.arange(out_cols), out_rows)  # shape (out_rows*out_cols,)

    # Broadcast (col vector) + (row vector) to the full index grids:
    # i, j: (fr*fc*n_in, out_rows*out_cols) absolute pixel coordinates.
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    # k: channel index for each X_col row -> (fr*fc*n_in, 1), broadcasts over j.
    k = np.repeat(np.arange(n_in), fr * fc).reshape(-1, 1)
    return k, i, j


def im2col(X, W_shape, pad, stride):
    """Unroll an NHWC image batch into the 2D matrix used by the conv matmul.

    Every receptive field (the fr x fc x n_in patch the kernel sees at one
    output position) becomes one column. After this, a convolution is just
    `W_col @ X_col`.

    :param X: input batch, NHWC -> (n_ex, in_rows, in_cols, n_in).
    :param W_shape: kernel shape (fr, fc, n_in, n_out).
    :return: (X_col, p) where
        X_col: (fr*fc*n_in, out_rows*out_cols*n_ex) -- one column per patch,
               one row per (channel, kernel_row, kernel_col).
        p:     the resolved padding 4-tuple (top, bottom, left, right).
    """
    fr, fc, n_in, n_out = W_shape
    s, p = stride, pad

    n_ex, in_rows, in_cols, n_in = X.shape
    # zero-pad the input (resolves "same"/int/tuple into a concrete p)
    X_pad, p = pad2D(X, p, W_shape[:2], stride=s)

    # shuffle to have channels as the first dim:
    # NHWC (n_ex, H, W, C) -> NCHW (n_ex, C, H, W), the layout the index
    # trick in _im2col_indices expects.
    X_pad = X_pad.transpose(0, 3, 1, 2)

    # get the indices for im2col (each of shape (fr*fc*n_in, out_rows*out_cols))
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, p, s)

    # Gather all patches at once via fancy indexing. The leading ':' keeps the
    # example axis, so:
    # X_col: (n_ex, fr*fc*n_in, out_rows*out_cols)
    X_col = X_pad[:, k, i, j]
    # Move the example axis to the back and flatten the patch positions and
    # examples into the columns:
    # transpose -> (fr*fc*n_in, out_rows*out_cols, n_ex)
    # reshape   -> (fr*fc*n_in, out_rows*out_cols*n_ex)
    X_col = X_col.transpose(1, 2, 0).reshape(fr * fc * n_in, -1)
    return X_col, p


def col2im(X_col, X_shape, W_shape, padding, stride):
    """Inverse of im2col: scatter columns back into an image-shaped tensor.

    Used in the conv backward pass to turn a column-form input gradient
    (dX_col) back into the spatial layout of the original input. It reuses the
    *same* (k, i, j) indices as im2col, so each column is added back to exactly
    the pixels it was gathered from.

    Crucial detail: we use `np.add.at` (a buffered, in-place add) rather than a
    plain assignment. Because neighbouring receptive fields overlap, a single
    input pixel may have contributed to several output positions; in the
    backward pass its gradient is therefore the *sum* of the gradients flowing
    back through all those positions. np.add.at performs exactly that
    accumulation (a normal `X_pad[idx] = ...` would overwrite, keeping only the
    last contribution and giving a wrong gradient).

    :return: gradient w.r.t. the (unpadded) input, NCHW
        -> (n_ex, n_in, in_rows, in_cols).
    """
    s = stride
    pad = padding

    # Normalise padding to the (top, bottom, left, right) 4-tuple, as in pad2D.
    if isinstance(pad, int):
        pad = (pad, pad, pad, pad)

    if isinstance(pad, tuple):
        if len(pad) == 2:
            pad = (pad[0], pad[0], pad[1], pad[1])

    pr1, pr2, pc1, pc2 = pad
    fr, fc, n_in, n_out = W_shape
    n_ex, in_rows, in_cols, n_in = X_shape

    # Allocate the padded gradient buffer (zeros) in NCHW; we will scatter-add
    # into it. Shape: (n_ex, n_in, in_rows+pr1+pr2, in_cols+pc1+pc2).
    X_pad = np.zeros((n_ex, n_in, in_rows + pr1 + pr2, in_cols + pc1 + pc2))
    # Same index arrays im2col used, so the scatter mirrors the gather exactly.
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, pad, s)
    # Undo im2col's final reshape/transpose so the columns line up with (k,i,j):
    # X_col: (n_in*fr*fc, out_rows*out_cols, n_ex)
    X_col_reshaped = X_col.reshape(n_in * fr * fc, -1, n_ex)
    # -> (n_ex, n_in*fr*fc, out_rows*out_cols), example axis first to match X_pad
    X_col_reshaped = X_col_reshaped.transpose(2, 0, 1)

    # Scatter-add every column back to its source pixels (sums on overlap).
    np.add.at(X_pad, (slice(None), k, i, j), X_col_reshaped)

    # Strip the padding we added in the forward pass so the gradient matches the
    # original input size. (None means "no trailing crop" when that pad was 0.)
    pr2 = None if pr2 == 0 else -pr2
    pc2 = None if pc2 == 0 else -pc2
    return X_pad[:, :, pr1:pr2, pc1:pc2]
