# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the im2col helpers (si.supervised.nn.im2col).
#
# These are the plumbing behind Conv2D: im2col flattens each sliding window
# into a column so a convolution becomes one matrix multiply, and col2im folds
# the result back. The padding solver decides how many zeros a "same"
# convolution needs to keep the spatial size unchanged.
#
# The convolution output-size formula, which every test here traces back to:
#     out = 1 + (in + 2*pad - kernel) / stride
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.supervised.nn.im2col import (
    calc_pad_dims_2D,
    pad2D,
    im2col,
    col2im,
)


def conv_out(in_dim, pad_total, kernel, stride):
    """The output size the formula predicts, used to check the solver."""
    return 1 + (in_dim + pad_total - kernel) // stride


class TestCalcPadDims(unittest.TestCase):
    def test_same_convolution_with_an_odd_kernel_is_symmetric(self):
        # A 3x3 kernel at stride 1 needs one row/col of zeros on each side to
        # keep 5x5 -> 5x5. Odd kernels divide evenly, so the padding is
        # symmetric.
        pad = calc_pad_dims_2D((2, 5, 5, 3), (5, 5), (3, 3), 1)
        self.assertEqual(pad, (1, 1, 1, 1))

    def test_larger_odd_kernel_needs_proportionally_more(self):
        # 5x5 kernel -> 2 on each side.
        self.assertEqual(calc_pad_dims_2D((1, 7, 7, 1), (7, 7), (5, 5), 1),
                         (2, 2, 2, 2))

    def test_even_kernel_pads_asymmetrically(self):
        # An even kernel cannot be centred: the total padding needed is odd, so
        # the solver puts the extra row/col on the bottom/right. This is the
        # asymmetric branch that `pr1, pr2` / `pc1, pc2` exist for.
        pad = calc_pad_dims_2D((1, 5, 5, 1), (5, 5), (2, 2), 1)
        top, bottom, left, right = pad
        self.assertEqual(top + bottom, 1)
        self.assertEqual(left + right, 1)
        self.assertGreaterEqual(bottom, top)
        self.assertGreaterEqual(right, left)

    def test_solved_padding_actually_produces_the_target_size(self):
        # The real contract: whatever the solver returns must satisfy the
        # output-size formula. Checked across kernels, strides and input sizes.
        for in_dim in (5, 6, 8, 11):
            for kernel in (2, 3, 5):
                for stride in (1, 2):
                    out = 1 + (in_dim - 1) // stride
                    with self.subTest(in_dim=in_dim, kernel=kernel, stride=stride):
                        try:
                            pad = calc_pad_dims_2D(
                                (1, in_dim, in_dim, 1), (out, out),
                                (kernel, kernel), stride)
                        except AssertionError:
                            # Not every combination is reachable; the solver
                            # says so rather than returning a wrong padding.
                            continue
                        self.assertEqual(
                            conv_out(in_dim, pad[0] + pad[1], kernel, stride), out)
                        self.assertEqual(
                            conv_out(in_dim, pad[2] + pad[3], kernel, stride), out)

    def test_unreachable_target_reports_what_is_achievable(self):
        # Asking for an output size the formula cannot produce used to raise a
        # bare AssertionError with no message at all.
        with self.assertRaises(AssertionError) as ctx:
            calc_pad_dims_2D((1, 5, 5, 1), (2, 2), (3, 3), 1)
        self.assertIn("closest achievable", str(ctx.exception))

    def test_negative_padding_is_rejected(self):
        # A target smaller than the input implies removing pixels, not padding.
        with self.assertRaises((ValueError, AssertionError)):
            calc_pad_dims_2D((1, 10, 10, 1), (1, 1), (3, 3), 1)

    def test_argument_types_are_validated(self):
        good = ((1, 5, 5, 1), (5, 5), (3, 3), 1)
        for i, bad in enumerate([[1, 5, 5, 1], [5, 5], [3, 3], 1.0]):
            args = list(good)
            args[i] = bad
            with self.subTest(argument=i):
                with self.assertRaises(ValueError):
                    calc_pad_dims_2D(*args)


class TestPad2D(unittest.TestCase):
    def setUp(self):
        # NHWC: 2 examples, 4x4 spatial, 3 channels.
        self.X = np.arange(2 * 4 * 4 * 3, dtype=float).reshape(2, 4, 4, 3)

    def test_int_pads_every_side_equally(self):
        out, p = pad2D(self.X, 1)
        self.assertEqual(out.shape, (2, 6, 6, 3))
        self.assertEqual(p, (1, 1, 1, 1))

    def test_two_tuple_is_rows_then_cols(self):
        out, p = pad2D(self.X, (1, 2))
        self.assertEqual(out.shape, (2, 4 + 2, 4 + 4, 3))
        self.assertEqual(p, (1, 1, 2, 2))

    def test_four_tuple_is_used_as_is(self):
        out, p = pad2D(self.X, (0, 1, 2, 3))
        self.assertEqual(out.shape, (2, 4 + 1, 4 + 5, 3))
        self.assertEqual(p, (0, 1, 2, 3))

    def test_padding_is_zeros_and_leaves_the_interior_intact(self):
        out, _ = pad2D(self.X, 2)
        # the original image sits in the middle, untouched
        np.testing.assert_array_equal(out[:, 2:-2, 2:-2, :], self.X)
        # and the border is all zeros
        self.assertTrue(np.all(out[:, :2, :, :] == 0))
        self.assertTrue(np.all(out[:, :, :2, :] == 0))

    def test_examples_and_channels_are_never_padded(self):
        out, _ = pad2D(self.X, 3)
        self.assertEqual(out.shape[0], self.X.shape[0])
        self.assertEqual(out.shape[3], self.X.shape[3])

    def test_same_makes_the_convolution_preserve_the_spatial_size(self):
        # "same" refers to the size of the CONVOLUTION OUTPUT, not of the padded
        # array -- the padded array is necessarily bigger. So a 4x4 input padded
        # for a same 3x3 stride-1 conv becomes 6x6, and convolving that 6x6 with
        # a 3x3 kernel yields 4x4 again.
        out, p = pad2D(self.X, "same", (3, 3), 1)
        self.assertEqual(p, (1, 1, 1, 1))
        self.assertEqual(out.shape, (2, 6, 6, 3))
        in_rows = self.X.shape[1]
        self.assertEqual(conv_out(in_rows, p[0] + p[1], 3, 1), in_rows)
        self.assertEqual(conv_out(self.X.shape[2], p[2] + p[3], 3, 1),
                         self.X.shape[2])

    def test_same_without_kernel_or_stride_explains_itself(self):
        # This used to raise UnboundLocalError from an unassigned X_pad, which
        # said nothing about the actual problem.
        for args in [(), ((3, 3),), (None, 1)]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError) as ctx:
                    pad2D(self.X, "same", *args)
                self.assertIn("kernel_shape", str(ctx.exception))

    def test_malformed_pad_is_rejected(self):
        for bad in ("valid", (1, 2, 3), [1, 1, 1, 1], None):
            with self.subTest(pad=bad):
                with self.assertRaises(ValueError):
                    pad2D(self.X, bad)

    def test_zero_padding_is_a_no_op(self):
        out, p = pad2D(self.X, 0)
        np.testing.assert_array_equal(out, self.X)
        self.assertEqual(p, (0, 0, 0, 0))


class TestIm2ColRoundTrip(unittest.TestCase):
    # Note im2col returns (X_col, resolved_padding), not just the matrix.

    def test_im2col_column_count_matches_the_output_positions(self):
        # One column per sliding-window position per example. With 4x4 input,
        # a 3x3 kernel, no padding and stride 1 there are 2x2 positions.
        X = np.random.rand(2, 4, 4, 3)
        W_shape = (3, 3, 3, 5)      # (kernel_rows, kernel_cols, in_ch, out_ch)
        cols, p = im2col(X, W_shape, 0, 1)
        # one row per kernel element across all input channels
        self.assertEqual(cols.shape[0], 3 * 3 * 3)
        # one column per window position per example
        self.assertEqual(cols.shape[1], 2 * 2 * 2)
        self.assertEqual(p, (0, 0, 0, 0))

    def test_im2col_columns_hold_the_actual_patches(self):
        # A single-channel 3x3 image with a 2x2 kernel gives four windows; the
        # first column must be the top-left 2x2 patch, read row by row.
        X = np.arange(9, dtype=float).reshape(1, 3, 3, 1)
        cols, _ = im2col(X, (2, 2, 1, 1), 0, 1)
        self.assertEqual(cols.shape, (4, 4))
        np.testing.assert_allclose(cols[:, 0], [0, 1, 3, 4])

    def test_stride_reduces_the_number_of_columns(self):
        X = np.random.rand(1, 5, 5, 1)
        one, _ = im2col(X, (3, 3, 1, 1), 0, 1)
        two, _ = im2col(X, (3, 3, 1, 1), 0, 2)
        self.assertEqual(one.shape[1], 3 * 3)   # 3x3 positions at stride 1
        self.assertEqual(two.shape[1], 2 * 2)   # 2x2 positions at stride 2

    def test_col2im_takes_nhwc_shape_but_returns_nchw(self):
        # A layout trap worth pinning: col2im is told the input's NHWC shape but
        # hands back NCHW (n_ex, channels, rows, cols). Every caller in cnn.py
        # therefore transposes (0, 2, 3, 1) immediately afterwards. Forgetting
        # that transpose silently produces a wrongly-oriented gradient.
        X = np.random.rand(2, 5, 5, 3)
        W_shape = (3, 3, 3, 4)
        cols, p = im2col(X, W_shape, 1, 1)
        back = col2im(cols, X.shape, W_shape, p, 1)
        n_ex, rows, cols_, ch = X.shape
        self.assertEqual(back.shape, (n_ex, ch, rows, cols_))
        # and the documented transpose recovers the original NHWC layout
        self.assertEqual(back.transpose(0, 2, 3, 1).shape, X.shape)

    def test_col2im_accumulates_overlapping_windows(self):
        # col2im is the TRANSPOSE of im2col, not its inverse: a pixel covered by
        # several windows receives a contribution from each. Feeding it a matrix
        # of ones therefore counts how many windows saw each pixel.
        X = np.ones((1, 4, 4, 1))
        W_shape = (3, 3, 1, 1)
        cols, p = im2col(X, W_shape, 0, 1)
        counts = col2im(np.ones_like(cols), X.shape, W_shape, p, 1)
        # NCHW indexing: [example, channel, row, col]
        # a 3x3 kernel over 4x4 at stride 1 gives 2x2 window positions, so the
        # corner pixel belongs to one window and the centre to all four
        self.assertAlmostEqual(counts[0, 0, 0, 0], 1.0)
        self.assertAlmostEqual(counts[0, 0, 1, 1], 4.0)
        self.assertGreater(counts[0, 0, 1, 1], counts[0, 0, 0, 0])

    def test_single_window_round_trip_is_exact(self):
        # When the kernel covers the whole image there is exactly one window, so
        # nothing overlaps and col2im returns the input unchanged -- up to the
        # NHWC -> NCHW layout flip.
        X = np.random.rand(1, 3, 3, 2)
        W_shape = (3, 3, 2, 1)
        cols, p = im2col(X, W_shape, 0, 1)
        back = col2im(cols, X.shape, W_shape, p, 1)
        np.testing.assert_allclose(back.transpose(0, 2, 3, 1), X)


if __name__ == "__main__":
    unittest.main()
