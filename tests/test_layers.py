# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Forward/backward round-trip tests for the neural network layers.
#
# Every layer is exercised through a forward pass and a backward pass so that
# shape contracts hold and the gradient flowing back matches the layer input.
# These tests would have caught the AveragePooling2D / ConstantPadding2D
# crash-on-first-use bugs.
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.supervised.nn.layers import (
    Dense,
    Flatten,
    Reshape,
    Dropout,
    BatchNormalization,
)
from si.supervised.nn.cnn import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    ConstantPadding2D,
)
from si.supervised.nn.activation import Sigmoid, ReLU, Tanh
from si.supervised.nn.optimizers import SGD


def seed():
    np.random.seed(42)


class TestDense(unittest.TestCase):
    def setUp(self):
        # A fully-connected layer mapping 4 input features -> 3 outputs.
        # initialize(optimizer) allocates the weight/bias matrices and attaches
        # the optimizer used during the backward pass; a batch of 5 samples is
        # the test input.
        seed()
        self.layer = Dense(4, 3)
        self.layer.initialize(SGD())
        self.x = np.random.rand(5, 4)

    def test_forward_shape(self):
        # forward(X) of a Dense(4, 3) on a batch of 5 must yield (5, 3):
        # the row count (batch) is preserved, the column count becomes the
        # number of output units.
        out = self.layer.forward(self.x)
        self.assertEqual(out.shape, (5, 3))

    def test_backward_shape_matches_input(self):
        # backward() returns dL/dX, the gradient w.r.t. the layer's INPUT, so it
        # must have exactly the input shape so it can be passed to the previous
        # layer. A forward pass is required first to cache the input.
        self.layer.forward(self.x)
        grad = self.layer.backward(np.ones((5, 3)))
        self.assertEqual(grad.shape, self.x.shape)

    def test_backward_updates_weights(self):
        # backward() must also apply a learning step: with a non-zero incoming
        # gradient the weights should change. If they stayed identical, the
        # optimizer/parameter-update wiring would be broken.
        self.layer.forward(self.x)
        before = self.layer.weights.copy()
        self.layer.backward(np.ones((5, 3)))
        self.assertFalse(np.allclose(before, self.layer.weights))


class TestFlatten(unittest.TestCase):
    def test_round_trip(self):
        # Flatten just reshapes, so it carries no learnable parameters and must
        # be a pure round-trip: forward collapses the per-sample (4, 4, 3) cube
        # into a length-48 vector (4*4*3=48, batch dim of 2 kept), and backward
        # reshapes the gradient back to the original input shape unchanged.
        # Feeding the forward output back as the gradient must recover x exactly.
        seed()
        layer = Flatten()
        x = np.random.rand(2, 4, 4, 3)
        out = layer.forward(x)
        self.assertEqual(out.shape, (2, 48))
        grad = layer.backward(out)
        self.assertEqual(grad.shape, x.shape)
        self.assertTrue(np.allclose(grad, x))


class TestReshape(unittest.TestCase):
    def test_round_trip(self):
        # Reshape((3, 4)) turns each length-12 sample into a 3x4 block and
        # backward undoes that. Like Flatten it is parameter-free, so passing the
        # forward output back through backward must reconstruct the input exactly.
        seed()
        layer = Reshape((3, 4))
        x = np.random.rand(2, 12)
        out = layer.forward(x)
        self.assertEqual(out.shape, (2, 3, 4))
        grad = layer.backward(out)
        self.assertEqual(grad.shape, x.shape)
        self.assertTrue(np.allclose(grad, x))


class TestDropout(unittest.TestCase):
    def test_inference_is_passthrough(self):
        # Dropout is only active during training. At inference (training=False)
        # it must be the identity, otherwise predictions would be randomly
        # corrupted at test time.
        seed()
        layer = Dropout(prob=0.5)
        x = np.random.rand(10, 10)
        out = layer.forward(x, training=False)
        self.assertTrue(np.allclose(out, x))

    def test_training_applies_mask(self):
        # During training each unit is independently kept with probability `prob`
        # and zeroed otherwise. With an all-ones input and prob=0.5, a surviving
        # unit is scaled to 1/0.5 = 2.0 (inverted dropout) and a dropped unit is
        # 0.0, so every output value is either 0.0 or 2.0.
        seed()
        layer = Dropout(prob=0.5)
        x = np.ones((100, 100))
        out = layer.forward(x, training=True)
        self.assertEqual(out.shape, x.shape)
        # inverted dropout: surviving units are scaled by 1/prob, the rest zeroed
        self.assertTrue(set(np.unique(out)).issubset({0.0, 2.0}))
        grad = layer.backward(np.ones_like(x))
        self.assertEqual(grad.shape, x.shape)


class TestBatchNormalization(unittest.TestCase):
    def setUp(self):
        # 4-feature BatchNorm. The input is deliberately shifted/scaled
        # (mean ~10, spread ~10) so that "did it normalize?" is a meaningful
        # question: raw data is far from zero-mean/unit-variance.
        seed()
        self.layer = BatchNormalization(input_shape=4)
        self.layer.initialize(SGD())
        self.x = np.random.rand(20, 4) * 10 + 5

    def test_training_normalizes(self):
        # In training mode BatchNorm subtracts the batch mean and divides by the
        # batch std, then applies the learnable scale/shift (gamma, beta). At
        # init gamma=1 and beta=0, so the output of each feature must be
        # approximately zero-mean across the batch.
        out = self.layer.forward(self.x, training=True)
        self.assertEqual(out.shape, self.x.shape)
        # gamma=1, beta=0 at init, so output should be ~ zero-mean
        self.assertTrue(np.allclose(out.mean(axis=0), 0, atol=1e-6))

    def test_inference_uses_running_stats(self):
        # BatchNorm accumulates a running mean/var during training and uses those
        # frozen statistics at inference. A forward pass with training=False must
        # therefore NOT update the running mean, otherwise predictions would
        # depend on the batch they happen to be evaluated in.
        self.layer.forward(self.x, training=True)
        running_mean = self.layer.running_mean.copy()
        # an inference pass on different data must not move the running mean
        self.layer.forward(np.random.rand(20, 4), training=False)
        self.assertTrue(np.allclose(running_mean, self.layer.running_mean))

    def test_backward_shape(self):
        # The gradient w.r.t. the input must match the input shape so it can
        # propagate to the preceding layer.
        self.layer.forward(self.x, training=True)
        grad = self.layer.backward(np.ones_like(self.x))
        self.assertEqual(grad.shape, self.x.shape)


class TestConv2D(unittest.TestCase):
    def setUp(self):
        # Conv2D over 8x8x3 images with 3x3 kernels, 4 output filters, stride 1,
        # no padding. Input is a batch of 2 images.
        seed()
        self.layer = Conv2D((8, 8, 3), (3, 3), layer_depth=4, stride=1, padding=0)
        self.layer.initialize(SGD())
        self.x = np.random.rand(2, 8, 8, 3)

    def test_forward_shape(self):
        # With a 3x3 kernel, stride 1 and no padding, an 8x8 image shrinks to
        # 8-3+1 = 6 along each spatial axis, and the channel dim becomes the
        # number of filters (4): (2, 8, 8, 3) -> (2, 6, 6, 4).
        out = self.layer.forward(self.x)
        self.assertEqual(out.shape, (2, 6, 6, 4))

    def test_backward_shape_matches_input(self):
        # dL/dX must have the original image shape so it can flow to the layer
        # below.
        out = self.layer.forward(self.x)
        grad = self.layer.backward(np.ones_like(out))
        self.assertEqual(grad.shape, self.x.shape)

    def test_backward_matches_numerical_gradient(self):
        # The key correctness test for backprop: the analytic gradient produced
        # by backward() must agree with a finite-difference estimate. For the
        # loss L = sum(output), dL/dX[i] is estimated by nudging input element i
        # up and down by eps and measuring the change in the summed output
        # ((f(x+eps) - f(x-eps)) / 2*eps, the central difference). A frozen
        # optimizer (lr=0) keeps weights/bias still during backward so they
        # don't shift between the forward evaluations.
        seed()
        layer = Conv2D((5, 5, 2), (3, 3), layer_depth=3, stride=1, padding=0)
        layer.initialize(SGD(learning_rate=0.0))
        x = np.random.rand(1, 5, 5, 2)

        eps = 1e-5
        num = np.zeros_like(x)
        # Visit every input element once and fill in its numerical partial
        # derivative via the central-difference formula.
        it = np.nditer(x, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            xp = x.copy(); xp[idx] += eps
            xm = x.copy(); xm[idx] -= eps
            num[idx] = (layer.forward(xp).sum() - layer.forward(xm).sum()) / (2 * eps)
            it.iternext()

        # backward(ones) gives dL/dX for L = sum(output) (since dL/d(output) = 1).
        # It must match the numerical gradient to within finite-difference error.
        out = layer.forward(x)
        analytic = layer.backward(np.ones_like(out))
        self.assertEqual(analytic.shape, x.shape)
        self.assertTrue(np.allclose(analytic, num, atol=1e-4))


class TestMaxPooling2D(unittest.TestCase):
    def setUp(self):
        # 2x2 pooling window with stride 2 -> non-overlapping windows that
        # halve each spatial dimension.
        seed()
        self.layer = MaxPooling2D(size=2, stride=2)
        self.x = np.random.rand(2, 4, 4, 3)

    def test_forward_shape(self):
        # 2x2/stride-2 pooling halves height and width (4 -> 2) and leaves the
        # channel dimension alone: (2, 4, 4, 3) -> (2, 2, 2, 3).
        out = self.layer.forward(self.x)
        self.assertEqual(out.shape, (2, 2, 2, 3))

    def test_backward_shape_matches_input(self):
        # Pooling has no parameters; backward only routes the gradient back to
        # the chosen input positions, so dL/dX must match the input shape.
        out = self.layer.forward(self.x)
        grad = self.layer.backward(np.ones_like(out))
        self.assertEqual(grad.shape, self.x.shape)


class TestAveragePooling2D(unittest.TestCase):
    def setUp(self):
        # Same geometry as max pooling, but each window outputs its mean instead
        # of its maximum.
        seed()
        self.layer = AveragePooling2D(size=2, stride=2)
        self.x = np.random.rand(2, 4, 4, 3)

    def test_forward_shape(self):
        # Like max pooling, 2x2/stride-2 halves the spatial dims.
        out = self.layer.forward(self.x)
        self.assertEqual(out.shape, (2, 2, 2, 3))

    def test_forward_is_window_mean(self):
        # The output of average pooling is the mean over each window. A constant
        # window must average to that same constant, which directly verifies the
        # mean is being computed (and would catch a sum-instead-of-mean bug).
        # a constant input must average to that same constant
        x = np.full((1, 2, 2, 1), 7.0)
        layer = AveragePooling2D(size=2, stride=2)
        out = layer.forward(x)
        self.assertTrue(np.allclose(out, 7.0))

    def test_backward_shape_matches_input(self):
        # Gradient w.r.t. the input must match the input shape.
        out = self.layer.forward(self.x)
        grad = self.layer.backward(np.ones_like(out))
        self.assertEqual(grad.shape, self.x.shape)


class TestConstantPadding2D(unittest.TestCase):
    def setUp(self):
        seed()
        self.x = np.random.rand(2, 4, 4, 3)

    def test_forward_pads_spatial_dims(self):
        # padding=(1, 2) adds 1 row of padding on the top AND bottom (height
        # grows 4 -> 6) and 2 columns on the left AND right (width grows
        # 4 -> 8). Batch and channel dimensions are never padded.
        # pad height by (1, 1) and width by (2, 2); channels untouched
        layer = ConstantPadding2D(padding=(1, 2))
        out = layer.forward(self.x)
        self.assertEqual(out.shape, (2, 6, 8, 3))

    def test_backward_recovers_input(self):
        # Backward simply crops away the padding it added, so passing the padded
        # forward output back through backward must reconstruct the original
        # input exactly (padding has no learnable parameters).
        layer = ConstantPadding2D(padding=(1, 2))
        out = layer.forward(self.x)
        grad = layer.backward(out)
        self.assertEqual(grad.shape, self.x.shape)
        self.assertTrue(np.allclose(grad, self.x))

    def test_padding_value(self):
        # The newly added border cells must be filled with `padding_value`. The
        # top-left corner is entirely padding, so it must equal 9.0.
        layer = ConstantPadding2D(padding=(1, 1), padding_value=9.0)
        out = layer.forward(self.x)
        # the top-left corner is pure padding
        self.assertEqual(out[0, 0, 0, 0], 9.0)


class TestActivations(unittest.TestCase):
    def test_sigmoid(self):
        # sigmoid(0) = 1/(1+e^0) = 0.5, a known fixed point that pins down the
        # forward formula; backward must return a gradient shaped like the input.
        layer = Sigmoid()
        out = layer.forward(np.zeros((3, 3)))
        self.assertTrue(np.allclose(out, 0.5))
        grad = layer.backward(np.ones((3, 3)))
        self.assertEqual(grad.shape, (3, 3))

    def test_relu(self):
        # ReLU(x) = max(0, x): negative inputs are clamped to 0, positives pass
        # through unchanged. The handpicked input checks both branches at once.
        layer = ReLU()
        x = np.array([[-1.0, 2.0, -3.0, 4.0]])
        out = layer.forward(x)
        self.assertTrue(np.allclose(out, [[0.0, 2.0, 0.0, 4.0]]))
        grad = layer.backward(np.ones_like(x))
        self.assertEqual(grad.shape, x.shape)

    def test_tanh(self):
        # tanh(0) = 0, another known fixed point; backward must preserve shape.
        layer = Tanh()
        out = layer.forward(np.zeros((2, 2)))
        self.assertTrue(np.allclose(out, 0.0))
        grad = layer.backward(np.ones((2, 2)))
        self.assertEqual(grad.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
