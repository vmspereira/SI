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
        seed()
        self.layer = Dense(4, 3)
        self.layer.initialize(SGD())
        self.x = np.random.rand(5, 4)

    def test_forward_shape(self):
        out = self.layer.forward(self.x)
        self.assertEqual(out.shape, (5, 3))

    def test_backward_shape_matches_input(self):
        self.layer.forward(self.x)
        grad = self.layer.backward(np.ones((5, 3)))
        self.assertEqual(grad.shape, self.x.shape)

    def test_backward_updates_weights(self):
        self.layer.forward(self.x)
        before = self.layer.weights.copy()
        self.layer.backward(np.ones((5, 3)))
        self.assertFalse(np.allclose(before, self.layer.weights))


class TestFlatten(unittest.TestCase):
    def test_round_trip(self):
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
        seed()
        layer = Dropout(prob=0.5)
        x = np.random.rand(10, 10)
        out = layer.forward(x, training=False)
        self.assertTrue(np.allclose(out, x))

    def test_training_applies_mask(self):
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
        seed()
        self.layer = BatchNormalization(input_shape=4)
        self.layer.initialize(SGD())
        self.x = np.random.rand(20, 4) * 10 + 5

    def test_training_normalizes(self):
        out = self.layer.forward(self.x, training=True)
        self.assertEqual(out.shape, self.x.shape)
        # gamma=1, beta=0 at init, so output should be ~ zero-mean
        self.assertTrue(np.allclose(out.mean(axis=0), 0, atol=1e-6))

    def test_inference_uses_running_stats(self):
        self.layer.forward(self.x, training=True)
        running_mean = self.layer.running_mean.copy()
        # an inference pass on different data must not move the running mean
        self.layer.forward(np.random.rand(20, 4), training=False)
        self.assertTrue(np.allclose(running_mean, self.layer.running_mean))

    def test_backward_shape(self):
        self.layer.forward(self.x, training=True)
        grad = self.layer.backward(np.ones_like(self.x))
        self.assertEqual(grad.shape, self.x.shape)


class TestConv2D(unittest.TestCase):
    def setUp(self):
        seed()
        self.layer = Conv2D((8, 8, 3), (3, 3), layer_depth=4, stride=1, padding=0)
        self.layer.initialize(SGD())
        self.x = np.random.rand(2, 8, 8, 3)

    def test_forward_shape(self):
        out = self.layer.forward(self.x)
        self.assertEqual(out.shape, (2, 6, 6, 4))

    def test_backward_shape_matches_input(self):
        out = self.layer.forward(self.x)
        grad = self.layer.backward(np.ones_like(out))
        self.assertEqual(grad.shape, self.x.shape)

    def test_backward_matches_numerical_gradient(self):
        # finite-difference check of dL/dX for loss = sum(output); a frozen
        # optimizer (lr=0) keeps weights/bias still during backward.
        seed()
        layer = Conv2D((5, 5, 2), (3, 3), layer_depth=3, stride=1, padding=0)
        layer.initialize(SGD(learning_rate=0.0))
        x = np.random.rand(1, 5, 5, 2)

        eps = 1e-5
        num = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            xp = x.copy(); xp[idx] += eps
            xm = x.copy(); xm[idx] -= eps
            num[idx] = (layer.forward(xp).sum() - layer.forward(xm).sum()) / (2 * eps)
            it.iternext()

        out = layer.forward(x)
        analytic = layer.backward(np.ones_like(out))
        self.assertEqual(analytic.shape, x.shape)
        self.assertTrue(np.allclose(analytic, num, atol=1e-4))


class TestMaxPooling2D(unittest.TestCase):
    def setUp(self):
        seed()
        self.layer = MaxPooling2D(size=2, stride=2)
        self.x = np.random.rand(2, 4, 4, 3)

    def test_forward_shape(self):
        out = self.layer.forward(self.x)
        self.assertEqual(out.shape, (2, 2, 2, 3))

    def test_backward_shape_matches_input(self):
        out = self.layer.forward(self.x)
        grad = self.layer.backward(np.ones_like(out))
        self.assertEqual(grad.shape, self.x.shape)


class TestAveragePooling2D(unittest.TestCase):
    def setUp(self):
        seed()
        self.layer = AveragePooling2D(size=2, stride=2)
        self.x = np.random.rand(2, 4, 4, 3)

    def test_forward_shape(self):
        out = self.layer.forward(self.x)
        self.assertEqual(out.shape, (2, 2, 2, 3))

    def test_forward_is_window_mean(self):
        # a constant input must average to that same constant
        x = np.full((1, 2, 2, 1), 7.0)
        layer = AveragePooling2D(size=2, stride=2)
        out = layer.forward(x)
        self.assertTrue(np.allclose(out, 7.0))

    def test_backward_shape_matches_input(self):
        out = self.layer.forward(self.x)
        grad = self.layer.backward(np.ones_like(out))
        self.assertEqual(grad.shape, self.x.shape)


class TestConstantPadding2D(unittest.TestCase):
    def setUp(self):
        seed()
        self.x = np.random.rand(2, 4, 4, 3)

    def test_forward_pads_spatial_dims(self):
        # pad height by (1, 1) and width by (2, 2); channels untouched
        layer = ConstantPadding2D(padding=(1, 2))
        out = layer.forward(self.x)
        self.assertEqual(out.shape, (2, 6, 8, 3))

    def test_backward_recovers_input(self):
        layer = ConstantPadding2D(padding=(1, 2))
        out = layer.forward(self.x)
        grad = layer.backward(out)
        self.assertEqual(grad.shape, self.x.shape)
        self.assertTrue(np.allclose(grad, self.x))

    def test_padding_value(self):
        layer = ConstantPadding2D(padding=(1, 1), padding_value=9.0)
        out = layer.forward(self.x)
        # the top-left corner is pure padding
        self.assertEqual(out[0, 0, 0, 0], 9.0)


class TestActivations(unittest.TestCase):
    def test_sigmoid(self):
        layer = Sigmoid()
        out = layer.forward(np.zeros((3, 3)))
        self.assertTrue(np.allclose(out, 0.5))
        grad = layer.backward(np.ones((3, 3)))
        self.assertEqual(grad.shape, (3, 3))

    def test_relu(self):
        layer = ReLU()
        x = np.array([[-1.0, 2.0, -3.0, 4.0]])
        out = layer.forward(x)
        self.assertTrue(np.allclose(out, [[0.0, 2.0, 0.0, 4.0]]))
        grad = layer.backward(np.ones_like(x))
        self.assertEqual(grad.shape, x.shape)

    def test_tanh(self):
        layer = Tanh()
        out = layer.forward(np.zeros((2, 2)))
        self.assertTrue(np.allclose(out, 0.0))
        grad = layer.backward(np.ones((2, 2)))
        self.assertEqual(grad.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
