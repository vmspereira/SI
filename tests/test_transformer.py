# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the transformer building blocks: Embedding, PositionalEncoding and
# the TransformerBlock composite.
#
# The block is where the residual bookkeeping lives, so the end-to-end gradient
# check is the important one: a residual sends the incoming gradient down BOTH
# the shortcut and the sub-layer, and dropping either term still yields
# plausible-looking numbers.
#
# Gradient checks use a random incoming error, and parameter gradients are
# recovered from what the layer actually applied rather than recomputed here.
# ----------------------------------------------------------------------------
import unittest

import numpy as np

from si.data import Dataset
from si.supervised.nn import NN
from si.supervised.nn.layers import Dense
from si.supervised.nn.optimizers import SGD, Adam
from si.supervised.nn.transformer import (
    Embedding,
    PositionalEncoding,
    TransformerBlock,
)


def frozen_block(d_model, n_heads, d_ff=None, causal=False, seed=0):
    """Factory for identically-initialised TransformerBlocks with a frozen
    optimizer, so repeated forward passes are comparable."""
    np.random.seed(seed)
    reference = TransformerBlock(d_model, n_heads, d_ff=d_ff, causal=causal)
    reference.initialize(SGD(learning_rate=0.0))

    norms = {name: (getattr(reference, name).gamma.copy(),
                    getattr(reference, name).beta.copy())
             for name in ('norm1', 'norm2')}
    denses = {name: (getattr(reference, name).weights.copy(),
                     getattr(reference, name).bias.copy())
              for name in ('ff_in', 'ff_out')}
    projections = {name: (getattr(reference.attention, name).weights.copy(),
                          getattr(reference.attention, name).bias.copy())
                   for name in ('W_q', 'W_k', 'W_v', 'W_o')}

    def fresh():
        block = TransformerBlock(d_model, n_heads, d_ff=d_ff, causal=causal)
        block.initialize(SGD(learning_rate=0.0))
        for name, (gamma, beta) in norms.items():
            layer = getattr(block, name)
            layer.gamma, layer.beta = gamma.copy(), beta.copy()
        for name, (weights, bias) in denses.items():
            layer = getattr(block, name)
            layer.weights, layer.bias = weights.copy(), bias.copy()
        for name, (weights, bias) in projections.items():
            layer = getattr(block.attention, name)
            layer.weights, layer.bias = weights.copy(), bias.copy()
        return block

    return fresh


class TestEmbedding(unittest.TestCase):
    """A lookup table: the rows ARE the weights of an imaginary Dense layer
    applied to a one-hot input, with the multiplication skipped."""

    def setUp(self):
        np.random.seed(0)
        self.layer = Embedding(7, 4)
        self.layer.initialize(SGD())
        self.ids = np.array([[0, 3, 3], [6, 0, 1]])

    def test_lookup_shape_and_values(self):
        out = self.layer.forward(self.ids)
        self.assertEqual(out.shape, (2, 3, 4))
        # the vector for a token is literally that row of the table
        np.testing.assert_allclose(out[0, 0], self.layer.weights[0])
        np.testing.assert_allclose(out[1, 0], self.layer.weights[6])

    def test_the_same_token_maps_to_the_same_vector(self):
        out = self.layer.forward(self.ids)
        np.testing.assert_allclose(out[0, 1], out[0, 2])   # both are token 3

    def test_gradient_reaches_the_embedding_matrix(self):
        # Recovered from what the layer applied: plain SGD updates
        # w <- w - lr*grad, so grad = (before - after)/lr.
        vocab, width, lr = 7, 4, 0.1
        rng = np.random.RandomState(1)
        error = rng.randn(2, 3, width)
        reference = Embedding(vocab, width)
        reference.initialize(SGD(learning_rate=0.0))
        base = reference.weights.copy()

        def fresh(weights=None, learning_rate=0.0):
            layer = Embedding(vocab, width)
            layer.initialize(SGD(learning_rate=learning_rate, momentum=0))
            layer.weights = (base if weights is None else weights).copy()
            return layer

        layer = fresh(learning_rate=lr)
        layer.forward(self.ids)
        before = layer.weights.copy()
        layer.backward(error.copy())
        applied = (before - layer.weights) / lr

        h = 1e-6
        numerical = np.zeros_like(base)
        for idx in np.ndindex(base.shape):
            up, down = base.copy(), base.copy()
            up[idx] += h
            down[idx] -= h
            numerical[idx] = ((fresh(up).forward(self.ids) * error).sum()
                              - (fresh(down).forward(self.ids) * error).sum()) / (2 * h)
        np.testing.assert_allclose(applied, numerical, atol=1e-6)

    def test_unused_tokens_receive_no_gradient(self):
        lr = 0.1
        layer = Embedding(7, 4)
        layer.initialize(SGD(learning_rate=lr, momentum=0))
        layer.forward(self.ids)
        before = layer.weights.copy()
        layer.backward(np.random.RandomState(2).randn(2, 3, 4))
        # tokens 2, 4 and 5 never appear in self.ids
        for token in (2, 4, 5):
            with self.subTest(token=token):
                np.testing.assert_allclose(before[token], layer.weights[token])

    def test_a_repeated_token_accumulates_every_use(self):
        # np.add.at rather than indexed assignment: token 3 appears twice, and
        # plain assignment would keep only the last write.
        lr = 1.0
        layer = Embedding(7, 4)
        layer.initialize(SGD(learning_rate=lr, momentum=0))
        layer.forward(np.array([[3, 3]]))
        before = layer.weights.copy()
        # both occurrences contribute the same gradient of ones
        layer.backward(np.ones((1, 2, 4)))
        applied = (before - layer.weights) / lr
        np.testing.assert_allclose(applied[3], np.full(4, 2.0))

    def test_backward_returns_something_shaped_like_the_ids(self):
        # There is no derivative with respect to an integer id, but the
        # container chains the return value, so it must not be None.
        out = self.layer.forward(self.ids)
        grad = self.layer.backward(np.random.randn(*out.shape))
        self.assertEqual(grad.shape, self.ids.shape)
        np.testing.assert_allclose(grad, 0.0)

    def test_out_of_range_ids_are_rejected(self):
        for bad in (np.array([[7]]), np.array([[-1]])):
            with self.subTest(ids=bad.tolist()):
                with self.assertRaises(ValueError):
                    self.layer.forward(bad)

    def test_non_integer_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            self.layer.forward(np.array([[0.5]]))

    def test_float_valued_integers_are_accepted(self):
        # ids often arrive as floats from array plumbing; unambiguous, so allowed
        np.testing.assert_allclose(self.layer.forward(np.array([[3.0]]))[0, 0],
                                   self.layer.weights[3])

    def test_invalid_sizes_are_rejected(self):
        for kwargs in ({'vocab_size': 0, 'd_model': 4},
                       {'vocab_size': 4, 'd_model': 0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    Embedding(**kwargs)


class TestPositionalEncoding(unittest.TestCase):
    """Attention sees a SET of vectors, so order has to be added to them."""

    def setUp(self):
        self.layer = PositionalEncoding(50, 8)
        self.layer.initialize(SGD())

    def test_every_position_gets_a_distinct_code(self):
        # If two positions shared a code the model could not tell them apart.
        codes = self.layer.encoding[:20]
        self.assertEqual(len(np.unique(codes, axis=0)), 20)

    def test_encoding_is_bounded(self):
        # sin and cos, so it cannot dominate the embeddings it is added to.
        self.assertGreaterEqual(self.layer.encoding.min(), -1.0)
        self.assertLessEqual(self.layer.encoding.max(), 1.0)

    def test_it_adds_rather_than_replaces(self):
        x = np.zeros((2, 5, 8))
        np.testing.assert_allclose(self.layer.forward(x),
                                   np.broadcast_to(self.layer.encoding[:5],
                                                   (2, 5, 8)))

    def test_the_same_position_gets_the_same_code_in_every_example(self):
        out = self.layer.forward(np.zeros((3, 4, 8)))
        np.testing.assert_allclose(out[0], out[1])
        np.testing.assert_allclose(out[0], out[2])

    def test_gradient_passes_straight_through(self):
        # Adding a constant does not change the derivative.
        error = np.random.RandomState(0).randn(2, 5, 8)
        np.testing.assert_allclose(self.layer.backward(error), error)

    def test_a_too_long_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            self.layer.forward(np.zeros((1, 99, 8)))

    def test_it_has_no_parameters(self):
        self.assertFalse(hasattr(self.layer, 'weights'))


class TestTransformerBlock(unittest.TestCase):
    def test_output_keeps_the_input_shape(self):
        block = TransformerBlock(8, 2)
        block.initialize(SGD())
        self.assertEqual(block.forward(np.random.randn(2, 5, 8)).shape, (2, 5, 8))

    def test_feed_forward_widens_by_four_by_default(self):
        self.assertEqual(TransformerBlock(8, 2).d_ff, 32)
        self.assertEqual(TransformerBlock(8, 2, d_ff=5).d_ff, 5)

    def zeroed_block(self):
        """A block whose two sub-layer branches contribute nothing, so only the
        residual path and the normalization remain."""
        np.random.seed(0)
        block = TransformerBlock(8, 2)
        block.initialize(SGD())
        block.attention.W_o.weights[:] = 0
        block.attention.W_o.bias[:] = 0
        block.ff_out.weights[:] = 0
        block.ff_out.bias[:] = 0
        return block

    def test_zeroed_branches_leave_only_the_normalizations(self):
        # POST-norm puts LayerNorm ON the residual path, so even a block that
        # contributes nothing does not pass its input through untouched: it
        # returns LayerNorm(LayerNorm(x)).
        #
        # This is the arrangement of the original paper, and this test is where
        # the cost of it is visible. Under PRE-norm the same block would return x
        # exactly, because the shortcut would be a clean sum.
        block = self.zeroed_block()
        x = np.random.randn(2, 5, 8)
        out = block.forward(x)
        self.assertFalse(np.allclose(out, x))
        np.testing.assert_allclose(
            out, block.norm2.forward(block.norm1.forward(x)))

    def test_the_residual_shortcut_still_carries_the_signal(self):
        # The residual is doing its job even with both branches dead: the output
        # is a normalised version of the input, so its ORDERING is preserved --
        # information has passed through rather than been destroyed.
        block = self.zeroed_block()
        x = np.random.randn(2, 5, 8)
        out = block.forward(x)
        for example in range(x.shape[0]):
            for position in range(x.shape[1]):
                with self.subTest(example=example, position=position):
                    np.testing.assert_array_equal(
                        np.argsort(x[example, position]),
                        np.argsort(out[example, position]))

    def test_gradient_matches_numerical(self):
        # The end-to-end check. Each residual copies the incoming gradient to
        # both the shortcut and the sub-layer; dropping either term still
        # produces plausible numbers, so only this catches it.
        for causal in (False, True):
            for n_heads, d_ff in [(2, None), (1, 6), (4, 16)]:
                with self.subTest(causal=causal, n_heads=n_heads, d_ff=d_ff):
                    fresh = frozen_block(8, n_heads, d_ff=d_ff, causal=causal)
                    rng = np.random.RandomState(11)
                    x = rng.randn(2, 4, 8)
                    error = rng.randn(2, 4, 8)
                    block = fresh()
                    block.forward(x)
                    analytic = block.backward(error.copy())
                    h = 1e-6
                    numerical = np.zeros_like(x)
                    for idx in np.ndindex(x.shape):
                        up, down = x.copy(), x.copy()
                        up[idx] += h
                        down[idx] -= h
                        numerical[idx] = (
                            (fresh().forward(up) * error).sum()
                            - (fresh().forward(down) * error).sum()) / (2 * h)
                    np.testing.assert_allclose(analytic, numerical, atol=1e-5)

    def test_causal_blocks_do_not_leak_the_future(self):
        np.random.seed(5)
        block = TransformerBlock(8, 2, causal=True)
        block.initialize(SGD(learning_rate=0.0))
        x = np.random.randn(1, 5, 8)
        base = block.forward(x)
        disturbed = x.copy()
        disturbed[0, -1, :] += 100.0
        after = block.forward(disturbed)
        for position in range(4):
            with self.subTest(position=position):
                np.testing.assert_allclose(base[0, position], after[0, position],
                                           atol=1e-9)

    def test_backward_updates_every_sublayer(self):
        block = TransformerBlock(8, 2)
        block.initialize(SGD(learning_rate=0.1, momentum=0))
        before = {
            'norm1': block.norm1.gamma.copy(),
            'norm2': block.norm2.gamma.copy(),
            'ff_in': block.ff_in.weights.copy(),
            'ff_out': block.ff_out.weights.copy(),
            'W_q': block.attention.W_q.weights.copy(),
        }
        x = np.random.RandomState(0).randn(2, 4, 8)
        out = block.forward(x)
        block.backward(np.random.RandomState(1).randn(*out.shape))
        after = {
            'norm1': block.norm1.gamma,
            'norm2': block.norm2.gamma,
            'ff_in': block.ff_in.weights,
            'ff_out': block.ff_out.weights,
            'W_q': block.attention.W_q.weights,
        }
        for name in before:
            with self.subTest(sublayer=name):
                self.assertFalse(np.allclose(before[name], after[name]))


class TestTransformerTrainsThroughTheContainer(unittest.TestCase):
    """The composite design justified: a block drops into the plain sequential
    NN container, which knows nothing about residuals or masking."""

    def build_task(self, vocab=12, seq_len=6, n=192):
        # Sequences counting upward mod vocab. Predicting the next token
        # requires actually reading the current one, so a model that ignores its
        # input cannot score well.
        rng = np.random.RandomState(0)
        starts = rng.randint(0, vocab, n)
        X = np.stack([(starts + i) % vocab for i in range(seq_len)], axis=1)
        y = np.stack([(starts + i + 1) % vocab for i in range(seq_len)], axis=1)
        return X, y, vocab, seq_len

    def test_a_small_transformer_learns_next_token_prediction(self):
        X, y, vocab, seq_len = self.build_task()
        np.random.seed(0)
        net = NN(epochs=60, batch_size=64, verbose=False,
                 loss="softmax-cross-entropy", optimizer=Adam(0.01))
        net.add(Embedding(vocab, 16))
        net.add(PositionalEncoding(seq_len, 16))
        net.add(TransformerBlock(16, 2, d_ff=32, causal=True))
        net.add(Dense(16, vocab))
        net.fit(Dataset(X, y))

        first, last = net.history[1][0], net.history[60][0]
        self.assertLess(last, first / 10)
        accuracy = (net.predict(X).argmax(axis=-1) == y).mean()
        self.assertGreater(accuracy, 0.95)

    def test_the_stack_reports_its_layers(self):
        net = NN(verbose=False)
        net.add(Embedding(12, 8))
        net.add(PositionalEncoding(6, 8))
        net.add(TransformerBlock(8, 2, causal=True))
        self.assertEqual(len(str(net).splitlines()), 3)
        self.assertIn('TransformerBlock', str(net))


if __name__ == "__main__":
    unittest.main()
