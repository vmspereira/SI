# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# The pieces that turn attention into a transformer.
#
# Attention alone can only average existing values: it moves information between
# positions but never transforms it. A transformer block pairs it with three
# other ingredients:
#
#   * a position-wise feed-forward network, which does the actual computation on
#     what attention gathered;
#   * residual connections, so every sub-layer only has to learn a CORRECTION to
#     its input rather than reproduce it, which is what makes deep stacks
#     trainable;
#   * layer normalization, which keeps activations at a workable scale.
#
# Plus the two layers that turn discrete tokens into something a network can
# consume: an embedding table and a positional encoding.
# ---------------------------------------------------------------------------
import numpy as np

from .layers import Layer, Dense, LayerNorm
from .activation import ReLU
from .attention import MultiHeadAttention


class Embedding(Layer):
    """A lookup table mapping integer token ids to learned vectors.

    A token id is a name, not a quantity: character 42 is not "greater than"
    character 7, and feeding the raw integer to a Dense layer would invent an
    ordering that does not exist. One-hot encoding avoids that but is wasteful --
    multiplying a one-hot row by a weight matrix just SELECTS a row of it.

    An embedding skips the multiplication and does the selection directly, so
    the table IS the weight matrix of that imaginary Dense layer. Each row is a
    learned vector for one token, adjusted by training like any other parameter.

    :param int vocab_size: number of distinct tokens.
    :param int d_model: width of each token's vector.
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        if vocab_size < 1 or d_model < 1:
            raise ValueError(
                f"vocab_size and d_model must both be at least 1; got "
                f"{vocab_size} and {d_model}."
            )
        self.vocab_size = vocab_size
        self.d_model = d_model

    def initialize(self, optimizer):
        from copy import copy
        # Small random vectors, as elsewhere in this library: large ones would
        # dominate the positional encoding added on top.
        self.weights = np.random.rand(self.vocab_size, self.d_model) - 0.5
        self.w_opt = copy(optimizer)

    def forward(self, input, training=True):
        ids = np.asarray(input)
        if not np.issubdtype(ids.dtype, np.integer):
            if np.all(ids == np.floor(ids)):
                ids = ids.astype(int)
            else:
                raise ValueError(
                    f"Embedding expects integer token ids; got dtype {ids.dtype}."
                )
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ValueError(
                f"token ids must lie in [0, {self.vocab_size - 1}]; got a range "
                f"of [{ids.min()}, {ids.max()}]."
            )
        self.ids = ids
        # Fancy indexing does the whole lookup: (b, t) -> (b, t, d_model).
        return self.weights[ids]

    def backward(self, output_error):
        # Only the rows that were actually looked up receive gradient, and a
        # token appearing several times accumulates all of its occurrences --
        # hence np.add.at rather than plain indexed assignment, which would keep
        # only the last write.
        grad = np.zeros_like(self.weights)
        np.add.at(grad, self.ids, output_error)
        self.weights = self.w_opt.update(self.weights, grad)
        # The input is a set of integer ids. There is no meaningful derivative
        # with respect to them, so nothing flows further back; zeros keep the
        # container's chain intact if anything precedes this layer.
        return np.zeros(self.ids.shape)

    def __str__(self):
        return f"Embedding({self.vocab_size}, {self.d_model})"


class PositionalEncoding(Layer):
    """Adds a fixed, position-dependent signal to the embeddings.

    Attention is permutation-invariant: it sees a SET of vectors, with no notion
    of which came first. Shuffle the input and the outputs shuffle with it, so
    "dog bites man" and "man bites dog" would be indistinguishable. Order has to
    be put into the vectors themselves.

    This uses the original sinusoidal scheme: for position pos and dimension i,

        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    Each dimension is a sinusoid of a different wavelength, from ~2 up to
    ~10000*2*pi, so a position is encoded by a unique pattern across dimensions
    -- rather like a binary counter with smooth digits. It has no parameters and
    extends to any length up to max_len, and because the offset between two
    positions is a fixed linear map of the encodings, relative position is
    recoverable too.

    :param int max_len: longest sequence this layer will be asked to encode.
    :param int d_model: embedding width, matching the layer it feeds.
    """

    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        position = np.arange(max_len)[:, None]                 # (max_len, 1)
        index = np.arange(d_model)[None, :]                    # (1, d_model)
        # 2i for both members of each sin/cos pair, so the pair shares a wavelength
        angle = position / np.power(10000.0, (2 * (index // 2)) / d_model)
        self.encoding = np.where(index % 2 == 0, np.sin(angle), np.cos(angle))

    def initialize(self, optimizer):
        pass

    def forward(self, input, training=True):
        seq_len = input.shape[1]
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence of length {seq_len} exceeds max_len={self.max_len}."
            )
        return input + self.encoding[:seq_len]

    def backward(self, output_error):
        # Adding a constant does not change the gradient: d(x + c)/dx = 1.
        return output_error

    def __str__(self):
        return f"PositionalEncoding(max_len={self.max_len}, d_model={self.d_model})"


class TransformerBlock(Layer):
    """One transformer layer: attention, then a feed-forward, both residual.

    Uses the PRE-norm arrangement:

        h   = x + attention(LayerNorm(x))
        out = h + feedforward(LayerNorm(h))

    rather than the post-norm of the original paper, x = LayerNorm(x + attn(x)).
    The difference matters in practice: with pre-norm the residual path from
    input to output is a clean sum with nothing in the way, so gradients reach
    early layers undiminished and training is stable without a learning-rate
    warmup. Post-norm needs that warmup, which is a lot of machinery to explain
    for a teaching implementation.

    Note what the residuals buy. Each sub-layer computes a CORRECTION added to
    its input, so a block that has learned nothing yet is roughly the identity
    and passes its input through unharmed. Stacking is then safe: depth cannot
    make things worse before training makes them better.

    The feed-forward is applied position-wise -- the same small network at every
    position, which is exactly what a Dense layer over the last axis now does.
    It widens to d_ff (conventionally 4x d_model) and back, and it is where most
    of a transformer's parameters live.

    This is a COMPOSITE layer: it owns its sub-layers and performs the residual
    additions inside its own forward and backward. That is deliberate -- the NN
    container is strictly sequential and cannot express a skip connection, so
    keeping the branching internal lets a block drop straight into it.

    :param int d_model: embedding width in and out.
    :param int n_heads: attention heads.
    :param int d_ff: hidden width of the feed-forward. Defaults to 4 * d_model.
    :param bool causal: if True, no position attends to a later one.
    """

    def __init__(self, d_model, n_heads, d_ff=None, causal=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = 4 * d_model if d_ff is None else d_ff
        self.causal = causal

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.norm2 = LayerNorm(d_model)
        self.ff_in = Dense(d_model, self.d_ff)
        self.relu = ReLU()
        self.ff_out = Dense(self.d_ff, d_model)

    def _sublayers(self):
        return (self.norm1, self.attention, self.norm2,
                self.ff_in, self.relu, self.ff_out)

    def initialize(self, optimizer):
        for sublayer in self._sublayers():
            sublayer.initialize(optimizer)

    def forward(self, input, training=True):
        # First residual branch: attention over the normalised input.
        attended = self.attention.forward(
            self.norm1.forward(input, training), training)
        hidden = input + attended

        # Second residual branch: position-wise feed-forward.
        transformed = self.ff_out.forward(
            self.relu.forward(
                self.ff_in.forward(self.norm2.forward(hidden, training),
                                   training),
                training),
            training)
        return hidden + transformed

    def backward(self, output_error):
        # out = hidden + transformed, so the incoming gradient reaches BOTH the
        # residual shortcut and the feed-forward branch. Gradients through a sum
        # are copied, not split.
        d_transformed = output_error
        d_hidden = output_error + self.norm2.backward(
            self.ff_in.backward(
                self.relu.backward(
                    self.ff_out.backward(d_transformed))))

        # hidden = input + attended: the same pattern one level up.
        d_attended = d_hidden
        return d_hidden + self.norm1.backward(
            self.attention.backward(d_attended))

    def __str__(self):
        kind = "causal" if self.causal else "full"
        return (f"TransformerBlock(d_model={self.d_model}, "
                f"heads={self.n_heads}, d_ff={self.d_ff}, {kind})")
