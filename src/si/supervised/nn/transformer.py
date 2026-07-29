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

    Uses the POST-norm arrangement of the original paper, "Attention Is All You
    Need" (Vaswani et al., 2017) -- normalization AFTER the residual addition:

        h   = LayerNorm(x + attention(x))
        out = LayerNorm(h + feedforward(h))

    Each sub-layer computes a CORRECTION which is added to its input, so the
    network learns what to CHANGE rather than having to reproduce what it was
    given. That is what makes deep stacks trainable at all: the shortcut carries
    the signal, and a sub-layer only has to contribute a refinement.

    Where the normalization sits, relative to that addition, is a real design
    choice and worth being explicit about:

    * POST-norm (here, and as published) puts LayerNorm ON the residual path.
      Nothing passes from input to output untouched, so a freshly initialised
      block is NOT the identity -- it returns LayerNorm(LayerNorm(x)).
    * PRE-norm, the later variant, normalises the sub-layer's INPUT instead
      (h = x + attention(LayerNorm(x))), leaving the shortcut a clean sum from
      input to output. It trains more forgivingly and is what most modern
      implementations use.

    That difference is not academic, and it shows up in THIS implementation. On
    the next-token task in the tests, with Adam(0.01) over three seeds:

        1 block   loss ~0.002   accuracy 1.000
        2 blocks  loss ~0.002   accuracy 1.000
        4 blocks  loss ~0.002   accuracy 1.000
        6 blocks  loss 1.6-2.5  accuracy 0.09-0.34   <-- collapses

    Chance on that task is 1/12 = 0.083, so six stacked post-norm blocks learn
    close to nothing, while four are perfectly fine. Depth is what breaks it: the
    normalization sits between the input and the output at every level, so the
    shortcut is attenuated once per block and the signal reaching the earliest
    layers decays with depth. Pre-norm, or a learning-rate warmup, is the usual
    remedy.

    Worth knowing before stacking these: at four blocks or fewer this trains
    without ceremony.

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
        # First sub-layer: attend, add the shortcut, THEN normalise.
        attended = self.attention.forward(input, training)
        hidden = self.norm1.forward(input + attended, training)

        # Second sub-layer: position-wise feed-forward, same pattern.
        transformed = self.ff_out.forward(
            self.relu.forward(self.ff_in.forward(hidden, training), training),
            training)
        return self.norm2.forward(hidden + transformed, training)

    def backward(self, output_error):
        # Mirror of forward, outermost first. The normalization is now INSIDE the
        # residual path, so the gradient passes through it before reaching the
        # branch point -- which is precisely why post-norm is harder to train
        # than pre-norm, where the shortcut is untouched.
        d_sum = self.norm2.backward(output_error)

        # sum = hidden + transformed: a sum COPIES its gradient to both inputs
        # rather than splitting it, so each branch receives d_sum in full.
        d_transformed = d_sum
        d_hidden = d_sum + self.ff_in.backward(
            self.relu.backward(self.ff_out.backward(d_transformed)))

        # hidden = LayerNorm(input + attended): the same pattern one level up.
        d_inner_sum = self.norm1.backward(d_hidden)
        d_attended = d_inner_sum
        return d_inner_sum + self.attention.backward(d_attended)

    def __str__(self):
        kind = "causal" if self.causal else "full"
        return (f"TransformerBlock(d_model={self.d_model}, "
                f"heads={self.n_heads}, d_ff={self.d_ff}, {kind})")
