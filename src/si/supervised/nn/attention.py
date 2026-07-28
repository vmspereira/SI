# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Scaled dot-product self-attention.
#
# Every layer before this one mixes information along the FEATURE axis: Dense
# combines the features of one sample, Conv2D combines a small neighbourhood of
# pixels, and the RNN carries a single hidden state forward one step at a time.
# Attention mixes along the SEQUENCE axis instead, and does it in one shot: each
# position looks at every other position and takes a weighted average of them,
# with the weights computed from the data itself.
#
# That is the whole idea. The rest is bookkeeping.
#
# Tensor layout is (n_examples, seq_len, d_model) -- a batch of sequences, the
# same convention the RNN uses.
# ---------------------------------------------------------------------------
import numpy as np

from .layers import Layer, Dense


def softmax(z, axis=-1):
    """Softmax along one axis, computed stably.

    Subtracts the maximum before exponentiating: the result is unchanged, since
    a constant cancels between the numerator and the denominator, but every
    exponent becomes <= 0 and so cannot overflow. Attention scores grow with the
    magnitude of the inputs, so this is not a theoretical concern.
    """
    shifted = z - np.max(z, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def softmax_backward(probabilities, output_error, axis=-1):
    """Gradient through a softmax, given dE/dp and p = softmax(z).

    The Jacobian of softmax is a full matrix, not a diagonal:

        dp_i / dz_j = p_i * (delta_ij - p_j)

    so changing ONE logit moves EVERY probability -- they are tied together by
    the normalisation. Contracting dE/dp with that Jacobian collapses to

        dE/dz = p * (dE/dp - sum_j(dE/dp_j * p_j))

    which is what this returns, and it costs no more than the elementwise form.

    Note this is deliberately NOT the `SoftMax` activation in activation.py.
    That one returns only the diagonal, p*(1-p), because the framework's
    element-wise `backward` contract cannot express a full Jacobian; it is
    documented and tested as such. It is fine where softmax is fused with a
    cross-entropy loss (the two simplify to p - y), but attention applies
    softmax in the MIDDLE of the network, where the off-diagonal terms matter.
    """
    weighted = np.sum(output_error * probabilities, axis=axis, keepdims=True)
    return probabilities * (output_error - weighted)


class SelfAttention(Layer):
    """Single-head scaled dot-product self-attention.

    Each position emits three vectors, obtained by three learned projections of
    its own embedding:

        query  -- what this position is looking for
        key    -- what this position offers to others
        value  -- what this position actually contributes if attended to

    Position i then scores every position j by how well its query matches j's
    key (a dot product), turns those scores into weights with a softmax, and
    outputs the weighted average of the values:

        scores = Q K^T / sqrt(d_k)          (seq_len, seq_len) per example
        weights = softmax(scores)           each row sums to 1
        output = weights V                  then a final output projection

    Why divide by sqrt(d_k)? The dot product of two d_k-dimensional vectors with
    unit-variance entries has variance d_k, so without the scaling the scores
    grow with the model size, the softmax saturates, and its gradient vanishes.
    Dividing by sqrt(d_k) keeps the scores at roughly unit variance whatever
    d_k is.

    :param int d_model: size of the input and output embeddings.
    :param int d_k: internal query/key/value width. Defaults to d_model.
    :param bool causal: if True, position i may only attend to positions <= i.
        This is what makes a language model possible: predicting the next token
        is only a real task if the model cannot read it first.
    """

    def __init__(self, d_model, d_k=None, causal=False):
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be at least 1; got {d_model}.")
        self.d_model = d_model
        self.d_k = d_model if d_k is None else d_k
        if self.d_k < 1:
            raise ValueError(f"d_k must be at least 1; got {self.d_k}.")
        self.causal = causal
        # The four projections are Dense layers, which keeps the affine maths and
        # the optimizer wiring in one place. They rely on Dense accepting a
        # (batch, seq, features) input and applying itself per position.
        self.W_q = Dense(d_model, self.d_k)
        self.W_k = Dense(d_model, self.d_k)
        self.W_v = Dense(d_model, self.d_k)
        self.W_o = Dense(self.d_k, d_model)

    def initialize(self, optimizer):
        for projection in (self.W_q, self.W_k, self.W_v, self.W_o):
            projection.initialize(optimizer)

    def _causal_mask(self, seq_len):
        """True where attention is FORBIDDEN: strictly above the diagonal.

        Row i is allowed columns 0..i, so a position can attend to itself and to
        everything before it, never to the future.
        """
        return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    def forward(self, input, training=True):
        if input.ndim != 3:
            raise ValueError(
                "SelfAttention expects (n_examples, seq_len, d_model); got "
                f"{input.ndim} dimension(s) with shape {input.shape}."
            )
        if input.shape[2] != self.d_model:
            raise ValueError(
                f"This SelfAttention was built for d_model={self.d_model}, but "
                f"the input has {input.shape[2]} features."
            )
        _, seq_len, _ = input.shape

        # Project each position independently into query/key/value space.
        self.Q = self.W_q.forward(input, training)   # (n, t, d_k)
        self.K = self.W_k.forward(input, training)   # (n, t, d_k)
        self.V = self.W_v.forward(input, training)   # (n, t, d_k)

        # scores[n, i, j] = how much position i of example n attends to j.
        # matmul batches over the leading axis, so this is one (t, d_k) @ (d_k, t)
        # product per example.
        self.scale = 1.0 / np.sqrt(self.d_k)
        scores = np.matmul(self.Q, np.swapaxes(self.K, -1, -2)) * self.scale

        if self.causal:
            # -inf before the softmax becomes exactly 0 after it, so a forbidden
            # position contributes nothing AND receives no gradient. Masking
            # after the softmax instead would leave the weights un-normalised.
            scores = np.where(self._causal_mask(seq_len), -np.inf, scores)

        # Cached for backward: each row is a probability distribution over the
        # positions this one is allowed to see.
        self.weights = softmax(scores, axis=-1)
        context = np.matmul(self.weights, self.V)    # (n, t, d_k)
        return self.W_o.forward(context, training)

    def backward(self, output_error):
        # Back through the output projection first; Dense updates its own
        # parameters and hands back dE/d(context).
        d_context = self.W_o.backward(output_error)          # (n, t, d_k)

        # context = weights @ V
        d_weights = np.matmul(d_context, np.swapaxes(self.V, -1, -2))
        d_V = np.matmul(np.swapaxes(self.weights, -1, -2), d_context)

        # Through the softmax, using the full-Jacobian form.
        d_scores = softmax_backward(self.weights, d_weights, axis=-1)

        if self.causal:
            # Masked entries were -inf, so their weight is 0 and they must
            # receive no gradient. softmax_backward already yields ~0 there
            # (p == 0 multiplies the whole term), but zeroing explicitly keeps a
            # 0 * inf from ever producing a NaN.
            d_scores = np.where(self._causal_mask(d_scores.shape[-1]),
                                0.0, d_scores)

        # scores = Q K^T * scale
        d_scores = d_scores * self.scale
        d_Q = np.matmul(d_scores, self.K)
        d_K = np.matmul(np.swapaxes(d_scores, -1, -2), self.Q)

        # Each of Q, K and V is a projection of the SAME input, so the three
        # paths all lead back to it and their gradients add.
        return (self.W_q.backward(d_Q)
                + self.W_k.backward(d_K)
                + self.W_v.backward(d_V))

    def __str__(self):
        kind = "causal" if self.causal else "full"
        return f"SelfAttention(d_model={self.d_model}, d_k={self.d_k}, {kind})"
