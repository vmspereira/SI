# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A character-level language model, assembled from the layers in this package.
#
# A language model answers one question, over and over: given the text so far,
# which character comes next? That is all next-token prediction is, and it is
# enough -- run it repeatedly, feeding each prediction back in, and the model
# writes.
#
# Nothing here is new machinery. It is Embedding, PositionalEncoding, a stack of
# causal TransformerBlocks and a Dense projection to the vocabulary, trained by
# the ordinary NN container with softmax cross entropy. The only additions are
# the plumbing every language model needs: turning text into integers, cutting
# it into training windows, and sampling from the model to generate more.
#
# On expectations: this is NumPy on a CPU, with a few thousand characters of
# training text. Expect the loss to fall clearly and the samples to acquire the
# STATISTICS of the text -- plausible letter pairs, word-like runs, line breaks
# in roughly the right places. Do not expect fluent English. The gap between the
# two is compute and data, not a defect in the implementation.
# ---------------------------------------------------------------------------
import numpy as np

from .network import NN
from .layers import Dense
from .optimizers import Adam
from .transformer import Embedding, PositionalEncoding, TransformerBlock


class CharTokenizer:
    """Maps characters to integer ids and back.

    The vocabulary is DERIVED FROM THE TEXT, so it contains exactly the
    characters that occur, in sorted order. Nothing is normalised away: case,
    punctuation and newlines are all just characters the model has to learn,
    which is what makes character-level modelling the simplest place to start --
    no tokeniser training, no out-of-vocabulary handling, no word segmentation.

    The price is that the model must spend capacity learning spelling before it
    can learn anything about words.

    :param str text: the corpus whose characters define the vocabulary.
    """

    def __init__(self, text):
        if not text:
            raise ValueError("cannot build a vocabulary from empty text.")
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self._to_id = {char: index for index, char in enumerate(self.chars)}
        self._to_char = {index: char for char, index in self._to_id.items()}

    def encode(self, text):
        """Text -> a 1-D array of integer ids."""
        unknown = sorted(set(text) - set(self._to_id))
        if unknown:
            raise ValueError(
                f"cannot encode character(s) {unknown}: absent from the "
                "vocabulary, which was built from the training text. Build the "
                "tokenizer on text covering everything you intend to encode."
            )
        return np.array([self._to_id[char] for char in text], dtype=int)

    def decode(self, ids):
        """A sequence of integer ids -> text."""
        return ''.join(self._to_char[int(i)] for i in np.asarray(ids).ravel())

    def __len__(self):
        return self.vocab_size

    def __str__(self):
        return f"CharTokenizer({self.vocab_size} chars)"


def make_windows(ids, seq_len, stride=1):
    """Cuts a stream of token ids into overlapping next-character examples.

    For seq_len=4 and ids = [h, e, l, l, o, ...] the first example is

        X = [h, e, l, l]
        y = [e, l, l, o]

    so y is X shifted one step left: at every position the target is the NEXT
    character. One window therefore yields seq_len training signals, not one,
    and because the model is CAUSAL each position may only use what precedes it
    -- position 0 predicts from `h` alone, position 3 from `hell`. Without the
    causal mask this task would be trivial and the model would learn nothing.

    :param ids: 1-D array of token ids.
    :param int seq_len: window length.
    :param int stride: step between window starts. 1 gives maximum overlap and
        the most examples; larger strides trade examples for speed.
    :returns: (X, y), both (n_windows, seq_len) integer arrays.
    """
    ids = np.asarray(ids).ravel()
    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1; got {seq_len}.")
    if stride < 1:
        raise ValueError(f"stride must be at least 1; got {stride}.")
    # +1 because every window needs one character beyond its end to be the last
    # target.
    if len(ids) < seq_len + 1:
        raise ValueError(
            f"need at least seq_len + 1 = {seq_len + 1} tokens to build a "
            f"single window; got {len(ids)}."
        )
    starts = range(0, len(ids) - seq_len, stride)
    X = np.stack([ids[s:s + seq_len] for s in starts])
    y = np.stack([ids[s + 1:s + seq_len + 1] for s in starts])
    return X, y


def load_text(filename):
    """Reads a UTF-8 text corpus.

    Dataset.from_data cannot be used here: it goes through np.genfromtxt, which
    parses numbers.
    """
    with open(filename, encoding='utf-8') as handle:
        return handle.read()


def build_language_model(vocab_size, seq_len, d_model=64, n_heads=4,
                         n_blocks=2, d_ff=None, epochs=200, batch_size=64,
                         learning_rate=0.005, verbose=False):
    """Assembles a causal character-level transformer.

        Embedding -> PositionalEncoding -> n_blocks x TransformerBlock -> Dense

    The Dense projects to vocab_size logits per position, one score per possible
    next character, which softmax cross entropy consumes directly.

    Defaults are chosen so a run finishes in a reasonable time in pure NumPy;
    they are small enough that the result is a demonstration rather than a
    useful model. n_blocks defaults to 2 deliberately: TransformerBlock uses
    post-norm, which trains cleanly up to about four blocks and then degrades
    sharply -- see its docstring for the measured figures.

    :returns: an unfitted NN, ready for .fit(Dataset(X, y)).
    """
    if n_blocks > 4:
        # Not forbidden, but the caller should know what they are in for.
        import warnings
        warnings.warn(
            f"n_blocks={n_blocks}: post-norm blocks train cleanly up to about "
            "four and then degrade sharply (6 blocks reached only 0.09-0.34 "
            "accuracy on the next-token test task). Consider fewer blocks.",
            stacklevel=2,
        )
    net = NN(epochs=epochs, batch_size=batch_size, verbose=verbose,
             loss="softmax-cross-entropy", optimizer=Adam(learning_rate),
             step=max(1, epochs // 10))
    net.add(Embedding(vocab_size, d_model))
    net.add(PositionalEncoding(seq_len, d_model))
    for _ in range(n_blocks):
        net.add(TransformerBlock(d_model, n_heads, d_ff=d_ff, causal=True))
    net.add(Dense(d_model, vocab_size))
    return net


def generate(model, tokenizer, prompt, n_chars=200, seq_len=None,
             temperature=1.0, random_state=None):
    """Continues `prompt` one character at a time.

    The loop is the whole idea of autoregressive generation: predict the next
    character, append it to the context, predict again. The model only ever does
    what it was trained to do -- score the next character -- and repetition turns
    that into text.

    Only the LAST position's logits are used. The others predict characters we
    already know; during training they provided extra signal, but here they are
    history.

    :param model: a fitted language model, as built by build_language_model.
    :param CharTokenizer tokenizer: the tokenizer the model was trained with.
    :param str prompt: text to continue. Must be non-empty -- the model
        conditions on context, so there has to be some.
    :param int n_chars: how many characters to produce.
    :param int seq_len: how much context to keep. Longer prompts are truncated
        from the left to this many characters, which must not exceed the
        positional encoding's max_len. Defaults to the full prompt length.
    :param float temperature: sharpens (< 1) or flattens (> 1) the distribution
        before sampling. 0 means greedy -- always the single most likely
        character, deterministic and prone to repetition.
    :param int random_state: seed, for reproducible samples.
    :returns: the prompt followed by the generated text.
    """
    if not prompt:
        raise ValueError("prompt must be non-empty: the model needs context.")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0; got {temperature}.")
    rng = np.random.default_rng(random_state)

    context = list(tokenizer.encode(prompt))
    window = len(context) if seq_len is None else seq_len
    generated = []

    for _ in range(n_chars):
        # Keep only the most recent `window` characters: the model cannot attend
        # further back than its positional encoding covers.
        recent = np.array(context[-window:], dtype=int)[None, :]
        logits = model.predict(recent)[0, -1]

        if temperature == 0:
            next_id = int(np.argmax(logits))
        else:
            # Stable softmax over the tempered logits. Dividing by a small
            # temperature makes large logits dominate, so the sample approaches
            # the greedy choice; a large one flattens towards uniform.
            scaled = logits / temperature
            scaled = scaled - np.max(scaled)
            probabilities = np.exp(scaled)
            probabilities /= probabilities.sum()
            next_id = int(rng.choice(len(probabilities), p=probabilities))

        context.append(next_id)
        generated.append(next_id)

    return prompt + tokenizer.decode(generated)
