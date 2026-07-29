# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Tests for the character-level language model.
#
# The transformer layers themselves are covered in test_transformer.py and
# test_attention.py. What is left here is the language-model plumbing -- the
# tokenizer, the windowing, the generation loop -- plus one end-to-end test that
# the assembled stack can actually learn to predict text.
#
# That end-to-end test overfits a single short string deliberately. Overfitting
# is normally a failure, but as a test it is ideal: it is fast, it has an
# unambiguous pass condition (the model reproduces the string it memorised), and
# it fails if ANY link in the chain is broken. Judging fluency would be slow and
# subjective; judging memorisation is neither.
# ----------------------------------------------------------------------------
import os
import unittest
import warnings

import numpy as np

from si.data import Dataset
from si.supervised.nn.language_model import (
    CharTokenizer,
    build_language_model,
    generate,
    load_text,
    make_windows,
)


CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'datasets', 'tiny-text.txt')


class TestCharTokenizer(unittest.TestCase):
    def test_vocabulary_is_derived_from_the_text_and_sorted(self):
        # Sorted so the ids are stable: the same text always produces the same
        # vocabulary, whatever order the characters happened to appear in.
        tokenizer = CharTokenizer("banana")
        self.assertEqual(tokenizer.chars, ['a', 'b', 'n'])
        self.assertEqual(tokenizer.vocab_size, 3)
        self.assertEqual(len(tokenizer), 3)

    def test_round_trip(self):
        text = "To be, or not to be"
        tokenizer = CharTokenizer(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_nothing_is_normalised_away(self):
        # Case, punctuation and newlines are all just characters to learn.
        tokenizer = CharTokenizer("Aa\n!")
        self.assertEqual(tokenizer.vocab_size, 4)
        self.assertEqual(tokenizer.decode(tokenizer.encode("Aa\n!")), "Aa\n!")

    def test_encode_returns_integer_ids(self):
        ids = CharTokenizer("abc").encode("cab")
        self.assertTrue(np.issubdtype(ids.dtype, np.integer))
        np.testing.assert_array_equal(ids, [2, 0, 1])

    def test_unknown_characters_are_reported(self):
        tokenizer = CharTokenizer("abc")
        with self.assertRaises(ValueError) as ctx:
            tokenizer.encode("axc")
        self.assertIn("'x'", str(ctx.exception))

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            CharTokenizer("")

    def test_the_shipped_corpus_tokenizes(self):
        text = load_text(CORPUS)
        tokenizer = CharTokenizer(text)
        self.assertGreater(len(text), 1000)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


class TestMakeWindows(unittest.TestCase):
    def test_targets_are_the_inputs_shifted_one_left(self):
        # The definition of next-character prediction.
        ids = np.arange(10)
        X, y = make_windows(ids, seq_len=4)
        np.testing.assert_array_equal(X[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(y[0], [1, 2, 3, 4])
        np.testing.assert_array_equal(X[:, 1:], y[:, :-1])

    def test_one_window_yields_seq_len_training_signals(self):
        X, y = make_windows(np.arange(10), seq_len=4)
        self.assertEqual(X.shape, y.shape)
        self.assertEqual(X.shape[1], 4)

    def test_window_count_follows_the_stride(self):
        ids = np.arange(21)
        self.assertEqual(len(make_windows(ids, 4, stride=1)[0]), 17)
        self.assertEqual(len(make_windows(ids, 4, stride=4)[0]), 5)

    def test_needs_one_token_beyond_the_window(self):
        # The last position of a window needs a target after it.
        with self.assertRaises(ValueError):
            make_windows(np.arange(4), seq_len=4)
        # one more token is enough
        self.assertEqual(len(make_windows(np.arange(5), seq_len=4)[0]), 1)

    def test_invalid_arguments_are_rejected(self):
        for kwargs in ({'seq_len': 0}, {'seq_len': 4, 'stride': 0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    make_windows(np.arange(10), **kwargs)


class TestBuildLanguageModel(unittest.TestCase):
    def test_architecture(self):
        model = build_language_model(vocab_size=20, seq_len=8, d_model=16,
                                     n_heads=2, n_blocks=2)
        description = str(model)
        self.assertIn('Embedding(20, 16)', description)
        self.assertIn('PositionalEncoding', description)
        self.assertEqual(description.count('TransformerBlock'), 2)
        self.assertIn('Dense (16, 20)', description)

    def test_blocks_are_causal(self):
        # A language model whose attention could see ahead would be predicting
        # characters it has already been shown.
        model = build_language_model(vocab_size=20, seq_len=8, n_blocks=1)
        blocks = [layer for layer in model.layers
                  if type(layer).__name__ == 'TransformerBlock']
        self.assertTrue(all(block.causal for block in blocks))

    def test_deep_stacks_warn(self):
        # TransformerBlock is post-norm, which degrades sharply past about four
        # blocks; the caller should be told rather than left puzzled.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_language_model(vocab_size=20, seq_len=8, n_blocks=6)
        self.assertTrue(any('post-norm' in str(w.message) for w in caught))

    def test_shallow_stacks_do_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_language_model(vocab_size=20, seq_len=8, n_blocks=2)
        self.assertEqual([w for w in caught if 'post-norm' in str(w.message)], [])


class TestGeneration(unittest.TestCase):
    """Generation is checked on an UNTRAINED model where that is enough.

    Shape, determinism and temperature handling do not depend on the model being
    any good, so they can be verified in microseconds. Whether generation
    produces the RIGHT characters is the end-to-end test's job.
    """

    def setUp(self):
        np.random.seed(0)
        self.tokenizer = CharTokenizer("abcdefgh ")
        self.model = build_language_model(len(self.tokenizer), seq_len=6,
                                          d_model=8, n_heads=2, n_blocks=1)
        # generate() calls predict, which requires a fitted model
        self.model.is_fitted = True

    def test_produces_the_requested_number_of_characters(self):
        out = generate(self.model, self.tokenizer, "abc", n_chars=10, seq_len=6)
        self.assertEqual(len(out), len("abc") + 10)
        self.assertTrue(out.startswith("abc"))

    def test_output_is_always_in_the_vocabulary(self):
        out = generate(self.model, self.tokenizer, "abc", n_chars=20, seq_len=6,
                       random_state=0)
        self.assertTrue(set(out) <= set(self.tokenizer.chars))

    def test_greedy_is_deterministic(self):
        first = generate(self.model, self.tokenizer, "abc", n_chars=15,
                         seq_len=6, temperature=0)
        second = generate(self.model, self.tokenizer, "abc", n_chars=15,
                          seq_len=6, temperature=0)
        self.assertEqual(first, second)

    def test_sampling_is_reproducible_with_a_seed(self):
        kwargs = dict(n_chars=15, seq_len=6, temperature=1.0)
        first = generate(self.model, self.tokenizer, "abc",
                         random_state=7, **kwargs)
        second = generate(self.model, self.tokenizer, "abc",
                          random_state=7, **kwargs)
        self.assertEqual(first, second)

    def test_different_seeds_give_different_samples(self):
        kwargs = dict(n_chars=40, seq_len=6, temperature=1.0)
        first = generate(self.model, self.tokenizer, "abc",
                         random_state=1, **kwargs)
        second = generate(self.model, self.tokenizer, "abc",
                          random_state=2, **kwargs)
        self.assertNotEqual(first, second)

    def test_a_very_low_temperature_reproduces_greedy(self):
        # The property that actually pins temperature to the sampling
        # distribution. Dividing by a tiny temperature makes the largest logit
        # dominate completely, so sampling collapses onto the argmax.
        #
        # Without this, a `generate` that ignored `temperature` entirely still
        # passed every other test here: determinism at 0 and seeded
        # reproducibility hold whatever the distribution is.
        greedy = generate(self.model, self.tokenizer, "abc", n_chars=60,
                          seq_len=6, temperature=0)
        nearly_greedy = generate(self.model, self.tokenizer, "abc", n_chars=60,
                                 seq_len=6, temperature=0.01, random_state=0)
        self.assertEqual(nearly_greedy, greedy)

    def test_agreement_with_greedy_falls_as_temperature_rises(self):
        # Higher temperature flattens the distribution, so samples wander further
        # from the most likely character.
        greedy = generate(self.model, self.tokenizer, "abc", n_chars=120,
                          seq_len=6, temperature=0)[3:]

        def agreement(temperature):
            sampled = generate(self.model, self.tokenizer, "abc", n_chars=120,
                               seq_len=6, temperature=temperature,
                               random_state=0)[3:]
            return sum(a == b for a, b in zip(sampled, greedy)) / len(greedy)

        low, middle, high = agreement(0.1), agreement(1.0), agreement(5.0)
        self.assertGreater(low, middle)
        self.assertGreater(middle, high)

    def test_higher_temperature_uses_more_of_the_vocabulary(self):
        def distinct(temperature):
            return len(set(generate(self.model, self.tokenizer, "abc",
                                    n_chars=120, seq_len=6,
                                    temperature=temperature,
                                    random_state=0)[3:]))

        self.assertLess(distinct(0.1), distinct(1.0))

    def test_a_long_prompt_is_truncated_to_the_context_window(self):
        # The model cannot attend further back than its positional encoding
        # covers, so a longer prompt must not be fed in whole.
        long_prompt = "abcdefgh" * 3          # 24 characters, window is 6
        out = generate(self.model, self.tokenizer, long_prompt, n_chars=5,
                       seq_len=6, temperature=0)
        self.assertEqual(len(out), len(long_prompt) + 5)

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            generate(self.model, self.tokenizer, "", n_chars=5)

    def test_negative_temperature_is_rejected(self):
        with self.assertRaises(ValueError):
            generate(self.model, self.tokenizer, "abc", n_chars=5,
                     temperature=-1.0)


class TestEndToEnd(unittest.TestCase):
    """One test that the whole assembly learns. Deliberately an overfit.

    Memorising a single string is a poor model and an excellent test: fast,
    with an unambiguous pass condition, and sensitive to a break anywhere in the
    chain -- tokenizer, windowing, embedding, attention, causal mask, loss or
    generation.
    """

    def test_it_memorises_a_short_string(self):
        text = "abcabcabcabcabcabcabcabcabcabc"
        tokenizer = CharTokenizer(text)
        seq_len = 6
        X, y = make_windows(tokenizer.encode(text), seq_len)

        np.random.seed(0)
        model = build_language_model(len(tokenizer), seq_len, d_model=16,
                                     n_heads=2, n_blocks=1, d_ff=32,
                                     epochs=80, batch_size=16,
                                     learning_rate=0.01)
        model.fit(Dataset(X, y))

        first = model.history[1][0]
        last = model.history[80][0]
        self.assertLess(last, first / 5)

        # Every next character in this text is fully determined, so a model that
        # has learned it should predict them all.
        accuracy = (model.predict(X).argmax(axis=-1) == y).mean()
        self.assertGreater(accuracy, 0.95)

        # And greedy continuation should carry the pattern on.
        continued = generate(model, tokenizer, "abc", n_chars=9,
                             seq_len=seq_len, temperature=0)
        self.assertEqual(continued, "abc" + "abcabcabc")

    def test_an_untrained_model_does_not_pass_that_bar(self):
        # Control: the assertion above is not something any model satisfies.
        text = "abcabcabcabcabcabcabcabcabcabc"
        tokenizer = CharTokenizer(text)
        seq_len = 6
        X, y = make_windows(tokenizer.encode(text), seq_len)
        np.random.seed(0)
        model = build_language_model(len(tokenizer), seq_len, d_model=16,
                                     n_heads=2, n_blocks=1, d_ff=32)
        model.is_fitted = True
        accuracy = (model.predict(X).argmax(axis=-1) == y).mean()
        self.assertLess(accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()
