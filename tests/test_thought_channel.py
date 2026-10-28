"""h1ok: Gemma 4's native thought channel travels as reasoning_content.

With <|think|> at the top of the system turn, Gemma 4 writes
<|channel>thought\\n...<channel|> before its answer. The torch path decoded
with skip_special_tokens, which dropped the markers and ran the thought into
the answer; gaia_cpp rendered the markers into the answer text. Both now
split the channel off.
"""

from gaia_engine.core import _split_thought_tokens
from gaia_engine.cpp.backend import _split_thought_text

OPEN, CLOSE = 100, 101


class Tok:
    """Word-per-id tokenizer with Gemma 4's two channel markers."""
    unk_token_id = 3
    vocab = {"<|channel>": OPEN, "<channel|>": CLOSE, "thought": 10, "\n": 11,
             "two": 12, "plus": 13, "is": 14, "four.": 15, "Four.": 16}
    inv = {v: k for k, v in vocab.items()}

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)

    def decode(self, ids, skip_special_tokens=True):
        words = [self.inv[i] for i in ids if not (skip_special_tokens and i in (OPEN, CLOSE))]
        return " ".join(words).replace(" \n ", "\n")


def test_tokens_split_thought_from_answer():
    gen = [OPEN, 10, 11, 12, 13, 12, 14, 15, CLOSE, 16]
    thought, answer = _split_thought_tokens(Tok(), gen)
    assert thought == "two plus two is four."
    assert answer == [16]


def test_tokens_without_channel_are_untouched():
    gen = [16]
    assert _split_thought_tokens(Tok(), gen) == (None, gen)


def test_tokens_cut_off_inside_thought_leave_no_answer():
    thought, answer = _split_thought_tokens(Tok(), [OPEN, 10, 11, 12, 13])
    assert thought == "two plus" and answer == []


def test_text_split_thought_from_answer():
    thought, answer = _split_thought_text("<|channel>thought\nCount the r's: three.<channel|>There are three.")
    assert thought == "Count the r's: three."
    assert answer == "There are three."


def test_text_without_channel_is_untouched():
    assert _split_thought_text("There are three.") == (None, "There are three.")


def test_text_cut_off_inside_thought_is_all_thought():
    assert _split_thought_text("<|channel>thought\nStill counting") == ("Still counting", "")
