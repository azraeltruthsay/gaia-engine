"""f8x8: every Gemma prompt starts with <bos>, on every path."""
import os

import pytest

from gaia_engine.core import always_bos


class _Tok:
    def __init__(self, bos="<bos>", bos_id=2, adds=False):
        self.bos_token, self.bos_token_id, self.add_bos_token = bos, bos_id, adds

    def encode(self, text, add_special_tokens=True):
        ids = [99]
        if add_special_tokens and self.add_bos_token and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return ids


def test_turns_bos_on_when_missing():
    t = _Tok(adds=False)
    assert always_bos(t) is True
    assert t.encode("x") == [2, 99]
    assert t.encode("x", add_special_tokens=False) == [99]


def test_leaves_a_tokenizer_that_adds_it():
    t = _Tok(adds=True)
    assert always_bos(t) is False and t.encode("x") == [2, 99]


def test_leaves_a_tokenizer_without_bos():  # Qwen
    t = _Tok(bos=None, bos_id=None)
    assert always_bos(t) is False and t.add_bos_token is False


_GEMMA12B = "/models/Gemma4-12B-GAIA-Core-v2"


@pytest.mark.skipif(not os.path.isdir(_GEMMA12B), reason="12B tokenizer not mounted")
def test_real_gemma_12b_tokenizer():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(_GEMMA12B)
    always_bos(t)
    assert t.encode("hi")[0] == t.bos_token_id
    assert t.encode("hi", add_special_tokens=False)[0] != t.bos_token_id
