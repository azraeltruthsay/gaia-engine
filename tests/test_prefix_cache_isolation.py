"""r4p6: the model appends each request's tokens to the KV cache it is
given, in place. PrefixCache.get_kv() handed out its stored cache, so one
request's conversation leaked into the next ("The secret word in the text
is ZEBRA" answering an unrelated question). Each call must get a copy."""

import torch
from transformers import DynamicCache

from gaia_engine.core import PrefixCache


class _Out:
    def __init__(self, kv):
        self.past_key_values = kv


class FakeModel:
    """Writes one KV entry per input token into a DynamicCache, like HF."""

    def __call__(self, ids, past_key_values=None, use_cache=True, **_):
        kv = past_key_values if past_key_values is not None else DynamicCache()
        n = ids.shape[1]
        k = torch.zeros(1, 1, n, 2)
        kv.update(k, k.clone(), 0)
        return _Out(kv)


class FakeTokenizer:
    def encode(self, text, return_tensors="pt", **_):
        return torch.zeros(1, len(text.split()), dtype=torch.long)


class FakeFormatter:
    def format_system(self, text):
        return text


def _cache():
    pc = PrefixCache(FakeModel(), FakeTokenizer(), device="cpu", formatter=FakeFormatter())
    pc.update_segment("identity", "you are gaia here")
    return pc


def test_generation_does_not_grow_the_cached_prefix():
    pc = _cache()
    model = pc.model
    kv, n = pc.get_kv()
    assert kv.get_seq_length() == n
    # A request's prompt + generated tokens extend the kv it was handed...
    model(torch.zeros(1, 50, dtype=torch.long), past_key_values=kv)
    assert kv.get_seq_length() == n + 50
    # ...but the next request starts from the clean prefix.
    kv2, n2 = pc.get_kv()
    assert n2 == n and kv2.get_seq_length() == n
    assert kv2 is not kv


def test_miss_path_also_returns_a_private_copy():
    pc = _cache()
    kv, n = pc.get_kv()  # computed on this call
    pc.model(torch.zeros(1, 7, dtype=torch.long), past_key_values=kv)
    assert pc._cached_kv.get_seq_length() == n
