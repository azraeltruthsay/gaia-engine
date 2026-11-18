"""xani: the CUDA-graph static KV grows for a long prompt and shrinks back
after a few short ones (it only grew; Prime fell from 47 to 14 t/s)."""
import types

import pytest

from gaia_engine import core


class _FakeDecoder:
    def __init__(self, model, max_len, device):
        self.model, self.max_len, self.failed = model, max_len, set()


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(core, "GRAPH_DECODE_ENABLED", True)
    monkeypatch.setattr(core, "_GraphDecoder", _FakeDecoder)
    monkeypatch.setattr(core, "_static_kv_bytes_per_token", lambda cfg: 1)
    monkeypatch.setattr(core.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(core.torch.cuda, "mem_get_info", lambda: (10 ** 12, 10 ** 12))
    model_type = next(iter(core._GRAPH_VERIFIED_MODEL_TYPES))
    model = types.SimpleNamespace(config=types.SimpleNamespace(
        model_type=model_type, get_text_config=lambda: None), device="cuda")
    e = core.GAIAEngine.__new__(core.GAIAEngine)
    e.device, e.model = "cuda", model
    return e


def test_grows_then_shrinks_after_short_requests(engine):
    assert engine._graph_decoder_for(9000).max_len == 16384
    for _ in range(core._GRAPH_SHRINK_AFTER - 1):
        assert engine._graph_decoder_for(1000).max_len == 16384
    assert engine._graph_decoder_for(1000).max_len == core._GRAPH_MIN_LEN


def test_a_mid_sized_request_keeps_it(engine):
    engine._graph_decoder_for(9000)
    for _ in range(5):
        assert engine._graph_decoder_for(6000).max_len == 16384
