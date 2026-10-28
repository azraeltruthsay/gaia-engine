"""uapv: static-KV / CUDA-graph decode — the parts checkable without a GPU.

The decode itself is validated on the GPU (teacher-forced: the graph is
bit-exact to the same static cache run eagerly; that differs from the dynamic
cache only by bf16 rounding). Here: KV sizing and when the engine declines to
use the graph path at all.
"""

from types import SimpleNamespace

from gaia_engine import core
from gaia_engine.core import GAIAEngine, _static_kv_bytes_per_token

# Gemma4-E4B-shaped: 42 layers, the last 18 share KV, every 6th is full attention.
CORE_TEXT = SimpleNamespace(
    num_hidden_layers=42, num_kv_shared_layers=18, num_key_value_heads=2,
    head_dim=256, global_head_dim=512,
    layer_types=["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(42)],
)


def test_kv_bytes_count_only_layers_that_own_a_cache():
    # 24 owning layers: 20 sliding (2 heads x 256 x K,V x bf16) + 4 full (x 512)
    assert _static_kv_bytes_per_token(CORE_TEXT) == 20 * 2 * 256 * 2 * 2 + 4 * 2 * 512 * 2 * 2


def _engine(device="cuda", model_type="gemma4", **extra):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type, get_text_config=lambda: CORE_TEXT))
    return SimpleNamespace(device=device, model=model, _graph_decoder=None, **extra)


def test_cpu_engine_never_uses_graphs():
    assert GAIAEngine._graph_decoder_for(_engine(device="cpu"), 1000) is None


def test_unverified_architecture_keeps_dynamic_loop():
    assert GAIAEngine._graph_decoder_for(_engine(model_type="qwen3_vl"), 1000) is None


def test_disabled_after_a_failure_stays_disabled():
    assert GAIAEngine._graph_decoder_for(_engine(_graph_disabled=True), 1000) is None


def test_env_switch_turns_it_off(monkeypatch):
    monkeypatch.setattr(core, "GRAPH_DECODE_ENABLED", False)
    assert GAIAEngine._graph_decoder_for(_engine(), 1000) is None


def test_existing_decoder_is_reused_when_it_fits():
    eng = _engine()
    eng._graph_decoder = SimpleNamespace(model=eng.model, max_len=4096, failed=set())
    assert GAIAEngine._graph_decoder_for(eng, 3000) is eng._graph_decoder


class _RefusesGlobalHeadDim(SimpleNamespace):
    """transformers 5.17's Gemma4TextConfig: head_dim varies per layer, so a
    global read raises; per_layer_config carries the real sizes."""
    def __getattribute__(self, name):
        if name == "head_dim":
            raise AttributeError("'head_dim' is a per-layer attribute")
        return super().__getattribute__(name)


def test_kv_bytes_from_per_layer_config():
    per_layer = [SimpleNamespace(head_dim=512 if t == "full_attention" else 256, num_key_value_heads=2)
                 for t in CORE_TEXT.layer_types]
    cfg = _RefusesGlobalHeadDim(num_hidden_layers=42, num_kv_shared_layers=18, num_key_value_heads=2,
                                layer_types=CORE_TEXT.layer_types, per_layer_config=per_layer)
    assert _static_kv_bytes_per_token(cfg) == _static_kv_bytes_per_token(CORE_TEXT)


def test_key_whose_capture_failed_uses_dynamic_cache():
    eng = _engine()
    eng._graph_decoder = SimpleNamespace(model=eng.model, max_len=4096, failed={("core_thinking_v1", True)})
    assert GAIAEngine._graph_decoder_for(eng, 3000, ("core_thinking_v1", True)) is None
    assert GAIAEngine._graph_decoder_for(eng, 3000, (None, True)) is eng._graph_decoder
