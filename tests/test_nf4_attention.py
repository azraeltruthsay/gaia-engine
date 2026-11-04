"""Eager attention is for Gemma only: on Qwen3-VL a long prompt broke the worker."""
import json

from gaia_engine.core import _nf4_attention


def _model(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(tmp_path)


def test_gemma_keeps_eager(tmp_path):
    assert _nf4_attention(_model(tmp_path, "gemma4")) == "eager"


def test_others_use_sdpa(tmp_path):
    assert _nf4_attention(_model(tmp_path, "qwen3_vl")) == "sdpa"


def test_an_unreadable_config_falls_back_to_what_is_known_to_load(tmp_path):
    assert _nf4_attention(str(tmp_path)) == "eager"
