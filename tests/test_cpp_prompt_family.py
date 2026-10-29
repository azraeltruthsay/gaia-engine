"""gaia_cpp prompts in the model's own turn format (found during isep).

The family check read an attribute nothing set, so every model — Gemma 4
included — was prompted in ChatML. The format now follows the GGUF's
general.architecture, with the resolved file name only as a fallback.
"""

from types import SimpleNamespace

from gaia_engine.cpp.backend import GaiaCppBackendAdapter

MSGS = [{"role": "system", "content": "You are GAIA."}, {"role": "user", "content": "hi"}]


def _adapter(arch=None, path="/models/core.gguf"):
    a = GaiaCppBackendAdapter.__new__(GaiaCppBackendAdapter)
    a.model_path = path
    backend = SimpleNamespace()
    if arch is not None:
        backend.architecture = lambda: arch
    a._backend = backend
    return a


def test_gemma_architecture_gets_gemma_turns_despite_symlink_name():
    p = _adapter("gemma4")._build_prompt(MSGS)
    assert p.startswith("<|turn>system<turn|>\nYou are GAIA.") and "<|im_start|>" not in p


def test_qwen_architecture_gets_chatml():
    p = _adapter("qwen3vl", "/models/prime.gguf")._build_prompt(MSGS)
    assert p.startswith("<|im_start|>system\nYou are GAIA.")


def test_older_extension_falls_back_to_resolved_file_name(tmp_path):
    real = tmp_path / "Gemma4-E4B-GAIA-Core-IDENTITY_V3-Q8_0.gguf"
    real.write_bytes(b"")
    link = tmp_path / "core.gguf"
    link.symlink_to(real)
    assert _adapter(None, str(link))._family() == "gemma"
