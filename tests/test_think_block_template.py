"""Thinking off writes an empty <think></think> only for a model whose own
template knows think blocks (2026-11-23). Prime v2b (Qwen3-VL Instruct) has
none; given one, it answered "Assistant\\nWashington." or a URL fragment."""
import struct
from types import SimpleNamespace

from gaia_engine.core import ChatFormatter
from gaia_engine.cpp.backend import GaiaCppBackendAdapter
from gaia_engine.prompt_format import gguf_string, template_thinks

INSTRUCT = "{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>{% endfor %}"
HYBRID = INSTRUCT + "{%- if enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}"
MSGS = [{"role": "user", "content": "hi"}]


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf(path, template=None):
    kv = [_s("general.architecture") + struct.pack("<I", 8) + _s("qwen3vl"),
          _s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 3) + _s("a") + _s("<think>") + _s("c"),
          _s("tokenizer.ggml.scores") + struct.pack("<IIQ", 9, 6, 3) + struct.pack("<fff", 0, 0, 0),
          _s("general.alignment") + struct.pack("<II", 4, 32)]
    if template is not None:
        kv.append(_s("tokenizer.chat_template") + struct.pack("<I", 8) + _s(template))
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
    return str(path)


def test_template_thinks():
    assert not template_thinks(INSTRUCT)
    assert template_thinks(HYBRID)
    assert template_thinks(None) and template_thinks("")   # unknown: as before


def test_gguf_string_reads_past_arrays(tmp_path):
    assert gguf_string(_gguf(tmp_path / "m.gguf", INSTRUCT), "tokenizer.chat_template") == INSTRUCT
    assert gguf_string(_gguf(tmp_path / "n.gguf"), "tokenizer.chat_template") is None
    assert gguf_string(str(tmp_path / "missing.gguf"), "tokenizer.chat_template") is None


def _fmt(template):
    tok = SimpleNamespace(vocab={"<|im_start|>": 1}, chat_template=template)
    return ChatFormatter(tok)


def test_formatter_writes_no_think_block_for_an_instruct_template():
    assert _fmt(INSTRUCT).assistant_prefix(enable_thinking=False) == "<|im_start|>assistant\n"
    assert _fmt(HYBRID).assistant_prefix(enable_thinking=False).endswith("<think>\n\n</think>\n\n")
    assert _fmt(INSTRUCT).assistant_prefix(enable_thinking=True, native_thinking=True).endswith("<think>\n")


def _adapter(path):
    a = GaiaCppBackendAdapter.__new__(GaiaCppBackendAdapter)
    a.model_path = path
    a._backend = SimpleNamespace(architecture=lambda: "qwen3vl")
    return a


def test_cpp_prompt_follows_the_gguf_template(tmp_path):
    p = _adapter(_gguf(tmp_path / "i.gguf", INSTRUCT))._build_prompt(MSGS, enable_thinking=False)
    assert p.endswith("<|im_start|>assistant\n") and "<think>" not in p
    p = _adapter(_gguf(tmp_path / "h.gguf", HYBRID))._build_prompt(MSGS, enable_thinking=False)
    assert p.endswith("<think>\n\n</think>\n\n")
