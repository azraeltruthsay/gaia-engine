"""Prompt formats for Gemma 4 (2qus, 2026-11-06).

GAIA's own Core was trained on a house turn format: ``<|turn>role<turn|>\\n``
with an empty ``<|think|>`` block to suppress thinking. A stock Gemma 4 model
(gemma-4-12B-it) speaks Google's format instead: ``<|turn>role\\n...<turn|>``,
the assistant is ``model``, and ``<|think|>`` *switches thinking on*. Given the
house format, the stock 12B thought for 300 tokens and never answered.

``house`` stays the default. A stock model is loaded with
``prompt_format="stock"``. ``<bos>`` is left out: both tokenizers add it.
"""
from typing import Iterable

HOUSE = "house"
STOCK = "stock"


def normalize(prompt_format: str) -> str:
    return STOCK if str(prompt_format or "").lower() == STOCK else HOUSE


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def stock_gemma4(messages: Iterable[dict], enable_thinking: bool = False,
                 add_generation_prompt: bool = True) -> str:
    """The text gemma-4-12B-it's own chat template renders, without <bos>."""
    msgs = list(messages)
    out = []
    has_system = bool(msgs) and msgs[0].get("role") in ("system", "developer")
    if enable_thinking and not has_system:
        out.append("<|turn>system\n<|think|>\n<turn|>\n")
    for i, m in enumerate(msgs):
        role = {"assistant": "model", "developer": "system"}.get(m.get("role"), m.get("role", "user"))
        content = _text(m.get("content"))
        if i == 0 and has_system and enable_thinking:
            content = "<|think|>\n" + content
        out.append(f"<|turn>{role}\n{content}<turn|>\n")
    if add_generation_prompt:
        out.append("<|turn>model\n" + ("" if enable_thinking else "<|channel>thought\n<channel|>"))
    return "".join(out)


def template_thinks(template) -> bool:
    """Whether a ChatML model's own chat template knows think blocks.

    Thinking off writes an empty ``<think></think>`` after the assistant start:
    right for Qwen3's hybrid models, whose template does the same. Qwen3-VL
    *Instruct* (Prime v2b) never saw one, and given it answered
    "Assistant\\nWashington." instead of "Washington.", or a URL fragment, or
    nothing (2026-11-23). No template known: keep the block, as before.
    """
    text = str(template or "")
    return not text.strip() or "<think>" in text


def gguf_string(path: str, key: str):
    """A string value from a GGUF file's metadata (e.g.
    ``tokenizer.chat_template``), or None. Reads the header only."""
    import struct
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF" or struct.unpack("<I", f.read(4))[0] < 2:
                return None
            _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

            def text() -> bytes:
                return f.read(struct.unpack("<Q", f.read(8))[0])

            def skip(t: int) -> None:
                if t == 8:
                    f.seek(struct.unpack("<Q", f.read(8))[0], 1)
                elif t == 9:
                    et, n = struct.unpack("<IQ", f.read(12))
                    if et == 8:
                        for _ in range(n):
                            f.seek(struct.unpack("<Q", f.read(8))[0], 1)
                    elif et in sizes:
                        f.seek(sizes[et] * n, 1)
                    else:
                        raise ValueError(f"nested GGUF array type {et}")
                else:
                    f.seek(sizes[t], 1)

            for _ in range(n_kv):
                k = text().decode("utf-8", "replace")
                t = struct.unpack("<I", f.read(4))[0]
                if k == key and t == 8:
                    return text().decode("utf-8", "replace")
                skip(t)
    except (OSError, ValueError, KeyError, struct.error):
        return None
    return None
