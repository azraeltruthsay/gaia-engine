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
