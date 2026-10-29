"""Separate a model's native thought from its answer, in rendered text.

Two native formats reach the engine:
  - Gemma 4's thought channel: <|channel>thought ...<channel|> (special tokens;
    gaia_cpp renders them, the torch path splits them by token id instead —
    core._split_thought_tokens);
  - Qwen3's <think>...</think> (added but non-special tokens, so they survive
    decoding as text on every backend).

Both come back to callers as message.reasoning_content, apart from the answer
(h1ok, isep). A thought cut off before its close marker is all thought and
leaves no answer — it is never voiced as one. Torch-free so the gaia_cpp
backend can use it.
"""
import re
from typing import Optional, Tuple

_CHANNEL_RE = re.compile(r"<\|channel>(?:thought)?\s*(.*?)(?:<channel\|>|\Z)", re.DOTALL)
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def split_channel_text(text: str) -> Tuple[Optional[str], str]:
    """Gemma 4 thought channel: (thought, answer); thought None if absent."""
    m = _CHANNEL_RE.search(text or "")
    if not m:
        return None, text
    return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()


def split_think_text(text: str) -> Tuple[Optional[str], str]:
    """Qwen3 <think> block: (thought, answer); thought None if absent.

    Handles a closed block, a block cut off before </think> (all thought), and
    a lone </think> (the opening <think> was part of the prompt prefill).
    """
    text = text or ""
    start = text.find(_THINK_OPEN)
    end = text.find(_THINK_CLOSE)
    if start < 0 and end < 0:
        return None, text
    if start < 0:                       # opened by the prefill
        return text[:end].strip(), text[end + len(_THINK_CLOSE):].strip()
    body_start = start + len(_THINK_OPEN)
    end = text.find(_THINK_CLOSE, body_start)
    if end < 0:                         # cut off inside the thought
        return text[body_start:].strip(), text[:start].strip()
    answer = (text[:start] + text[end + len(_THINK_CLOSE):]).strip()
    return text[body_start:end].strip(), answer


def split_thought_text(text: str) -> Tuple[Optional[str], str]:
    """Either native format, whichever the text carries."""
    thought, answer = split_channel_text(text)
    if thought is None:
        thought, answer = split_think_text(text)
    return thought, answer
