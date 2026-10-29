"""ecnc: `stop` strings. gaia-core stops Core at "</tool_call>": what the
model writes after its call (a made-up <tool_result>, the call again) is
waste and imitation. The engine ignored `stop` entirely."""

import threading

import torch

from gaia_engine.core import GAIAEngine, _cut_at_stop, _normalize_stop

SCRIPT = 'Sure. <tool_call>{"tool": "web"}</tool_call><tool_result>fake result</tool_result> more'
EOS = len(SCRIPT)


class _Out:
    def __init__(self, logits):
        self.logits = logits
        self.past_key_values = None


class ScriptedModel:
    """Greedy-decodes SCRIPT one character per token, then EOS."""
    device = "cpu"

    def __init__(self):
        self.step = 0
        self.calls = 0

    def __call__(self, ids, **_):
        self.calls += 1
        nxt = self.step if self.step < len(SCRIPT) else EOS
        self.step += 1
        logits = torch.zeros(1, 1, EOS + 1)
        logits[0, -1, nxt] = 1.0
        return _Out(logits)


class CharTokenizer:
    def encode(self, text, return_tensors=None, add_special_tokens=True):
        return torch.zeros(1, 3, dtype=torch.long) if return_tensors else []

    def decode(self, ids, skip_special_tokens=True):
        return "".join(SCRIPT[i] for i in ids if i < len(SCRIPT))


class Formatter:
    stop_token_ids = {EOS}
    eos_token_id = EOS
    think_token = "<think>"

    def format_conversation(self, msgs, enable_thinking=True, native_thinking=False):
        return "prompt"


class _Monitor:
    enabled = False


def _engine():
    eng = object.__new__(GAIAEngine)
    eng._lock = threading.Lock()
    eng.device = "cpu"
    eng.model = ScriptedModel()
    eng.tokenizer = CharTokenizer()
    eng.formatter = Formatter()
    eng.monitor = _Monitor()
    eng.has_vision = False
    eng._request_count = 0
    eng._total_tokens = 0
    return eng


def _stream(**kw):
    eng = _engine()
    chunks = list(eng.generate_stream([{"role": "user", "content": "hi"}], max_tokens=200,
                                      temperature=0, **kw))
    text = "".join(c["choices"][0].get("delta", {}).get("content", "") for c in chunks)
    return text, chunks[-1], eng.model.calls


def test_helpers():
    assert _normalize_stop("x") == ["x"] and _normalize_stop(["", "a", None]) == ["a"]
    assert _normalize_stop(None) == []
    assert _cut_at_stop("ab</t>cd</t>", ["</t>"], False) == ("ab", True)
    assert _cut_at_stop("ab</t>cd", ["</t>"], True) == ("ab</t>", True)
    assert _cut_at_stop("abXcdY", ["Y", "X"], False) == ("ab", True)  # earliest wins
    assert _cut_at_stop("abc", ["Z"], False) == ("abc", False)


def test_stream_without_stop_runs_to_the_end():
    text, last, _ = _stream()
    assert text == SCRIPT and last["choices"][0]["finish_reason"] == "stop"


def test_stream_stops_at_the_tool_call_and_keeps_it_when_asked():
    text, last, calls = _stream(stop=["</tool_call>"], include_stop_str_in_output=True)
    assert text == 'Sure. <tool_call>{"tool": "web"}</tool_call>'
    assert last["choices"][0]["message"]["content"] == text
    assert calls < len(SCRIPT)  # generation actually stopped early


def test_stream_excluded_stop_string_never_reaches_the_client():
    text, _, _ = _stream(stop="</tool_call>")
    assert text == 'Sure. <tool_call>{"tool": "web"}'
    assert "</tool" not in text


def test_stream_holdback_is_flushed_when_no_stop_matches():
    text, _, _ = _stream(stop=["NEVER-APPEARS"])
    assert text == SCRIPT


def test_manager_cut_for_the_cpp_backend():
    from gaia_engine.manager import _apply_stop
    text = 'Sure. <tool_call>{"tool": "web"}</tool_call><tool_result>fake'
    assert _apply_stop(text, ["</tool_call>"], True) == 'Sure. <tool_call>{"tool": "web"}</tool_call>'
    assert _apply_stop(text, "</tool_call>", False) == 'Sure. <tool_call>{"tool": "web"}'
    assert _apply_stop(text, ["NOPE"], True) == text
