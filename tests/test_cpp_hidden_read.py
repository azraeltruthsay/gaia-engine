"""d3px: /hidden/read gives the last-token hidden state of chosen layers for
a chat prompt, generating nothing, and puts the capture layers back."""
import threading
import types

import pytest

backend = pytest.importorskip("gaia_engine.cpp.backend")


class FakeBackend:
    def __init__(self):
        self.layers = [0, 4, 8]
        self.calls = []

    def n_layer(self):
        return 12

    def n_embd(self):
        return 3

    def set_capture_layers(self, layers):
        self.layers = list(layers)

    def generate(self, prompt, max_tokens, temperature, top_p, top_k, capture):
        self.calls.append((prompt, max_tokens, capture, list(self.layers)))
        return types.SimpleNamespace(hidden_states={x: [float(x), 1.0, 2.0] for x in self.layers},
                                     prompt_tokens=5)


def _adapter():
    a = object.__new__(backend.GaiaCppBackendAdapter)
    a._backend = FakeBackend()
    a._request_lock = threading.Lock()
    a.capture_layers = [0, 4, 8]
    a._build_prompt = lambda messages, enable_thinking=True, media=None: "PROMPT:" + messages[-1]["content"]
    return a


def test_reads_the_asked_layers_without_generating_and_restores_capture():
    a = _adapter()
    got = a.read_hidden([{"role": "user", "content": "hi"}], layers=[6])
    assert got["layers"] == {6: [6.0, 1.0, 2.0]} and got["prompt_tokens"] == 5 and got["n_embd"] == 3
    prompt, max_tokens, capture, layers = a._backend.calls[0]
    assert prompt == "PROMPT:hi" and max_tokens == 0 and capture and layers == [6]
    assert a._backend.layers == [0, 4, 8]


def test_default_layers_and_bad_layers():
    a = _adapter()
    assert sorted(a.read_hidden([{"role": "user", "content": "hi"}])["layers"]) == [0, 4, 8]
    with pytest.raises(ValueError):
        a.read_hidden([{"role": "user", "content": "hi"}], layers=[99])


def test_the_native_reader_is_used_without_the_generation_lock():
    a = _adapter()
    seen = []

    def read_hidden(prompt, layers):
        seen.append((prompt, list(layers), a._request_lock.locked()))
        return {"layers": {x: [1.0, 2.0, 3.0] for x in layers}, "prompt_tokens": 4}
    a._backend.read_hidden = read_hidden
    with a._request_lock:     # a generation in progress
        got = a.read_hidden([{"role": "user", "content": "hi"}], layers=[4])
    assert got["reader"] and got["layers"] == {4: [1.0, 2.0, 3.0]} and seen[0][:2] == ("PROMPT:hi", [4])
    assert not a._backend.calls   # the main context wasn't used
