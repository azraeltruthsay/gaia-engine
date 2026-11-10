"""frlp: gaia_cpp's image/audio path in the adapter — content parts become
prompt text plus media markers and bytes, media requests go to
generate_media, and the repetition penalty is worked out as the torch worker
does and handed to the extension (older extensions: without it)."""

import base64
import threading
import types

import pytest

backend = pytest.importorskip("gaia_engine.cpp.backend")
manager = pytest.importorskip("gaia_engine.manager")

PNG = b"\x89PNG\r\n\x1a\nfake"
WAV = b"RIFF....WAVEfake"


def _adapter(marker="<M>"):
    a = object.__new__(backend.GaiaCppBackendAdapter)
    a._media_marker = marker
    a.prompt_format = "house"
    a._family = lambda: "gemma"
    return a


def _uri(mime, data):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def test_parts_become_text_markers_and_bytes():
    a = _adapter()
    media = []
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": _uri("image/png", PNG)}},
        {"type": "text", "text": " And this?"},
        {"type": "audio_url", "audio_url": {"url": _uri("audio/wav", WAV)}},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(WAV).decode(), "format": "wav"}},
    ]}]
    prompt = a._build_prompt(msgs, media=media)
    assert "<|turn>user<turn|>\nWhat is this?<M> And this?<M><M>" in prompt
    assert media == [PNG, WAV, WAV]


def test_without_projector_media_is_dropped_not_repr():
    a = _adapter(marker="")
    media = []
    prompt = a._build_prompt([{"role": "user", "content": [
        {"type": "text", "text": "Hi"},
        {"type": "image_url", "image_url": {"url": _uri("image/png", PNG)}}]}], media=media)
    assert media == [] and "Hi" in prompt and "image_url" not in prompt and "[{" not in prompt


def test_text_only_callers_get_text_parts_joined():
    a = _adapter()
    prompt = a._build_prompt([{"role": "user", "content": [{"type": "text", "text": "a"},
                                                           {"type": "text", "text": "b"}]}])
    assert "<|turn>user<turn|>\nab" in prompt


def test_media_bytes_from_path_and_bad_url(tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(PNG)
    assert backend._media_bytes({"type": "image_url", "image_url": str(f)}) == PNG
    assert backend._media_bytes({"type": "text", "text": "x"}) is None
    with pytest.raises(ValueError):
        backend._media_bytes({"type": "image_url", "image_url": {"url": "http://example.com/a.png"}})


def test_repetition_penalty_matches_torch_worker():
    rp = manager._repetition_penalty
    assert rp({}) == 1.1
    assert rp({"repetition_penalty": 1.3}) == 1.3
    assert rp({"frequency_penalty": 0.5}) == pytest.approx(1.05)
    assert rp({"frequency_penalty": 0}) == 1.1


def test_media_request_goes_to_generate_media_with_penalty():
    calls = {}

    class Ext:
        def generate_media(self, *args):
            calls["media"] = args
            return types.SimpleNamespace(text="A dog.", prompt_tokens=300, completion_tokens=3,
                                         reused_tokens=0, evaluated_tokens=300, snapshot_restored=False,
                                         draft_tokens=0, draft_accepted=0, hidden_states={})

    a = _adapter()
    a._backend = Ext()
    a._request_lock = threading.Lock()
    a._kv_control = False
    a.capture_layers = []
    out = a.generate_json([{"role": "user", "content": [
        {"type": "text", "text": "What?"},
        {"type": "image_url", "image_url": {"url": _uri("image/png", PNG)}}]}],
        max_tokens=10, temperature=0.0, repetition_penalty=1.1)
    args = calls["media"]
    assert args[1] == [PNG] and args[-1] == 1.1
    assert out["choices"][0]["message"]["content"] == "A dog."


def test_older_extension_called_without_penalty():
    seen = []

    def old_generate(*args):
        if len(args) > 6:
            raise TypeError("generate(): incompatible function arguments")
        seen.append(args)
        return "ok"

    assert backend.GaiaCppBackendAdapter._call(old_generate, 1.1, "p", 10, 0.7, 0.9, 0, False) == "ok"
    assert len(seen) == 1
