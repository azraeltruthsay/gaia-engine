"""ojwf: the gaia_cpp adapter's b11804 additions — separate prompt-read
threads, MTP speculative decoding enabled from the environment, MTP stats in
/health — and clean fallbacks on an extension built before them."""

import threading
import types

import pytest

backend = pytest.importorskip("gaia_engine.cpp.backend")


class FakeExt:
    """Stands in for gaia_cpp.LlamaCppBackend (b11804 build)."""

    instances = []

    def __init__(self, model_path, n_gpu_layers, capture_layers, n_ctx, n_threads, n_threads_batch=None):
        self.args = (model_path, n_gpu_layers, n_ctx, n_threads, n_threads_batch)
        self.mtp = None
        FakeExt.instances.append(self)

    def n_layer(self): return 48
    def n_embd(self): return 3840
    def n_vocab(self): return 262144
    def has_gpu(self): return False
    def active_adapter_count(self): return 0
    def set_capture_layers(self, layers): self.capture = layers
    def set_prefix_reuse(self, on): pass
    def set_snapshot_slots(self, n): pass
    def kv_stats(self): return {"snapshot_slots": 2}

    def enable_mtp(self, path, n_max):
        self.mtp = (path, n_max)
        return ""

    def mtp_stats(self):
        return {"enabled": self.mtp is not None, "n_max": self.mtp[1] if self.mtp else 0}


class OldExt:
    """An extension from before b11804: five constructor args, no MTP API."""

    def __init__(self, model_path, n_gpu_layers, capture_layers, n_ctx, n_threads):
        self.args = (model_path, n_gpu_layers, n_ctx, n_threads, None)

    def n_layer(self): return 48
    def n_embd(self): return 3840
    def set_capture_layers(self, layers): self.capture = layers
    def set_prefix_reuse(self, on): pass
    def set_snapshot_slots(self, n): pass
    def kv_stats(self): return {"snapshot_slots": 2}


@pytest.fixture
def ext(monkeypatch):
    FakeExt.instances.clear()
    monkeypatch.setattr(backend, "_CPP_AVAILABLE", True)
    monkeypatch.delenv("GAIA_CPP_MTP_DRAFT", raising=False)
    monkeypatch.delenv("GAIA_CPP_MTP_N", raising=False)
    return monkeypatch


def _make(**kw):
    return backend.GaiaCppBackendAdapter("/models/core.gguf", n_threads=12, **kw)


def test_prompt_read_threads_reach_the_extension(ext):
    ext.setattr(backend, "LlamaCppBackend", FakeExt)
    a = _make(n_threads_batch=24)
    assert a._backend.args[3:] == (12, 24)


def test_older_extension_takes_five_args(ext):
    ext.setattr(backend, "LlamaCppBackend", OldExt)
    a = _make(n_threads_batch=24)
    assert a._backend.args[3] == 12 and a._backend.args[4] is None
    assert a.mtp_stats() == {}


def test_mtp_enabled_from_env(ext, tmp_path):
    draft = tmp_path / "assistant.gguf"
    draft.write_bytes(b"gguf")
    ext.setattr(backend, "LlamaCppBackend", FakeExt)
    ext.setenv("GAIA_CPP_MTP_DRAFT", str(draft))
    ext.setenv("GAIA_CPP_MTP_N", "4")
    a = _make()
    assert a._backend.mtp == (str(draft), 4)
    assert a.health()["mtp"]["enabled"] is True


def test_mtp_off_without_env_or_with_missing_file(ext, tmp_path):
    ext.setattr(backend, "LlamaCppBackend", FakeExt)
    assert _make()._backend.mtp is None
    ext.setenv("GAIA_CPP_MTP_DRAFT", str(tmp_path / "missing.gguf"))
    assert _make()._backend.mtp is None


def test_mtp_env_on_old_extension_is_ignored(ext, tmp_path):
    draft = tmp_path / "assistant.gguf"
    draft.write_bytes(b"gguf")
    ext.setattr(backend, "LlamaCppBackend", OldExt)
    ext.setenv("GAIA_CPP_MTP_DRAFT", str(draft))
    a = _make()  # must not raise
    assert a.mtp_stats() == {}


def test_mtp_stats_never_wait_on_a_running_request():
    a = object.__new__(backend.GaiaCppBackendAdapter)
    a._backend = types.SimpleNamespace(mtp_stats=lambda: {"enabled": True, "accepted_total": 7})
    a._request_lock = threading.Lock()
    a._mtp_last = {}
    assert a.mtp_stats()["accepted_total"] == 7
    a._backend.mtp_stats = lambda: pytest.fail("mtp_stats called during a request")
    with a._request_lock:
        got = a.mtp_stats()
    assert got["accepted_total"] == 7 and got["busy"] is True
