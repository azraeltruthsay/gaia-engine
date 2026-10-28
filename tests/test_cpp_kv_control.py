"""d12t: the gaia_cpp adapter's KV controls — prefix reuse configured from
the environment, stats in /health, save/load wrappers — and a clean
fallback when the compiled extension predates them."""

import types

import pytest

backend = pytest.importorskip("gaia_engine.cpp.backend")


class FakeKvBackend:
    def __init__(self):
        self.calls = []
        self.stats = {"prefix_reuse": True, "cached_tokens": 0, "snapshot_slots": 2}

    def kv_stats(self):
        return dict(self.stats)

    def clear_kv(self):
        self.calls.append(("clear",))

    def save_state(self, path):
        self.calls.append(("save", path))
        return 42

    def load_state(self, path):
        self.calls.append(("load", path))
        return 42


def _adapter(fake):
    import threading
    a = object.__new__(backend.GaiaCppBackendAdapter)
    a._backend = fake
    a._request_lock = threading.Lock()
    a._kv_control = hasattr(fake, "kv_stats")
    return a


def test_kv_wrappers_call_through(tmp_path):
    fake = FakeKvBackend()
    a = _adapter(fake)
    assert a.kv_stats()["snapshot_slots"] == 2
    assert a.clear_kv() is True
    p = tmp_path / "states" / "b.kvstate"
    assert a.save_kv(str(p)) == 42 and p.parent.is_dir()
    assert a.load_kv(str(p)) == 0          # file doesn't exist: no load attempted
    p.write_bytes(b"x")
    assert a.load_kv(str(p)) == 42
    assert ("clear",) in fake.calls and ("load", str(p)) in fake.calls


def test_older_extension_without_kv_api_degrades_quietly(tmp_path):
    a = _adapter(types.SimpleNamespace())  # no kv_* methods
    assert a.kv_stats() == {} and a.clear_kv() is False
    assert a.save_kv(str(tmp_path / "x.kvstate")) == 0 and a.load_kv(str(tmp_path / "x.kvstate")) == 0
