"""Adapter selection survives a backend swap, and gaia_cpp's /adapter/set acts.

Clients cache "already loaded" and send only /adapter/set afterwards. When a
gear shift swapped Core's backend, the new one never got the adapter; and on
gaia_cpp /adapter/set answered ok without loading or clearing anything.
"""

import json

from gaia_engine.manager import EngineManager


class FakeCpp:
    def __init__(self):
        self._lora_path = None
        self.loads = []

    def load_adapter(self, path, scale=1.0):
        self.loads.append(path)
        self._lora_path = path
        return True

    def unload_adapter(self):
        self._lora_path = None

    def health(self):
        return {"lora_active": int(bool(self._lora_path))}


def _mgr(tmp_path):
    adir = tmp_path / "core_thinking_v1"
    adir.mkdir()
    (adir / "adapter.gguf").write_bytes(b"gguf")
    m = EngineManager.__new__(EngineManager)
    m._adapter_paths = {}
    return m, str(adir)


def _post(m, cpp, path, body):
    status, _, out = m._proxy_cpp("POST", path, json.dumps(body).encode(), cpp)
    return status, json.loads(out)


def test_set_loads_a_remembered_adapter_and_none_clears_it(tmp_path):
    m, adir = _mgr(tmp_path)
    m._remember_adapter(json.dumps({"name": "core_thinking_v1", "path": adir}).encode())
    cpp = FakeCpp()
    status, out = _post(m, cpp, "/adapter/set", {"name": "core_thinking_v1"})
    assert status == 200 and out["active"] == "core_thinking_v1"
    assert cpp._lora_path.endswith("adapter.gguf")
    _post(m, cpp, "/adapter/set", {"name": "core_thinking_v1"})
    assert len(cpp.loads) == 1          # already active: no reload
    status, out = _post(m, cpp, "/adapter/set", {"name": None})
    assert status == 200 and cpp._lora_path is None


def test_set_unknown_adapter_is_an_error_not_a_silent_ok(tmp_path):
    m, _ = _mgr(tmp_path)
    status, out = _post(m, FakeCpp(), "/adapter/set", {"name": "never_loaded"})
    assert status == 404 and out["ok"] is False


def test_load_replaces_instead_of_stacking(tmp_path):
    m, adir = _mgr(tmp_path)
    other = tmp_path / "other.gguf"
    other.write_bytes(b"gguf")
    cpp = FakeCpp()
    _post(m, cpp, "/adapter/load", {"name": "core_thinking_v1", "path": adir})
    _post(m, cpp, "/adapter/load", {"adapter_path": str(other)})
    assert cpp._lora_path == str(other) and len(cpp.loads) == 2


def test_worker_set_loads_adapter_it_lacks(tmp_path):
    m, adir = _mgr(tmp_path)
    m._adapter_paths["core_thinking_v1"] = adir
    calls = []
    answers = iter([{"ok": False, "error": "adapter 'core_thinking_v1' not loaded"}, {"ok": True}, {"ok": True}])

    def fake_proxy(method, path, headers, body=b"", direct=False):
        calls.append((path, direct))
        return 200, {}, json.dumps(next(answers)).encode()
    m.proxy_to_worker = fake_proxy
    status, _, out = m._set_adapter_on_worker({}, json.dumps({"name": "core_thinking_v1"}).encode())
    assert [c[0] for c in calls] == ["/adapter/set", "/adapter/load", "/adapter/set"]
    assert all(direct for _, direct in calls) and json.loads(out)["ok"]


def test_cpp_health_identifies_the_gaia_engine(tmp_path):
    m, _ = _mgr(tmp_path)
    m._draining, m._active_inference_count = False, 0
    cpp = FakeCpp()
    cpp.health = lambda: {"has_gpu": False}
    status, _, out = m._proxy_cpp("GET", "/health", b"", cpp)
    assert json.loads(out)["engine"] == "gaia"


def test_manager_health_identifies_the_gaia_engine_on_cpp(tmp_path):
    import threading
    m, _ = _mgr(tmp_path)
    m._lock, m._draining, m._active_inference_count = threading.Lock(), False, 0
    m.backend, m.model_path, m.device = "cpp", "/models/core.gguf", "cpu"
    cpp = FakeCpp()
    cpp.health = lambda: {"has_gpu": False}
    m._cpp_backend = cpp
    assert m._build_health_response()["engine"].startswith("gaia")
