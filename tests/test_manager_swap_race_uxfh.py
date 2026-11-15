"""uxfh (2026-11-10): while a swap's worker loaded, /health said standby; the
orchestrator reconciled with its own swap; the first swap's 180 s timeout then
stopped the CURRENT worker -- the one the orchestrator's swap had started.

- /health reports mode=loading with the target model during a load.
- A swap waits for the one in progress.
- A timed-out load stops only the worker it spawned.
"""
import threading
import time

from gaia_engine import manager as M
from gaia_engine.manager import EngineManager


class FakeProc:
    _next_pid = 1000

    def __init__(self, *a, **k):
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.killed = False
        self.stdout = self.stderr = None

    def poll(self):
        return 0 if self.killed else None

    def terminate(self):
        self.killed = True

    kill = terminate

    def wait(self, timeout=None):
        return 0


def _mgr():
    m = EngineManager(port=0)
    m._stream_output = staticmethod(lambda *a: None)
    return m


def test_health_reports_loading_while_a_load_runs():
    m = _mgr()
    m._loading = {"model_path": "/models/e4b", "device": "cuda", "since": time.time() - 3}
    h = m.health_response()
    assert h["mode"] == "loading" and h["model_loaded"] is False
    assert h["loading_model"] == "/models/e4b" and h["loading_for_s"] >= 3


def test_health_answers_loading_even_while_state_lock_is_held():
    """An in-process gaia_cpp load holds _lock for the whole load."""
    m = _mgr()
    m._loading = {"model_path": "/models/core.gguf", "device": "cuda", "since": time.time()}
    with m._lock:
        out = {}
        t = threading.Thread(target=lambda: out.update(m.health_response()))
        t.start()
        t.join(1)
    assert out.get("mode") == "loading"


def test_start_worker_sets_and_clears_loading(monkeypatch):
    m = _mgr()
    seen = {}
    def fake_start(*a, **k):
        seen["mode"] = m.health_response()["mode"]
        return {"ok": True}
    monkeypatch.setattr(m, "_start_worker", fake_start)
    m.start_worker("/models/x", "cuda")
    assert seen["mode"] == "loading" and m._loading is None


def test_timed_out_load_leaves_a_newer_worker_alone(monkeypatch):
    m = _mgr()
    monkeypatch.setattr(M.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(M, "_find_free_port", lambda: 5555)
    newer = FakeProc()

    def slow_health(port, timeout):
        # While this load waits, something stops it and a newer worker takes over.
        with m._lock:
            m.worker_process.killed = True
            m.worker_process = newer
            m.worker_port = 6666
        return False

    monkeypatch.setattr(M, "_wait_for_health", slow_health)
    r = m._start_worker("/models/e4b", "cpu")
    assert r["ok"] is False and "superseded" in r["error"]
    assert m.worker_process is newer and not newer.killed and m.worker_port == 6666


def test_timed_out_load_cleans_up_its_own_worker(monkeypatch):
    m = _mgr()
    monkeypatch.setattr(M.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(M, "_find_free_port", lambda: 5555)
    monkeypatch.setattr(M, "_wait_for_health", lambda port, timeout: False)
    r = m._start_worker("/models/e4b", "cpu")
    assert r["ok"] is False and m.worker_process is None


def test_a_second_swap_waits_for_the_first(monkeypatch):
    m = _mgr()
    order = []
    monkeypatch.setattr(m, "stop_worker", lambda: {"ok": True})

    def fake_start(path, *a, **k):
        order.append(("begin", path))
        time.sleep(0.2)
        order.append(("end", path))
        return {"ok": True}

    monkeypatch.setattr(m, "start_worker", fake_start)
    t1 = threading.Thread(target=m.swap_worker, args=("/models/e4b",))
    t1.start()
    time.sleep(0.05)
    t2 = threading.Thread(target=m.swap_worker, args=("/models/core",))
    t2.start()
    t1.join(2)
    t2.join(2)
    assert order == [("begin", "/models/e4b"), ("end", "/models/e4b"),
                     ("begin", "/models/core"), ("end", "/models/core")]
