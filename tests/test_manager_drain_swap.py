"""sy0s: a direct /model/swap or /model/unload drains in-flight inference
before killing the worker, and a fresh worker never inherits a stale drain."""

import threading
import time

from gaia_engine import manager as M
from gaia_engine.manager import EngineManager


def _mgr():
    m = EngineManager.__new__(EngineManager)
    m._lock = threading.Lock()
    m._draining = False
    m._drain_event = threading.Event()
    m._drain_event.set()
    m._active_inference_count = 0
    m.model_path = "/models/old"
    m.events = []
    m.stop_worker = lambda: m.events.append(("stop", m._active_inference_count)) or {"ok": True}
    m.start_worker = lambda *a, **k: m.events.append(("start", m._draining)) or {"ok": True}
    return m


def _finish_after(m, delay):
    def run():
        time.sleep(delay)
        with m._lock:
            m._active_inference_count = 0
        m._drain_event.set()  # what the proxy does when the last request ends
    threading.Thread(target=run, daemon=True).start()


def test_swap_waits_for_the_live_generation():
    m = _mgr()
    m._active_inference_count = 1
    _finish_after(m, 0.2)
    m.swap_worker("/models/new")
    assert m.events[0] == ("stop", 0)  # stopped only once nothing was in flight
    assert m._draining is False        # resumed afterwards


def test_swap_inside_an_orchestrator_drain_leaves_resuming_to_it():
    m = _mgr()
    m.drain(1)                          # the orchestrator's drain
    m.swap_worker("/models/new")
    assert m._draining is True and m.events == [("stop", 0), ("start", True)]


def test_unload_drains_too(monkeypatch):
    monkeypatch.setattr(M, "DRAIN_TIMEOUT_S", 0.1)
    m = _mgr()
    m._active_inference_count = 1      # never finishes: the timeout bounds the wait
    t = time.time()
    assert m.stop_worker_drained()["ok"]
    assert 0.08 < time.time() - t < 2 and m._draining is False


class _Stop(Exception):
    pass


def test_start_worker_clears_a_stale_drain(monkeypatch):
    m = EngineManager.__new__(EngineManager)
    m._lock, m._drain_event = threading.Lock(), threading.Event()
    m._draining, m.worker_process = True, None

    def stop_here():
        raise _Stop  # the flag reset comes before the spawn; stop there
    monkeypatch.setattr(M, "_find_free_port", stop_here)
    try:
        m.start_worker("/models/x", "cpu")
    except _Stop:
        pass
    assert m._draining is False and m._drain_event.is_set()
