"""Tests: EngineManager.health_response()'s CUDA augmentation (GAIA_Project-t0f2).

The bug: a stale CUDA driver context (host reboot, driver reset) left every
health check reporting "healthy" right up until a real inference request
crashed -- /health never exercised CUDA. health_response() now probes
torch.cuda.is_available() (via gaia_engine.core.cuda_health_snapshot) when
the manager's tracked device is "cuda", and downgrades status to "degraded"
if it's dead. gaia-doctor's existing check_health() already treats any
status=="degraded" body as unhealthy, so this plugs into the existing
restart pipeline with no gaia-doctor changes needed.
"""

import torch

from gaia_engine.manager import EngineManager


def _manager() -> EngineManager:
    return EngineManager(port=9999)


class TestHealthResponseCudaAugmentation:
    def test_cpu_device_untouched_regardless_of_cuda_state(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        m = _manager()
        m.device = "cpu"
        data = m.health_response()
        assert data["status"] == "ok"
        assert "cuda_healthy" not in data

    def test_no_model_loaded_device_none_untouched(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        m = _manager()
        data = m.health_response()
        assert data["device"] == "none"
        assert data["status"] == "ok"
        assert "cuda_healthy" not in data

    def test_cuda_device_and_cuda_available_stays_healthy(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        m = _manager()
        m.device = "cuda"
        data = m.health_response()
        assert data["status"] == "ok"
        assert data["cuda_healthy"] is True

    def test_cuda_device_but_cuda_unavailable_degrades_status(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        m = _manager()
        m.device = "cuda"
        data = m.health_response()
        assert data["status"] == "degraded"
        assert data["cuda_healthy"] is False
        assert "is_available() is False" in data["cuda_detail"]

    def test_cuda_device_but_is_available_raises_degrades_status(self, monkeypatch):
        # t0f2's actual repro: torch.cuda.is_available() raised, not just
        # returned False.
        def _boom():
            raise RuntimeError("CUDA unknown error... Setting the available devices to be zero.")
        monkeypatch.setattr(torch.cuda, "is_available", _boom)
        m = _manager()
        m.device = "cuda"
        data = m.health_response()
        assert data["status"] == "degraded"
        assert data["cuda_healthy"] is False
        assert "RuntimeError" in data["cuda_detail"]

    def test_cuda_check_failure_itself_fails_open(self, monkeypatch):
        # If cuda_health_snapshot itself breaks (import error, internal bug),
        # don't let that unrelated problem report every health check as
        # degraded -- only a confirmed-dead GPU should do that.
        import gaia_engine.core as core_mod

        def _raise(*a, **k):
            raise RuntimeError("simulated failure inside cuda_health_snapshot")
        monkeypatch.setattr(core_mod, "cuda_health_snapshot", _raise)

        m = _manager()
        m.device = "cuda"
        data = m.health_response()
        assert data["status"] == "ok"
        assert data["cuda_healthy"] is True
