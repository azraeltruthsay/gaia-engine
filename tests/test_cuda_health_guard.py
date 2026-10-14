"""Tests: _require_cuda_healthy / cuda_health_snapshot / GPUUnavailableError
(GAIA_Project-448j, GAIA_Project-t0f2).

The bug: PrefixCache.get_kv() (and generate()/generate_stream()) did
ids.to(self.device) etc. unconditionally. When a host reboot or driver
reset leaves torch.cuda.is_available() == False while the engine still
thinks device == "cuda" (GAIA_Project-t0f2), that call throws a cryptic,
many-frames-deep torch RuntimeError that takes down every inference call
with no clear signal of what actually went wrong. The fix fails fast with
a dedicated, catchable exception instead (448j), and separately reports
the same condition non-fatally for health checks / gaia-doctor (t0f2),
via cuda_health_snapshot().

t0f2's original incident report is explicit that torch.cuda.is_available()
itself RAISED ("CUDA unknown error... Setting the available devices to be
zero.") rather than cleanly returning False -- the original 448j tests only
covered the return-False case, not the raises case. That gap is covered
here.
"""

import pytest
import torch

from gaia_engine.core import GPUUnavailableError, _require_cuda_healthy, cuda_health_snapshot


def test_passes_when_device_is_cpu(monkeypatch):
    # CPU-backed engines must never trip this check, regardless of CUDA state.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _require_cuda_healthy("cpu")  # no raise


def test_passes_when_cuda_and_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _require_cuda_healthy("cuda")  # no raise


def test_raises_when_cuda_expected_but_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(GPUUnavailableError):
        _require_cuda_healthy("cuda")


def test_raises_when_is_available_itself_raises(monkeypatch):
    # t0f2's actual repro: torch.cuda.is_available() raised, didn't just
    # return False. Must still surface as GPUUnavailableError, not the raw
    # torch exception.
    def _boom():
        raise RuntimeError("CUDA unknown error... Setting the available devices to be zero.")
    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    with pytest.raises(GPUUnavailableError):
        _require_cuda_healthy("cuda")


def test_gpu_unavailable_error_is_a_runtime_error():
    # Callers that already catch generic RuntimeError (if any) keep working;
    # callers can also catch this one specifically to distinguish it.
    assert issubclass(GPUUnavailableError, RuntimeError)


class TestCudaHealthSnapshot:
    """Non-raising counterpart, used by health-check/observability callers."""

    def test_cpu_device_always_healthy(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        healthy, detail = cuda_health_snapshot("cpu")
        assert healthy is True

    def test_none_device_always_healthy(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        healthy, detail = cuda_health_snapshot("none")
        assert healthy is True

    def test_cuda_available_is_healthy(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        healthy, detail = cuda_health_snapshot("cuda")
        assert healthy is True
        assert "available" in detail

    def test_cuda_unavailable_is_unhealthy_with_detail(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        healthy, detail = cuda_health_snapshot("cuda")
        assert healthy is False
        assert "is_available() is False" in detail

    def test_cuda_raising_is_unhealthy_with_detail(self, monkeypatch):
        def _boom():
            raise RuntimeError("CUDA unknown error")
        monkeypatch.setattr(torch.cuda, "is_available", _boom)
        healthy, detail = cuda_health_snapshot("cuda")
        assert healthy is False
        assert "RuntimeError" in detail
        assert "CUDA unknown error" in detail

    def test_never_raises_regardless_of_device(self, monkeypatch):
        def _boom():
            raise RuntimeError("boom")
        monkeypatch.setattr(torch.cuda, "is_available", _boom)
        # Must not raise for any device value -- that's the whole point of
        # the non-raising snapshot variant.
        cuda_health_snapshot("cuda")
        cuda_health_snapshot("cpu")
        cuda_health_snapshot("")
