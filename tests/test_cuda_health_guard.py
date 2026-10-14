"""Tests: _require_cuda_healthy / GPUUnavailableError (GAIA_Project-448j).

The bug: PrefixCache.get_kv() (and generate()/generate_stream()) did
ids.to(self.device) etc. unconditionally. When a host reboot or driver
reset leaves torch.cuda.is_available() == False while the engine still
thinks device == "cuda" (GAIA_Project-t0f2), that call throws a cryptic,
many-frames-deep torch RuntimeError that takes down every inference call
with no clear signal of what actually went wrong. The fix fails fast with
a dedicated, catchable exception instead.
"""

import pytest
import torch

from gaia_engine.core import GPUUnavailableError, _require_cuda_healthy


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


def test_gpu_unavailable_error_is_a_runtime_error():
    # Callers that already catch generic RuntimeError (if any) keep working;
    # callers can also catch this one specifically to distinguish it.
    assert issubclass(GPUUnavailableError, RuntimeError)
