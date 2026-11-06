"""A GPU fault restarts the worker instead of 500s forever (pqpo, 2026-11-06).

Prime's worker loaded an adapter trained for another model; the load failed
half-way and left LoRA layers in the base model, the next generation sampled
from inf/nan, and a device-side assert wedged CUDA. torch.cuda.is_available()
stayed True, so health said "ok" while every request failed until someone
restarted the container — twice in one day.
"""
import pytest
import torch
from torch import nn

from gaia_engine import core
from gaia_engine.core import _strip_injected_lora, cuda_health_snapshot, note_cuda_failure
from gaia_engine.manager import EngineManager


@pytest.fixture(autouse=True)
def _clean_wedge(monkeypatch):
    monkeypatch.setattr(core, "_CUDA_WEDGED", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


class AcceleratorError(RuntimeError):
    pass


def test_gpu_faults_are_recognised_and_ordinary_errors_are_not():
    assert not note_cuda_failure(ValueError("prompt too long"))
    assert cuda_health_snapshot("cuda")[0]
    assert note_cuda_failure(AcceleratorError("CUDA error: device-side assert triggered"))
    ok, detail = cuda_health_snapshot("cuda")
    assert not ok and "wedged" in detail and "device-side assert" in detail


def test_a_wedged_context_fails_the_next_request_fast():
    note_cuda_failure(RuntimeError("CUDA error: an illegal memory access was encountered"))
    with pytest.raises(core.GPUUnavailableError):
        core._require_cuda_healthy("cuda")
    assert cuda_health_snapshot("cpu")[0]  # a CPU engine doesn't care


class LoraLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.ModuleDict({"x": nn.Linear(4, 2)})


def test_half_loaded_lora_layers_are_removed():
    model = nn.Module()
    model.attn = nn.Module()
    q, v = nn.Linear(4, 4), nn.Linear(4, 4)
    model.attn.q_proj, model.attn.v_proj = LoraLinear(q), v
    model.peft_config = {"x": object()}
    assert _strip_injected_lora(model) == 1
    assert model.attn.q_proj is q and model.attn.v_proj is v
    assert not hasattr(model, "peft_config")
    assert _strip_injected_lora(model) == 0


def test_the_manager_replaces_a_wedged_worker(monkeypatch):
    m = EngineManager(port=9999)
    m.device, m.model_path = "cuda", "/models/prime-nf4"
    calls = []
    monkeypatch.setattr(m, "_build_health_response", lambda: {
        "status": "ok", "device": "cuda", "cuda_healthy": False,
        "cuda_detail": "CUDA context wedged by an earlier GPU fault: AcceleratorError"})
    monkeypatch.setattr(m, "stop_worker", lambda: calls.append("stop") or {"ok": True})
    monkeypatch.setattr(m, "start_worker", lambda path, device="cuda", **kw: calls.append(("start", path, device, kw.get("prompt_format"))) or {"ok": True})
    m._prompt_format = "stock"  # a respawn keeps the model's prompt format
    data = m.health_response()
    assert data["status"] == "degraded" and data["respawning"] is True
    for _ in range(50):
        if len(calls) == 2:
            break
        import time; time.sleep(0.02)
    assert calls == ["stop", ("start", "/models/prime-nf4", "cuda", "stock")]
    # Once a minute at most: a worker that wedges again isn't restarted in a loop.
    assert m.health_response()["respawning"] is False


def test_no_model_means_nothing_to_restart():
    m = EngineManager(port=9999)
    assert m.respawn_wedged_worker("x") is False
