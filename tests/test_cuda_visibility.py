"""Tests: CUDA visibility save/mask/restore (GAIA_Project-3tch).

The bug: repeated CPU/GGUF loads across gear cycles saved
ORIGINAL_CUDA_VISIBLE_DEVICES="" (the mask itself), and every later cuda
worker spawn faithfully "restored" blindness — torch saw zero GPUs, NF4
fell back to a bf16 CPU grind. Empty string must count as ABSENT.
"""

from gaia_engine.manager import _expose_cuda, _hide_cuda


def test_hide_then_expose_roundtrip_with_real_value():
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    _hide_cuda(env)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["ORIGINAL_CUDA_VISIBLE_DEVICES"] == "0"
    _expose_cuda(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_hide_when_never_set_leaves_no_original():
    env = {}
    _hide_cuda(env)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "ORIGINAL_CUDA_VISIBLE_DEVICES" not in env
    _expose_cuda(env)
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_3tch_poison_cycle_double_hide_then_cuda_spawn():
    """The exact production sequence: park (hide), park again (hide),
    then awake (expose for the cuda worker). The old code saved
    ORIGINAL="" on the second hide and restored blindness on expose."""
    env = {}                    # container starts with no CUDA var
    _hide_cuda(env)             # first CPU/GGUF load
    _hide_cuda(env)             # second CPU/GGUF load (sleep/park cycle)
    assert env.get("ORIGINAL_CUDA_VISIBLE_DEVICES", "") == "", \
        "mask must never be saved as the original"
    _expose_cuda(env)           # cuda worker spawn env
    assert "CUDA_VISIBLE_DEVICES" not in env or env["CUDA_VISIBLE_DEVICES"] != "", \
        "cuda spawn must not inherit the mask"


def test_many_cycles_preserve_real_original():
    env = {"CUDA_VISIBLE_DEVICES": "0,1"}
    for _ in range(5):
        _hide_cuda(env)
        assert env["ORIGINAL_CUDA_VISIBLE_DEVICES"] == "0,1"
    _expose_cuda(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_expose_with_empty_original_unsets():
    # A pre-fix poisoned state on disk/in-process: expose must clean it up,
    # not restore blindness.
    env = {"CUDA_VISIBLE_DEVICES": "", "ORIGINAL_CUDA_VISIBLE_DEVICES": ""}
    _expose_cuda(env)
    assert "CUDA_VISIBLE_DEVICES" not in env
