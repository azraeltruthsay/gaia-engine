"""Tests for the lifecycle state machine definitions."""

from gaia_engine.lifecycle.states import (
    LifecycleState,
    TransitionTrigger,
    validate_transition,
    available_transitions,
    TRANSITIONS,
    TIER_EXPECTATIONS,
)
from gaia_engine.lifecycle.snapshot import LifecycleSnapshot, TierLiveStatus


def test_lifecycle_states():
    assert LifecycleState.AWAKE.value == "awake"
    assert LifecycleState.FOCUSING.value == "focusing"
    assert LifecycleState.DEEP_SLEEP.value == "deep_sleep"


def test_valid_transition():
    target = validate_transition(LifecycleState.AWAKE, TransitionTrigger.IDLE_TIMEOUT)
    assert target == LifecycleState.SLEEP


def test_invalid_transition():
    target = validate_transition(LifecycleState.AWAKE, TransitionTrigger.WAKE_SIGNAL)
    assert target is None  # Can't wake from AWAKE


def test_user_request_transition():
    target = validate_transition(
        LifecycleState.AWAKE, TransitionTrigger.USER_REQUEST,
        target=LifecycleState.FOCUSING)
    assert target == LifecycleState.FOCUSING


def test_user_request_invalid_target():
    target = validate_transition(
        LifecycleState.SLEEP, TransitionTrigger.USER_REQUEST,
        target=LifecycleState.FOCUSING)
    assert target is None  # Can't go from SLEEP to FOCUSING directly


def test_available_transitions_awake():
    avail = available_transitions(LifecycleState.AWAKE)
    triggers = [t["trigger"] for t in avail]
    assert "voice_join" in triggers
    assert "escalation_needed" in triggers
    assert "idle_timeout" in triggers
    assert "user_request" in triggers


def test_available_transitions_transitioning():
    avail = available_transitions(LifecycleState.TRANSITIONING)
    assert avail == []


def test_tier_expectations_awake():
    exp = TIER_EXPECTATIONS[LifecycleState.AWAKE]
    assert exp["core"].device == "gpu"
    assert exp["core"].required is True
    assert exp["prime"].device == "unloaded"
    assert exp["prime"].required is False


def test_tier_expectations_focusing():
    exp = TIER_EXPECTATIONS[LifecycleState.FOCUSING]
    assert exp["prime"].device == "gpu"
    assert exp["prime"].required is True
    assert exp["core"].device == "unloaded"


def test_snapshot_defaults():
    snap = LifecycleSnapshot()
    assert snap.state == LifecycleState.AWAKE
    assert snap.audio_stt is False
    assert "core" in snap.tiers
    assert snap.vram_used_mb == 0


def test_tier_live_status():
    status = TierLiveStatus(device="gpu", model_loaded=True, vram_mb=3600)
    assert status.device == "gpu"
    assert status.model_loaded is True
    assert status.vram_mb == 3600
