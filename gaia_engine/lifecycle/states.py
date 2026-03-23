"""
Lifecycle state definitions — the single source of truth for GAIA's GPU lifecycle.

States, transitions, triggers, and per-state tier expectations.
Imported by both the orchestrator (authority) and gaia-core (consumer).
"""

from enum import Enum
from typing import Dict, List, Optional, Set


class LifecycleState(str, Enum):
    """Primary lifecycle states for GAIA's GPU allocation."""
    AWAKE = "awake"              # Core + Nano on GPU. Default cognitive operation.
    LISTENING = "listening"      # Core + Nano + Audio STT on GPU.
    FOCUSING = "focusing"        # Prime GPTQ on GPU. Core/Nano off GPU.
    MEDITATION = "meditation"    # Study owns GPU for training. All cognitive tiers off.
    SLEEP = "sleep"              # Core + Nano in CPU RAM. GPU empty.
    DEEP_SLEEP = "deep_sleep"    # Core unloaded from RAM. Nano minimal reflex. GPU empty.
    TRANSITIONING = "transitioning"  # Handoff in progress.


class TransitionTrigger(str, Enum):
    """Events that trigger lifecycle state transitions."""
    IDLE_TIMEOUT = "idle_timeout"            # No user activity for threshold minutes
    WAKE_SIGNAL = "wake_signal"              # Message received during sleep
    VOICE_JOIN = "voice_join"                # User joined Discord voice channel
    VOICE_LEAVE = "voice_leave"              # User left Discord voice channel
    ESCALATION_NEEDED = "escalation_needed"  # Complex query requires Prime
    TASK_COMPLETE = "task_complete"           # Prime finished complex reasoning
    TRAINING_SCHEDULED = "training_scheduled" # Study needs GPU for QLoRA/merge
    TRAINING_COMPLETE = "training_complete"   # Study finished training
    USER_REQUEST = "user_request"            # Manual transition from dashboard/API
    EXTENDED_IDLE = "extended_idle"           # Long idle in SLEEP → DEEP_SLEEP
    PREEMPT = "preempt"                      # Wake signal during MEDITATION


class TierExpectation:
    """Expected state of a tier in a given lifecycle state."""
    __slots__ = ("device", "required")

    def __init__(self, device: str, required: bool = True):
        self.device = device      # "gpu", "cpu", "unloaded"
        self.required = required  # Must this tier be in this state?

    def __repr__(self):
        return f"TierExpectation({self.device}, required={self.required})"


# ── Transition Table ──────────────────────────────────────────────────────────
# Dict[source_state] → Dict[trigger] → target_state
# USER_REQUEST can target multiple states, handled specially in validate_transition.

TRANSITIONS: Dict[LifecycleState, Dict[TransitionTrigger, LifecycleState]] = {
    LifecycleState.AWAKE: {
        TransitionTrigger.VOICE_JOIN: LifecycleState.LISTENING,
        TransitionTrigger.ESCALATION_NEEDED: LifecycleState.FOCUSING,
        TransitionTrigger.IDLE_TIMEOUT: LifecycleState.SLEEP,
        TransitionTrigger.TRAINING_SCHEDULED: LifecycleState.MEDITATION,
        # USER_REQUEST handled in validate_transition
    },
    LifecycleState.LISTENING: {
        TransitionTrigger.VOICE_LEAVE: LifecycleState.AWAKE,
        TransitionTrigger.ESCALATION_NEEDED: LifecycleState.FOCUSING,
    },
    LifecycleState.FOCUSING: {
        TransitionTrigger.TASK_COMPLETE: LifecycleState.AWAKE,
        TransitionTrigger.VOICE_JOIN: LifecycleState.LISTENING,
        TransitionTrigger.TRAINING_SCHEDULED: LifecycleState.MEDITATION,
        TransitionTrigger.IDLE_TIMEOUT: LifecycleState.SLEEP,
    },
    LifecycleState.MEDITATION: {
        TransitionTrigger.TRAINING_COMPLETE: LifecycleState.AWAKE,
        TransitionTrigger.PREEMPT: LifecycleState.AWAKE,
        TransitionTrigger.WAKE_SIGNAL: LifecycleState.AWAKE,
    },
    LifecycleState.SLEEP: {
        TransitionTrigger.WAKE_SIGNAL: LifecycleState.AWAKE,
        TransitionTrigger.EXTENDED_IDLE: LifecycleState.DEEP_SLEEP,
        TransitionTrigger.TRAINING_SCHEDULED: LifecycleState.MEDITATION,
    },
    LifecycleState.DEEP_SLEEP: {
        TransitionTrigger.WAKE_SIGNAL: LifecycleState.AWAKE,
    },
    LifecycleState.TRANSITIONING: {
        # No triggers — transitions out are handled by the machine itself
    },
}

# States reachable via USER_REQUEST from each state
USER_REQUEST_TARGETS: Dict[LifecycleState, Set[LifecycleState]] = {
    LifecycleState.AWAKE: {
        LifecycleState.FOCUSING, LifecycleState.SLEEP,
        LifecycleState.DEEP_SLEEP, LifecycleState.MEDITATION,
    },
    LifecycleState.LISTENING: {
        LifecycleState.AWAKE, LifecycleState.FOCUSING,
    },
    LifecycleState.FOCUSING: {
        LifecycleState.AWAKE, LifecycleState.SLEEP, LifecycleState.DEEP_SLEEP,
    },
    LifecycleState.MEDITATION: {
        LifecycleState.AWAKE,
    },
    LifecycleState.SLEEP: {
        LifecycleState.AWAKE, LifecycleState.DEEP_SLEEP,
    },
    LifecycleState.DEEP_SLEEP: {
        LifecycleState.AWAKE, LifecycleState.SLEEP,
    },
    LifecycleState.TRANSITIONING: set(),
}


# ── Tier Expectations Per State ───────────────────────────────────────────────
# Defines what each tier should look like in each lifecycle state.
# The lifecycle machine uses this to determine which load/unload actions to take.

TIER_EXPECTATIONS: Dict[LifecycleState, Dict[str, TierExpectation]] = {
    LifecycleState.AWAKE: {
        "core":  TierExpectation("gpu", required=True),
        "nano":  TierExpectation("gpu", required=False),  # may be llama-server (unmanaged)
        "prime": TierExpectation("unloaded", required=False),
        "study": TierExpectation("unloaded", required=False),
    },
    LifecycleState.LISTENING: {
        "core":  TierExpectation("gpu", required=True),
        "nano":  TierExpectation("gpu", required=False),  # may be llama-server (unmanaged)
        "prime": TierExpectation("unloaded", required=False),
        "study": TierExpectation("unloaded", required=False),
        # audio_stt flag handled separately
    },
    LifecycleState.FOCUSING: {
        "core":  TierExpectation("unloaded", required=False),
        "nano":  TierExpectation("unloaded", required=False),
        "prime": TierExpectation("gpu", required=True),
        "study": TierExpectation("unloaded", required=False),
    },
    LifecycleState.MEDITATION: {
        "core":  TierExpectation("unloaded", required=False),
        "nano":  TierExpectation("unloaded", required=False),
        "prime": TierExpectation("unloaded", required=False),
        "study": TierExpectation("gpu", required=True),
    },
    LifecycleState.SLEEP: {
        "core":  TierExpectation("cpu", required=True),
        "nano":  TierExpectation("cpu", required=True),
        "prime": TierExpectation("unloaded", required=False),
        "study": TierExpectation("unloaded", required=False),
    },
    LifecycleState.DEEP_SLEEP: {
        "core":  TierExpectation("unloaded", required=False),
        "nano":  TierExpectation("cpu", required=True),  # minimal reflex
        "prime": TierExpectation("unloaded", required=False),
        "study": TierExpectation("unloaded", required=False),
    },
    LifecycleState.TRANSITIONING: {
        # No expectations — transitioning is temporary
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_transition(
    current: LifecycleState,
    trigger: TransitionTrigger,
    target: Optional[LifecycleState] = None,
) -> Optional[LifecycleState]:
    """Validate and resolve a transition. Returns target state or None if invalid.

    For USER_REQUEST triggers, `target` must be specified.
    For all other triggers, the target is determined by the transition table.
    """
    if current == LifecycleState.TRANSITIONING:
        return None  # Can't transition while already transitioning

    if trigger == TransitionTrigger.USER_REQUEST:
        if target is None:
            return None
        valid_targets = USER_REQUEST_TARGETS.get(current, set())
        return target if target in valid_targets else None

    table = TRANSITIONS.get(current, {})
    return table.get(trigger)


def available_transitions(current: LifecycleState) -> List[dict]:
    """Return list of available transitions from the current state.

    Each item: {"trigger": str, "target": str} or
               {"trigger": "user_request", "targets": [str, ...]}
    """
    if current == LifecycleState.TRANSITIONING:
        return []

    result = []
    table = TRANSITIONS.get(current, {})
    for trigger, target in table.items():
        result.append({
            "trigger": trigger.value,
            "target": target.value,
        })

    # Add user_request targets
    user_targets = USER_REQUEST_TARGETS.get(current, set())
    if user_targets:
        result.append({
            "trigger": TransitionTrigger.USER_REQUEST.value,
            "targets": sorted(t.value for t in user_targets),
        })

    return result
