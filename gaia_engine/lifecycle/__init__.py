"""
GAIA Lifecycle — unified GPU lifecycle state machine types.

Shared by orchestrator (authority) and gaia-core (consumer).
"""

from gaia_engine.lifecycle.states import (
    LifecycleState,
    TransitionTrigger,
    TRANSITIONS,
    TIER_EXPECTATIONS,
    TierExpectation,
    available_transitions,
    validate_transition,
)
from gaia_engine.lifecycle.snapshot import (
    LifecycleSnapshot,
    TierLiveStatus,
    TransitionRecord,
    TransitionResult,
)

__all__ = [
    "LifecycleState",
    "TransitionTrigger",
    "TRANSITIONS",
    "TIER_EXPECTATIONS",
    "TierExpectation",
    "available_transitions",
    "validate_transition",
    "LifecycleSnapshot",
    "TierLiveStatus",
    "TransitionRecord",
    "TransitionResult",
]
