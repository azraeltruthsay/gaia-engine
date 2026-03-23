"""
Lifecycle snapshot models — shared state representation.

These pydantic models are the canonical format for lifecycle state,
returned by the orchestrator's /lifecycle/* endpoints and consumed
by gaia-core, gaia-web, and the dashboard.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from gaia_engine.lifecycle.states import LifecycleState, TransitionTrigger


class TierLiveStatus(BaseModel):
    """Live status of a single cognitive tier."""
    device: str = "unloaded"        # "gpu", "cpu", "unloaded"
    model_loaded: bool = False
    model_path: str = ""
    vram_mb: int = 0
    managed: bool = False           # Using managed engine (subprocess isolation)?
    healthy: bool = False           # Health endpoint responded?
    endpoint: str = ""              # Engine endpoint URL


class TransitionRecord(BaseModel):
    """Record of a completed lifecycle transition."""
    from_state: str
    to_state: str
    trigger: str
    reason: str = ""
    target: Optional[str] = None    # For user_request: explicit target
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_s: float = 0.0
    actions: List[str] = Field(default_factory=list)  # e.g. ["core:unload_gpu", "prime:load_gpu"]
    error: Optional[str] = None


class LifecycleSnapshot(BaseModel):
    """Complete lifecycle state snapshot — the single source of truth."""
    state: LifecycleState = LifecycleState.AWAKE

    # Audio capability flags (not separate states)
    audio_stt: bool = False         # STT model loaded (1.8GB)
    audio_tts: bool = False         # TTS model loaded (4.3GB)

    # Per-tier live status
    tiers: Dict[str, TierLiveStatus] = Field(default_factory=lambda: {
        "core": TierLiveStatus(),
        "nano": TierLiveStatus(),
        "prime": TierLiveStatus(),
        "study": TierLiveStatus(),
    })

    # GPU memory
    vram_total_mb: int = 15833      # RTX 5080 usable
    vram_used_mb: int = 0
    vram_free_mb: int = 15833

    # Transition metadata (populated during TRANSITIONING)
    transition_from: Optional[LifecycleState] = None
    transition_to: Optional[LifecycleState] = None
    transition_phase: Optional[str] = None  # e.g. "unloading_core", "loading_prime"
    transition_error: Optional[str] = None

    # Timestamps
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_transition_at: Optional[datetime] = None
    last_transition_trigger: Optional[str] = None

    # Recent history
    history: List[TransitionRecord] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class TransitionResult(BaseModel):
    """Result of a transition request."""
    ok: bool
    from_state: str = ""
    to_state: str = ""
    trigger: str = ""
    elapsed_s: float = 0.0
    actions: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    snapshot: Optional[LifecycleSnapshot] = None
