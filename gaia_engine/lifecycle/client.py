"""
Lifecycle client — HTTP interface for querying and requesting transitions.

Used by gaia-core (and any other service) to interact with the authoritative
lifecycle state machine running in the orchestrator.

Provides both async (for FastAPI services) and sync (for threaded code like
agent_core.py and sleep_cycle_loop.py) interfaces.
"""

import json
import logging
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from gaia_engine.lifecycle.snapshot import LifecycleSnapshot, TransitionResult
from gaia_engine.lifecycle.states import (
    LifecycleState,
    TransitionTrigger,
    available_transitions,
)

logger = logging.getLogger("GAIA.LifecycleClient")


class LifecycleClient:
    """HTTP client for the orchestrator's lifecycle state machine.

    Designed for gaia-core and other services that need to query or
    request lifecycle transitions. Stdlib-only (no httpx dependency).
    """

    def __init__(self, orchestrator_url: str = "http://gaia-orchestrator:6410"):
        self._url = orchestrator_url.rstrip("/")
        self._cached_state: Optional[LifecycleSnapshot] = None

    # ── Sync Interface (for threaded code) ────────────────────────────────

    def get_state_sync(self, timeout: float = 5.0) -> LifecycleSnapshot:
        """Get current lifecycle state (synchronous). Falls back to cache on error."""
        try:
            req = Request(f"{self._url}/lifecycle/state")
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                snapshot = LifecycleSnapshot(**data)
                self._cached_state = snapshot
                return snapshot
        except Exception as e:
            logger.debug("Lifecycle state query failed: %s — using cache", e)
            if self._cached_state:
                return self._cached_state
            # Return a default snapshot if we've never connected
            return LifecycleSnapshot()

    def request_transition_sync(
        self,
        trigger: TransitionTrigger,
        target: Optional[LifecycleState] = None,
        reason: str = "",
        timeout: float = 300.0,
    ) -> TransitionResult:
        """Request a lifecycle transition (synchronous)."""
        body = {
            "trigger": trigger.value if isinstance(trigger, TransitionTrigger) else trigger,
            "reason": reason,
        }
        if target is not None:
            body["target"] = target.value if isinstance(target, LifecycleState) else target

        try:
            data = json.dumps(body).encode()
            req = Request(
                f"{self._url}/lifecycle/transition",
                data=data,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            with urlopen(req, timeout=timeout) as resp:
                result_data = json.loads(resp.read())
                return TransitionResult(**result_data)
        except URLError as e:
            return TransitionResult(ok=False, error=f"orchestrator unreachable: {e}")
        except Exception as e:
            return TransitionResult(ok=False, error=str(e))

    def reconcile_sync(self, timeout: float = 30.0) -> dict:
        """Force state reconciliation (synchronous)."""
        try:
            req = Request(f"{self._url}/lifecycle/reconcile", method="POST")
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Async Interface (for FastAPI services) ────────────────────────────

    async def get_state(self, timeout: float = 5.0) -> LifecycleSnapshot:
        """Get current lifecycle state (async)."""
        import asyncio
        return await asyncio.to_thread(self.get_state_sync, timeout)

    async def request_transition(
        self,
        trigger: TransitionTrigger,
        target: Optional[LifecycleState] = None,
        reason: str = "",
        timeout: float = 300.0,
    ) -> TransitionResult:
        """Request a lifecycle transition (async)."""
        import asyncio
        return await asyncio.to_thread(
            self.request_transition_sync, trigger, target, reason, timeout)

    async def reconcile(self, timeout: float = 30.0) -> dict:
        """Force state reconciliation (async)."""
        import asyncio
        return await asyncio.to_thread(self.reconcile_sync, timeout)

    # ── Convenience Properties ────────────────────────────────────────────

    @property
    def current_state(self) -> LifecycleState:
        """Last known state (from cache). Call get_state_sync() first for fresh data."""
        if self._cached_state:
            return LifecycleState(self._cached_state.state)
        return LifecycleState.AWAKE

    @property
    def is_prime_available(self) -> bool:
        """Whether Prime is currently loaded on GPU (from cache)."""
        if self._cached_state:
            return LifecycleState(self._cached_state.state) == LifecycleState.FOCUSING
        return False

    @property
    def is_gpu_active(self) -> bool:
        """Whether any cognitive tier has GPU (from cache)."""
        if self._cached_state:
            state = LifecycleState(self._cached_state.state)
            return state in (
                LifecycleState.AWAKE, LifecycleState.LISTENING, LifecycleState.FOCUSING)
        return False
