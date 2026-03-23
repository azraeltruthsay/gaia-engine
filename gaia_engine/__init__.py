"""
GAIA Inference Engine — shared cognitive inference library.

Used by all tier containers (gaia-core, gaia-nano, and Prime when loaded).
Each container imports this library and serves one model through it.

    from gaia_engine import GAIAEngine, serve

    engine = GAIAEngine("/models/Qwen3.5-2B-GAIA-Core-v3", device="cuda")
    result = engine.generate(messages=[...])

    # Or run as a standalone server:
    serve("/models/Qwen3.5-2B-GAIA-Core-v3", port=8092)
"""

from gaia_engine.core import GAIAEngine, serve
from gaia_engine.manager import EngineManager, serve_managed, set_event_logger
from gaia_engine.thought_composer import compose_thoughts, estimate_composed_size

__version__ = "1.0.0"

__all__ = [
    "GAIAEngine", "serve",
    "EngineManager", "serve_managed", "set_event_logger",
    "compose_thoughts", "estimate_composed_size",
]
