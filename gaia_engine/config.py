"""
gaia_engine/config.py — Centralized configuration for the GAIA Engine.

All environment variable lookups and defaults in one place.
Import from here instead of scattering os.environ.get() across modules.
"""

import os

# ── Inference ────────────────────────────────────────────────────────────────

GGUF_CTX_SIZE: int = int(os.environ.get("GGUF_CTX_SIZE", os.environ.get("CORE_CPU_CTX", "16384")))
GGUF_THREADS: int = int(os.environ.get("GGUF_THREADS", "16"))

# ── Identity ─────────────────────────────────────────────────────────────────

ENGINE_TIER: str = os.environ.get("GAIA_ENGINE_TIER", "prime")

# ── SAE / Activation Streaming ───────────────────────────────────────────────

SAE_SAMPLE_RATE: int = int(os.environ.get("GAIA_SAE_SAMPLE_RATE", "4"))
SAE_STREAM_EVERY_N: int = int(os.environ.get("GAIA_SAE_STREAM_EVERY_N", "8"))
ACTIVATION_STREAM_PATH: str = os.environ.get("ACTIVATION_STREAM_PATH", "/logs/activation_stream.jsonl")

# ── Timezone (for awareness injection) ───────────────────────────────────────

LOCAL_TZ_OFFSET: int = int(os.environ.get("GAIA_LOCAL_TZ_OFFSET", "-7"))
LOCAL_TZ_LABEL: str = os.environ.get("GAIA_LOCAL_TZ_LABEL", "PDT")

# ── Paths ────────────────────────────────────────────────────────────────────

AWARENESS_DIR: str = os.environ.get("AWARENESS_DIR", "/knowledge/awareness")
