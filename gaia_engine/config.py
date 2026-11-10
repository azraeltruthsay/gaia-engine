"""
gaia_engine/config.py — Centralized configuration for the GAIA Engine.

All environment variable lookups and defaults in one place.
Import from here instead of scattering os.environ.get() across modules.
"""

import os

# ── Inference ────────────────────────────────────────────────────────────────

GGUF_CTX_SIZE: int = int(os.environ.get("GGUF_CTX_SIZE", os.environ.get("CORE_CPU_CTX", "16384")))
GGUF_THREADS: int = int(os.environ.get("GGUF_THREADS", "16"))
# Threads for reading the prompt (-1 = same as GGUF_THREADS). Generation is
# memory-bandwidth bound and peaks near the physical core count; prompt reading
# is compute bound and keeps scaling (ojwf, 12B Q8 on a 16-core 9950X3D:
# generation best at 12, prefill 57 tok/s at 16 threads vs 78 at 24).
GGUF_THREADS_BATCH: int = int(os.environ.get("GGUF_THREADS_BATCH", "-1"))

# GPU offload layer count for GGUF models when device="cuda". 999 = "all layers"
# (llama.cpp clamps to the real count). NOTE: Gemma 4 GGUF crashes in the CUDA
# graph when the FINAL layer/output is offloaded (llama.cpp abort in
# llama_decode); set this to (num_layers - 1) — e.g. 33 for Gemma4-E4B — to
# offload all-but-last and keep the crashing op on CPU. Validated 2026-06-09
# for the Discord-voice GGUF-on-GPU "voice gear" (GAIA_Project-a1t): E4B GGUF
# at 33 layers ≈ 3.7GB VRAM, 30-48 tok/s, no crash. Other archs are fine at 999.
GGUF_GPU_LAYERS: int = int(os.environ.get("GGUF_GPU_LAYERS", "999"))

# ── Identity ─────────────────────────────────────────────────────────────────

ENGINE_TIER: str = os.environ.get("GAIA_ENGINE_TIER", "prime")

# ── SAE / Activation Streaming ───────────────────────────────────────────────

SAE_SAMPLE_RATE: int = int(os.environ.get("GAIA_SAE_SAMPLE_RATE", "4"))
SAE_STREAM_EVERY_N: int = int(os.environ.get("GAIA_SAE_STREAM_EVERY_N", "8"))
# The atlas recorded on this model's weights, read live into the activation stream.
SAE_ATLAS_PATH: str = os.environ.get("GAIA_SAE_ATLAS", "")
SAE_ATLAS_LAYERS: list = [int(x) for x in os.environ.get("GAIA_SAE_ATLAS_LAYERS", "12,24,35").split(",") if x.strip()]
ACTIVATION_STREAM_PATH: str = os.environ.get("ACTIVATION_STREAM_PATH", "/logs/activation_stream.jsonl")

# ── Prompt-shaping features ──────────────────────────────────────────────────
# Dynamic awareness injection (a world-state prompt segment) and CogPacket
# system-prompt compression (drops sections judged already known). Both change
# what the model is shown. Until the engine was mounted as one directory they
# were only installed in gaia-prime, so Core never ran them; they are explicit
# settings now, and turning them on for a tier is a deliberate, evaluated change.

def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "off")


AWARENESS_ENABLED: bool = _flag("GAIA_ENGINE_AWARENESS")
COGPACKET_COMPRESS_ENABLED: bool = _flag("GAIA_ENGINE_COGPACKET_COMPRESS")

# ── Timezone (for awareness injection) ───────────────────────────────────────

LOCAL_TZ_OFFSET: int = int(os.environ.get("GAIA_LOCAL_TZ_OFFSET", "-7"))
LOCAL_TZ_LABEL: str = os.environ.get("GAIA_LOCAL_TZ_LABEL", "PDT")


def local_now():
    """(now in the operator's timezone, its label). A named zone follows
    daylight saving: the fixed offset said PDT all winter. The offset is only
    the fallback for an image without tzdata."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        local = now.astimezone(ZoneInfo(os.environ.get("GAIA_USER_TZ") or "America/Los_Angeles"))
        return local, local.tzname()
    except Exception:
        return now.astimezone(timezone(timedelta(hours=LOCAL_TZ_OFFSET))), LOCAL_TZ_LABEL

# ── Paths ────────────────────────────────────────────────────────────────────

AWARENESS_DIR: str = os.environ.get("AWARENESS_DIR", "/knowledge/awareness")
