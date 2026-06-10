"""
gaia_engine/config.py — Centralized configuration for the GAIA Engine.

All environment variable lookups and defaults in one place.
Import from here instead of scattering os.environ.get() across modules.
"""

import os

# ── Inference ────────────────────────────────────────────────────────────────

GGUF_CTX_SIZE: int = int(os.environ.get("GGUF_CTX_SIZE", os.environ.get("CORE_CPU_CTX", "16384")))
GGUF_THREADS: int = int(os.environ.get("GGUF_THREADS", "16"))

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
ACTIVATION_STREAM_PATH: str = os.environ.get("ACTIVATION_STREAM_PATH", "/logs/activation_stream.jsonl")

# ── Timezone (for awareness injection) ───────────────────────────────────────

LOCAL_TZ_OFFSET: int = int(os.environ.get("GAIA_LOCAL_TZ_OFFSET", "-7"))
LOCAL_TZ_LABEL: str = os.environ.get("GAIA_LOCAL_TZ_LABEL", "PDT")

# ── Paths ────────────────────────────────────────────────────────────────────

AWARENESS_DIR: str = os.environ.get("AWARENESS_DIR", "/knowledge/awareness")
