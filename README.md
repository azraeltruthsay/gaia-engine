# GAIA Engine

Purpose-built inference engine for self-aware AI systems.

Not a general-purpose inference server. Optimized for single-user, single-GPU deployments with deep introspection capabilities.

## Features

- **Subprocess Isolation** — Engine manager starts with zero GPU. Model loading spawns a worker subprocess that owns the CUDA context. Unloading kills the process — guaranteed zero VRAM.
- **KV Prefix Caching** — Pre-computed identity/context as frozen KV states. Sub-100ms latency on cache hits.
- **Thought Snapshots** — Hold, resume, compose KV cache states across conversations.
- **Activation Monitoring (Polygraph)** — Hidden state capture at every layer for lie detection and self-analysis.
- **LoRA Hot-Swap** — Dynamic adapter loading/unloading without model restart.
- **SAE Atlas** — Sparse autoencoder training on model activations for feature discovery.
- **ROME Editing** — Rank-One Model Editing for surgical weight corrections.
- **GPU Lifecycle** — Unified state machine (AWAKE/FOCUSING/SLEEP/DEEP_SLEEP/MEDITATION) with automatic tier management.
- **GPTQ/NF4/int8** — Automatic quantization detection and loading.
- **SSE Streaming** — True per-token streaming via manager proxy passthrough.

## Quick Start

```python
from gaia_engine import GAIAEngine, serve

engine = GAIAEngine("/models/Qwen3.5-2B", device="cuda")
result = engine.generate(
    messages=[{"role": "user", "content": "Who are you?"}],
    max_tokens=128,
)
print(result["choices"][0]["message"]["content"])

# As a server
serve("/models/Qwen3.5-2B", port=8092)
```

## Managed Mode (Zero-GPU Standby)

```bash
# Start with zero GPU footprint
gaia-engine --managed --port 8092

# Load model via HTTP
curl -X POST http://localhost:8092/model/load \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/Qwen3.5-2B", "device": "cuda"}'

# Inference
curl -X POST http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'

# Unload — kills subprocess, frees ALL GPU memory
curl -X POST http://localhost:8092/model/unload
```

## Installation

```bash
pip install gaia-engine

# With GPTQ support
pip install gaia-engine[gptq]

# With vision support
pip install gaia-engine[vision]

# With training support (datasets, trl)
pip install gaia-engine[training]

# Everything
pip install gaia-engine[all]
```

## License

Apache 2.0
