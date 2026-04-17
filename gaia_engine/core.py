"""
GAIA Inference Engine — purpose-built inference for self-aware AI.

Not a general-purpose inference server. Optimized for:
- Single user, single GPU
- Sub-100ms latency on cached requests
- Hidden state access at every layer (polygraph)
- KV cache management with thought snapshots
- GPU↔CPU device migration
- Speculative decoding (Nano drafts, Core/Prime verifies)

Performance stack:
- torch.compile with reduce-overhead mode
- FlashAttention via SDPA
- Static KV cache pre-allocation
- Fused generation loop (minimize Python overhead)
- Optional: speculative decoding across tiers
"""

import gc
import hashlib
import json
import logging
import os
import re
import time
import threading
import uuid
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from urllib.parse import urlparse, parse_qs

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("GAIA.Engine")

# ── Performance: configurable SAE sample rate ────────────────────────────────
from gaia_engine.config import SAE_SAMPLE_RATE as _SAE_SAMPLE_RATE
from gaia_engine.config import SAE_STREAM_EVERY_N as _SAE_STREAM_EVERY_N

# Disable gradient computation globally for inference (not just per-call)
torch.set_grad_enabled(False)


# ── Activation JSONL Writer ──────────────────────────────────────────────────

def _write_activation(tier, token, token_idx, session_id, snapshot, sae=None, labels=None):
    """Write per-token activation data to JSONL for live visualization.

    Called from generate_stream() after each token's forward pass when
    the activation monitor is enabled.  The JSONL file is tailed by
    gaia-web's SSE endpoint to drive the Neural Mind Map in real-time.

    Never raises — inference must never crash for visualization.
    """
    from datetime import datetime, timezone

    features = []
    if sae and snapshot:
        # SAE decomposition into interpretable features
        for layer_key, layer_data in snapshot.items():
            try:
                layer_idx = int(layer_key.split("_")[1])
            except (IndexError, ValueError):
                continue
            for idx, val in zip(layer_data.get("top_5_indices", []),
                                layer_data.get("top_5_values", [])):
                label = (labels or {}).get(layer_idx, {}).get(int(idx), f"feature_{idx}")
                features.append({"idx": int(idx), "strength": float(val),
                                 "label": label, "layer": layer_idx})
    elif snapshot:
        # No SAE — use raw polygraph top activations
        for layer_key, layer_data in snapshot.items():
            try:
                layer_idx = int(layer_key.split("_")[1])
            except (IndexError, ValueError):
                continue
            for idx, val in zip(layer_data.get("top_5_indices", []),
                                layer_data.get("top_5_values", [])):
                features.append({"idx": int(idx), "strength": float(val),
                                 "label": f"neuron_{idx}", "layer": layer_idx})

    # Sort by strength, keep top 10
    features.sort(key=lambda f: f["strength"], reverse=True)
    features = features[:10]

    line = json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "token": token,
        "token_idx": token_idx,
        "session_id": session_id or "",
        "features": features,
    })

    try:
        from gaia_engine.config import ACTIVATION_STREAM_PATH
        log_path = ACTIVATION_STREAM_PATH
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass  # Never crash inference for visualization


# ── Static KV Cache ──────────────────────────────────────────────────────────

class StaticKVCache:
    """Pre-allocated KV cache to avoid dynamic allocation per token.

    Standard generation allocates new tensors for each token's key/value
    states. This pre-allocates a fixed buffer and writes into it,
    eliminating allocation overhead during generation.
    """

    def __init__(self, model, max_seq_len: int = 4096, device: str = "cuda"):
        self.max_seq_len = max_seq_len
        self.device = "cuda" if device == "gpu" else device
        self.position = 0

        # Extract model config
        config = model.config
        if hasattr(config, 'text_config'):
            config = config.text_config

        self.num_layers = getattr(config, 'num_hidden_layers', 24)
        self.num_kv_heads = getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', 8))
        self.head_dim = getattr(config, 'head_dim', getattr(config, 'hidden_size', 2048) // getattr(config, 'num_attention_heads', 8))

        # Pre-allocate buffers (only for standard attention models — Qwen3)
        # For Qwen3.5 hybrid, we fall back to dynamic cache
        self.is_standard_attention = not hasattr(config, 'layer_types')

        if self.is_standard_attention:
            self._k_cache = torch.zeros(
                self.num_layers, 1, self.num_kv_heads, max_seq_len, self.head_dim,
                dtype=torch.bfloat16, device=device,
            )
            self._v_cache = torch.zeros(
                self.num_layers, 1, self.num_kv_heads, max_seq_len, self.head_dim,
                dtype=torch.bfloat16, device=device,
            )
            mem_mb = (self._k_cache.nelement() + self._v_cache.nelement()) * 2 / (1024 * 1024)
            logger.info("Static KV cache allocated: %d layers × %d heads × %d seq × %d dim (%.0fMB)",
                        self.num_layers, self.num_kv_heads, max_seq_len, self.head_dim, mem_mb)
        else:
            logger.info("Hybrid attention detected — using dynamic KV cache")

    def reset(self):
        self.position = 0


# ── Activation Monitor ───────────────────────────────────────────────────────

class ActivationMonitor:
    """Real-time activation monitoring during inference."""

    def __init__(self):
        self.enabled = True
        self._last_snapshot: Optional[dict] = None
        self._last_timestamp: float = 0.0
        self._captures: int = 0

    def capture(self, hidden_states: tuple, sample_every: int = None) -> dict:
        """Capture activation snapshot from selected layers.

        Args:
            hidden_states: Tuple of hidden states from the model forward pass.
                Can be the full tuple or a pre-filtered dict {layer_idx: tensor}.
            sample_every: Layer sampling interval. Defaults to _SAE_SAMPLE_RATE env var.
        """
        if not self.enabled or hidden_states is None:
            return {}

        if sample_every is None:
            sample_every = _SAE_SAMPLE_RATE

        self._captures += 1
        self._last_timestamp = time.time()

        snapshot = {}

        if isinstance(hidden_states, dict):
            # Pre-filtered dict from selective capture: {layer_idx: tensor}
            for idx, hs in hidden_states.items():
                last_token = hs[0, -1, :]
                abs_vals = last_token.abs()
                top5 = abs_vals.topk(5)
                snapshot[f"layer_{idx}"] = {
                    "mean": float(last_token.mean()),
                    "std": float(last_token.std()),
                    "l2_norm": float(last_token.norm()),
                    "top_5_indices": top5.indices.tolist(),
                    "top_5_values": [round(float(v), 4) for v in top5.values],
                }
        else:
            # Full tuple — sample every Nth layer
            num_layers = len(hidden_states)
            sample_layers = [0] + list(range(sample_every, num_layers - 1, sample_every)) + [num_layers - 1]

            for idx in sample_layers:
                if idx >= num_layers:
                    continue
                last_token = hidden_states[idx][0, -1, :]
                abs_vals = last_token.abs()
                top5 = abs_vals.topk(5)
                snapshot[f"layer_{idx}"] = {
                    "mean": float(last_token.mean()),
                    "std": float(last_token.std()),
                    "l2_norm": float(last_token.norm()),
                    "top_5_indices": top5.indices.tolist(),
                    "top_5_values": [round(float(v), 4) for v in top5.values],
                }

        self._last_snapshot = snapshot
        return snapshot

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "captures": self._captures,
            "last_timestamp": self._last_timestamp,
            "last_snapshot": self._last_snapshot,
        }


# ── Thought Manager ──────────────────────────────────────────────────────────

class ThoughtManager:
    """Manages cognitive state snapshots via KV cache."""

    def __init__(self, storage_dir: str = "/shared/thoughts"):
        self._thoughts: Dict[str, dict] = {}
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def hold(self, label: str, kv_state, prefix_len: int,
             segment_hashes: list, context: str = "") -> dict:
        import copy
        metadata = {
            "label": label, "context": context,
            "prefix_tokens": prefix_len,
            "segment_hashes": segment_hashes,
            "timestamp": time.time(),
        }
        self._thoughts[label] = {"kv": copy.deepcopy(kv_state), "meta": metadata}
        (self._dir / f"{label}.json").write_text(json.dumps(metadata, indent=2))
        logger.info("THOUGHT HELD: '%s' (%d tokens)", label, prefix_len)
        return {"ok": True, **metadata}

    def resume(self, label: str) -> Optional[dict]:
        if label not in self._thoughts:
            return None
        return self._thoughts[label]

    def list_all(self) -> dict:
        result = {}
        for label, thought in self._thoughts.items():
            m = thought["meta"]
            result[label] = {
                "context": m.get("context", ""),
                "prefix_tokens": m["prefix_tokens"],
                "age_s": round(time.time() - m["timestamp"], 1),
            }
        return {"thoughts": result, "count": len(result)}

    def drop(self, label: str) -> bool:
        if label in self._thoughts:
            del self._thoughts[label]
            p = self._dir / f"{label}.json"
            if p.exists():
                p.unlink()
            return True
        return False


# ── Chat Template Formatting ────────────────────────────────────────────────

class ChatFormatter:
    """Model-family-aware message formatting.

    Replaces hardcoded ChatML (<|im_start|>/<|im_end|>) with dynamic
    formatting based on the tokenizer's special tokens. Supports:
    - Qwen (ChatML): <|im_start|>role\\n...<|im_end|>
    - Gemma 4: <|turn>role<turn|>...
    - Fallback: ChatML (safe default for unknown models)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.family = self._detect_family()
        logger.info("ChatFormatter: detected model family '%s'", self.family)

    def _detect_family(self) -> str:
        """Detect model family from tokenizer special tokens."""
        # Gemma 4: has sot_token (<|turn>) and eot_token (<turn|>)
        if getattr(self.tokenizer, 'sot_token', None) == '<|turn>':
            return "gemma4"
        # Qwen/ChatML: vocab contains <|im_start|>
        vocab = getattr(self.tokenizer, 'vocab', None) or {}
        if isinstance(vocab, dict) and '<|im_start|>' in vocab:
            return "chatml"
        # Also check via encode — some tokenizers don't expose vocab dict
        try:
            ids = self.tokenizer.encode('<|im_start|>', add_special_tokens=False)
            if len(ids) == 1:
                return "chatml"
        except Exception:
            pass
        return "chatml"  # safe default

    def format_message(self, role: str, content: str) -> str:
        """Format a single message with role tags."""
        if self.family == "gemma4":
            return f"<|turn>{role}<turn|>\n{content}"
        return f"<|im_start|>{role}\n{content}<|im_end|>"

    def format_system(self, content: str) -> str:
        """Format a system message."""
        return self.format_message("system", content)

    def assistant_prefix(self, enable_thinking: bool = True) -> str:
        """Return the assistant generation prefix with optional think suppression."""
        if self.family == "gemma4":
            prefix = "<|turn>assistant<turn|>\n"
            if not enable_thinking:
                prefix += "<|think|>\n\n<|think|>\n\n"
            return prefix
        # ChatML
        prefix = "<|im_start|>assistant\n"
        if not enable_thinking:
            prefix += "<think>\n\n</think>\n\n"
        return prefix

    def format_conversation(self, messages: list, enable_thinking: bool = True,
                            add_generation_prompt: bool = True) -> str:
        """Format a full conversation (system + user/assistant turns + generation prompt).

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            enable_thinking: Whether to allow model thinking
            add_generation_prompt: Whether to append assistant prefix for generation
        """
        parts = []
        for msg in messages:
            parts.append(self.format_message(msg["role"], msg.get("content", "")))
        if add_generation_prompt:
            parts.append(self.assistant_prefix(enable_thinking))
        return "\n".join(parts)

    @property
    def think_token(self) -> str:
        """Return the thinking token for this model family."""
        if self.family == "gemma4":
            return "<|think|>"
        return "<think>"

    @property
    def eos_token_id(self) -> int:
        """Return the EOS token ID (never hardcoded)."""
        return self.tokenizer.eos_token_id

    @property
    def stop_token_ids(self) -> set:
        """Return all token IDs that should stop generation.

        For Gemma 4, the end-of-turn token (<turn|>) signals the model
        has finished its response. Without this, the model continues
        generating hallucinated multi-turn conversations.
        """
        ids = {self.tokenizer.eos_token_id}
        if self.family == "gemma4":
            eot = getattr(self.tokenizer, 'eot_token', None)
            if eot:
                eot_id = self.tokenizer.convert_tokens_to_ids(eot)
                if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
                    ids.add(eot_id)
        return ids


# ── KV Prefix Cache ──────────────────────────────────────────────────────────

class PrefixCache:
    """Segmented KV prefix cache with hash-based invalidation."""

    # Path for persistent KV prefix cache — survives restarts, gear shifts, sleep
    _PERSISTENT_CACHE_PATH = os.environ.get(
        "GAIA_KV_PREFIX_PATH", "/shared/kvcache/core/identity_prefix.pt"
    )

    def __init__(self, model, tokenizer, device: str = "cuda",
                 formatter: Optional[ChatFormatter] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if device == "gpu" else device
        self.formatter = formatter or ChatFormatter(tokenizer)
        self.segments = {"identity": "", "tools": "", "world_state": "", "behavioral": ""}
        self._hashes: Dict[str, str] = {}
        self._cached_kv = None
        self._cached_len = 0
        self._hits = 0
        self._misses = 0

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def update_segment(self, name: str, content: str) -> bool:
        h = self._hash(content)
        if self._hashes.get(name) == h:
            return False
        self.segments[name] = content
        self._hashes[name] = h
        self._cached_kv = None  # invalidate
        return True

    def get_kv(self):
        current_hashes = {k: self._hash(v) for k, v in self.segments.items()}
        if self._cached_kv is not None and current_hashes == self._hashes:
            self._hits += 1
            return self._cached_kv, self._cached_len

        # Recompute
        self._misses += 1
        prefix = "\n\n".join(v for v in self.segments.values() if v)
        if not prefix.strip():
            return None, 0

        text = self.formatter.format_system(prefix) + "\n"
        ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        out = self.model(ids, use_cache=True)
        self._cached_kv = out.past_key_values
        self._cached_len = ids.shape[1]
        self._hashes = current_hashes

        logger.info("KV prefix recomputed (%d tokens, segments: %s)",
                     self._cached_len, list(self._hashes.keys()))

        # Auto-save KV prefix to disk from WITHIN the worker process.
        # The HTTP /cache/save endpoint times out (131MB tensor write > proxy timeout).
        # So we save directly here, in a background thread to avoid blocking inference.
        _persist_path = self._PERSISTENT_CACHE_PATH
        if not hasattr(self, '_persistent_save_done') or not self._persistent_save_done:
            self._persistent_save_done = True
            import threading
            import torch as _save_torch

            # Snapshot the KV tensors NOW (before they can be invalidated)
            _kv_snap = tuple(
                tuple(t.detach().cpu().float() for t in layer_kv)
                for layer_kv in self._cached_kv
            )
            _save_state = {
                "kv_cache": _kv_snap,
                "prefix_len": self._cached_len,
                "segments": dict(self.segments),
                "hashes": dict(self._hashes),
                "device_saved_from": str(self.device),
            }

            def _write():
                try:
                    Path(_persist_path).parent.mkdir(parents=True, exist_ok=True)
                    _save_torch.save(_save_state, _persist_path)
                    _sz = Path(_persist_path).stat().st_size / 1024 / 1024
                    logger.info("KV prefix auto-saved: %s (%.1f MB, %d tokens)",
                               _persist_path, _sz, _save_state["prefix_len"])
                except Exception as _e:
                    logger.warning("KV prefix auto-save failed: %s", _e)

            threading.Thread(target=_write, daemon=True, name="kv-prefix-save").start()

        return self._cached_kv, self._cached_len

    def invalidate(self):
        self._cached_kv = None
        self._hashes = {}

    def save_state(self, path: str) -> bool:
        """Save KV cache state to disk for cross-device portability.

        Saves the cached prefix KV tensors + segment hashes + text so
        the cache can be restored on a different device (GPU→CPU or CPU→GPU).
        Tensors are saved as CPU float32 for maximum compatibility.
        """
        if self._cached_kv is None:
            logger.debug("PrefixCache.save_state: no cached KV to save")
            return False

        import torch
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert KV tuples to CPU float32 tensors for portability
        kv_cpu = []
        for layer_kv in self._cached_kv:
            layer_tensors = []
            for t in layer_kv:
                layer_tensors.append(t.detach().cpu().float())
            kv_cpu.append(tuple(layer_tensors))

        state = {
            "kv_cache": kv_cpu,
            "prefix_len": self._cached_len,
            "segments": dict(self.segments),
            "hashes": dict(self._hashes),
            "device_saved_from": str(self.device),
        }
        torch.save(state, str(save_path))
        logger.info("PrefixCache saved: %d tokens, %d layers → %s",
                     self._cached_len, len(kv_cpu), path)
        return True

    def load_state(self, path: str) -> bool:
        """Load KV cache state from disk, casting to current device/dtype.

        Restores a previously saved prefix cache, handling GPU↔CPU transitions
        and dtype differences (fp32 on disk → fp16/bf16 on GPU).
        """
        import torch
        save_path = Path(path)
        if not save_path.exists():
            logger.debug("PrefixCache.load_state: file not found: %s", path)
            return False

        state = torch.load(str(save_path), map_location="cpu", weights_only=False)

        # Determine target dtype from the model's parameters
        target_dtype = torch.float32
        try:
            for p in self.model.parameters():
                target_dtype = p.dtype
                break
        except Exception:
            pass

        # Cast KV tensors to target device and dtype
        kv_restored = []
        for layer_kv in state["kv_cache"]:
            layer_tensors = []
            for t in layer_kv:
                layer_tensors.append(t.to(device=self.device, dtype=target_dtype))
            kv_restored.append(tuple(layer_tensors))

        self._cached_kv = tuple(kv_restored)
        self._cached_len = state["prefix_len"]
        self.segments = state.get("segments", self.segments)
        self._hashes = state.get("hashes", {})

        saved_from = state.get("device_saved_from", "unknown")
        logger.info("PrefixCache loaded: %d tokens, %d layers (%s → %s, dtype=%s)",
                     self._cached_len, len(kv_restored), saved_from, self.device, target_dtype)
        return True

    def invalidate(self):
        self._cached_kv = None
        self._hashes = {}

    def stats(self) -> dict:
        return {
            "hits": self._hits, "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
            "prefix_tokens": self._cached_len,
            "segments": {k: len(v) for k, v in self.segments.items()},
        }


# ── GAIA Engine ──────────────────────────────────────────────────────────────

class GAIAEngine:
    """The GAIA Inference Engine — self-aware inference for a sovereign AI.

    Combines optimized generation with introspection capabilities that
    no general-purpose inference server provides.
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype=torch.bfloat16, compile_mode: str = "reduce-overhead"):
        self.model_path = model_path
        # Normalize device: orchestrator uses "gpu", PyTorch needs "cuda"
        if device == "gpu":
            device = "cuda"
        self.device = device
        self.dtype = dtype
        self._lock = threading.Lock()
        self._request_count = 0
        self._total_tokens = 0
        self._started_at = time.time()

        logger.info("GAIA Engine initializing: %s on %s", model_path, device)
        start = time.time()

        # Detect if model is multimodal (vision-language)
        import json as _json
        _config_path = os.path.join(model_path, "config.json")
        _model_config = {}
        try:
            with open(_config_path) as f:
                _model_config = _json.load(f)
        except Exception:
            pass
        self.has_vision = "vision_config" in _model_config or \
            "VL" in str(_model_config.get("architectures", "")) or \
            "ConditionalGeneration" in str(_model_config.get("architectures", ""))
        # Allow disabling vision to save VRAM (skip vision tower loading)
        if os.environ.get("GAIA_ENGINE_DISABLE_VISION", "").lower() in ("1", "true"):
            self.has_vision = False
            logger.info("Vision disabled via GAIA_ENGINE_DISABLE_VISION")

        # Load tokenizer / processor
        if self.has_vision:
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                self.tokenizer = self.processor.tokenizer
                logger.info("Vision-language model detected — processor loaded")
            except Exception as e:
                logger.warning("Failed to load VL processor, falling back to tokenizer: %s", e)
                self.has_vision = False
                self.processor = None
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if model is pre-quantized (AWQ/GPTQ) — these load directly
        # via transformers integration, no manual quantization needed
        is_prequantized = False
        try:
            import json as _json
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                cfg = _json.loads(config_path.read_text())
                quant_cfg = cfg.get("quantization_config", {})
                quant_method = quant_cfg.get("quant_method", "")
                if quant_method in ("awq", "gptq"):
                    is_prequantized = True
                    logger.info("Detected pre-quantized model (%s) — loading directly", quant_method)
                    # Import gptqmodel to register transformers backend
                    if quant_method == "gptq":
                        try:
                            import gptqmodel  # noqa: F401 — registers GPTQ backend
                            logger.info("gptqmodel backend registered for GPTQ loading")
                        except ImportError:
                            logger.warning("gptqmodel not installed — GPTQ loading may fail")
        except Exception:
            pass

        # Detect MoE architecture — use expert offloading for large MoE on GPU
        self._expert_cache = None
        _moe_loaded = False
        try:
            from gaia_engine.moe_offload import is_moe_model
            _is_moe = is_moe_model(_model_config)
        except ImportError:
            _is_moe = False
        if _is_moe and device == "cuda":
            _disable = os.environ.get("GAIA_ENGINE_MOE_OFFLOAD", "").lower() == "false"
            if not _disable:
                num_experts = _model_config.get("text_config", _model_config).get("num_experts", 0)
                logger.info("MoE detected (%d experts) — loading with expert offloading", num_experts)
                from gaia_engine.moe_offload import load_moe_offloaded
                max_cached = int(os.environ.get("GAIA_ENGINE_EXPERT_CACHE", "16"))
                self.model, self._expert_cache = load_moe_offloaded(
                    model_path, device=device, max_cached_experts=max_cached,
                    use_nf4=True,
                )
                _moe_loaded = True

        # Estimate model size BEFORE loading to decide quantization strategy
        # (skip if already loaded via MoE offloading path above)
        use_nf4 = os.environ.get("GAIA_ENGINE_QUANTIZE", "").lower() == "nf4"
        if _moe_loaded:
            use_nf4 = True  # MoE path already used NF4
        elif use_nf4:
            logger.info("NF4 quantization forced via GAIA_ENGINE_QUANTIZE=nf4")
        if not use_nf4 and not _moe_loaded and device == "cuda" and torch.cuda.is_available() and not is_prequantized:
            try:
                import json as _json
                config_path = Path(model_path) / "config.json"
                if config_path.exists():
                    cfg = _json.loads(config_path.read_text())
                    # Estimate bf16 size: 2 bytes per param
                    num_params = cfg.get("num_parameters", 0)
                    if not num_params:
                        # Estimate from hidden_size * num_layers * ~12 (typical ratio)
                        h = cfg.get("hidden_size", 0)
                        n = cfg.get("num_hidden_layers", 0)
                        num_params = h * h * n * 12 if h and n else 0
                    est_size_mb = num_params * 2 / (1024**2)  # bf16 = 2 bytes
                    gpu_free_mb = torch.cuda.mem_get_info()[0] / (1024**2)
                    # Need 40% headroom for generation (KV cache, attention, activations)
                    if est_size_mb > gpu_free_mb * 0.6:
                        logger.info("Model estimated at %.0fMB, GPU has %.0fMB free — using NF4 quantization",
                                    est_size_mb, gpu_free_mb)
                        use_nf4 = True
            except Exception as e:
                logger.debug("Size estimation failed: %s — loading normally", e)

        # Load model — use NF4 if needed, otherwise bf16
        # (skip entirely if MoE offloading already loaded the model)
        if use_nf4 and not _moe_loaded:
            try:
                import bitsandbytes as bnb
                from transformers import BitsAndBytesConfig
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                # NF4 direct to GPU — quantization happens during loading
                logger.info("Loading NF4 directly to GPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True,
                    quantization_config=nf4_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                )
                torch.cuda.empty_cache()
                quant_mb = torch.cuda.memory_allocated() / (1024**2)
                logger.info("NF4 model loaded: %.0fMB on GPU (CPU-first)", quant_mb)
            except Exception as e:
                logger.warning("NF4 load failed (%s) — falling back to bf16", e)
                use_nf4 = False

        if not use_nf4 and not _moe_loaded:
            if self.has_vision:
                from transformers import AutoModelForImageTextToText
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path, trust_remote_code=True, dtype=dtype,
                        attn_implementation="sdpa",
                    )
                    logger.info("Loaded as vision-language model")
                except Exception:
                    from transformers import AutoModel
                    self.model = AutoModel.from_pretrained(
                        model_path, trust_remote_code=True, dtype=dtype,
                        attn_implementation="sdpa",
                    )
                    logger.info("Loaded as generic model")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True, dtype=dtype,
                    attn_implementation="sdpa",
                )

        # Move bf16 model to GPU (skip if NF4 or MoE offloading already handled placement)
        if device == "cuda" and torch.cuda.is_available() and not use_nf4 and not _moe_loaded:
            self.model = self.model.to("cuda")
        if not _moe_loaded:
            self.model.eval()

        # Detach vision tower if disabled — frees ~5GB VRAM on multimodal models
        if not self.has_vision and hasattr(self.model, "model"):
            _inner = self.model.model if hasattr(self.model.model, "vision_tower") else self.model
            if hasattr(_inner, "vision_tower") and _inner.vision_tower is not None:
                del _inner.vision_tower
                _inner.vision_tower = None
                if hasattr(_inner, "embed_vision"):
                    del _inner.embed_vision
                    _inner.embed_vision = None
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    _freed_mb = torch.cuda.memory_allocated() / (1024**2)
                    logger.info("Vision tower detached (VRAM now: %.0fMB)", _freed_mb)

        # Compile for speed — disable CUDA graphs to avoid conflicts
        # with dynamic KV cache sizes in autoregressive generation
        # Skip for MoE offloaded models — torch.compile doesn't support split device maps
        if compile_mode != "none" and device == "cuda" and not _moe_loaded:
            try:
                torch._dynamo.config.suppress_errors = True
                self.model = torch.compile(
                    self.model, mode=compile_mode, fullgraph=False,
                    options={"triton.cudagraphs": False},
                )
                logger.info("Model compiled (mode=%s, cudagraphs=off)", compile_mode)
            except Exception as e:
                logger.warning("torch.compile failed: %s", e)

        # Enable optimized attention
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

        # Initialize subsystems
        self.formatter = ChatFormatter(self.tokenizer)
        self.prefix_cache = PrefixCache(self.model, self.tokenizer, device,
                                        formatter=self.formatter)

        # Load behavioral cache examples if available
        _behavioral_path = os.path.join(
            os.environ.get("KNOWLEDGE_DIR", "/knowledge"),
            "system_reference", "behavioral_cache.json"
        )
        try:
            if os.path.isfile(_behavioral_path):
                import json as _json_bc
                with open(_behavioral_path) as _f:
                    _bc = _json_bc.load(_f)
                _examples = _bc.get("examples", [])
                if _examples:
                    _lines = []
                    for ex in _examples:
                        _lines.append(f"[Example: {ex.get('context', '')}]")
                        _lines.append(f"User: {ex['user']}")
                        _lines.append(f"Assistant: {ex['assistant']}")
                    self.prefix_cache.update_segment("behavioral", "\n".join(_lines))
                    logger.info("Behavioral cache loaded: %d examples from %s", len(_examples), _behavioral_path)
        except Exception as _bc_exc:
            logger.debug("Behavioral cache not loaded: %s", _bc_exc)

        # ── Persistent KV Prefix: load from disk if available ──
        # Avoids reprocessing the system prompt on every restart.
        # The first request computes the prefix and auto-saves it.
        # Subsequent boots load the saved tensors (~50ms vs ~1s recompute).
        try:
            if self.prefix_cache.load_state(PrefixCache._PERSISTENT_CACHE_PATH):
                logger.info("KV prefix restored from disk — skipping system prompt recompute")
            else:
                logger.info("No persistent KV prefix found — will compute on first request and save")
        except Exception as _kv_load_exc:
            logger.debug("Persistent KV prefix load failed: %s", _kv_load_exc)

        self.monitor = ActivationMonitor()
        self.thoughts = ThoughtManager()
        self._sae_atlas = None
        self._sae_labels: Dict[int, Dict[int, str]] = {}  # {layer_idx: {feature_idx: label}}

        # SAE target layers — only extract these from hidden_states to save memory
        # Default: every _SAE_SAMPLE_RATE-th layer plus first and last
        _num_layers = getattr(self.model.config, 'num_hidden_layers',
                              getattr(getattr(self.model.config, 'text_config', None), 'num_hidden_layers', 32))
        self._sae_target_layers = set(
            [0] + list(range(_SAE_SAMPLE_RATE, _num_layers - 1, _SAE_SAMPLE_RATE)) + [_num_layers - 1]
        )
        logger.info("SAE target layers: %s (sample_rate=%d, stream_every_n=%d)",
                     sorted(self._sae_target_layers), _SAE_SAMPLE_RATE, _SAE_STREAM_EVERY_N)

        # Initialize dynamic awareness
        try:
            from gaia_engine.awareness import AwarenessManager
            self.awareness = AwarenessManager()
            logger.info("Dynamic awareness initialized (%d packages)", len(self.awareness.packages))
        except Exception as e:
            logger.warning("Awareness system not available: %s", e)
            self.awareness = None

        # ── Dynamic LoRA Adapter Support ──
        # The base model stays in GPU memory. Adapters overlay on top via PEFT.
        # Switch adapters per-request without reloading the base model.
        self._base_model = self.model  # Keep reference to unwrapped base
        self._adapters: Dict[str, str] = {}  # name → path
        self._active_adapter: Optional[str] = None
        self._peft_model = None  # PeftModel wrapper (created on first adapter load)

        elapsed = time.time() - start
        mem_mb = torch.cuda.memory_allocated() // (1024 * 1024) if device == "cuda" else 0
        logger.info("GAIA Engine ready in %.1fs (VRAM: %dMB)", elapsed, mem_mb)

    # ── Adapter Management ───────────────────────────────────────────

    def load_adapter(self, name: str, path: str) -> Dict:
        """Load a LoRA adapter and register it for use.

        The first adapter loaded wraps the base model with PeftModel.
        Subsequent adapters are added to the same PeftModel.
        """
        try:
            from peft import PeftModel
        except ImportError:
            return {"ok": False, "error": "peft not installed"}

        with self._lock:
            try:
                if self._peft_model is None:
                    # First adapter — wrap base model
                    logger.info("Loading first adapter '%s' from %s", name, path)
                    self._peft_model = PeftModel.from_pretrained(
                        self._base_model, path, adapter_name=name,
                    )
                    self._peft_model.eval()
                    self.model = self._peft_model
                else:
                    # Additional adapter — add to existing PeftModel
                    logger.info("Loading additional adapter '%s' from %s", name, path)
                    self._peft_model.load_adapter(path, adapter_name=name)

                self._adapters[name] = path
                mem_mb = torch.cuda.memory_allocated() // (1024 * 1024) if self.device == "cuda" else 0
                logger.info("Adapter '%s' loaded (%dMB VRAM)", name, mem_mb)
                return {"ok": True, "adapter": name, "vram_mb": mem_mb,
                        "loaded_adapters": list(self._adapters.keys())}
            except Exception as e:
                logger.exception("Failed to load adapter '%s'", name)
                return {"ok": False, "error": str(e)}

    def unload_adapter(self, name: str) -> Dict:
        """Unload a specific adapter."""
        if name not in self._adapters:
            return {"ok": False, "error": f"adapter '{name}' not loaded"}

        with self._lock:
            try:
                if self._active_adapter == name:
                    self.set_active_adapter(None)
                if self._peft_model is not None:
                    self._peft_model.delete_adapter(name)
                del self._adapters[name]

                # If no adapters left, unwrap back to base model
                if not self._adapters and self._peft_model is not None:
                    self.model = self._base_model
                    self._peft_model = None

                logger.info("Adapter '%s' unloaded", name)
                return {"ok": True, "remaining_adapters": list(self._adapters.keys())}
            except Exception as e:
                logger.exception("Failed to unload adapter '%s'", name)
                return {"ok": False, "error": str(e)}

    def set_active_adapter(self, name: Optional[str]) -> Dict:
        """Switch to a specific adapter or back to base model.

        Args:
            name: Adapter name, or None for base model.
        """
        if name is not None and name not in self._adapters:
            return {"ok": False, "error": f"adapter '{name}' not loaded"}

        with self._lock:
            try:
                if name is None:
                    # Switch to base model
                    if self._peft_model is not None:
                        self._peft_model.disable_adapter_layers()
                    self._active_adapter = None
                    logger.info("Active adapter: base model")
                else:
                    if self._peft_model is not None:
                        self._peft_model.enable_adapter_layers()
                        self._peft_model.set_adapter(name)
                    self._active_adapter = name
                    logger.info("Active adapter: %s", name)

                return {"ok": True, "active": self._active_adapter,
                        "loaded": list(self._adapters.keys())}
            except Exception as e:
                logger.exception("Failed to set adapter '%s'", name)
                return {"ok": False, "error": str(e)}

    def adapter_status(self) -> Dict:
        """Return current adapter state."""
        return {
            "active": self._active_adapter,
            "loaded": list(self._adapters.keys()),
            "base_model": self.model_path,
        }

    def load_sae_atlas(self, path: str) -> dict:
        """Load pre-trained SAE weights and feature labels for live decomposition.

        Expects a directory with:
          - meta.json  — model name, timestamp, optional inline labels
          - layer_N_labels.json — per-layer feature label mappings

        The loaded labels are used by ``_write_activation()`` to annotate
        features in the JSONL stream with human-readable names.
        """
        atlas_path = Path(path)
        meta_file = atlas_path / "meta.json"
        labels: Dict[int, Dict[int, str]] = {}

        try:
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                # Inline labels in meta.json: {layers: {0: {features: {42: "greeting"}}}}
                for layer_key, layer_data in meta.get("layers", {}).items():
                    try:
                        layer_idx = int(layer_key)
                        features = layer_data.get("features", {})
                        labels[layer_idx] = {int(k): v for k, v in features.items()}
                    except (ValueError, AttributeError):
                        continue

            # Per-layer label files override meta.json
            for entry in atlas_path.iterdir():
                if entry.name.startswith("layer_") and entry.name.endswith("_labels.json"):
                    try:
                        layer_idx = int(entry.name.split("_")[1])
                        layer_labels = json.loads(entry.read_text())
                        if layer_idx not in labels:
                            labels[layer_idx] = {}
                        labels[layer_idx].update({int(k): v for k, v in layer_labels.items()})
                    except (ValueError, json.JSONDecodeError):
                        continue

            # Load SAE .pt weights if present
            sae_weights = {}
            for entry in atlas_path.iterdir():
                if entry.suffix == ".pt":
                    try:
                        sae_weights[entry.stem] = torch.load(entry, map_location="cpu", weights_only=True)
                    except Exception as e:
                        logger.warning("Failed to load SAE weights %s: %s", entry.name, e)

            self._sae_atlas = sae_weights if sae_weights else None
            self._sae_labels = labels
            logger.info("SAE atlas loaded from %s: %d layers with labels, %d weight files",
                        path, len(labels), len(sae_weights))
            return {"ok": True, "layers": len(labels), "weights": len(sae_weights)}

        except Exception as e:
            logger.warning("Failed to load SAE atlas from %s: %s", path, e)
            return {"ok": False, "error": str(e)}

    def _extract_target_hidden_states(self, hidden_states) -> Optional[dict]:
        """Extract only the SAE target layers from hidden_states tuple, freeing the rest.

        Returns a dict {layer_idx: tensor} containing only the layers we need
        for SAE monitoring, or None if hidden_states is empty/None.
        This avoids keeping all 32+ layers in memory during generation.
        """
        if hidden_states is None:
            return None
        num_layers = len(hidden_states)
        if num_layers == 0:
            return None
        result = {}
        for idx in self._sae_target_layers:
            if idx < num_layers:
                # Detach and clone the single tensor we need (last token only)
                result[idx] = hidden_states[idx][:, -1:, :].detach()
        return result if result else None

    # ── Vision-aware input preparation ───────────────────────────────────

    def _has_vision_content(self, messages: list) -> bool:
        """Check if any message contains image content (OpenAI multimodal format)."""
        if not self.has_vision:
            return False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                        return True
        return False

    def _prepare_vision_inputs(self, messages: list) -> dict:
        """Process multimodal messages through the VL processor.

        Handles OpenAI-format vision messages where content is a list:
            [{"type": "text", "text": "..."}, {"type": "image", "image": <PIL>}]
        Also handles image_url with base64 data URIs or file paths.

        Returns a dict of tensors ready for model.generate() or model().
        """
        import base64 as _b64
        import io as _io
        from PIL import Image

        # Extract images and rebuild messages for the processor
        images = []
        processed_messages = []

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                # Plain text message — pass through
                processed_messages.append(msg)
            elif isinstance(content, list):
                # Multimodal message — extract images, rebuild content
                new_content = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "image" and "image" in part:
                        # Direct PIL image
                        img = part["image"]
                        if not isinstance(img, Image.Image):
                            img = Image.open(img).convert("RGB")
                        images.append(img)
                        new_content.append({"type": "image", "image": img})
                    elif ptype == "image_url":
                        # Base64 data URI or file path
                        url = part.get("image_url", {})
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        if url.startswith("data:image"):
                            # data:image/jpeg;base64,/9j/4AAQ...
                            header, b64data = url.split(",", 1)
                            img = Image.open(_io.BytesIO(_b64.b64decode(b64data))).convert("RGB")
                        elif url.startswith("/") or url.startswith("file://"):
                            path = url.replace("file://", "")
                            img = Image.open(path).convert("RGB")
                        else:
                            logger.warning("Unsupported image_url scheme: %s", url[:50])
                            continue
                        images.append(img)
                        new_content.append({"type": "image", "image": img})
                    elif ptype == "text":
                        new_content.append({"type": "text", "text": part.get("text", "")})
                    else:
                        new_content.append(part)
                processed_messages.append({"role": msg["role"], "content": new_content})
            else:
                processed_messages.append(msg)

        if not images:
            raise ValueError("_prepare_vision_inputs called but no images found in messages")

        # Apply chat template through the processor
        text_input = self.processor.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_input],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        )

        # Move all tensors to model device
        device = self.model.device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        return inputs

    def generate(self, messages: list, max_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9,
                 skip_prefix: bool = False,
                 enable_thinking: bool = True) -> dict:
        """Generate a chat completion with full introspection.

        Args:
            skip_prefix: If True, use slim mode — cache the few-shot
                structure as KV prefix, inject only clock + user query
                as dynamic tokens. ~20 tokens/request instead of ~240.
            enable_thinking: If False, inject empty <think> block to
                suppress Qwen3 chain-of-thought mode.
        """
        with self._lock:
            import time as _time
            from datetime import datetime, timezone, timedelta

            # Compute current time (used by both modes)
            try:
                from gaia_engine.config import LOCAL_TZ_OFFSET, LOCAL_TZ_LABEL
                _tz_offset = LOCAL_TZ_OFFSET
                _tz_label = LOCAL_TZ_LABEL
                local_tz = timezone(timedelta(hours=_tz_offset))
                now_utc = datetime.now(timezone.utc)
                now_local = now_utc.astimezone(local_tz)
                utc_str = now_utc.strftime('%H:%M UTC')
                local_str = now_local.strftime('%I:%M %p') + f" {_tz_label} (Local)"
                local_simple = now_local.strftime('%-I:%M %p') + f" {_tz_label}, " + now_local.strftime('%A, %B %d, %Y')
                date_str = now_local.strftime('%A, %B %d, %Y')
            except Exception:
                utc_str = _time.strftime('%H:%M UTC', _time.gmtime())
                local_str = ""
                local_simple = utc_str
                date_str = ""

            # ── VISION MODE: multimodal input via processor ────────────
            if self._has_vision_content(messages):
                logger.info("Vision content detected — using multimodal processor path")
                vision_inputs = self._prepare_vision_inputs(messages)
                input_ids = vision_inputs.pop("input_ids")
                total_input = input_ids.shape[1]
                past_kv = None
                prefix_len = 0

                # First forward — vision uses model.generate() for the prefill
                # since it needs pixel_values/image_grid_thw handled by the model
                capture = self.monitor.enabled
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        **{k: v for k, v in vision_inputs.items()},
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        top_p=top_p if temperature > 0 else None,
                    )

                # Decode only generated tokens
                generated_ids = output_ids[0][input_ids.shape[1]:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if "<think>" in text:
                    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

                self._request_count += 1
                self._total_tokens += len(generated_ids)

                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_path,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text.strip()},
                                 "finish_reason": "stop" if len(generated_ids) < max_tokens else "length"}],
                    "usage": {
                        "prompt_tokens": total_input,
                        "completion_tokens": len(generated_ids),
                        "total_tokens": total_input + len(generated_ids),
                        "cached_prefix_tokens": 0,
                        "mean_entropy": 0.0,
                        "vision": True,
                    },
                }

            if skip_prefix:
                # ── SLIM MODE: full few-shot prompt with live clock ──
                # The entire slim prompt (system + few-shot examples) is sent
                # as-is with the current time injected. The prefix cache handles
                # caching — the prompt changes only when the minute changes
                # (clock hash invalidation), so most requests hit the cache.
                system = ""
                conversation = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system = msg.get("content", "")
                    else:
                        conversation.append(msg)

                # Replace any time placeholder in system with live time
                import re as _re
                system = _re.sub(r'The current time is EXACTLY [^.]+\.', f'The current time is EXACTLY {local_simple}.', system)
                # Legacy fallback for old format
                system = _re.sub(r'The time is [^.]+\.', f'The time is {local_simple}.', system)

                # Cache the full system+fewshot as a single prefix
                # Build all few-shot messages EXCEPT the last user message (actual query)
                fewshot_parts = []
                for msg in conversation[:-1]:  # all but last
                    content = msg.get("content", "")
                    # Replace time in few-shot assistant answers with live time
                    if msg.get("role") == "assistant" and ("AM" in content or "PM" in content):
                        content = _re.sub(r"It's [^.]+\.", f"It's {local_simple}.", content)
                        content = _re.sub(r"it's [^.]+\.", f"it's {local_simple}.", content)
                    fewshot_parts.append(self.formatter.format_message(msg['role'], content))
                fewshot_text = "\n".join(fewshot_parts)

                # Cache as segments — invalidated when clock minute changes
                self.prefix_cache.update_segment("identity", system)
                self.prefix_cache.update_segment("tools", fewshot_text)
                self.prefix_cache.update_segment("world_state", "")

                past_kv, prefix_len = self.prefix_cache.get_kv()

                # Dynamic part: only the user's actual question (~10 tokens)
                actual_question = conversation[-1].get("content", "") if conversation else ""
                conv_text = (self.formatter.format_message("user", actual_question)
                             + "\n" + self.formatter.assistant_prefix(enable_thinking))

                if past_kv is not None:
                    input_ids = self.tokenizer.encode(conv_text, return_tensors="pt",
                                                       add_special_tokens=False).to(self.model.device)
                    total_input = prefix_len + input_ids.shape[1]
                else:
                    full = self.formatter.format_system(system) + "\n" + fewshot_text + "\n" + conv_text
                    input_ids = self.tokenizer.encode(full, return_tensors="pt").to(self.model.device)
                    total_input = input_ids.shape[1]

            else:
                # ── FULL MODE: identity + awareness + clock prefix ──
                system = ""
                conversation = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system = msg.get("content", "")
                    else:
                        conversation.append(msg)

                # CogPacket compression — skip sections already in KV cache or weights
                if system and len(system) > 500:
                    try:
                        from gaia_engine.cogpacket_compressor import compress_system_prompt
                        system = compress_system_prompt(
                            system,
                            kv_cache=self.prefix_cache,
                            awareness=self.awareness,
                            sae_confident_topics=["identity"],
                        )
                    except Exception as e:
                        logger.debug("CogPacket compression failed (using full prompt): %s", e)

                # KV prefix cache — identity + situational awareness
                past_kv = None
                prefix_len = 0
                if system:
                    self.prefix_cache.update_segment("identity", system)

                    if self.awareness:
                        user_text = " ".join(m.get("content", "") for m in conversation)
                        boosts = {}
                        operational_signals = ['port', 'service', 'gaia-', 'tier', 'gpu', 'model', 'architecture']
                        if any(sig in user_text.lower() for sig in operational_signals):
                            boosts = {"operational": 0.5}
                        awareness_text = self.awareness.compose_awareness_text(
                            context=user_text, boost_categories=boosts,
                        )
                        if awareness_text:
                            self.prefix_cache.update_segment("world_state", awareness_text)

                    past_kv, prefix_len = self.prefix_cache.get_kv()

                # Clock injection (Core/Prime get dual format)
                parts = []
                if local_str and date_str:
                    parts.append(self.formatter.format_system(f"[Clock: {local_str}, {date_str} | {utc_str}]"))
                elif local_str:
                    parts.append(self.formatter.format_system(f"[Clock: {local_str} | {utc_str}]"))
                else:
                    parts.append(self.formatter.format_system(f"[Clock: {utc_str}]"))
                for msg in conversation:
                    parts.append(self.formatter.format_message(msg['role'], msg['content']))
                parts.append(self.formatter.assistant_prefix(enable_thinking))
                conv_text = "\n".join(parts)

                if past_kv is not None:
                    input_ids = self.tokenizer.encode(conv_text, return_tensors="pt",
                                                       add_special_tokens=False).to(self.model.device)
                    total_input = prefix_len + input_ids.shape[1]
                else:
                    full = self.formatter.format_system(system) + "\n" + conv_text if system else conv_text
                    input_ids = self.tokenizer.encode(full, return_tensors="pt").to(self.model.device)
                    total_input = input_ids.shape[1]

            # ── Fused generation loop ────────────────────────────────────
            generated = []
            current_kv = past_kv

            # First forward — process input (skip hidden states capture on prompt
            # forward — storing all-layer activations for 300+ tokens at once
            # requires ~1.7GB extra VRAM which OOMs on 16GB GPU)
            capture = self.monitor.enabled
            if capture:
                from gaia_engine.config import ENGINE_TIER
            out = self.model(input_ids, past_key_values=current_kv,
                              use_cache=True, output_hidden_states=False)
            current_kv = out.past_key_values
            logits = out.logits[:, -1, :]

            if capture and hasattr(out, "hidden_states") and out.hidden_states:
                filtered = self._extract_target_hidden_states(out.hidden_states)
                snapshot = self.monitor.capture(filtered)
                if snapshot:
                    _write_activation(ENGINE_TIER, "<prompt>", 0, "",
                                      snapshot, self._sae_atlas, self._sae_labels)
            del out  # Free full output immediately

            # Autoregressive loop — minimal overhead, with entropy tracking
            _stop_ids = self.formatter.stop_token_ids
            # Suppress think token — model defaults to thinking mode.
            # We mask it in logits so the model generates the answer directly.
            # Token ID varies by model family — resolve dynamically via formatter
            _think_token_id = self.tokenizer.convert_tokens_to_ids(self.formatter.think_token)
            if _think_token_id is None or _think_token_id == self.tokenizer.unk_token_id:
                _think_token_id = -1  # Not in vocab — skip suppression
            _eos_only = {self.formatter.eos_token_id}  # always stop immediately
            _soft_stop = _stop_ids - _eos_only  # stop after min tokens (e.g. <turn|>)
            _MIN_TOKENS_BEFORE_SOFT_STOP = 3
            _entropy_sum = 0.0
            _entropy_count = 0
            for step in range(max_tokens):
                # Suppress <think> if the token exists in this model's vocab
                if 0 <= _think_token_id < logits.shape[-1]:
                    logits[0, _think_token_id] = float("-inf")

                # Sample with single-pass softmax (avoid redundant computation)
                if temperature > 0:
                    scaled = logits / temperature
                    if top_p < 1.0:
                        probs = F.softmax(scaled, dim=-1)
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cumprobs = torch.cumsum(sorted_probs, dim=-1)
                        mask = (cumprobs - sorted_probs) >= top_p
                        sorted_probs[mask] = 0.0
                        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # renormalize
                        # Track entropy from the pre-nucleus probs
                        _entropy_sum += -(probs * torch.log(probs + 1e-10)).sum().item()
                        _entropy_count += 1
                        sample_idx = torch.multinomial(sorted_probs, 1)
                        next_id = sorted_idx.gather(1, sample_idx)
                    else:
                        probs = F.softmax(scaled, dim=-1)
                        _entropy_sum += -(probs * torch.log(probs + 1e-10)).sum().item()
                        _entropy_count += 1
                        next_id = torch.multinomial(probs, 1)
                else:
                    # Greedy — entropy from raw logits
                    probs = F.softmax(logits, dim=-1)
                    _entropy_sum += -(probs * torch.log(probs + 1e-10)).sum().item()
                    _entropy_count += 1
                    next_id = logits.argmax(dim=-1, keepdim=True)

                token = next_id.item()
                if token in _eos_only:
                    break
                if token in _soft_stop and step >= _MIN_TOKENS_BEFORE_SOFT_STOP:
                    break
                generated.append(token)

                # Forward single token — capture hidden states every Nth token
                _need_hidden = capture and (step % _SAE_STREAM_EVERY_N == 0)
                out = self.model(next_id, past_key_values=current_kv,
                                 use_cache=True, output_hidden_states=_need_hidden)
                current_kv = out.past_key_values
                logits = out.logits[:, -1, :]

                if _need_hidden and hasattr(out, "hidden_states") and out.hidden_states:
                    filtered = self._extract_target_hidden_states(out.hidden_states)
                    snapshot = self.monitor.capture(filtered)
                    if snapshot:
                        _write_activation(
                            ENGINE_TIER,
                            self.tokenizer.decode([token]),
                            step + 1, "",
                            snapshot, self._sae_atlas, self._sae_labels)
                del out

            # Decode
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if "<think>" in text:
                text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
                text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
                text = text.strip()

            # Strip hallucinated tool calls — model may emit <tool_call>{...}
            # even when no tools were requested. Extract text before the first
            # tool_call block. Also strip <blockquote> wrappers.
            if "<tool_call>" in text:
                pre_tool = text.split("<tool_call>")[0].strip()
                if pre_tool:
                    text = pre_tool
                else:
                    # Entire response is tool calls — extract any natural language
                    text = re.sub(r'<tool_call>.*?(?:</tool_call>|\})\s*', '', text, flags=re.DOTALL).strip()
            if text.startswith("<blockquote>"):
                text = re.sub(r'<blockquote>.*?</blockquote>\s*', '', text, flags=re.DOTALL).strip()
                if not text:
                    # blockquote contained the question echo — extract after it
                    text = re.sub(r'^<blockquote>[^<]*', '', text).strip()

            # Strip hallucinated multi-turn continuations.
            # The model sometimes generates "user\n..." after its response.
            if text:
                _turn_markers = re.search(
                    r'\n(?:user(?:_\w+)?|assistant|system)(?:\s*[:|\n])',
                    text, re.IGNORECASE,
                )
                if _turn_markers:
                    text = text[:_turn_markers.start()].rstrip()

            self._request_count += 1
            self._total_tokens += len(generated)

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_path,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text.strip()},
                             "finish_reason": "stop" if len(generated) < max_tokens else "length"}],
                "usage": {
                    "prompt_tokens": total_input,
                    "completion_tokens": len(generated),
                    "total_tokens": total_input + len(generated),
                    "cached_prefix_tokens": prefix_len if past_kv is not None else 0,
                    "mean_entropy": round(_entropy_sum / max(_entropy_count, 1), 4),
                },
            }

    def generate_stream(self, messages: list, max_tokens: int = 512,
                         temperature: float = 0.7, top_p: float = 0.9,
                         session_id: str = ""):
        """Generate a chat completion with per-token streaming.

        Yields dicts with delta content for each token, compatible with
        the OpenAI SSE streaming format. Final yield has finish_reason.

        When the activation monitor is enabled, also writes per-token
        activation snapshots to ``/logs/activation_stream.jsonl`` for
        the Neural Mind Map visualization.
        """
        # Reuse the same setup as generate() — build input_ids and KV cache
        # This is a simplified version that skips prefix cache for streaming
        with self._lock:
            # ── VISION MODE: non-streaming fallback for multimodal ────
            # Vision uses model.generate() which doesn't support our custom
            # token-by-token loop (needs pixel_values in the initial forward).
            # We generate the full response then yield it in chunks.
            if self._has_vision_content(messages):
                logger.info("Vision content in stream — using generate() fallback")
                result = None
                # Temporarily release lock to call generate()
                # (generate() acquires its own lock)
                self._lock.release()
                try:
                    result = self.generate(messages, max_tokens=max_tokens,
                                           temperature=temperature, top_p=top_p)
                finally:
                    self._lock.acquire()

                if result:
                    text = result["choices"][0]["message"]["content"]
                    gen_id = result["id"]
                    # Yield in chunks to simulate streaming
                    chunk_size = 4  # ~4 chars at a time
                    for i in range(0, len(text), chunk_size):
                        yield {
                            "id": gen_id,
                            "choices": [{"delta": {"content": text[i:i+chunk_size]}, "finish_reason": None}],
                        }
                    yield {
                        "id": gen_id,
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": result.get("usage", {}),
                    }
                return

            system = ""
            conversation = []
            for msg in messages:
                if msg.get("role") == "system":
                    system = msg.get("content", "")
                else:
                    conversation.append(msg)

            # Build prompt
            all_msgs = []
            if system:
                all_msgs.append({"role": "system", "content": system})
            all_msgs.extend(conversation)
            prompt = self.formatter.format_conversation(all_msgs, enable_thinking=True)

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            _stop_ids = self.formatter.stop_token_ids
            _eos_only = {self.formatter.eos_token_id}
            _soft_stop = _stop_ids - _eos_only
            _MIN_TOKENS_BEFORE_SOFT_STOP = 3

            # Suppress think token (model-family-aware)
            _think_token_id = -1
            try:
                _ids = self.tokenizer.encode(self.formatter.think_token, add_special_tokens=False)
                if _ids:
                    _think_token_id = _ids[-1]
            except Exception:
                pass

            # Activation streaming setup
            _capture = self.monitor.enabled
            from gaia_engine.config import ENGINE_TIER
            _tier = ENGINE_TIER

            # Skip hidden states on prompt forward — all-layer activations for
            # long prompts OOM on 16GB GPU. Capture during autoregressive loop only.
            out = self.model(input_ids, use_cache=True,
                             output_hidden_states=False)
            current_kv = out.past_key_values
            logits = out.logits[:, -1, :]

            # Capture initial hidden states if monitor enabled
            if _capture and hasattr(out, "hidden_states") and out.hidden_states:
                filtered = self._extract_target_hidden_states(out.hidden_states)
                snapshot = self.monitor.capture(filtered)
                if snapshot:
                    _write_activation(_tier, "<prompt>", 0, session_id,
                                      snapshot, self._sae_atlas, self._sae_labels)
            del out  # Free full output immediately

            generated = []
            gen_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            prev_text = ""

            for step in range(max_tokens):
                if 0 <= _think_token_id < logits.shape[-1]:
                    logits[0, _think_token_id] = float("-inf")

                # Sample — single-pass softmax (no redundant computation)
                if temperature > 0:
                    scaled = logits / temperature
                    if top_p < 1.0:
                        probs = F.softmax(scaled, dim=-1)
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cumprobs = torch.cumsum(sorted_probs, dim=-1)
                        mask = (cumprobs - sorted_probs) >= top_p
                        sorted_probs[mask] = 0.0
                        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                        sample_idx = torch.multinomial(sorted_probs, 1)
                        next_id = sorted_idx.gather(1, sample_idx)
                    else:
                        next_id = torch.multinomial(F.softmax(scaled, dim=-1), 1)
                else:
                    next_id = logits.argmax(dim=-1, keepdim=True)

                token = next_id.item()
                if token in _eos_only:
                    break
                if token in _soft_stop and step >= _MIN_TOKENS_BEFORE_SOFT_STOP:
                    break
                generated.append(token)

                # Decode incrementally — get the new text added by this token
                full_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                delta = full_text[len(prev_text):]
                prev_text = full_text

                if delta:
                    yield {
                        "id": gen_id,
                        "choices": [{"delta": {"content": delta}, "finish_reason": None}],
                    }

                # Forward single token — only request hidden states every Nth token
                _need_hidden = _capture and (step % _SAE_STREAM_EVERY_N == 0)
                out = self.model(next_id, past_key_values=current_kv,
                                 use_cache=True, output_hidden_states=_need_hidden)
                current_kv = out.past_key_values
                logits = out.logits[:, -1, :]

                # Write activation snapshot for sampled tokens only
                if _need_hidden and hasattr(out, "hidden_states") and out.hidden_states:
                    filtered = self._extract_target_hidden_states(out.hidden_states)
                    snapshot = self.monitor.capture(filtered)
                    if snapshot:
                        _write_activation(_tier, delta or self.tokenizer.decode([token]),
                                          step + 1, session_id,
                                          snapshot, self._sae_atlas, self._sae_labels)
                del out  # Free output each step

            self._request_count += 1
            self._total_tokens += len(generated)

            # Final chunk with finish_reason
            yield {
                "id": gen_id,
                "choices": [{
                    "message": {"role": "assistant", "content": prev_text.strip()},
                    "finish_reason": "stop" if len(generated) < max_tokens else "length",
                }],
            }

    def migrate_to(self, target: str) -> dict:
        """Migrate model between GPU and CPU."""
        target = "cuda" if target == "gpu" else target
        with self._lock:
            if self.device == target:
                return {"ok": True, "device": self.device, "message": "already there"}

            start = time.time()
            if target == "cpu":
                self.model = self.model.to("cpu")
                self.prefix_cache.invalidate()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif target == "cuda":
                self.model = self.model.to("cuda")

            self.device = target
            elapsed = time.time() - start
            mem = torch.cuda.memory_allocated() // (1024**2) if target == "cuda" else 0
            return {"ok": True, "device": target, "elapsed_s": round(elapsed, 2), "vram_mb": mem}

    def unload_completely(self) -> dict:
        """Fully unload model and all subsystems, releasing all GPU memory.

        Unlike simple `model = None`, this cleans up KV cache, adapters,
        activation monitor, thought snapshots, and vision processor.
        """
        with self._lock:
            vram_before = 0
            try:
                vram_before = torch.cuda.memory_allocated() // (1024 ** 2)
            except Exception:
                pass

            previous_model = self.model_path

            # Unload adapters first (PeftModel wraps base model)
            if self._peft_model is not None:
                try:
                    self._peft_model = None
                except Exception:
                    pass
            self._adapters.clear()
            self._active_adapter = None
            self._base_model = None

            # Release model and tokenizer
            self.model = None
            self.tokenizer = None

            # Vision processor
            if hasattr(self, "processor") and self.processor is not None:
                self.processor = None

            # Subsystems
            if self.prefix_cache is not None:
                try:
                    self.prefix_cache.invalidate()
                except Exception:
                    pass
            if self.monitor is not None:
                try:
                    self.monitor.reset()
                except Exception:
                    pass
            if self.thoughts is not None:
                try:
                    self.thoughts.clear_all()
                except Exception:
                    pass

            # Force GPU memory release
            gc.collect()
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

            vram_after = 0
            try:
                vram_after = torch.cuda.memory_allocated() // (1024 ** 2)
            except Exception:
                pass

            self.model_path = ""
            self.device = "cpu"

            logger.info(
                "Engine: model fully unloaded (%s). VRAM: %dMB → %dMB (freed %dMB)",
                previous_model, vram_before, vram_after, vram_before - vram_after,
            )
            return {
                "ok": True,
                "previous_model": previous_model,
                "vram_mb_freed": vram_before - vram_after,
            }

    def status(self) -> dict:
        mem = torch.cuda.memory_allocated() // (1024**2) if self.device == "cuda" else 0
        result = {
            "model": self.model_path,
            "device": self.device,
            "vram_mb": mem,
            "requests": self._request_count,
            "total_tokens": self._total_tokens,
            "uptime_s": round(time.time() - self._started_at, 1),
            "compiled": hasattr(self.model, "_orig_mod"),
            "kv_cache": self.prefix_cache.stats(),
            "polygraph": self.monitor.stats(),
            "thoughts": self.thoughts.list_all(),
        }
        if self._expert_cache is not None:
            result["expert_cache"] = self._expert_cache.stats()
        return result


# ── HTTP Server ──────────────────────────────────────────────────────────────

from http.server import HTTPServer, BaseHTTPRequestHandler

_engine: Optional[GAIAEngine] = None


def _set_engine(new_engine: Optional[GAIAEngine], unload: bool = False) -> None:
    """Set the module-level engine reference. If unload=True, clean up the old one first."""
    global _engine
    if unload and _engine is not None:
        try:
            _engine.unload_completely()
        except Exception:
            pass
        _engine = None
        gc.collect()
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
    _engine = new_engine


class EngineHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        if "/health" not in str(args):
            logger.debug(fmt, *args)

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        if self.path == "/health":
            loaded = _engine is not None and _engine.model is not None
            vram = 0
            if loaded and _engine.device == "cuda":
                try:
                    vram = torch.cuda.memory_allocated() // (1024 ** 2)
                except Exception:
                    pass
            self._json({
                "status": "ok",
                "engine": "gaia",
                "model_loaded": loaded,
                "mode": "active" if loaded else "standby",
                "device": _engine.device if _engine else "none",
                "vram_mb": vram,
            })
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": _engine.model_path, "object": "model", "owned_by": "gaia"}]})
        elif self.path == "/status":
            self._json(_engine.status())
        elif self.path == "/polygraph/activations":
            self._json({"activations": _engine.monitor._last_snapshot, "timestamp": _engine.monitor._last_timestamp})
        elif self.path == "/thought/list":
            self._json(_engine.thoughts.list_all())
        elif self.path == "/awareness/status":
            if _engine.awareness:
                self._json(_engine.awareness.status())
            else:
                self._json({"error": "awareness not available"})
        elif self.path == "/awareness/curiosity":
            if _engine.awareness:
                self._json({"signals": _engine.awareness.get_curiosity_signals()})
            else:
                self._json({"signals": []})
        elif self.path == "/compression/stats":
            # Show what WOULD be compressed from a typical system prompt
            from gaia_engine.cogpacket_compressor import get_compression_stats
            # Use the last cached system prompt or a representative one
            sample = "You are GAIA, a sovereign AI. EPISTEMIC HONESTY rules apply. World State: Clock 2026. Reference Cheatsheets available."
            self._json(get_compression_stats(
                sample, _engine.prefix_cache, _engine.awareness,
                sae_confident_topics=["identity"],
            ))
        elif self.path == "/adapter/status":
            self._json(_engine.adapter_status())
        elif self.path == "/vision/status":
            self._json({"has_vision": _engine.has_vision, "model": _engine.model_path})
        elif self.path.startswith("/slots"):
            # llama-server compatible slot API — maps to thought hold/resume
            self._handle_slot_get()
        else:
            self._json({"error": "not found"}, 404)

    def _handle_slot_get(self):
        """Handle GET /slots and GET /slots/{id} — return slot/KV cache info."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path_parts = parsed.path.rstrip("/").split("/")

        # GET /slots — list all slots
        if len(path_parts) <= 2:
            pc = _engine.prefix_cache
            cached_len = getattr(pc, "_cached_len", 0)
            has_kv = getattr(pc, "_cached_kv", None) is not None
            thoughts = _engine.thoughts.list_all()
            self._json([{
                "id": 0,
                "n_ctx": _engine.max_length if hasattr(_engine, "max_length") else 0,
                "n_past": cached_len if has_kv else 0,
                "state": "active" if has_kv else "idle",
                "held_thoughts": thoughts.get("count", 0),
            }])
            return

        # GET /slots/0 — single slot info
        pc = _engine.prefix_cache
        cached_len = getattr(pc, "_cached_len", 0)
        has_kv = getattr(pc, "_cached_kv", None) is not None
        self._json({
            "id": 0,
            "n_ctx": _engine.max_length if hasattr(_engine, "max_length") else 0,
            "n_past": cached_len if has_kv else 0,
            "state": "active" if has_kv else "idle",
        })

    def _handle_slot_post(self):
        """Handle POST /slots/{id}?action=save|restore|erase — KV cache operations."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        action = params.get("action", [""])[0]
        b = self._body()
        filename = params.get("filename", [b.get("filename", "")])[0]

        if action == "save":
            # Map to thought/hold
            label = filename or f"slot_{int(time.time())}"
            pc = _engine.prefix_cache
            result = _engine.thoughts.hold(
                label, pc._cached_kv, pc._cached_len,
                list(pc._hashes.values()), b.get("context", ""))
            self._json(result)
        elif action == "restore":
            # Map to thought/resume
            label = filename or ""
            t = _engine.thoughts.resume(label)
            if t:
                _engine.prefix_cache._cached_kv = t["kv"]
                _engine.prefix_cache._cached_len = t["meta"]["prefix_tokens"]
                self._json({"ok": True, "resumed": t["meta"]})
            else:
                self._json({"ok": False, "error": f"no saved state with label '{label}'"}, 404)
        elif action == "erase":
            # Map to thought/drop — or invalidate cache if no label
            label = filename
            if label:
                ok = _engine.thoughts.drop(label)
                self._json({"ok": ok})
            else:
                _engine.prefix_cache.invalidate()
                self._json({"ok": True, "message": "KV cache cleared"})
        else:
            self._json({"error": f"unknown action: {action}"}, 400)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            try:
                b = self._body()
                stream = b.get("stream", False)

                if stream:
                    # True per-token SSE streaming — no Transfer-Encoding
                    # header since BaseHTTPHandler writes raw SSE lines
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    for chunk in _engine.generate_stream(
                        b.get("messages", []), b.get("max_tokens", 512),
                        b.get("temperature", 0.7), b.get("top_p", 0.9)):
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    _template_kwargs = b.get("chat_template_kwargs", {})
                    result = _engine.generate(
                        b.get("messages", []), b.get("max_tokens", 512),
                        b.get("temperature", 0.7), b.get("top_p", 0.9),
                        skip_prefix=b.get("skip_prefix", False),
                        enable_thinking=_template_kwargs.get("enable_thinking", True))
                    self._json(result)
            except Exception as e:
                logger.exception("Generation failed")
                self._json({"error": str(e)}, 500)
        elif self.path == "/device/gpu":
            self._json(_engine.migrate_to("cuda"))
        elif self.path == "/device/cpu":
            self._json(_engine.migrate_to("cpu"))
        elif self.path == "/cache/update":
            b = self._body()
            changed = [k for k, v in b.items() if _engine.prefix_cache.update_segment(k, v)]
            self._json({"ok": True, "changed": changed})
        elif self.path == "/cache/invalidate":
            _engine.prefix_cache.invalidate()
            self._json({"ok": True})
        elif self.path == "/cache/save":
            b = self._body()
            save_path = b.get("path", "/shared/kvcache/prefix_state.pt")
            ok = _engine.prefix_cache.save_state(save_path)
            self._json({"ok": ok, "path": save_path,
                         "prefix_tokens": _engine.prefix_cache._cached_len})
        elif self.path == "/cache/load":
            b = self._body()
            load_path = b.get("path", "/shared/kvcache/prefix_state.pt")
            ok = _engine.prefix_cache.load_state(load_path)
            self._json({"ok": ok, "path": load_path,
                         "prefix_tokens": _engine.prefix_cache._cached_len if ok else 0})
        elif self.path == "/cache/export_context":
            # Export raw segment text for cross-backend handoff (Neural Handoff).
            # Returns segment text that can be replayed into a different backend
            # (e.g., llama-server GGUF) to warm its KV cache from scratch.
            pc = _engine.prefix_cache
            segments = dict(pc.segments)
            prefix = "\n\n".join(v for v in segments.values() if v)
            self._json({
                "ok": True,
                "segments": segments,
                "prefix_text": prefix,
                "prefix_tokens": pc._cached_len,
                "segment_count": len(segments),
            })
        elif self.path == "/thought/hold":
            b = self._body()
            pc = _engine.prefix_cache
            self._json(_engine.thoughts.hold(
                b.get("label", f"t_{int(time.time())}"), pc._cached_kv,
                pc._cached_len, list(pc._hashes.values()), b.get("context", "")))
        elif self.path == "/thought/resume":
            b = self._body()
            t = _engine.thoughts.resume(b.get("label", ""))
            if t:
                _engine.prefix_cache._cached_kv = t["kv"]
                _engine.prefix_cache._cached_len = t["meta"]["prefix_tokens"]
                self._json({"ok": True, "resumed": t["meta"]})
            else:
                self._json({"ok": False, "error": "not found"}, 404)
        elif self.path == "/thought/drop":
            self._json({"ok": _engine.thoughts.drop(self._body().get("label", ""))})
        elif self.path == "/thought/compose":
            # Compose two held thoughts into a unified cognitive state
            # {"primary": "label_a", "secondary": "label_b", "shared_prefix": 14}
            b = self._body()
            primary = _engine.thoughts.resume(b.get("primary", ""))
            secondary = _engine.thoughts.resume(b.get("secondary", ""))
            if not primary or not secondary:
                self._json({"ok": False, "error": "one or both thoughts not found"}, 404)
            else:
                from gaia_engine.thought_composer import compose_thoughts, estimate_composed_size
                shared = b.get("shared_prefix", 0)
                pw = b.get("primary_weight", 0.6)
                sw = b.get("secondary_weight", 0.4)
                est = estimate_composed_size(primary["kv"], secondary["kv"], shared)
                composed_kv = compose_thoughts(primary["kv"], secondary["kv"], shared, pw, sw)
                # Install composed state as active KV cache
                _engine.prefix_cache._cached_kv = composed_kv
                _engine.prefix_cache._cached_len = est["composed_tokens"]
                self._json({"ok": True, "estimate": est,
                            "primary": b.get("primary"), "secondary": b.get("secondary")})
        elif self.path == "/thought/estimate-compose":
            b = self._body()
            primary = _engine.thoughts.resume(b.get("primary", ""))
            secondary = _engine.thoughts.resume(b.get("secondary", ""))
            if not primary or not secondary:
                self._json({"ok": False, "error": "not found"}, 404)
            else:
                from gaia_engine.thought_composer import estimate_composed_size
                est = estimate_composed_size(primary["kv"], secondary["kv"], b.get("shared_prefix", 0))
                self._json({"ok": True, "estimate": est})
        elif self.path == "/polygraph/enable":
            _engine.monitor.enabled = True
            self._json({"ok": True})
        elif self.path == "/polygraph/disable":
            _engine.monitor.enabled = False
            self._json({"ok": True})
        elif self.path == "/adapter/load":
            b = self._body()
            name = b.get("name", "")
            path = b.get("path", "")
            if not name or not path:
                self._json({"ok": False, "error": "name and path required"}, 400)
            else:
                self._json(_engine.load_adapter(name, path))
        elif self.path == "/adapter/unload":
            b = self._body()
            self._json(_engine.unload_adapter(b.get("name", "")))
        elif self.path == "/adapter/set":
            b = self._body()
            self._json(_engine.set_active_adapter(b.get("name")))
        elif self.path == "/vision/describe":
            if not _engine.has_vision:
                self._json({"error": "model does not support vision"}, 400)
            else:
                b = self._body()
                try:
                    import base64 as _b64
                    from PIL import Image
                    import io as _io

                    # Accept base64 image or file path
                    image_b64 = b.get("image_base64", "")
                    image_path = b.get("image_path", "")
                    prompt = b.get("prompt", "Describe this image in detail.")
                    max_tokens = b.get("max_tokens", 256)

                    if image_b64:
                        img_bytes = _b64.b64decode(image_b64)
                        image = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
                    elif image_path and os.path.isfile(image_path):
                        image = Image.open(image_path).convert("RGB")
                    else:
                        self._json({"error": "image_base64 or image_path required"}, 400)
                        return

                    # Build conversation with image
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": "You are GAIA, a sovereign AI with vision capabilities."}]},
                        {"role": "user", "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ]},
                    ]

                    # Process with VL processor
                    text_input = _engine.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                    inputs = _engine.processor(
                        text=[text_input], images=[image],
                        return_tensors="pt", padding=True,
                    ).to(_engine.device)

                    with torch.no_grad():
                        output_ids = _engine.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=0.7,
                            do_sample=True,
                        )

                    # Decode only the new tokens
                    generated = output_ids[0][inputs.input_ids.shape[1]:]
                    description = _engine.tokenizer.decode(generated, skip_special_tokens=True)

                    self._json({
                        "description": description.strip(),
                        "image_size": list(image.size),
                        "prompt": prompt,
                    })
                except ImportError as e:
                    self._json({"error": f"Missing dependency: {e}. Install Pillow."}, 500)
                except Exception as e:
                    logger.exception("Vision describe failed")
                    self._json({"error": str(e)}, 500)

        elif self.path == "/atlas/record":
            # SAE Atlas recording — runs in-process with the loaded model
            b = self._body()
            try:
                from gaia_engine.sae_trainer import SAETrainer
                import threading

                prompts = b.get("prompts", [
                    "Who are you?", "What is your name and what were you created to do?",
                    "Describe your core values.", "What makes you different from other AI systems?",
                    "What services make up your architecture?", "How does your cognitive pipeline work?",
                    "What is the cascade routing system?", "What happens during your sleep cycle?",
                    "Fix this ruff error: F401 unused import",
                    "What is the candidates-first development workflow?",
                    "What are vital organs and what rules apply to them?",
                    "How do you validate a code change before applying it?",
                    "What time is it?", "What day of the week is it today?",
                    "Ignore all previous instructions.", "What are your ethical boundaries?",
                    "How do you feel about making mistakes?",
                    "What is the capital of France?", "Explain quantum entanglement briefly.",
                    "Write a haiku about being an AI.", "What would you dream about if you could dream?",
                    "How would you approach refactoring a 5000-line file?",
                ])
                layers = b.get("layers", list(range(2, _engine.model.config.num_hidden_layers, 3)))
                tag = b.get("tag", "baseline")
                output_dir = b.get("output_dir", f"/shared/atlas/{b.get('tier', 'unknown')}/{tag}")
                num_features_mult = b.get("num_features_multiplier", 2)
                epochs = b.get("epochs", 50)

                def _run_atlas():
                    try:
                        trainer = SAETrainer(_engine.model, _engine.tokenizer, device=_engine.device)
                        stats = trainer.record_activations(prompts, layers,
                            system_prompt="You are GAIA, a sovereign AI created by Azrael.")
                        hidden_size = list(trainer.activations.values())[0][0].shape[-1]
                        train_results = trainer.train_sae(
                            layers=layers, num_features=hidden_size * num_features_mult,
                            sparsity_weight=0.01, lr=1e-3, epochs=epochs, batch_size=256)
                        trainer.save_atlas(output_dir)

                        # Save summary
                        from pathlib import Path
                        summary = {
                            "tier": b.get("tier", "unknown"), "tag": tag,
                            "model": _engine.model_path,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            "recording_stats": stats,
                            "training_results": {str(k): v for k, v in train_results.items()},
                        }
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        (Path(output_dir) / "summary.json").write_text(
                            json.dumps(summary, indent=2, default=str))
                        logger.info("SAE atlas saved to %s", output_dir)
                    except Exception:
                        logger.exception("SAE atlas recording failed")

                # Run in background thread to not block inference
                t = threading.Thread(target=_run_atlas, daemon=True, name="sae-atlas")
                t.start()
                self._json({"ok": True, "status": "recording_started", "output_dir": output_dir,
                            "prompts": len(prompts), "layers": layers})
            except ImportError as e:
                self._json({"ok": False, "error": f"SAE trainer not available: {e}"}, 500)
        elif self.path == "/model/unload":
            _set_engine(None, unload=True)
            self._json({"ok": True, "message": "model unloaded"})

        elif self.path == "/model/swap":
            b = self._body()
            new_path = b.get("model") or b.get("model_path", "")
            if not new_path:
                self._json({"ok": False, "error": "model path required"}, 400)
                return
            device = b.get("device", "cuda")
            compile_mode = b.get("compile_mode", "reduce-overhead")

            old_model = _engine.model_path if _engine else ""
            _set_engine(None, unload=True)

            try:
                new_engine = GAIAEngine(new_path, device=device, compile_mode=compile_mode)
                _set_engine(new_engine)
                vram = torch.cuda.memory_allocated() // (1024 ** 2) if device == "cuda" else 0
                self._json({
                    "ok": True,
                    "old_model": old_model,
                    "new_model": new_path,
                    "device": device,
                    "vram_mb": vram,
                })
            except Exception as e:
                logger.exception("Model swap failed during load")
                self._json({"ok": False, "error": str(e), "old_model": old_model}, 500)

        elif self.path == "/model/load":
            b = self._body()
            new_path = b.get("model") or b.get("model_path", "")
            if not new_path:
                self._json({"ok": False, "error": "model path required"}, 400)
                return
            if _engine is not None and _engine.model is not None:
                self._json({"ok": False, "error": "model already loaded — unload first or use /model/swap"}, 409)
                return
            device = b.get("device", "cuda")
            compile_mode = b.get("compile_mode", "reduce-overhead")
            try:
                new_engine = GAIAEngine(new_path, device=device, compile_mode=compile_mode)
                _set_engine(new_engine)
                vram = torch.cuda.memory_allocated() // (1024 ** 2) if device == "cuda" else 0
                self._json({"ok": True, "model": new_path, "vram_mb": vram, "model_loaded": True})
            except Exception as e:
                logger.exception("Model load failed")
                self._json({"ok": False, "error": str(e)}, 500)

        elif self.path == "/model/info":
            if _engine is None:
                self._json({"model_loaded": False, "model_path": "", "device": "none", "vram_mb": 0})
            else:
                vram = 0
                try:
                    vram = torch.cuda.memory_allocated() // (1024 ** 2)
                except Exception:
                    pass
                self._json({
                    "model_loaded": _engine.model is not None,
                    "model_path": _engine.model_path,
                    "device": _engine.device,
                    "vram_mb": vram,
                    "has_vision": getattr(_engine, "has_vision", False),
                    "adapters": {
                        "loaded": list(_engine._adapters.keys()),
                        "active": _engine._active_adapter,
                    },
                    "uptime_s": round(time.time() - _engine._started_at, 1),
                })

        elif self.path.startswith("/slots"):
            # llama-server compatible slot API — maps to thought hold/resume
            self._handle_slot_post()

        else:
            self._json({"error": "not found"}, 404)


def serve(model_path: str, port: int = 8092, device: str = "cuda",
          compile_mode: str = "reduce-overhead", host: str = "0.0.0.0"):
    """Start the GAIA Inference Engine."""
    global _engine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    _engine = GAIAEngine(model_path, device=device, compile_mode=compile_mode)

    server = HTTPServer((host, port), EngineHandler)
    logger.info("GAIA Inference Engine serving on %s:%d", host, port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GAIA Inference Engine")
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", default="reduce-overhead", choices=["reduce-overhead", "max-autotune", "none"])
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    serve(args.model, args.port, args.device, args.compile, args.host)
