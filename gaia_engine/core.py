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

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("GAIA.Engine")

# ── Static KV Cache ──────────────────────────────────────────────────────────

class StaticKVCache:
    """Pre-allocated KV cache to avoid dynamic allocation per token.

    Standard generation allocates new tensors for each token's key/value
    states. This pre-allocates a fixed buffer and writes into it,
    eliminating allocation overhead during generation.
    """

    def __init__(self, model, max_seq_len: int = 4096, device: str = "cuda"):
        self.max_seq_len = max_seq_len
        self.device = device
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

    def capture(self, hidden_states: tuple, sample_every: int = 4) -> dict:
        if not self.enabled or hidden_states is None:
            return {}

        self._captures += 1
        self._last_timestamp = time.time()
        num_layers = len(hidden_states)

        snapshot = {}
        sample_layers = [0] + list(range(sample_every, num_layers - 1, sample_every)) + [num_layers - 1]

        for idx in sample_layers:
            if idx >= num_layers:
                continue
            last_token = hidden_states[idx][0, -1, :]
            snapshot[f"layer_{idx}"] = {
                "mean": float(last_token.mean()),
                "std": float(last_token.std()),
                "l2_norm": float(last_token.norm()),
                "top_5_indices": last_token.abs().topk(5).indices.tolist(),
                "top_5_values": [round(float(v), 4) for v in last_token.abs().topk(5).values],
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


# ── KV Prefix Cache ──────────────────────────────────────────────────────────

class PrefixCache:
    """Segmented KV prefix cache with hash-based invalidation."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.segments = {"identity": "", "tools": "", "world_state": ""}
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

        text = f"<|im_start|>system\n{prefix}<|im_end|>\n"
        ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(ids, use_cache=True)
            self._cached_kv = out.past_key_values
            self._cached_len = ids.shape[1]
            self._hashes = current_hashes

        logger.info("KV prefix recomputed (%d tokens, segments: %s)",
                     self._cached_len, list(self._hashes.keys()))
        return self._cached_kv, self._cached_len

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

        # Estimate model size BEFORE loading to decide quantization strategy
        # Check config.json for num_params or estimate from file sizes
        use_nf4 = False
        if device == "cuda" and torch.cuda.is_available() and not is_prequantized:
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
        if use_nf4:
            try:
                import bitsandbytes as bnb
                from transformers import BitsAndBytesConfig
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                # CPU-first NF4: load to CPU, quantize there, then dispatch to GPU
                # This never holds bf16 weights on GPU — only the 4-bit version
                logger.info("Loading NF4 to CPU first (avoids bf16 GPU peak)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True,
                    quantization_config=nf4_config,
                    device_map={"": "cpu"},
                    low_cpu_mem_usage=True,
                )
                # Move quantized model to GPU via accelerate
                from accelerate import dispatch_model, infer_auto_device_map
                device_map = infer_auto_device_map(self.model, max_memory={0: f"{int(gpu_free_mb * 0.8)}MB", "cpu": "32GB"})
                self.model = dispatch_model(self.model, device_map)
                quant_mb = torch.cuda.memory_allocated() / (1024**2)
                logger.info("NF4 model loaded: %.0fMB on GPU (CPU-first)", quant_mb)
            except Exception as e:
                logger.warning("NF4 load failed (%s) — falling back to bf16", e)
                use_nf4 = False

        if not use_nf4:
            if self.has_vision:
                from transformers import AutoModelForImageTextToText
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path, trust_remote_code=True, dtype=dtype,
                    )
                    logger.info("Loaded as vision-language model")
                except Exception:
                    from transformers import AutoModel
                    self.model = AutoModel.from_pretrained(
                        model_path, trust_remote_code=True, dtype=dtype,
                    )
                    logger.info("Loaded as generic model")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True, dtype=dtype,
                )

        # Move bf16 model to GPU (skip if NF4 already loaded on GPU)
        if device == "cuda" and torch.cuda.is_available() and not use_nf4:
            self.model = self.model.to("cuda")
        self.model.eval()

        # Compile for speed — disable CUDA graphs to avoid conflicts
        # with dynamic KV cache sizes in autoregressive generation
        if compile_mode != "none" and device == "cuda":
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
        self.prefix_cache = PrefixCache(self.model, self.tokenizer, device)
        self.monitor = ActivationMonitor()
        self.thoughts = ThoughtManager()

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

    def generate(self, messages: list, max_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9,
                 skip_prefix: bool = False) -> dict:
        """Generate a chat completion with full introspection.

        Args:
            skip_prefix: If True, use slim mode — cache the few-shot
                structure as KV prefix, inject only clock + user query
                as dynamic tokens. ~20 tokens/request instead of ~240.
        """
        with self._lock:
            import time as _time
            from datetime import datetime, timezone, timedelta

            # Compute current time (used by both modes)
            try:
                _tz_offset = int(os.environ.get("GAIA_LOCAL_TZ_OFFSET", "-7"))
                _tz_label = os.environ.get("GAIA_LOCAL_TZ_LABEL", "PDT")
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
                    fewshot_parts.append(f"<|im_start|>{msg['role']}\n{content}<|im_end|>")
                fewshot_text = "\n".join(fewshot_parts)

                # Cache as segments — invalidated when clock minute changes
                self.prefix_cache.update_segment("identity", system)
                self.prefix_cache.update_segment("tools", fewshot_text)
                self.prefix_cache.update_segment("world_state", "")

                past_kv, prefix_len = self.prefix_cache.get_kv()

                # Dynamic part: only the user's actual question (~10 tokens)
                actual_question = conversation[-1].get("content", "") if conversation else ""
                conv_text = f"<|im_start|>user\n{actual_question}<|im_end|>\n<|im_start|>assistant\n"

                if past_kv is not None:
                    input_ids = self.tokenizer.encode(conv_text, return_tensors="pt",
                                                       add_special_tokens=False).to(self.model.device)
                    total_input = prefix_len + input_ids.shape[1]
                else:
                    full = f"<|im_start|>system\n{system}<|im_end|>\n{fewshot_text}\n{conv_text}"
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
                    parts.append(f"<|im_start|>system\n[Clock: {local_str}, {date_str} | {utc_str}]<|im_end|>")
                elif local_str:
                    parts.append(f"<|im_start|>system\n[Clock: {local_str} | {utc_str}]<|im_end|>")
                else:
                    parts.append(f"<|im_start|>system\n[Clock: {utc_str}]<|im_end|>")
                for msg in conversation:
                    parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
                parts.append("<|im_start|>assistant\n")
                conv_text = "\n".join(parts)

                if past_kv is not None:
                    input_ids = self.tokenizer.encode(conv_text, return_tensors="pt",
                                                       add_special_tokens=False).to(self.model.device)
                    total_input = prefix_len + input_ids.shape[1]
                else:
                    full = f"<|im_start|>system\n{system}<|im_end|>\n{conv_text}" if system else conv_text
                    input_ids = self.tokenizer.encode(full, return_tensors="pt").to(self.model.device)
                    total_input = input_ids.shape[1]

            # ── Fused generation loop ────────────────────────────────────
            generated = []
            current_kv = past_kv

            # First forward — process input + capture activations
            capture = self.monitor.enabled
            with torch.no_grad():
                out = self.model(input_ids, past_key_values=current_kv,
                                  use_cache=True, output_hidden_states=capture)
            current_kv = out.past_key_values
            logits = out.logits[:, -1, :]

            if capture and hasattr(out, "hidden_states") and out.hidden_states:
                self.monitor.capture(out.hidden_states)

            # Autoregressive loop — minimal overhead, with entropy tracking
            eos_id = self.tokenizer.eos_token_id
            # Suppress <think> token — Qwen3.5 defaults to thinking mode.
            # We mask it in logits so the model generates the answer directly.
            # Token ID varies by model family — resolve dynamically
            _think_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
            if _think_token_id is None or _think_token_id == self.tokenizer.unk_token_id:
                _think_token_id = -1  # Not in vocab — skip suppression
            _entropy_sum = 0.0
            _entropy_count = 0
            for step in range(max_tokens):
                # Suppress <think> if the token exists in this model's vocab
                if 0 <= _think_token_id < logits.shape[-1]:
                    logits[0, _think_token_id] = float("-inf")

                # Track per-token entropy (uncertainty signal)
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-10)
                token_entropy = -(probs * log_probs).sum().item()
                _entropy_sum += token_entropy
                _entropy_count += 1

                # Sample
                if temperature > 0:
                    scaled = logits / temperature
                    if top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        mask = (cumprobs - F.softmax(sorted_logits, dim=-1)) >= top_p
                        sorted_logits[mask] = float("-inf")
                        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                    next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
                else:
                    next_id = logits.argmax(dim=-1, keepdim=True)

                token = next_id.item()
                if token == eos_id:
                    break
                generated.append(token)

                # Forward single token (no hidden states capture for speed)
                with torch.no_grad():
                    out = self.model(next_id, past_key_values=current_kv, use_cache=True)
                current_kv = out.past_key_values
                logits = out.logits[:, -1, :]

            # Decode
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if "<think>" in text:
                text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

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

    def migrate_to(self, target: str) -> dict:
        """Migrate model between GPU and CPU."""
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
        return {
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
            self._json({
                "status": "ok",
                "engine": "gaia",
                "model_loaded": loaded,
                "mode": "active" if loaded else "standby",
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
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            try:
                b = self._body()
                stream = b.get("stream", False)
                result = _engine.generate(
                    b.get("messages", []), b.get("max_tokens", 512),
                    b.get("temperature", 0.7), b.get("top_p", 0.9),
                    skip_prefix=b.get("skip_prefix", False))

                if stream:
                    # SSE format — compatible with vllm_remote_model._stream_chat()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    # Send content as a single SSE chunk (true token streaming is future work)
                    chunk = json.dumps({
                        "choices": [{"delta": {"content": content}, "finish_reason": None}]
                    })
                    self.wfile.write(f"data: {chunk}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
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
