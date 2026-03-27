"""
gaia_engine/cpp/backend.py — GaiaCppBackendAdapter

Wraps LlamaCppBackend (C++ pybind11 extension) with:
  - Token streaming via direct write_fn callback (no daemon thread)
  - Hidden state → snapshot format compatible with _write_activation()
  - LoRA adapter load/unload matching GAIAEngine.load_adapter() interface
  - Activation stream write (same JSONL path as core.py)

Used by manager.py as the 'cpp' backend when gaia_cpp extension is available.
Falls back gracefully if the extension is missing (non-Prime tiers).
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import numpy as np

logger = logging.getLogger("GAIA.CppBackend")

try:
    from gaia_engine.cpp.gaia_cpp import LlamaCppBackend  # compiled .so
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    LlamaCppBackend = None  # type: ignore[assignment,misc]


def is_available() -> bool:
    return _CPP_AVAILABLE


from gaia_engine.config import SAE_SAMPLE_RATE as _SAE_SAMPLE_RATE
from gaia_engine.config import ACTIVATION_STREAM_PATH as _ACTIVATION_STREAM_PATH


def _snapshot_from_hidden_states(hidden_states: dict) -> dict:
    """
    Convert C++ hidden states {layer_idx: np.ndarray(n_embd,)}
    into the snapshot format that _write_activation() / SAE expect:
      {"layer_N": {"mean": float, "std": float, "l2_norm": float,
                   "top_5_indices": [int, ...], "top_5_values": [float, ...]}}
    """
    snapshot = {}
    for layer_idx, arr in hidden_states.items():
        if arr is None or arr.size == 0:
            continue
        abs_arr = np.abs(arr)
        n = len(abs_arr)
        k = min(5, n)
        top_k_idx = np.argpartition(abs_arr, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(abs_arr[top_k_idx])[::-1]]
        snapshot[f"layer_{layer_idx}"] = {
            "mean":          float(arr.mean()),
            "std":           float(arr.std()),
            "l2_norm":       float(np.linalg.norm(arr)),
            "top_5_indices": top_k_idx.tolist(),
            "top_5_values":  [round(float(abs_arr[i]), 4) for i in top_k_idx],
        }
    return snapshot


def _write_activation(tier: str, token: str, token_idx: int,
                       session_id: str, snapshot: dict) -> None:
    """Write activation JSONL matching core.py's _write_activation() format."""
    if not snapshot:
        return

    features = []
    for layer_key, layer_data in snapshot.items():
        try:
            layer_idx = int(layer_key.split("_")[1])
        except (IndexError, ValueError):
            continue
        for idx, val in zip(layer_data.get("top_5_indices", []),
                             layer_data.get("top_5_values", [])):
            features.append({
                "idx":      int(idx),
                "strength": float(val),
                "label":    f"neuron_{idx}",
                "layer":    layer_idx,
            })

    features.sort(key=lambda f: f["strength"], reverse=True)
    features = features[:10]

    line = json.dumps({
        "ts":         datetime.now(timezone.utc).isoformat(),
        "tier":       tier,
        "token":      token,
        "token_idx":  token_idx,
        "session_id": session_id or "",
        "features":   features,
    })

    try:
        with open(_ACTIVATION_STREAM_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


class GaiaCppBackendAdapter:
    """
    In-process GGUF backend using gaia_cpp C++ extension.

    Provides HTTP-server-like methods that manager.py calls instead of
    proxying to a llama-server subprocess. Same external behavior, zero
    subprocess overhead, full hidden state access.

    Activation stream: writes to /logs/activation_stream.jsonl every
    GAIA_SAE_SAMPLE_RATE tokens during generation (same file as core.py).
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        n_threads: int = 8,
        capture_layers: Optional[List[int]] = None,
        tier: str = "prime",
    ):
        if not _CPP_AVAILABLE:
            raise RuntimeError("gaia_cpp C++ extension not available in this environment")

        self.model_path = model_path
        self.tier = tier

        logger.info(
            "Loading GGUF via gaia_cpp: %s (gpu_layers=%d, ctx=%d, threads=%d)",
            model_path, n_gpu_layers, n_ctx, n_threads,
        )

        # Create C++ backend with empty capture layers initially
        self._backend = LlamaCppBackend(
            model_path,
            n_gpu_layers,
            [],          # resolved below after knowing n_layer
            n_ctx,
            n_threads,
        )

        # Resolve capture layers based on actual model depth
        n_layers = self._backend.n_layer()
        if capture_layers is not None:
            resolved = capture_layers
        else:
            resolved = (
                [0]
                + list(range(_SAE_SAMPLE_RATE, n_layers - 1, _SAE_SAMPLE_RATE))
                + [n_layers - 1]
            )
        self._backend.set_capture_layers(resolved)
        self.capture_layers = resolved

        logger.info(
            "gaia_cpp ready: %d layers, embd=%d, capture=%s",
            n_layers, self._backend.n_embd(), resolved,
        )

        self._request_lock = threading.Lock()  # single active request guard
        self._lora_path: Optional[str] = None

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        return {
            "status":    "ok",
            "backend":   "cpp",
            "model":     self.model_path,
            "n_layer":   self._backend.n_layer(),
            "n_embd":    self._backend.n_embd(),
            "n_vocab":   self._backend.n_vocab(),
            "has_gpu":   self._backend.has_gpu(),
            "tier":      self.tier,
            "lora_active": self._backend.active_adapter_count(),
            "lora_path":   self._lora_path,
        }

    # ── Streaming generation ──────────────────────────────────────────────────

    def stream_to_writer(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        session_id: str = "",
        write_fn=None,
    ) -> None:
        """
        Stream SSE directly to write_fn.  No daemon thread, no generator.

        Called from the HTTP handler thread (which has a valid GIL state) so
        pybind11's py::gil_scoped_release works correctly on Python 3.11.

        This is the primary streaming path used by _proxy_stream_cpp().
        Writes activation JSONL after generation completes.
        """
        prompt = self._build_prompt(messages)
        gen_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        def token_callback(delta: str) -> None:
            if write_fn is None:
                return
            chunk = {
                "id":      gen_id,
                "object":  "chat.completion.chunk",
                "choices": [{"delta": {"content": delta}, "finish_reason": None}],
            }
            write_fn(f"data: {json.dumps(chunk)}\n\n".encode())

        with self._request_lock:
            result = self._backend.generate_stream(
                prompt,
                token_callback,
                max_tokens,
                temperature,
                top_p,
                0,      # top_k
                bool(self.capture_layers),
            )

        # Write activation log from final result (hidden states of last token)
        if result.hidden_states:
            snapshot = _snapshot_from_hidden_states(result.hidden_states)
            _write_activation(
                tier=self.tier,
                token=result.text[-1] if result.text else "",
                token_idx=result.completion_tokens - 1,
                session_id=session_id,
                snapshot=snapshot,
            )

        # Final SSE chunk with full message + usage
        if write_fn is not None:
            final = {
                "id":      gen_id,
                "object":  "chat.completion.chunk",
                "choices": [{
                    "delta":         {},
                    "finish_reason": "stop",
                    "message":       {"role": "assistant", "content": result.text.strip()},
                }],
                "usage": {
                    "prompt_tokens":     result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
            }
            write_fn(f"data: {json.dumps(final)}\n\n".encode())
            write_fn(b"data: [DONE]\n\n")

    def generate_stream_sse(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        session_id: str = "",
    ) -> Iterator[bytes]:
        """
        Generator that yields SSE bytes.  Buffers all tokens first then yields.
        Use stream_to_writer() for the HTTP handler path (avoids daemon threads).
        """
        chunks: list = []
        self.stream_to_writer(
            messages, max_tokens, temperature, top_p, session_id,
            write_fn=chunks.append,
        )
        yield from chunks

    def generate_json(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """Non-streaming generation, returns OpenAI-compatible JSON dict."""
        prompt = self._build_prompt(messages)

        with self._request_lock:
            result = self._backend.generate(
                prompt, max_tokens, temperature, top_p, 0,
                bool(self.capture_layers),
            )

        return {
            "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object":  "chat.completion",
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": result.text.strip()},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
        }

    # ── Adapter management ────────────────────────────────────────────────────

    def load_adapter(self, adapter_path: str, scale: float = 1.0) -> bool:
        """Load a GGUF LoRA adapter. Returns True on success."""
        success = self._backend.load_lora(adapter_path, scale)
        if success:
            self._lora_path = adapter_path
            logger.info("Loaded LoRA adapter: %s (scale=%.2f)", adapter_path, scale)
        else:
            logger.warning("Failed to load LoRA adapter: %s", adapter_path)
        return success

    def unload_adapter(self) -> None:
        """Unload all LoRA adapters."""
        self._backend.clear_lora()
        self._lora_path = None
        logger.info("LoRA adapters cleared")

    # ── Prompt formatting ─────────────────────────────────────────────────────

    def _build_prompt(self, messages: list) -> str:
        """
        Build ChatML-formatted prompt for Qwen3/Qwen2 models.
        Matches the template llama-server uses for these model families.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
