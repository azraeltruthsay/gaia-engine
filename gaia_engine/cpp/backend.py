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
import re
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



# Native thought (Gemma 4's channel, Qwen3's <think>) -> reasoning_content.
from gaia_engine.thoughts import split_thought_text as _split_thought_text  # noqa: E402


def _media_bytes(part: dict) -> Optional[bytes]:
    """File bytes of an OpenAI-style image/audio content part, or None if the
    part isn't media. Accepts image_url / audio_url ({"url": ...} or a bare
    string; a data: URI or a local path, as the HF engine path does) and
    input_audio ({"data": base64, "format": ...})."""
    import base64
    ptype = part.get("type", "")
    if ptype == "input_audio":
        return base64.b64decode(part.get("input_audio", {}).get("data", ""))
    if ptype not in ("image_url", "audio_url"):
        return None
    url = part.get(ptype, "")
    if isinstance(url, dict):
        url = url.get("url", "")
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        if ";base64" not in head:
            raise ValueError("data URI is not base64")
        return base64.b64decode(data)
    if url.startswith("/") and os.path.isfile(url):
        with open(url, "rb") as f:
            return f.read()
    raise ValueError(f"unsupported media url: {url[:40]}")


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
        prompt_format: str = "house",
        n_threads_batch: int = -1,
    ):
        from gaia_engine.prompt_format import normalize
        self.prompt_format = normalize(prompt_format)
        if not _CPP_AVAILABLE:
            raise RuntimeError("gaia_cpp C++ extension not available in this environment")

        self.model_path = model_path
        self.tier = tier

        logger.info(
            "Loading GGUF via gaia_cpp: %s (gpu_layers=%d, ctx=%d, threads=%d, batch threads=%s)",
            model_path, n_gpu_layers, n_ctx, n_threads,
            n_threads_batch if n_threads_batch > 0 else "same",
        )

        # Create C++ backend with empty capture layers initially. ojwf: the
        # prompt-read thread count is separate since the b11804 build; an
        # older extension takes five arguments and reads with n_threads.
        try:
            self._backend = LlamaCppBackend(
                model_path, n_gpu_layers, [], n_ctx, n_threads, n_threads_batch)
        except TypeError:
            self._backend = LlamaCppBackend(model_path, n_gpu_layers, [], n_ctx, n_threads)

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
        self._kv_last: dict = {}  # last kv_stats read, served while a request runs
        self._lora_path: Optional[str] = None

        # d12t: KV reuse between requests (prefix reuse + in-memory
        # snapshots). GAIA_CPP_PREFIX_REUSE=0 restores the old stateless
        # behaviour; GAIA_CPP_KV_SNAPSHOTS sets the snapshot slots;
        # GAIA_CPP_KV_BASELINE names a saved state to load at start.
        self._kv_control = hasattr(self._backend, "kv_stats")  # older .so: no KV API
        if self._kv_control:
            reuse = os.environ.get("GAIA_CPP_PREFIX_REUSE", "1") != "0"
            self._backend.set_prefix_reuse(reuse)
            self._backend.set_snapshot_slots(int(os.environ.get("GAIA_CPP_KV_SNAPSHOTS", "2")))
            baseline = os.environ.get("GAIA_CPP_KV_BASELINE", "")
            if reuse and baseline and os.path.exists(baseline):
                n = self._backend.load_state(baseline)
                if n:
                    logger.info("gaia_cpp: KV baseline loaded from %s (%d tokens)", baseline, n)
                else:
                    logger.warning("gaia_cpp: could not load KV baseline %s; starting cold", baseline)
            logger.info("gaia_cpp: prefix reuse %s, %d snapshot slot(s)",
                        "on" if reuse else "off", self._backend.kv_stats()["snapshot_slots"])
        else:
            logger.info("gaia_cpp: extension predates KV control (d12t); every request reads its full prompt")

        # frlp: images and audio. GAIA_CPP_MMPROJ names the projector GGUF
        # exported from the same weights as the model; GAIA_CPP_IMAGE_TOKENS
        # pins every image to that many tokens (Gemma 4 12B: 280, as its HF
        # processor does; 0 = the projector's own default).
        self._media_marker = ""
        mmproj = os.environ.get("GAIA_CPP_MMPROJ", "")
        if mmproj:
            if not hasattr(self._backend, "load_mmproj"):
                logger.warning("gaia_cpp: GAIA_CPP_MMPROJ set, but this extension predates media support")
            elif not os.path.exists(mmproj):
                logger.warning("gaia_cpp: projector %s not found; text only", mmproj)
            else:
                err = self._backend.load_mmproj(mmproj, int(os.environ.get("GAIA_CPP_IMAGE_TOKENS", "0")))
                if err:
                    logger.warning("gaia_cpp: projector not loaded (%s); text only", err)
                else:
                    self._media_marker = self._backend.media_marker()
                    info = self._backend.media_info()
                    logger.info("gaia_cpp: projector %s (vision=%s, audio=%s)",
                                mmproj, info["vision"], info["audio"])

        # ojwf: MTP speculative decoding. GAIA_CPP_MTP_DRAFT names a draft
        # "assistant" GGUF made for this model (Gemma 4 12B:
        # google/gemma-4-12B-it-assistant); GAIA_CPP_MTP_N is the draft length.
        # Output follows the same distribution as without it; only speed changes.
        self._mtp_last: dict = {}
        draft = os.environ.get("GAIA_CPP_MTP_DRAFT", "")
        if draft:
            if not hasattr(self._backend, "enable_mtp"):
                logger.warning("gaia_cpp: GAIA_CPP_MTP_DRAFT set, but this extension predates MTP")
            elif not os.path.exists(draft):
                logger.warning("gaia_cpp: MTP draft %s not found; running without MTP", draft)
            else:
                err = self._backend.enable_mtp(draft, int(os.environ.get("GAIA_CPP_MTP_N", "3")))
                if err:
                    logger.warning("gaia_cpp: MTP not enabled (%s)", err)
                else:
                    logger.info("gaia_cpp: MTP on, draft %s, n_max %s", draft,
                                self._backend.mtp_stats()["n_max"])

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
            "kv":          self.kv_stats(),
            "mtp":         self.mtp_stats(),
            "media":       (self._backend.media_info() if hasattr(self._backend, "media_info") else {}),
        }

    def mtp_stats(self) -> dict:
        """MTP draft counters ({} on an extension without MTP). Like kv_stats,
        never waits on a running request."""
        if not hasattr(self._backend, "mtp_stats"):
            return {}
        if not self._request_lock.acquire(blocking=False):
            return {**self._mtp_last, "busy": True}
        try:
            self._mtp_last = dict(self._backend.mtp_stats())
            return self._mtp_last
        except Exception as e:  # never fail /health over stats
            logger.debug("mtp_stats failed: %s", e)
            return {"error": str(e)}
        finally:
            self._request_lock.release()

    # ── KV cache control (d12t) ───────────────────────────────────────────────

    def kv_stats(self) -> dict:
        """Prefix-reuse / snapshot counters, or {} on an older extension.

        Never waits on a running request: the extension's kv_stats takes the
        inference mutex and is called with the GIL held, so during a
        generation it stalled every thread in the process for the whole
        generation — /health with it, and gaia-core's readiness gate turned
        users away ("my inference engine is loading") whenever Core's CPU
        gear was busy. While a request runs, the last reading is served.
        """
        if not self._kv_control:
            return {}
        if not self._request_lock.acquire(blocking=False):
            return {**self._kv_last, "busy": True}
        try:
            self._kv_last = dict(self._backend.kv_stats())
            return self._kv_last
        except Exception as e:  # never fail /health over stats
            logger.debug("kv_stats failed: %s", e)
            return {"error": str(e)}
        finally:
            self._request_lock.release()

    def clear_kv(self) -> bool:
        if not self._kv_control:
            return False
        with self._request_lock:
            self._backend.clear_kv()
        logger.info("gaia_cpp: KV cache and snapshots cleared")
        return True

    def save_kv(self, path: str) -> int:
        """Save the live KV (last prompt + reply) to `path`; tokens saved."""
        if not self._kv_control:
            return 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._request_lock:
            n = self._backend.save_state(path)
        logger.info("gaia_cpp: KV state saved to %s (%d tokens)", path, n)
        return n

    def load_kv(self, path: str) -> int:
        """Load a saved KV state as the live cache and a pinned baseline."""
        if not self._kv_control or not os.path.exists(path):
            return 0
        with self._request_lock:
            n = self._backend.load_state(path)
        if n:
            logger.info("gaia_cpp: KV state loaded from %s (%d tokens)", path, n)
        else:
            logger.warning("gaia_cpp: KV state %s did not load", path)
        return n

    def _log_reuse(self, result) -> None:
        if self._kv_control and result.prompt_tokens:
            logger.debug("gaia_cpp: prompt %d tok — %d reused%s, %d evaluated%s",
                         result.prompt_tokens, result.reused_tokens,
                         " (snapshot)" if result.snapshot_restored else "",
                         result.evaluated_tokens,
                         f"; MTP {result.draft_accepted}/{result.draft_tokens} drafted accepted"
                         if getattr(result, "draft_tokens", 0) else "")

    @staticmethod
    def _call(fn, repetition_penalty: float, *args):
        """Call a generate function with the repetition penalty (frlp) as its
        last argument; an extension built before it takes one argument fewer."""
        if repetition_penalty == 1.0:
            return fn(*args)
        try:
            return fn(*args, repetition_penalty)
        except TypeError as e:
            if "incompatible function arguments" not in str(e):
                raise
            return fn(*args)

    # ── Streaming generation ──────────────────────────────────────────────────

    def stream_to_writer(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        session_id: str = "",
        write_fn=None,
        repetition_penalty: float = 1.0,
    ) -> None:
        """
        Stream SSE directly to write_fn.  No daemon thread, no generator.

        Called from the HTTP handler thread (which has a valid GIL state) so
        pybind11's py::gil_scoped_release works correctly on Python 3.11.

        This is the primary streaming path used by _proxy_stream_cpp().
        Writes activation JSONL after generation completes.
        """
        media: list = []
        prompt = self._build_prompt(messages, media=media)
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
            if media:
                result = self._call(self._backend.generate_media, repetition_penalty,
                                    prompt, media, token_callback, max_tokens, temperature, top_p, 0,
                                    bool(self.capture_layers))
            else:
                result = self._call(self._backend.generate_stream, repetition_penalty,
                                    prompt, token_callback, max_tokens, temperature, top_p,
                                    0,  # top_k
                                    bool(self.capture_layers))
        self._log_reuse(result)

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

        # Strip <think> blocks from final output
        final_text = re.sub(r"<think>.*?</think>\s*", "", result.text.strip(), flags=re.DOTALL)

        # Final SSE chunk with full message + usage
        if write_fn is not None:
            final = {
                "id":      gen_id,
                "object":  "chat.completion.chunk",
                "choices": [{
                    "delta":         {},
                    "finish_reason": "stop",
                    "message":       {"role": "assistant", "content": final_text},
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
        enable_thinking: bool = True,
        repetition_penalty: float = 1.0,
    ) -> dict:
        """Non-streaming generation, returns OpenAI-compatible JSON dict."""
        media: list = []
        prompt = self._build_prompt(messages, enable_thinking=enable_thinking, media=media)

        with self._request_lock:
            if media:
                result = self._call(self._backend.generate_media, repetition_penalty,
                                    prompt, media, None, max_tokens, temperature, top_p, 0,
                                    bool(self.capture_layers))
            else:
                result = self._call(self._backend.generate, repetition_penalty,
                                    prompt, max_tokens, temperature, top_p, 0,
                                    bool(self.capture_layers))
        self._log_reuse(result)

        # The native thought (Gemma 4 channel or Qwen3 <think>, closed or cut
        # off) travels as reasoning_content; it used to be stripped and lost.
        thought, text = _split_thought_text(result.text.strip())
        text = text.strip()

        return {
            "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object":  "chat.completion",
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": text,
                                  **({"reasoning_content": thought} if thought is not None else {})},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                **({"cached_prefix_tokens": result.reused_tokens} if self._kv_control else {}),
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

    def _family(self) -> str:
        """'gemma' or 'chatml', from the GGUF's general.architecture.

        This used to test the path for 'gemma' — via an attribute nothing ever
        set, so every model, Gemma included, was prompted in ChatML. The
        resolved file name is only a fallback for an extension built before
        architecture() existed (core.gguf is a symlink that says nothing).
        """
        family = getattr(self, "_family_cache", None)
        if family is None:
            arch = ""
            try:
                arch = (self._backend.architecture() or "").lower()
            except AttributeError:
                pass
            name = arch or os.path.basename(os.path.realpath(self.model_path or "")).lower()
            family = "gemma" if name.startswith("gemma") or "gemma" in name else "chatml"
            self._family_cache = family
            logger.info("gaia_cpp prompt format: %s (architecture=%r)", family, arch or None)
        return family


    def _content(self, content, media: Optional[list]) -> str:
        """A message's content as prompt text. A list of OpenAI-style parts
        (frlp) gives its text parts in order; each image or audio part becomes
        the projector's media marker, with the file's bytes appended to
        `media`, when a projector is loaded and `media` is given. Otherwise
        media parts are left out (they used to reach the prompt as a Python
        list repr)."""
        if not isinstance(content, list):
            return "" if content is None else str(content)
        marker = getattr(self, "_media_marker", "") if media is not None else ""
        out = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "text":
                out.append(part.get("text", ""))
                continue
            data = None
            try:
                data = _media_bytes(part)
            except Exception as e:  # undecodable media: say so in place
                logger.warning("gaia_cpp: media part not decoded: %s", e)
            if data is None:
                continue
            if marker:
                media.append(data)
                out.append(marker)
            else:
                logger.warning("gaia_cpp: %s part dropped (no projector loaded)", ptype)
        return "".join(out)

    def _build_prompt(self, messages: list, enable_thinking: bool = True,
                      media: Optional[list] = None) -> str:
        """Build formatted prompt for the model family.

        Currently supports ChatML (Qwen) and Gemma 4 turn format.
        The cpp/GGUF backend detects model family from the model path.

        When enable_thinking=False, injects an empty think block after the
        assistant start token to suppress chain-of-thought mode.

        `media` (frlp): pass a list to collect image/audio bytes for
        generate_media(); the prompt then carries a marker for each.
        """
        messages = [{**m, "content": self._content(m.get("content", ""), media)} for m in messages]
        is_gemma = self._family() == "gemma"

        if is_gemma and getattr(self, "prompt_format", "house") == "stock":
            from gaia_engine.prompt_format import stock_gemma4
            return stock_gemma4(messages, enable_thinking=enable_thinking)

        if is_gemma:
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|turn>{role}<turn|>\n{content}")
            if enable_thinking:
                parts.append("<|turn>assistant<turn|>\n")
            else:
                parts.append("<|turn>assistant<turn|>\n<|think|>\n\n<|think|>\n\n")
            return "\n".join(parts)

        # ChatML (Qwen/default)
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if enable_thinking:
            parts.append("<|im_start|>assistant\n")
        else:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "\n".join(parts)
