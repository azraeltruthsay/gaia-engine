"""
GAIA Engine Manager — Zero-GPU subprocess isolation.

Starts as a pure Python HTTP server with ZERO GPU footprint.
No torch, no transformers, no CUDA context.

Model loading spawns a worker subprocess that owns the CUDA context
and runs the real GAIAEngine + EngineHandler on an internal port.

Model unloading kills the subprocess — all GPU memory freed, guaranteed.

Architecture:
    ┌─────────────────────────────────────┐
    │  EngineManager (this file, no CUDA) │
    │  - Public port (8092/7777/8080)     │
    │  - Proxies requests to worker       │
    │  - Spawns/kills worker subprocess   │
    ├─────────────────────────────────────┤
    │  Worker Subprocess (owns CUDA)      │
    │  - GAIAEngine + EngineHandler       │
    │  - Internal port (127.0.0.1:N)      │
    │  - Killed on unload = zero VRAM     │
    └─────────────────────────────────────┘
"""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("GAIA.EngineManager")

# Requests that should NOT be proxied but handled by the manager directly
_MANAGER_PATHS = {"/model/load", "/model/unload", "/model/swap", "/health", "/status", "/model/info",
                  "/adapter/load", "/adapter/unload", "/adapter/list", "/adapter/set",
                  "/inference/drain", "/inference/resume", "/inference/cancel",
                  "/model/migrate"}


def _find_free_port() -> int:
    """Find an available ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, timeout: int = 180) -> bool:
    """Poll the worker's /health endpoint until the model is actually loaded.

    The worker's HTTP server starts before weights are loaded, so a bare
    200-check returns too early.  We check two response formats:

    - GAIA Engine (Python): ``{"model_loaded": true, ...}``
    - llama-server (GGUF):  ``{"status": "ok"}``

    llama-server doesn't expose a ``model_loaded`` field — it returns
    ``{"status": "ok"}`` only after weights are loaded and slots are
    initialized, so ``status == "ok"`` is sufficient for GGUF workers.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = Request(f"http://127.0.0.1:{port}/health")
            with urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read().decode())
                    # GAIA Engine Python worker: explicit model_loaded field
                    if body.get("model_loaded"):
                        return True
                    # llama-server (GGUF): status=ok means ready to serve
                    if body.get("status") == "ok" and "model_loaded" not in body:
                        return True
                    # HTTP server up but model still loading — keep polling
        except Exception:
            pass
        time.sleep(0.5)
    return False


class EngineManager:
    """Zero-GPU engine manager with subprocess isolation."""

    def __init__(self, port: int, host: str = "0.0.0.0", max_concurrent: int = 1,
                 event_callback=None):
        self.port = port
        self.host = host
        self.worker_process = None
        self.worker_port = None
        self.model_path = None
        self.device = None
        self.backend = None  # 'engine' or 'gguf'
        self.event_callback = event_callback
        self._lock = threading.Lock()
        # Inference queue: limits concurrent requests to the worker.
        # Additional requests block until a slot opens. Prevents overwhelming
        # the model with rapid-fire Discord messages.
        self._inference_semaphore = threading.Semaphore(max_concurrent)
        # Drain state: graceful inference shutdown for model transitions
        self._draining = False
        self._drain_event = threading.Event()
        self._drain_event.set()  # Not draining initially
        self._active_inference_count = 0
        self._worker_stdout_thread = None
        self._worker_stderr_thread = None
        # gaia_cpp in-process backend (set when backend='cpp')
        self._cpp_backend = None

    def start_worker(self, model_path: str, device: str = "cuda",
                     compile_mode: str = "reduce-overhead",
                     quantize: str = "") -> dict:
        """Spawn a worker subprocess with the GAIA Engine.

        If model_path ends in .gguf, spawns llama-server instead of the
        Python engine. This provides fast CPU inference for GGUF models
        with the same OpenAI-compatible API.
        """
        with self._lock:
            if self.worker_process is not None:
                return {"ok": False, "error": "model already loaded — unload first or use /model/swap"}

            internal_port = _find_free_port()
            is_gguf = model_path.lower().endswith('.gguf')

            if is_gguf:
                # GGUF: try gaia_cpp in-process backend first (hidden states + speed)
                # Falls back to llama-server subprocess if unavailable.
                n_gpu_layers = 999 if device == "cuda" else 0
                from gaia_engine.config import GGUF_CTX_SIZE, GGUF_THREADS, ENGINE_TIER
                ctx_size = GGUF_CTX_SIZE
                threads = GGUF_THREADS
                tier = ENGINE_TIER

                try:
                    from gaia_engine.cpp import is_available, GaiaCppBackendAdapter
                    if is_available():
                        cpp = GaiaCppBackendAdapter(
                            model_path,
                            n_gpu_layers=n_gpu_layers,
                            n_ctx=ctx_size,
                            n_threads=threads,
                            tier=tier,
                        )
                        self._cpp_backend = cpp
                        self.worker_port = None
                        self.model_path = model_path
                        self.device = device
                        self.backend = 'cpp'
                        logger.info("GGUF loaded via gaia_cpp (in-process, gpu_layers=%d): %s",
                                    n_gpu_layers, model_path)
                        return {"ok": True, "model": model_path, "backend": "cpp", "model_loaded": True}
                except Exception as e:
                    logger.warning("gaia_cpp unavailable (%s) — falling back to llama-server", e)
                    self._cpp_backend = None

                # Fallback: llama-server subprocess
                cmd = [
                    "llama-server",
                    "--model", model_path,
                    "--host", "127.0.0.1",
                    "--port", str(internal_port),
                    "--n-gpu-layers", str(n_gpu_layers),
                    "--ctx-size", str(ctx_size),
                    "--threads", str(threads),
                    "--chat-template", "chatml",
                ]
                logger.info("GGUF mode: llama-server with %d GPU layers, ctx=%d, threads=%d",
                            n_gpu_layers, ctx_size, threads)
            else:
                # Safetensors/HF: use GAIA Engine Python worker
                cmd = [
                    sys.executable, "-m", "gaia_engine",
                    "--model", model_path,
                    "--port", str(internal_port),
                    "--device", device,
                    "--compile", compile_mode,
                    "--host", "127.0.0.1",
                ]

            env = os.environ.copy()
            # Pass quantize config via env if needed (avoids CLI arg complexity)
            if quantize:
                env["GAIA_ENGINE_QUANTIZE"] = quantize

            logger.info("Spawning worker: port=%d device=%s model=%s", internal_port, device, model_path)
            self.worker_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            # Stream worker stdout/stderr to our logger
            self._worker_stdout_thread = threading.Thread(
                target=self._stream_output, args=(self.worker_process.stdout, "worker.stdout"),
                daemon=True)
            self._worker_stderr_thread = threading.Thread(
                target=self._stream_output, args=(self.worker_process.stderr, "worker.stderr"),
                daemon=True)
            self._worker_stdout_thread.start()
            self._worker_stderr_thread.start()

        # Wait for health OUTSIDE the lock so we don't block unload
        logger.info("Waiting for worker health on port %d...", internal_port)
        if _wait_for_health(internal_port, timeout=180):
            with self._lock:
                # Check worker didn't die while we waited
                if self.worker_process is not None and self.worker_process.poll() is None:
                    self.worker_port = internal_port
                    self.model_path = model_path
                    self.device = device
                    self.backend = 'gguf' if is_gguf else 'engine'
                    logger.info("Worker healthy on port %d (backend=%s)", internal_port, self.backend)
                    return {"ok": True, "model": model_path, "worker_port": internal_port, "model_loaded": True}
                else:
                    return {"ok": False, "error": "worker process died during startup"}
        else:
            # Worker failed to start — clean up
            logger.error("Worker failed to become healthy within timeout")
            self._kill_worker_process()
            return {"ok": False, "error": "worker failed to start within 180s"}

    def stop_worker(self) -> dict:
        """Kill the worker subprocess (or release cpp backend) — frees ALL GPU/RAM."""
        with self._lock:
            # Handle gaia_cpp in-process backend
            if self._cpp_backend is not None:
                old_model = self.model_path
                logger.info("Releasing gaia_cpp backend for %s", old_model)
                try:
                    del self._cpp_backend
                except Exception:
                    pass
                self._cleanup_worker_state()
                return {"ok": True, "message": "model unloaded (cpp)", "old_model": old_model}

            if self.worker_process is None:
                return {"ok": True, "message": "already unloaded"}

            old_model = self.model_path
            # Log the caller for debugging silent unloads
            import traceback
            caller = "".join(traceback.format_stack()[-4:-1]).strip()
            logger.warning("Worker stop requested for %s — caller:\n%s", old_model, caller)

            # ── Controlled KV prefix save BEFORE kill ──
            # The worker has the KV tensors in memory. Save them to disk
            # now, while inference is stopped, for instant restoration next boot.
            if self.worker_port is not None:
                try:
                    from urllib.request import Request, urlopen
                    req = Request(
                        f"http://127.0.0.1:{self.worker_port}/cache/persist",
                        data=b'{}',
                        headers={"Content-Type": "application/json"},
                    )
                    with urlopen(req, timeout=30) as resp:
                        result = json.loads(resp.read())
                        if result.get("ok"):
                            logger.info("KV prefix persisted before unload: %s", result)
                        else:
                            logger.debug("KV prefix persist skipped: %s", result)
                except Exception as e:
                    logger.debug("KV prefix persist before unload failed: %s", e)

            self._kill_worker_process()
            logger.info("Worker killed — model %s unloaded, GPU memory freed", old_model)

            try:
                if self.event_callback:
                    self.event_callback("engine", f"Model unloaded: {old_model}",
                                        source="engine_manager", details={"caller": caller[:200]})
            except Exception:
                pass
            return {"ok": True, "message": "model unloaded", "old_model": old_model}

    def drain(self, timeout_s: float = 30.0) -> dict:
        """Stop accepting new inference, wait for in-flight requests to complete.

        The orchestrator calls this before model unload/swap to ensure no
        request is cut mid-generation. Returns when all active inference
        completes or timeout expires.
        """
        with self._lock:
            if self._draining:
                return {"ok": True, "message": "already draining",
                        "active": self._active_inference_count}
            self._draining = True
            self._drain_event.clear()
            active = self._active_inference_count

        if active == 0:
            self._drain_event.set()
            return {"ok": True, "message": "drained (no active requests)", "active": 0}

        logger.info("Draining: waiting for %d active inference request(s)...", active)
        drained = self._drain_event.wait(timeout=timeout_s)
        with self._lock:
            remaining = self._active_inference_count
        if drained:
            logger.info("Drain complete — all requests finished")
        else:
            logger.warning("Drain timeout after %.0fs — %d request(s) still active",
                           timeout_s, remaining)
        return {"ok": drained,
                "message": "drained" if drained else f"timeout ({remaining} active)",
                "active": remaining}

    def resume(self) -> dict:
        """Resume accepting inference requests after drain."""
        with self._lock:
            was_draining = self._draining
            self._draining = False
            self._drain_event.set()
        if was_draining:
            logger.info("Inference resumed")
        return {"ok": True, "was_draining": was_draining}

    def migrate_device(self, target_device: str) -> dict:
        """Migrate model between GPU and CPU without killing the worker.

        This is the fast path for FOCUSING transitions: instead of
        unloading and reloading (~95s), migrate the NF4 weights
        between CPU RAM and GPU VRAM (~5s).

        Only works with the Python GAIA Engine backend (safetensors).
        GGUF/cpp backends don't support migration.
        """
        if self.backend not in ("engine", None):
            return {"ok": False, "error": f"migration not supported for backend '{self.backend}'"}

        with self._lock:
            if self.worker_process is None or self.worker_port is None:
                return {"ok": False, "error": "no active worker"}
            port = self.worker_port

        # Normalize: worker expects /device/gpu or /device/cpu
        target = "gpu" if target_device in ("gpu", "cuda") else "cpu"

        try:
            endpoint = f"http://127.0.0.1:{port}/device/{target}"
            req = Request(endpoint, method="POST", data=b"",
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())

            if result.get("ok"):
                self.device = "cuda" if target == "gpu" else "cpu"
                logger.info("Model migrated to %s in %.1fs (vram=%sMB)",
                            target, result.get("elapsed_s", 0), result.get("vram_mb", 0))
            return result
        except Exception as e:
            logger.warning("Migration to %s failed: %s", target, e)
            return {"ok": False, "error": str(e)}

    def cancel_inference(self, respawn: bool = True) -> dict:
        """Cancel active generation by killing and optionally respawning the worker.

        Enables "stop generating" and "answer now" from UI.
        With respawn=True (default), the model is reloaded after kill.
        With respawn=False, the model is fully unloaded.
        """
        with self._lock:
            if self.worker_process is None and self._cpp_backend is None:
                return {"ok": False, "error": "no active worker"}
            model = self.model_path
            device = self.device

        logger.info("Cancelling inference (respawn=%s)", respawn)
        # Drain state cleanup — we're force-killing anyway
        self._draining = False
        self._drain_event.set()
        self._active_inference_count = 0

        self.stop_worker()

        if respawn and model:
            result = self.start_worker(model, device or "cuda")
            result["cancelled"] = True
            return result
        return {"ok": True, "cancelled": True, "respawned": False}

    def swap_worker(self, model_path: str, device: str = "cuda",
                    compile_mode: str = "reduce-overhead",
                    quantize: str = "") -> dict:
        """Swap models: kill old worker, spawn new one."""
        old_model = self.model_path
        self.stop_worker()
        result = self.start_worker(model_path, device, compile_mode, quantize)
        if result.get("ok"):
            result["old_model"] = old_model
        return result

    def proxy_to_worker(self, method: str, path: str, headers: dict,
                        body: bytes = b"") -> tuple:
        """Forward a request to the worker. Returns (status, headers_dict, body_bytes).

        Inference requests (/v1/chat/completions) are serialized through a
        semaphore to prevent overwhelming the model with concurrent requests.
        """
        # gaia_cpp in-process backend: handle directly without HTTP
        with self._lock:
            cpp = self._cpp_backend

        if cpp is not None:
            return self._proxy_cpp(method, path, body, cpp)

        with self._lock:
            port = self.worker_port
            if port is None:
                return (503, {"Content-Type": "application/json"},
                        json.dumps({"error": "no model loaded — engine is in standby"}).encode())

        # Queue inference requests — only one at a time
        is_inference = "/v1/chat/completions" in path
        if is_inference:
            # Reject new inference while draining (clutch engaged)
            if self._draining:
                return (503, {"Content-Type": "application/json"},
                        json.dumps({"error": "engine draining", "retry_after_ms": 500}).encode())
            acquired = self._inference_semaphore.acquire(timeout=120)
            if not acquired:
                return (429, {"Content-Type": "application/json"},
                        json.dumps({"error": "inference queue full — try again shortly"}).encode())
            with self._lock:
                self._active_inference_count += 1

        url = f"http://127.0.0.1:{port}{path}"

        # Retry loop: on proxy failure, verify worker health and retry once
        max_attempts = 2
        last_error = None
        try:
            for attempt in range(max_attempts):
                try:
                    req = Request(url, data=body if body else None, method=method)
                    for k, v in headers.items():
                        if k.lower() not in ("host", "connection", "transfer-encoding"):
                            req.add_header(k, v)

                    with urlopen(req, timeout=300) as resp:
                        resp_body = resp.read()
                        resp_headers = {k: v for k, v in resp.getheaders()}
                        return (resp.status, resp_headers, resp_body)
                except URLError as e:
                    last_error = e
                    # Worker might have crashed
                    if self.worker_process and self.worker_process.poll() is not None:
                        exit_code = self.worker_process.returncode
                        logger.error("Worker process died (exit code %d)", exit_code)
                        with self._lock:
                            self._cleanup_worker_state()
                        return (503, {"Content-Type": "application/json"},
                                json.dumps({"error": "worker process crashed"}).encode())

                    # Worker alive but proxy failed — check health before retry
                    if attempt < max_attempts - 1:
                        logger.warning("Proxy failed (attempt %d/%d): %s — retrying",
                                       attempt + 1, max_attempts, e)
                        if _wait_for_health(port, timeout=10):
                            import time as _time
                            _time.sleep(0.5)
                            continue
                        else:
                            break

                except Exception as e:
                    last_error = e
                    break

            return (502, {"Content-Type": "application/json"},
                    json.dumps({"error": f"worker proxy error: {last_error}"}).encode())
        finally:
            if is_inference:
                self._inference_semaphore.release()
                with self._lock:
                    self._active_inference_count -= 1
                    if self._draining and self._active_inference_count == 0:
                        self._drain_event.set()

    def health_response(self) -> dict:
        """Build health response based on worker state."""
        # gaia_cpp in-process backend
        with self._lock:
            cpp = self._cpp_backend

        if cpp is not None:
            h = cpp.health()
            h["managed"] = True
            h["model_loaded"] = True
            h["draining"] = self._draining
            h["active_inference"] = self._active_inference_count
            return h

        with self._lock:
            worker_alive = (self.worker_process is not None
                            and self.worker_process.poll() is None)
            # Detect dead worker: process object exists but has exited
            if self.worker_process is not None and not worker_alive:
                exit_code = self.worker_process.returncode
                logger.warning("Worker found dead (exit code %s) during health check — cleaning up", exit_code)
                self._cleanup_worker_state()
                worker_alive = False
            model_loaded = worker_alive and self.worker_port is not None

        if model_loaded:
            # Forward to worker for detailed health
            status, _, body = self.proxy_to_worker("GET", "/health", {})
            if status == 200:
                try:
                    data = json.loads(body)
                    data["managed"] = True
                    data["device"] = self.device or "unknown"
                    data["draining"] = self._draining
                    data["active_inference"] = self._active_inference_count
                    return data
                except Exception:
                    pass
            elif status in (502, 503) and worker_alive:
                # Worker process alive but proxy broken — stale connection
                logger.warning(
                    "Worker alive (PID %s) but proxy returned %d — connection may be stale. "
                    "Port %s may not be reachable.",
                    self.worker_process.pid if self.worker_process else "?",
                    status, self.worker_port,
                )

        return {
            "status": "ok",
            "engine": "gaia-managed",
            "backend": self.backend or "none",
            "model_loaded": model_loaded,
            "mode": "active" if model_loaded else "standby",
            "managed": True,
            "device": self.device or "none",
            "worker_pid": self.worker_process.pid if worker_alive else None,
            "draining": self._draining,
            "active_inference": self._active_inference_count,
        }

    def status_response(self) -> dict:
        """Build status response."""
        with self._lock:
            worker_alive = (self.worker_process is not None
                            and self.worker_process.poll() is None)

        if worker_alive and self.worker_port:
            status, _, body = self.proxy_to_worker("GET", "/status", {})
            if status == 200:
                try:
                    data = json.loads(body)
                    data["managed"] = True
                    return data
                except Exception:
                    pass

        return {"mode": "standby", "model_loaded": False, "managed": True}

    def model_info_response(self) -> dict:
        """Build model info response."""
        with self._lock:
            worker_alive = (self.worker_process is not None
                            and self.worker_process.poll() is None)

        if worker_alive and self.worker_port:
            status, _, body = self.proxy_to_worker("GET", "/model/info", {})
            if status == 200:
                try:
                    return json.loads(body)
                except Exception:
                    pass

        return {"model_loaded": False, "model_path": "", "device": "none", "vram_mb": 0}

    def _proxy_cpp(self, method: str, path: str, body: bytes, cpp) -> tuple:
        """Handle a request via the gaia_cpp in-process backend (non-streaming)."""
        ct = {"Content-Type": "application/json"}

        if path == "/health":
            h = cpp.health()
            h["managed"] = True
            h["model_loaded"] = True
            h["mode"] = "active"
            h["backend"] = "cpp"
            h["device"] = "cuda" if h.get("has_gpu") else "cpu"
            h["draining"] = self._draining
            h["active_inference"] = self._active_inference_count
            return (200, ct, json.dumps(h).encode())

        if path in ("/status", "/model/info"):
            h = cpp.health()
            return (200, ct, json.dumps({
                "model_loaded": True,
                "backend": "cpp",
                "device": "cuda" if h.get("has_gpu") else "cpu",
                "vram_mb": 0,  # cpp backend doesn't track VRAM per-model
                "model": h.get("model", ""),
            }).encode())

        if path == "/v1/chat/completions" and method == "POST" and body:
            try:
                req_data = json.loads(body)
                messages = req_data.get("messages", [])
                max_tokens = int(req_data.get("max_tokens", 512))
                temperature = float(req_data.get("temperature", 0.7))
                top_p = float(req_data.get("top_p", 0.9))
                # Extract enable_thinking from chat_template_kwargs
                template_kwargs = req_data.get("chat_template_kwargs", {})
                enable_thinking = template_kwargs.get("enable_thinking", True)
                result = cpp.generate_json(
                    messages, max_tokens, temperature, top_p,
                    enable_thinking=enable_thinking,
                )
                return (200, ct, json.dumps(result).encode())
            except Exception as e:
                return (500, ct, json.dumps({"error": str(e)}).encode())

        # ── LoRA adapter management ──────────────────────────────────────
        # Accepts both formats:
        #   cpp native: {"adapter_path": "/path/to/adapter.gguf", "scale": 1.0}
        #   unified:    {"name": "adapter_name", "path": "/path/to/adapter/dir"}
        #               (auto-finds adapter.gguf in the directory)
        if path == "/adapter/load" and method == "POST" and body:
            try:
                import os as _os
                req_data = json.loads(body)
                adapter_path = req_data.get("adapter_path", "")
                scale = float(req_data.get("scale", 1.0))

                # Unified format: resolve directory → adapter.gguf
                if not adapter_path:
                    dir_path = req_data.get("path", "")
                    if dir_path and _os.path.isdir(dir_path):
                        gguf_path = _os.path.join(dir_path, "adapter.gguf")
                        if _os.path.isfile(gguf_path):
                            adapter_path = gguf_path
                            logger.info("Resolved adapter dir → %s", adapter_path)
                        else:
                            return (400, ct, json.dumps({
                                "ok": False,
                                "error": f"No adapter.gguf found in {dir_path}. "
                                         f"Convert safetensors adapter to GGUF first."
                            }).encode())

                if not adapter_path:
                    return (400, ct, json.dumps({"ok": False, "error": "adapter_path or path required"}).encode())

                name = req_data.get("name", _os.path.basename(_os.path.dirname(adapter_path)))
                ok = cpp.load_adapter(adapter_path, scale)
                return (200, ct, json.dumps({
                    "ok": ok, "adapter": name, "adapter_path": adapter_path,
                    "scale": scale, "vram_mb": 0,
                    "loaded_adapters": [name] if ok else [],
                }).encode())
            except Exception as e:
                return (500, ct, json.dumps({"ok": False, "error": str(e)}).encode())

        if path == "/adapter/unload" and method == "POST":
            cpp.unload_adapter()
            return (200, ct, json.dumps({"ok": True}).encode())

        if path == "/adapter/set" and method == "POST" and body:
            # For cpp backend, set = load (only one active at a time)
            req_data = json.loads(body)
            name = req_data.get("name", "")
            h = cpp.health()
            return (200, ct, json.dumps({
                "ok": True, "active": name,
                "loaded": [name] if h.get("lora_active", 0) > 0 else [],
            }).encode())

        if path == "/adapter/list" and (method == "GET" or method == "POST"):
            h = cpp.health()
            return (200, ct, json.dumps({
                "active_count": h.get("lora_active", 0),
                "active_path": h.get("lora_path"),
            }).encode())

        return (404, ct, json.dumps({"error": f"cpp backend: path not found: {path}"}).encode())

    def _kill_worker_process(self):
        """Kill the worker process. Must be called with lock held or from locked context."""
        if self.worker_process is None:
            return

        pid = self.worker_process.pid
        logger.info("Terminating worker PID %d", pid)

        try:
            self.worker_process.terminate()
        except OSError:
            pass

        try:
            self.worker_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Worker PID %d did not exit after SIGTERM, sending SIGKILL", pid)
            try:
                self.worker_process.kill()
            except OSError:
                pass
            try:
                self.worker_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logger.error("Worker PID %d did not die after SIGKILL", pid)

        self._cleanup_worker_state()

    def _cleanup_worker_state(self):
        """Reset worker state variables."""
        self.worker_process = None
        self.worker_port = None
        self.model_path = None
        self.device = None
        self.backend = None
        self._cpp_backend = None

    @staticmethod
    def _stream_output(pipe, label):
        """Stream subprocess output to logger."""
        try:
            for line in iter(pipe.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.info("[%s] %s", label, text)
        except Exception:
            pass
        finally:
            pipe.close()


class ManagedEngineHandler(BaseHTTPRequestHandler):
    """HTTP handler for the managed engine. Routes to manager or proxies to worker."""

    manager: EngineManager = None  # Set by serve_managed()

    def log_message(self, fmt, *args):
        if "/health" not in str(args):
            logger.debug(fmt, *args)

    def _json(self, data, status=200):
        try:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            # Client disconnected before response was sent.
            # This is normal (timeout, cancelled request) — NOT a backend error.
            # Do NOT propagate — it would kill the handler thread and
            # BaseHTTPServer would log it as "Exception during processing".
            logger.debug("Client disconnected before response (BrokenPipe) — ignoring")

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def _body_json(self) -> dict:
        raw = self._read_body()
        return json.loads(raw) if raw else {}

    def do_GET(self):
        if self.path == "/health":
            self._json(self.manager.health_response())
        elif self.path == "/status":
            self._json(self.manager.status_response())
        elif self.path == "/model/info":
            self._json(self.manager.model_info_response())
        else:
            # Proxy all other GETs to worker
            headers = {k: v for k, v in self.headers.items()}
            status, resp_headers, body = self.manager.proxy_to_worker(
                "GET", self.path, headers)
            self._send_proxy_response(status, resp_headers, body)

    def do_POST(self):
        raw_body = self._read_body()

        if self.path == "/model/load":
            b = json.loads(raw_body) if raw_body else {}
            model_path = b.get("model") or b.get("model_path", "")
            if not model_path:
                self._json({"ok": False, "error": "model path required"}, 400)
                return
            device = b.get("device", "cuda")
            compile_mode = b.get("compile_mode", "reduce-overhead")
            quantize = b.get("quantize", "")
            result = self.manager.start_worker(model_path, device, compile_mode, quantize)
            status = 200 if result.get("ok") else 500
            if "already loaded" in result.get("error", ""):
                status = 409
            self._json(result, status)

        elif self.path == "/model/unload":
            self._json(self.manager.stop_worker())

        elif self.path == "/model/swap":
            b = json.loads(raw_body) if raw_body else {}
            model_path = b.get("model") or b.get("model_path", "")
            if not model_path:
                self._json({"ok": False, "error": "model path required"}, 400)
                return
            device = b.get("device", "cuda")
            compile_mode = b.get("compile_mode", "reduce-overhead")
            quantize = b.get("quantize", "")
            result = self.manager.swap_worker(model_path, device, compile_mode, quantize)
            status = 200 if result.get("ok") else 500
            self._json(result, status)

        elif self.path == "/inference/drain":
            b = json.loads(raw_body) if raw_body else {}
            timeout_s = float(b.get("timeout_s", 30))
            self._json(self.manager.drain(timeout_s))

        elif self.path == "/inference/resume":
            self._json(self.manager.resume())

        elif self.path == "/inference/cancel":
            # Cancel active generation — kill worker and respawn
            # Enables "stop generating" and "answer now" from UI
            b = json.loads(raw_body) if raw_body else {}
            respawn = b.get("respawn", True)
            result = self.manager.cancel_inference(respawn=respawn)
            self._json(result)

        elif self.path == "/model/migrate":
            # Migrate model between GPU and CPU without killing the worker
            # Fast path: ~5s instead of ~95s for FOCUSING transitions
            b = json.loads(raw_body) if raw_body else {}
            device = b.get("device", "cpu")
            result = self.manager.migrate_device(device)
            self._json(result)

        else:
            # Check if this is a streaming inference request
            is_stream = False
            if self.path == "/v1/chat/completions" and raw_body:
                try:
                    is_stream = json.loads(raw_body).get("stream", False)
                except Exception:
                    pass

            if is_stream:
                self._proxy_stream(raw_body)
            else:
                headers = {k: v for k, v in self.headers.items()}
                status, resp_headers, body = self.manager.proxy_to_worker(
                    "POST", self.path, headers, raw_body)
                self._send_proxy_response(status, resp_headers, body)

    def _proxy_stream(self, body: bytes):
        """Stream SSE response from worker to client, token by token."""
        import http.client

        # gaia_cpp in-process streaming path
        with self.manager._lock:
            cpp = self.manager._cpp_backend

        if cpp is not None:
            self._proxy_stream_cpp(body, cpp)
            return

        with self.manager._lock:
            port = self.manager.worker_port
        if port is None:
            self._json({"error": "no model loaded"}, 503)
            return

        # Reject new inference while draining (clutch engaged)
        if self.manager._draining:
            self._json({"error": "engine draining", "retry_after_ms": 500}, 503)
            return

        # Acquire inference semaphore
        acquired = self.manager._inference_semaphore.acquire(timeout=120)
        if not acquired:
            self._json({"error": "inference queue full"}, 429)
            return
        with self.manager._lock:
            self.manager._active_inference_count += 1

        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
            conn.request("POST", "/v1/chat/completions", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()

            # Forward headers to client
            self.send_response(resp.status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Stream chunks as they arrive
            while True:
                line = resp.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()

            conn.close()
        except Exception as e:
            logger.warning("Stream proxy error: %s", e)
            try:
                self.wfile.write(f"data: {{\"error\": \"{e}\"}}\n\n".encode())
                self.wfile.flush()
            except Exception:
                pass
        finally:
            self.manager._inference_semaphore.release()
            with self.manager._lock:
                self.manager._active_inference_count -= 1
                if self.manager._draining and self.manager._active_inference_count == 0:
                    self.manager._drain_event.set()

    def _proxy_stream_cpp(self, body: bytes, cpp):
        """Stream SSE directly from gaia_cpp in-process backend.

        Uses generate_stream_sse() in direct (write_fn) mode — no daemon thread.
        Runs on the HTTP handler thread (which has valid GIL state) so pybind11's
        py::gil_scoped_release works without crashing on Python 3.11.
        """
        # Reject new inference while draining (clutch engaged)
        if self.manager._draining:
            self._json({"error": "engine draining", "retry_after_ms": 500}, 503)
            return

        acquired = self.manager._inference_semaphore.acquire(timeout=120)
        if not acquired:
            self._json({"error": "inference queue full"}, 429)
            return
        with self.manager._lock:
            self.manager._active_inference_count += 1

        try:
            req_data = json.loads(body) if body else {}
            messages = req_data.get("messages", [])
            max_tokens = int(req_data.get("max_tokens", 512))
            temperature = float(req_data.get("temperature", 0.7))
            top_p = float(req_data.get("top_p", 0.9))
            session_id = req_data.get("session_id", "")

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def write_fn(data: bytes) -> None:
                self.wfile.write(data)
                self.wfile.flush()

            cpp.stream_to_writer(
                messages, max_tokens, temperature, top_p, session_id,
                write_fn=write_fn,
            )

        except Exception as e:
            logger.warning("cpp stream error: %s", e)
            try:
                self.wfile.write(
                    f"data: {{\"error\": \"{e}\"}}\n\n".encode()
                )
                self.wfile.flush()
            except Exception:
                pass
        finally:
            self.manager._inference_semaphore.release()
            with self.manager._lock:
                self.manager._active_inference_count -= 1
                if self.manager._draining and self.manager._active_inference_count == 0:
                    self.manager._drain_event.set()

    def _send_proxy_response(self, status: int, headers: dict, body: bytes):
        """Send a proxied response back to the client."""
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() not in ("transfer-encoding", "connection"):
                self.send_header(k, v)
        if "Content-Length" not in headers:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server so long-running requests (model load/unload)
    don't block health probes and other concurrent requests.

    daemon_threads=False: handler threads must NOT be daemon threads.
    pybind11's py::gil_scoped_release calls PyEval_SaveThread() which
    requires a non-NULL Python thread state. On Python 3.11, daemon
    threads can have a NULL thread state, causing a fatal crash.
    Using non-daemon threads keeps the thread state valid throughout.
    """
    daemon_threads = False


def serve_managed(port: int = 8092, host: str = "0.0.0.0"):
    """Start the managed engine server — zero GPU, subprocess isolation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    manager = EngineManager(port, host)

    # Ensure clean shutdown on SIGTERM
    def _shutdown(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        manager.stop_worker()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Create handler class with manager reference
    handler_class = type("Handler", (ManagedEngineHandler,), {"manager": manager})

    server = _ThreadingHTTPServer((host, port), handler_class)
    logger.info("GAIA Engine Manager (zero-GPU standby, threaded) on %s:%d", host, port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_worker()
        server.server_close()
