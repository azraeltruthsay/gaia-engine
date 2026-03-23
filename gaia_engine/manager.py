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
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("GAIA.EngineManager")

# Requests that should NOT be proxied but handled by the manager directly
_MANAGER_PATHS = {"/model/load", "/model/unload", "/model/swap", "/health", "/status", "/model/info"}


def _find_free_port() -> int:
    """Find an available ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, timeout: int = 180) -> bool:
    """Poll the worker's /health endpoint until it responds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = Request(f"http://127.0.0.1:{port}/health")
            with urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


class EngineManager:
    """Zero-GPU engine manager with subprocess isolation."""

    def __init__(self, port: int, host: str = "0.0.0.0", max_concurrent: int = 1):
        self.port = port
        self.host = host
        self.worker_process = None
        self.worker_port = None
        self.model_path = None
        self.device = None
        self._lock = threading.Lock()
        # Inference queue: limits concurrent requests to the worker.
        # Additional requests block until a slot opens. Prevents overwhelming
        # the model with rapid-fire Discord messages.
        self._inference_semaphore = threading.Semaphore(max_concurrent)
        self._worker_stdout_thread = None
        self._worker_stderr_thread = None

    def start_worker(self, model_path: str, device: str = "cuda",
                     compile_mode: str = "reduce-overhead",
                     quantize: str = "") -> dict:
        """Spawn a worker subprocess with the GAIA Engine."""
        with self._lock:
            if self.worker_process is not None:
                return {"ok": False, "error": "model already loaded — unload first or use /model/swap"}

            internal_port = _find_free_port()
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
                    logger.info("Worker healthy on port %d", internal_port)
                    return {"ok": True, "model": model_path, "worker_port": internal_port, "model_loaded": True}
                else:
                    return {"ok": False, "error": "worker process died during startup"}
        else:
            # Worker failed to start — clean up
            logger.error("Worker failed to become healthy within timeout")
            self._kill_worker_process()
            return {"ok": False, "error": "worker failed to start within 180s"}

    def stop_worker(self) -> dict:
        """Kill the worker subprocess — frees ALL GPU memory."""
        with self._lock:
            if self.worker_process is None:
                return {"ok": True, "message": "already unloaded"}

            old_model = self.model_path
            # Log the caller for debugging silent unloads
            import traceback
            caller = "".join(traceback.format_stack()[-4:-1]).strip()
            logger.warning("Worker stop requested for %s — caller:\n%s", old_model, caller)
            self._kill_worker_process()
            logger.info("Worker killed — model %s unloaded, GPU memory freed", old_model)

            try:
                if _event_logger: pass  # callback pattern
                log_event("engine", f"Model unloaded: {old_model}",
                          source="engine_manager", details={"caller": caller[:200]})
            except Exception:
                pass
            return {"ok": True, "message": "model unloaded", "old_model": old_model}

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
        with self._lock:
            port = self.worker_port
            if port is None:
                return (503, {"Content-Type": "application/json"},
                        json.dumps({"error": "no model loaded — engine is in standby"}).encode())

        # Queue inference requests — only one at a time
        is_inference = "/v1/chat/completions" in path
        if is_inference:
            acquired = self._inference_semaphore.acquire(timeout=120)
            if not acquired:
                return (429, {"Content-Type": "application/json"},
                        json.dumps({"error": "inference queue full — try again shortly"}).encode())

        url = f"http://127.0.0.1:{port}{path}"
        try:
            req = Request(url, data=body if body else None, method=method)
            for k, v in headers.items():
                # Skip hop-by-hop headers
                if k.lower() not in ("host", "connection", "transfer-encoding"):
                    req.add_header(k, v)

            with urlopen(req, timeout=300) as resp:
                resp_body = resp.read()
                resp_headers = {k: v for k, v in resp.getheaders()}
                return (resp.status, resp_headers, resp_body)
        except URLError as e:
            # Worker might have crashed
            if self.worker_process and self.worker_process.poll() is not None:
                exit_code = self.worker_process.returncode
                logger.error("Worker process died (exit code %d) — this causes 'silent unload'", exit_code)
                try:
                    if _event_logger: pass  # callback pattern
                    log_event("engine", f"Worker CRASHED (exit code {exit_code})",
                              source="engine_manager",
                              details={"model": self.model_path, "exit_code": exit_code})
                except Exception:
                    pass
                with self._lock:
                    self._cleanup_worker_state()
                return (503, {"Content-Type": "application/json"},
                        json.dumps({"error": "worker process crashed"}).encode())
            return (502, {"Content-Type": "application/json"},
                    json.dumps({"error": f"worker proxy error: {e}"}).encode())
        except Exception as e:
            return (502, {"Content-Type": "application/json"},
                    json.dumps({"error": f"proxy error: {e}"}).encode())
        finally:
            if is_inference:
                self._inference_semaphore.release()

    def health_response(self) -> dict:
        """Build health response based on worker state."""
        with self._lock:
            worker_alive = (self.worker_process is not None
                            and self.worker_process.poll() is None)
            model_loaded = worker_alive and self.worker_port is not None

        if model_loaded:
            # Forward to worker for detailed health
            status, _, body = self.proxy_to_worker("GET", "/health", {})
            if status == 200:
                try:
                    data = json.loads(body)
                    data["managed"] = True
                    return data
                except Exception:
                    pass

        return {
            "status": "ok",
            "engine": "gaia-managed",
            "model_loaded": model_loaded,
            "mode": "active" if model_loaded else "standby",
            "managed": True,
            "worker_pid": self.worker_process.pid if worker_alive else None,
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
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

        with self.manager._lock:
            port = self.manager.worker_port
        if port is None:
            self._json({"error": "no model loaded"}, 503)
            return

        # Acquire inference semaphore
        acquired = self.manager._inference_semaphore.acquire(timeout=120)
        if not acquired:
            self._json({"error": "inference queue full"}, 429)
            return

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

    server = HTTPServer((host, port), handler_class)
    logger.info("GAIA Engine Manager (zero-GPU standby) on %s:%d", host, port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_worker()
        server.server_close()
