# gaia_engine/cpp/__init__.py
# Import the compiled C extension, or fall back gracefully if unavailable.
# Import errors are expected where the extension isn't built (torch-only hosts).
#
# The extension and the llama.cpp libraries it was compiled against live next
# to this file (scripts/build_native.sh: ./gaia_cpp*.so, ./lib/*.so.0), so every
# container that mounts the gaia_engine directory gets the same build. Images
# carry their own, often newer, llama.cpp for llama-server; loading the bundled
# libraries first (by path, RTLD_GLOBAL) makes the extension bind to the build
# it was compiled against instead of whatever the image's loader path finds.

import ctypes as _ctypes
import logging as _logging
import os as _os

_LIB_DIR = _os.path.join(_os.path.dirname(__file__), "lib")
# Dependency order: each library's own dependencies first.
_BUNDLED = ("libggml-base.so.0", "libggml-cpu.so.0", "libggml-cuda.so.0", "libggml.so.0", "libllama.so.0")


def _load_bundled_libs():
    if not _os.path.isdir(_LIB_DIR):
        return
    for name in _BUNDLED:
        path = _os.path.join(_LIB_DIR, name)
        if not _os.path.exists(path):
            continue
        try:
            _ctypes.CDLL(path, mode=_ctypes.RTLD_GLOBAL)
        except OSError as e:  # e.g. no CUDA driver for libggml-cuda on a CPU-only host
            _logging.getLogger("GAIA.Engine.cpp").warning("bundled %s not loaded: %s", name, e)


try:
    _load_bundled_libs()
    from gaia_engine.cpp.gaia_cpp import LlamaCppBackend, GenerateResult  # noqa: F401
    from gaia_engine.cpp.backend import GaiaCppBackendAdapter, is_available  # noqa: F401
    _CPP_AVAILABLE = True
    __all__ = ["LlamaCppBackend", "GenerateResult", "GaiaCppBackendAdapter", "is_available"]
except ImportError:
    _CPP_AVAILABLE = False

    def is_available() -> bool:  # type: ignore[misc]
        return False

    GaiaCppBackendAdapter = None  # type: ignore[assignment]
    LlamaCppBackend = None        # type: ignore[assignment]
    GenerateResult = None         # type: ignore[assignment]
    __all__ = ["is_available", "GaiaCppBackendAdapter", "LlamaCppBackend", "GenerateResult"]
