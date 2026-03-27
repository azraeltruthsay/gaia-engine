# gaia_engine/cpp/__init__.py
# Import the compiled C extension, or fall back gracefully if unavailable.
# Import errors are expected on non-Prime tiers (Nano/Core use PyTorch path).

try:
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
