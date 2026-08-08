"""Backend-independent execution interface and built-in backends.

NumPy is imported eagerly because it is a core dependency. Optional backends
are loaded only when explicitly requested.
"""

from importlib import import_module
from importlib.util import find_spec
from typing import Any

from .base import (
    BackendError,
    BackendExecutionError,
    BackendNotFoundError,
    BackendUnavailableError,
    ExecutionBackend,
)
from .numpy import NumPyBackend


def get_backend(name: str, **options: Any) -> ExecutionBackend:
    """Return a backend by name, importing optional runtimes lazily."""
    if name == "numpy":
        return NumPyBackend(**options)
    if name == "torch":
        try:
            module = import_module(".torch", __name__)
        except (ImportError, OSError) as error:
            raise BackendUnavailableError(
                "PyTorch backend requested but PyTorch could not be imported; "
                "install tensorcontract[torch]"
            ) from error
        return module.TorchBackend(**options)
    if name == "triton":
        try:
            module = import_module(".triton", __name__)
        except (ImportError, OSError) as error:
            raise BackendUnavailableError(
                "Triton backend requested but PyTorch or Triton could not be imported; "
                "install tensorcontract[triton]"
            ) from error
        return module.TritonBackend(**options)
    raise BackendNotFoundError(
        f"unknown backend {name!r}; available backends: ['numpy']; "
        "optional registered backends: ['torch', 'triton']"
    )


def is_backend_available(name: str) -> bool:
    """Check backend availability without importing an optional runtime."""
    if name == "numpy":
        return True
    if name == "torch":
        try:
            return find_spec("torch") is not None
        except (ImportError, ValueError):
            return False
    if name == "triton":
        try:
            return find_spec("torch") is not None and find_spec("triton") is not None
        except (ImportError, ValueError):
            return False
    return False


def available_backends(*, include_optional: bool = False) -> tuple[str, ...]:
    """Return NumPy and, when requested and installed, optional backends.

    The default remains ``("numpy",)`` for Stage 4 compatibility and performs
    no optional-runtime probe.
    """
    if include_optional:
        result = ["numpy"]
        if is_backend_available("torch"):
            result.append("torch")
        if is_backend_available("triton"):
            result.append("triton")
        return tuple(result)
    return ("numpy",)


__all__ = [
    "BackendError",
    "BackendExecutionError",
    "BackendNotFoundError",
    "BackendUnavailableError",
    "ExecutionBackend",
    "NumPyBackend",
    "available_backends",
    "get_backend",
    "is_backend_available",
]
