"""Backend-independent execution interface and built-in backends.

Only the NumPy backend is registered here. Importing this package does not
import PyTorch, JAX, Triton, CUDA, or any other optional runtime.
"""

from .base import (
    BackendError,
    BackendExecutionError,
    BackendNotFoundError,
    ExecutionBackend,
)
from .numpy import NumPyBackend


def get_backend(name: str) -> ExecutionBackend:
    """Return a built-in backend by deterministic lowercase name."""
    if name == "numpy":
        return NumPyBackend()
    raise BackendNotFoundError(
        f"unknown backend {name!r}; available backends: ['numpy']"
    )


def available_backends() -> tuple[str, ...]:
    """Return built-in backends without probing optional dependencies."""
    return ("numpy",)


__all__ = [
    "BackendError",
    "BackendExecutionError",
    "BackendNotFoundError",
    "ExecutionBackend",
    "NumPyBackend",
    "available_backends",
    "get_backend",
]
