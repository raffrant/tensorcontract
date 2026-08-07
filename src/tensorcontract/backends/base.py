"""Backend-independent execution contracts and errors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tensorcontract.symbolics.ir import SymbolicGraph


class BackendError(RuntimeError):
    """Base class for execution-backend failures."""


class BackendNotFoundError(BackendError):
    """Raised when a requested backend is not registered."""


class BackendExecutionError(BackendError):
    """A runtime failure annotated with its operation and dependencies."""

    def __init__(
        self,
        backend: str,
        operation: str,
        operation_type: str,
        dependencies: tuple[str, ...],
        detail: str,
    ) -> None:
        self.backend = backend
        self.operation = operation
        self.operation_type = operation_type
        self.dependencies = dependencies
        self.detail = detail
        super().__init__(
            f"backend {backend!r} failed operation {operation!r} "
            f"({operation_type}) with tensors {list(dependencies)}: {detail}"
        )


@runtime_checkable
class ExecutionBackend(Protocol):
    """Minimal interface implemented by symbolic-graph execution backends."""

    name: str

    def execute_all(
        self,
        graph: "SymbolicGraph",
        bindings: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Execute a graph and return inputs plus all intermediate values."""
        ...

    def execute(
        self,
        graph: "SymbolicGraph",
        bindings: Mapping[str, Any],
        output: str | None = None,
    ) -> Any:
        """Execute and return one explicitly selected or unambiguous output."""
        ...
