"""Symbolic tensor construction, validation, and reference execution."""

from collections.abc import Mapping
from typing import Any

from .model import SymbolicNetwork, SymbolicNode, build_complete_five_node_network
from .errors import (
    DimensionMismatchError,
    InvalidOperationError,
    MissingIndexError,
    MissingValueError,
    RepeatedIndexError,
    SymbolicValidationError,
)
from .ir import (
    ContractionNode,
    ElementwiseKind,
    ElementwiseNode,
    IndexRole,
    ReductionNode,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicTensor,
    TransposeNode,
)
from .planner import (
    InvalidSymbolicGraphError,
    PlanningUnavailableError,
    SymbolicExecutionPlan,
    SymbolicPlanCost,
    SymbolicPlanStep,
    SymbolicPlannerOptions,
    SymbolicPlanningError,
    plan_symbolic_contractions,
)


def execute_symbolic_numpy(
    graph: SymbolicGraph,
    bindings: Mapping[str, Any],
    output: str | None = None,
) -> Any:
    """Execute one symbolic output with the lazily imported NumPy backend."""
    from .numpy_backend import execute_symbolic_numpy as execute

    return execute(graph, bindings, output)


def execute_symbolic_numpy_all(
    graph: SymbolicGraph,
    bindings: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Execute a symbolic graph and return all values using NumPy."""
    from .numpy_backend import execute_symbolic_numpy_all as execute_all

    return execute_all(graph, bindings)

__all__ = [
    "ContractionNode",
    "DimensionMismatchError",
    "ElementwiseKind",
    "ElementwiseNode",
    "IndexRole",
    "InvalidSymbolicGraphError",
    "InvalidOperationError",
    "MissingIndexError",
    "MissingValueError",
    "PlanningUnavailableError",
    "ReductionNode",
    "RepeatedIndexError",
    "SymbolicGraph",
    "SymbolicIndex",
    "SymbolicNetwork",
    "SymbolicNode",
    "SymbolicExecutionPlan",
    "SymbolicPlanCost",
    "SymbolicPlanStep",
    "SymbolicPlannerOptions",
    "SymbolicPlanningError",
    "SymbolicTensor",
    "SymbolicValidationError",
    "TransposeNode",
    "build_complete_five_node_network",
    "execute_symbolic_numpy",
    "execute_symbolic_numpy_all",
    "plan_symbolic_contractions",
]
