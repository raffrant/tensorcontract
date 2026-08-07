"""Deterministic contraction planning for the fixed-shape symbolic IR."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from math import prod

from .errors import SymbolicValidationError
from .ir import (
    ContractionNode, ElementwiseNode, OperationNode, ReductionNode,
    SymbolicGraph, TransposeNode,
)


class SymbolicPlanningError(ValueError):
    """Base class for symbolic planning failures."""


class InvalidSymbolicGraphError(SymbolicPlanningError):
    """Raised when planning is requested for an invalid symbolic graph."""


class PlanningUnavailableError(SymbolicPlanningError):
    """Raised when a requested planning strategy is unavailable."""


@dataclass(frozen=True, slots=True)
class SymbolicPlannerOptions:
    """Configuration for deterministic symbolic contraction planning."""

    strategy: str = "greedy"
    precision_bytes: int = 8
    max_intermediate_elements: int | None = None
    allow_fallback: bool = True

    def __post_init__(self) -> None:
        if self.precision_bytes < 1:
            raise SymbolicPlanningError("precision_bytes must be positive")
        if self.max_intermediate_elements is not None and self.max_intermediate_elements < 1:
            raise SymbolicPlanningError("max_intermediate_elements must be positive")


@dataclass(frozen=True, slots=True)
class SymbolicPlanStep:
    """One executable operation and its static cost estimates."""

    name: str
    source_operation: str
    operation_type: str
    dependencies: tuple[str, ...]
    output_indices: tuple[str, ...]
    output_shape: tuple[int, ...]
    estimated_flops: int
    output_elements: int
    materialized_elements: int

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.operation_type}({', '.join(self.dependencies)}) "
            f"-> {self.output_indices} shape={self.output_shape} "
            f"flops={self.estimated_flops} elements={self.output_elements}"
        )


@dataclass(frozen=True, slots=True)
class SymbolicPlanCost:
    """Aggregate cost estimates for a symbolic execution plan."""

    total_flops: int
    peak_live_elements: int
    peak_memory_bytes: int
    total_materialized_elements: int
    temporary_materialization_elements: int


@dataclass(frozen=True)
class SymbolicExecutionPlan:
    """A validated pairwise execution graph with explainable static costs."""

    graph: SymbolicGraph
    steps: tuple[SymbolicPlanStep, ...]
    cost: SymbolicPlanCost
    strategy: str
    precision_bytes: int
    fallback_used: bool = False
    fallback_reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Check graph, step, and aggregate-cost consistency."""
        try:
            self.graph.validate()
        except SymbolicValidationError as error:
            raise InvalidSymbolicGraphError(f"planned graph is invalid: {error}") from error
        operation_names = tuple(self.graph.operations)
        step_names = tuple(step.name for step in self.steps)
        if operation_names != step_names:
            raise SymbolicPlanningError(
                f"plan steps {step_names} do not match graph operations {operation_names}"
            )
        for step in self.steps:
            if step.output_shape != self.graph.value_shape(step.name):
                raise SymbolicPlanningError(f"step {step.name!r} has an inconsistent output shape")
            if step.output_elements != max(1, prod(step.output_shape)):
                raise SymbolicPlanningError(f"step {step.name!r} has an inconsistent output size")
            operation = self.graph.operations[step.name]
            if step.estimated_flops != _operation_flops(self.graph, operation):
                raise SymbolicPlanningError(
                    f"step {step.name!r} has an inconsistent FLOP estimate"
                )
            if step.materialized_elements != step.output_elements:
                raise SymbolicPlanningError(
                    f"step {step.name!r} has an inconsistent materialization estimate"
                )
            if isinstance(operation, ContractionNode) and len(operation.inputs) != 2:
                raise SymbolicPlanningError(f"planned contraction {step.name!r} is not pairwise")
        if self.cost.total_flops != sum(step.estimated_flops for step in self.steps):
            raise SymbolicPlanningError("aggregate FLOP estimate is inconsistent")
        if self.cost.total_materialized_elements != sum(
            step.materialized_elements for step in self.steps
        ):
            raise SymbolicPlanningError("aggregate materialization estimate is inconsistent")
        if self.cost.peak_memory_bytes != self.cost.peak_live_elements * self.precision_bytes:
            raise SymbolicPlanningError("peak byte estimate is inconsistent with precision")
        _, expected_cost = _build_steps_and_cost(
            self.graph,
            {step.name: step.source_operation for step in self.steps},
            self.precision_bytes,
        )
        if self.cost.peak_live_elements != expected_cost.peak_live_elements:
            raise SymbolicPlanningError("peak live-memory estimate is inconsistent")
        if (
            self.cost.temporary_materialization_elements
            != expected_cost.temporary_materialization_elements
        ):
            raise SymbolicPlanningError("temporary materialization estimate is inconsistent")

    def debug_string(self) -> str:
        """Return a stable multiline explanation of the selected plan."""
        lines = [
            f"SymbolicExecutionPlan(strategy={self.strategy!r}, "
            f"fallback_used={self.fallback_used}, steps={len(self.steps)})"
        ]
        lines.extend(f"  {number:02d}. {step}" for number, step in enumerate(self.steps, 1))
        lines.append(
            "  cost: "
            f"flops={self.cost.total_flops}, peak_elements={self.cost.peak_live_elements}, "
            f"peak_bytes={self.cost.peak_memory_bytes}, "
            f"materialized={self.cost.total_materialized_elements}, "
            f"temporary={self.cost.temporary_materialization_elements}"
        )
        if self.fallback_reason is not None:
            lines.append(f"  fallback_reason: {self.fallback_reason}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.debug_string()


def _elements(graph: SymbolicGraph, indices: tuple[str, ...]) -> int:
    return max(1, prod(graph.indices[index].dimension for index in indices))


def _pair_output_indices(
    left: tuple[str, ...],
    right: tuple[str, ...],
    other_signatures: tuple[tuple[str, ...], ...],
    final_output: tuple[str, ...],
) -> tuple[str, ...]:
    required = set(final_output)
    for signature in other_signatures:
        required.update(signature)
    union = tuple(dict.fromkeys(left + right))
    return tuple(index for index in union if index in required)


def _unique_intermediate_name(operation_name: str, number: int, reserved: set[str]) -> str:
    stem = f"__plan_{operation_name}_{number}"
    candidate = stem
    suffix = 0
    while candidate in reserved:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    reserved.add(candidate)
    return candidate


def _lower_contraction(
    target: SymbolicGraph,
    operation: ContractionNode,
    strategy: str,
    options: SymbolicPlannerOptions,
    reserved: set[str],
) -> dict[str, str]:
    live = {name: target.value_indices(name) for name in operation.inputs}
    sources: dict[str, str] = {}
    previous: str | None = None
    number = 0
    while len(live) > 1:
        keys = tuple(live)
        if strategy == "greedy":
            candidates: list[tuple[int, int, str, str, tuple[str, ...]]] = []
            for left, right in combinations(keys, 2):
                others = tuple(
                    signature for name, signature in live.items()
                    if name not in (left, right)
                )
                output = _pair_output_indices(
                    live[left], live[right], others, operation.output_indices
                )
                output_elements = _elements(target, output)
                union = tuple(dict.fromkeys(live[left] + live[right]))
                flops = _elements(target, union)
                if options.max_intermediate_elements is None or output_elements <= options.max_intermediate_elements:
                    candidates.append((output_elements, flops, left, right, output))
            if not candidates:
                raise SymbolicPlanningError(
                    f"contraction {operation.name!r} has no pair satisfying max_intermediate_elements"
                )
            _, _, left, right, output_indices = min(candidates)
        else:
            if previous is None:
                left, right = keys[0], keys[1]
            else:
                left = previous
                right = next(name for name in keys if name != previous)
            others = tuple(
                signature for name, signature in live.items()
                if name not in (left, right)
            )
            output_indices = _pair_output_indices(
                live[left], live[right], others, operation.output_indices
            )
            if (
                options.max_intermediate_elements is not None
                and _elements(target, output_indices) > options.max_intermediate_elements
            ):
                raise SymbolicPlanningError(
                    f"contraction {operation.name!r} exceeds max_intermediate_elements"
                )

        output_name = (
            operation.name
            if len(live) == 2
            else _unique_intermediate_name(operation.name, number, reserved)
        )
        pair = ContractionNode(
            output_name,
            (left, right),
            operation.output_indices if len(live) == 2 else output_indices,
            {**operation.metadata, "planned_from": operation.name},
        )
        target.add_operation(pair)
        sources[output_name] = operation.name
        del live[left], live[right]
        live[output_name] = pair.output_indices
        previous = output_name
        number += 1
    return sources


def _operation_flops(graph: SymbolicGraph, operation: OperationNode) -> int:
    if isinstance(operation, ContractionNode):
        union = tuple(
            dict.fromkeys(
                index for dependency in operation.inputs
                for index in graph.value_indices(dependency)
            )
        )
        return _elements(graph, union)
    if isinstance(operation, ElementwiseNode):
        return _elements(graph, operation.output_indices) * max(0, len(operation.inputs) - 1)
    if isinstance(operation, ReductionNode):
        return _elements(graph, graph.value_indices(operation.input))
    if isinstance(operation, TransposeNode):
        return 0
    raise SymbolicPlanningError(f"cannot cost operation {operation!r}")


def _build_steps_and_cost(
    graph: SymbolicGraph,
    sources: dict[str, str],
    precision_bytes: int,
) -> tuple[tuple[SymbolicPlanStep, ...], SymbolicPlanCost]:
    steps: list[SymbolicPlanStep] = []
    sizes = {name: _elements(graph, tensor.indices) for name, tensor in graph.tensors.items()}
    for name, operation in graph.operations.items():
        shape = graph.value_shape(name)
        output_elements = max(1, prod(shape))
        sizes[name] = output_elements
        steps.append(
            SymbolicPlanStep(
                name, sources.get(name, name), type(operation).__name__,
                operation.dependencies, operation.output_indices, shape,
                _operation_flops(graph, operation), output_elements, output_elements,
            )
        )

    remaining = Counter(
        dependency for operation in graph.operations.values()
        for dependency in operation.dependencies
    )
    final_outputs = set(graph.output_names)
    live_elements = sum(sizes[name] for name in graph.tensors)
    peak_elements = live_elements
    for operation in graph.operations.values():
        live_elements += sizes[operation.name]
        peak_elements = max(peak_elements, live_elements)
        for dependency in operation.dependencies:
            remaining[dependency] -= 1
            if remaining[dependency] == 0 and dependency not in final_outputs:
                live_elements -= sizes[dependency]

    total_materialized = sum(step.materialized_elements for step in steps)
    temporary = sum(
        step.materialized_elements for step in steps if step.name not in final_outputs
    )
    cost = SymbolicPlanCost(
        sum(step.estimated_flops for step in steps), peak_elements,
        peak_elements * precision_bytes, total_materialized, temporary,
    )
    return tuple(steps), cost


def plan_symbolic_contractions(
    graph: SymbolicGraph,
    options: SymbolicPlannerOptions | None = None,
) -> SymbolicExecutionPlan:
    """Lower a valid symbolic graph to a deterministic pairwise plan."""
    configured = options or SymbolicPlannerOptions()
    try:
        graph.validate()
    except SymbolicValidationError as error:
        raise InvalidSymbolicGraphError(
            f"cannot plan invalid symbolic graph: {error}"
        ) from error

    fallback_used = False
    fallback_reason: str | None = None
    if configured.strategy in ("greedy", "input-order"):
        strategy = configured.strategy
    elif configured.allow_fallback:
        strategy = "input-order"
        fallback_used = True
        fallback_reason = f"strategy {configured.strategy!r} is unavailable"
    else:
        raise PlanningUnavailableError(
            f"planning strategy {configured.strategy!r} is unavailable"
        )

    lowered = SymbolicGraph(metadata={**graph.metadata, "planned_strategy": strategy})
    for index in graph.indices.values():
        lowered.add_index(index)
    for tensor in graph.tensors.values():
        lowered.add_tensor(tensor)
    reserved = set(graph.tensors) | set(graph.operations)
    sources: dict[str, str] = {}
    for operation in graph.operations.values():
        if isinstance(operation, ContractionNode):
            sources.update(_lower_contraction(lowered, operation, strategy, configured, reserved))
        else:
            lowered.add_operation(operation)
            sources[operation.name] = operation.name

    steps, cost = _build_steps_and_cost(lowered, sources, configured.precision_bytes)
    plan = SymbolicExecutionPlan(
        lowered, steps, cost, strategy, configured.precision_bytes,
        fallback_used, fallback_reason,
        {"source_operation_count": len(graph.operations)},
    )
    plan.validate()
    return plan
