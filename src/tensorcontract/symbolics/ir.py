"""Minimal backend-neutral symbolic tensor intermediate representation.

This module deliberately supports fixed integer dimensions only. Symbolic
dimension binding and compilation belong to later stages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeAlias

from .errors import (
    DimensionMismatchError,
    InvalidOperationError,
    MissingIndexError,
    MissingValueError,
    RepeatedIndexError,
    SymbolicValidationError,
)


class IndexRole(str, Enum):
    """The intended role of a named index in a symbolic computation."""

    FREE = "free"
    CONTRACTED = "contracted"
    BATCH = "batch"


@dataclass(frozen=True, slots=True)
class SymbolicIndex:
    """A fixed-dimension named symbolic tensor index."""

    name: str
    dimension: int
    role: IndexRole = IndexRole.FREE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise SymbolicValidationError("a symbolic index requires a non-empty name")
        if self.dimension < 1:
            raise DimensionMismatchError(
                f"symbolic index {self.name!r} requires a positive dimension, got {self.dimension}"
            )

    def __str__(self) -> str:
        return f"{self.name}:{self.dimension}[{self.role.value}]"


@dataclass(frozen=True, slots=True)
class SymbolicTensor:
    """A symbolic input tensor with an ordered index signature."""

    name: str
    indices: tuple[str, ...]
    shape: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return ()

    def __str__(self) -> str:
        shape = "?" if self.shape is None else "×".join(str(value) for value in self.shape)
        return f"tensor {self.name}({', '.join(self.indices)}) shape={shape}"


@dataclass(frozen=True, slots=True)
class ContractionNode:
    """Contract one or more dependencies into an explicitly ordered output."""

    name: str
    inputs: tuple[str, ...]
    output_indices: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.inputs

    def __str__(self) -> str:
        return f"contract {self.name} = ({', '.join(self.inputs)}) -> ({', '.join(self.output_indices)})"


@dataclass(frozen=True, slots=True)
class TransposeNode:
    """Reorder all indices of one dependency without materializing semantics."""

    name: str
    input: str
    output_indices: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.input,)

    def __str__(self) -> str:
        return f"transpose {self.name} = {self.input} -> ({', '.join(self.output_indices)})"


class ElementwiseKind(str, Enum):
    """Elementwise operations supported by the minimal reference IR."""

    ADD = "add"
    MULTIPLY = "multiply"


@dataclass(frozen=True, slots=True)
class ElementwiseNode:
    """Apply an elementwise operation to equally indexed dependencies."""

    name: str
    inputs: tuple[str, ...]
    output_indices: tuple[str, ...]
    operation: ElementwiseKind
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.inputs

    def __str__(self) -> str:
        return (
            f"{self.operation.value} {self.name} = ({', '.join(self.inputs)}) "
            f"-> ({', '.join(self.output_indices)})"
        )


@dataclass(frozen=True, slots=True)
class ReductionNode:
    """Sum explicitly named indices from one dependency."""

    name: str
    input: str
    reduced_indices: tuple[str, ...]
    output_indices: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.input,)

    def __str__(self) -> str:
        return (
            f"sum {self.name} = {self.input} over ({', '.join(self.reduced_indices)}) "
            f"-> ({', '.join(self.output_indices)})"
        )


OperationNode: TypeAlias = ContractionNode | TransposeNode | ElementwiseNode | ReductionNode


def _reject_repeated(indices: tuple[str, ...], owner: str) -> None:
    repeated = sorted(name for name, count in Counter(indices).items() if count > 1)
    if repeated:
        raise RepeatedIndexError(f"{owner} repeats indices {repeated}; diagonal semantics must be explicit")


@dataclass
class SymbolicGraph:
    """A deterministic dependency graph of symbolic tensors and operations."""

    indices: dict[str, SymbolicIndex] = field(default_factory=dict)
    tensors: dict[str, SymbolicTensor] = field(default_factory=dict)
    operations: dict[str, OperationNode] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_index(self, index: SymbolicIndex) -> None:
        existing = self.indices.get(index.name)
        if existing is not None and existing.dimension != index.dimension:
            raise DimensionMismatchError(
                f"index {index.name!r} has incompatible dimensions "
                f"{existing.dimension} and {index.dimension}"
            )
        if existing is not None and existing.role != index.role:
            raise SymbolicValidationError(
                f"index {index.name!r} has conflicting roles {existing.role.value!r} and {index.role.value!r}"
            )
        self.indices[index.name] = index

    def add_tensor(self, tensor: SymbolicTensor) -> None:
        self._ensure_new_value(tensor.name)
        self._validate_signature(tensor.name, tensor.indices)
        expected = tuple(self.indices[name].dimension for name in tensor.indices)
        if tensor.shape is not None and tensor.shape != expected:
            raise DimensionMismatchError(
                f"tensor {tensor.name!r} shape {tensor.shape} is incompatible with "
                f"index dimensions {expected}"
            )
        self.tensors[tensor.name] = tensor

    def add_operation(self, operation: OperationNode) -> None:
        self._ensure_new_value(operation.name)
        for dependency in operation.dependencies:
            if not self.has_value(dependency):
                raise MissingValueError(
                    f"operation {operation.name!r} depends on missing value {dependency!r}"
                )
        self._validate_signature(operation.name, operation.output_indices)
        self._validate_operation(operation)
        self.operations[operation.name] = operation

    def has_value(self, name: str) -> bool:
        return name in self.tensors or name in self.operations

    def value_indices(self, name: str) -> tuple[str, ...]:
        if name in self.tensors:
            return self.tensors[name].indices
        if name in self.operations:
            return self.operations[name].output_indices
        raise MissingValueError(f"unknown symbolic value {name!r}")

    def value_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.indices[index].dimension for index in self.value_indices(name))

    @property
    def dependencies(self) -> dict[str, tuple[str, ...]]:
        return {name: operation.dependencies for name, operation in self.operations.items()}

    @property
    def output_names(self) -> tuple[str, ...]:
        consumed = {
            dependency
            for operation in self.operations.values()
            for dependency in operation.dependencies
        }
        return tuple(
            name for name in (*self.tensors, *self.operations)
            if name not in consumed
        )

    def contracted_indices(self, operation_name: str) -> tuple[str, ...]:
        operation = self.operations.get(operation_name)
        if not isinstance(operation, ContractionNode):
            raise InvalidOperationError(f"{operation_name!r} is not a contraction node")
        union = tuple(
            dict.fromkeys(
                index
                for dependency in operation.inputs
                for index in self.value_indices(dependency)
            )
        )
        return tuple(index for index in union if index not in operation.output_indices)

    def validate(self) -> None:
        """Revalidate all definitions and dependency ordering."""
        known: set[str] = set()
        for tensor in self.tensors.values():
            self._validate_signature(tensor.name, tensor.indices)
            expected = tuple(self.indices[name].dimension for name in tensor.indices)
            if tensor.shape is not None and tensor.shape != expected:
                raise DimensionMismatchError(
                    f"tensor {tensor.name!r} shape {tensor.shape} is incompatible with {expected}"
                )
            known.add(tensor.name)
        for operation in self.operations.values():
            missing = [name for name in operation.dependencies if name not in known]
            if missing:
                raise MissingValueError(
                    f"operation {operation.name!r} has unavailable dependencies {missing}"
                )
            self._validate_signature(operation.name, operation.output_indices)
            self._validate_operation(operation)
            known.add(operation.name)

    def debug_string(self) -> str:
        lines = ["SymbolicGraph(", "  indices:"]
        lines.extend(f"    {index}" for index in self.indices.values())
        lines.append("  values:")
        lines.extend(f"    {tensor}" for tensor in self.tensors.values())
        lines.extend(f"    {operation}" for operation in self.operations.values())
        lines.append(")")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.debug_string()

    def _ensure_new_value(self, name: str) -> None:
        if not name:
            raise SymbolicValidationError("symbolic values require a non-empty name")
        if self.has_value(name):
            raise SymbolicValidationError(f"duplicate symbolic value {name!r}")

    def _validate_signature(self, owner: str, indices: tuple[str, ...]) -> None:
        _reject_repeated(indices, owner)
        missing = sorted(set(indices) - self.indices.keys())
        if missing:
            raise MissingIndexError(f"{owner!r} references missing indices {missing}")

    def _validate_operation(self, operation: OperationNode) -> None:
        if isinstance(operation, ContractionNode):
            self._validate_contraction(operation)
        elif isinstance(operation, TransposeNode):
            source = self.value_indices(operation.input)
            if set(source) != set(operation.output_indices) or len(source) != len(operation.output_indices):
                raise InvalidOperationError(
                    f"transpose {operation.name!r} output {operation.output_indices} "
                    f"is not a permutation of {source}"
                )
        elif isinstance(operation, ElementwiseNode):
            if len(operation.inputs) < 1:
                raise InvalidOperationError(f"elementwise operation {operation.name!r} needs an input")
            for dependency in operation.inputs:
                signature = self.value_indices(dependency)
                if signature != operation.output_indices:
                    raise InvalidOperationError(
                        f"elementwise operation {operation.name!r} requires identical index order; "
                        f"{dependency!r} has {signature}, output has {operation.output_indices}"
                    )
        elif isinstance(operation, ReductionNode):
            _reject_repeated(operation.reduced_indices, f"reduction {operation.name!r}")
            source = self.value_indices(operation.input)
            missing = sorted(set(operation.reduced_indices) - set(source))
            if missing:
                raise MissingIndexError(
                    f"reduction {operation.name!r} cannot reduce missing indices {missing}"
                )
            expected = tuple(index for index in source if index not in operation.reduced_indices)
            if operation.output_indices != expected:
                raise InvalidOperationError(
                    f"reduction {operation.name!r} output must be {expected}, "
                    f"got {operation.output_indices}"
                )

    def _validate_contraction(self, operation: ContractionNode) -> None:
        if len(operation.inputs) < 2:
            raise InvalidOperationError(f"contraction {operation.name!r} needs at least two inputs")
        if len(set(operation.inputs)) != len(operation.inputs):
            raise InvalidOperationError(f"contraction {operation.name!r} repeats an input")
        signatures = [self.value_indices(name) for name in operation.inputs]
        occurrences = Counter(index for signature in signatures for index in signature)
        unknown_output = sorted(set(operation.output_indices) - occurrences.keys())
        if unknown_output:
            raise MissingIndexError(
                f"contraction {operation.name!r} output contains unavailable indices {unknown_output}"
            )
        contracted = set(occurrences) - set(operation.output_indices)
        invalid = sorted(index for index in contracted if occurrences[index] < 2)
        if invalid:
            raise InvalidOperationError(
                f"contraction {operation.name!r} would implicitly reduce unpaired indices {invalid}"
            )
