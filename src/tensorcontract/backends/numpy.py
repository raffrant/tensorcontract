"""NumPy reference implementation of the execution backend protocol."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tensorcontract.symbolics.errors import (
    DimensionMismatchError,
    MissingValueError,
    SymbolicValidationError,
)
from tensorcontract.symbolics.ir import (
    ContractionNode,
    ElementwiseKind,
    ElementwiseNode,
    ReductionNode,
    SymbolicGraph,
    TransposeNode,
)

from .base import BackendExecutionError
from .lowering import recognize_batched_matmul


class NumPyBackend:
    """Correctness-oriented interpreter for the current symbolic IR."""

    name = "numpy"

    def __init__(self, *, enable_batched_matmul: bool = True) -> None:
        self.enable_batched_matmul = enable_batched_matmul

    def contraction_implementation(
        self,
        graph: SymbolicGraph,
        operation: ContractionNode,
    ) -> str:
        """Return the selected eager implementation for debugging."""
        if self.enable_batched_matmul and recognize_batched_matmul(graph, operation):
            return "batched-matmul"
        return "einsum"

    def execute_all(
        self,
        graph: SymbolicGraph,
        bindings: Mapping[str, ArrayLike],
    ) -> dict[str, NDArray[np.generic]]:
        """Validate and execute every graph operation in dependency order."""
        graph.validate()
        values = self._bind_inputs(graph, bindings)
        for name, operation in graph.operations.items():
            try:
                if isinstance(operation, ContractionNode):
                    values[name] = self._contraction(graph, operation, values)
                elif isinstance(operation, TransposeNode):
                    source_indices = graph.value_indices(operation.input)
                    axes = tuple(source_indices.index(index) for index in operation.output_indices)
                    values[name] = np.asarray(np.transpose(values[operation.input], axes))
                elif isinstance(operation, ElementwiseNode):
                    values[name] = self._elementwise(operation, values)
                elif isinstance(operation, ReductionNode):
                    source_indices = graph.value_indices(operation.input)
                    axes = tuple(source_indices.index(index) for index in operation.reduced_indices)
                    values[name] = np.asarray(np.sum(values[operation.input], axis=axes))
                else:  # pragma: no cover - current OperationNode is a closed union
                    raise SymbolicValidationError(
                        f"unsupported symbolic operation {type(operation).__name__}"
                    )
            except (BackendExecutionError, SymbolicValidationError):
                raise
            except Exception as error:
                raise BackendExecutionError(
                    self.name,
                    name,
                    type(operation).__name__,
                    operation.dependencies,
                    str(error),
                ) from error
        return values

    def execute(
        self,
        graph: SymbolicGraph,
        bindings: Mapping[str, ArrayLike],
        output: str | None = None,
    ) -> NDArray[np.generic]:
        """Execute one selected output, inferring it only when unambiguous."""
        selected = output
        if selected is None:
            outputs = graph.output_names
            if len(outputs) != 1:
                raise MissingValueError(
                    f"output is ambiguous; graph has unconsumed values {list(outputs)}"
                )
            selected = outputs[0]
        if not graph.has_value(selected):
            raise MissingValueError(f"unknown requested output {selected!r}")
        return self.execute_all(graph, bindings)[selected]

    @staticmethod
    def _bind_inputs(
        graph: SymbolicGraph,
        bindings: Mapping[str, ArrayLike],
    ) -> dict[str, NDArray[np.generic]]:
        values: dict[str, NDArray[np.generic]] = {}
        for name in graph.tensors:
            if name not in bindings:
                raise MissingValueError(f"missing NumPy binding for symbolic tensor {name!r}")
            try:
                value = np.asarray(bindings[name])
            except Exception as error:
                raise BackendExecutionError(
                    "numpy", name, "SymbolicTensor", (name,), str(error)
                ) from error
            expected = graph.value_shape(name)
            if value.shape != expected:
                raise DimensionMismatchError(
                    f"NumPy binding for tensor {name!r} has shape {value.shape}, expected {expected}"
                )
            values[name] = value
        return values

    def _contraction(
        self,
        graph: SymbolicGraph,
        operation: ContractionNode,
        values: Mapping[str, NDArray[np.generic]],
    ) -> NDArray[np.generic]:
        lowering = (
            recognize_batched_matmul(graph, operation)
            if self.enable_batched_matmul
            else None
        )
        if lowering is not None:
            return np.asarray(
                np.matmul(values[lowering.left], values[lowering.right])
            )
        signatures = tuple(graph.value_indices(name) for name in operation.inputs)
        names = tuple(dict.fromkeys(index for signature in signatures for index in signature))
        labels = {name: position for position, name in enumerate(names)}
        arguments: list[object] = []
        for dependency, signature in zip(operation.inputs, signatures):
            arguments.extend(
                (values[dependency], [labels[index] for index in signature])
            )
        arguments.append([labels[index] for index in operation.output_indices])
        return np.asarray(np.einsum(*arguments, optimize=False))

    @staticmethod
    def _elementwise(
        operation: ElementwiseNode,
        values: Mapping[str, NDArray[np.generic]],
    ) -> NDArray[np.generic]:
        arrays = tuple(values[name] for name in operation.inputs)
        result = arrays[0].copy()
        function = np.add if operation.operation == ElementwiseKind.ADD else np.multiply
        for array in arrays[1:]:
            result = function(result, array)
        return np.asarray(result)
