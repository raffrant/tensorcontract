"""Optional PyTorch execution backend for the symbolic IR.

Import this module only after PyTorch availability has been established. The
base package and backend registry do not import it eagerly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

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

from .base import BackendExecutionError, BackendUnavailableError


class TorchBackend:
    """Execute symbolic operations with PyTorch while retaining autograd.

    Existing tensors are used without copying when ``device`` and ``dtype``
    are omitted. Supplying either option applies ``Tensor.to`` during binding,
    which remains differentiable but may create converted tensors.
    """

    name = "torch"

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        selected = None if device is None else torch.device(device)
        if selected is not None and selected.type == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailableError(
                f"PyTorch CUDA device {str(selected)!r} was requested but CUDA is unavailable"
            )
        self.device = selected
        self.dtype = dtype

    def execute_all(
        self,
        graph: SymbolicGraph,
        bindings: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Validate and execute every operation without disabling autograd."""
        graph.validate()
        values = self._bind_inputs(graph, bindings)
        for name, operation in graph.operations.items():
            try:
                if isinstance(operation, ContractionNode):
                    values[name] = self._contraction(graph, operation, values)
                elif isinstance(operation, TransposeNode):
                    source_indices = graph.value_indices(operation.input)
                    axes = tuple(source_indices.index(index) for index in operation.output_indices)
                    values[name] = values[operation.input].permute(axes)
                elif isinstance(operation, ElementwiseNode):
                    values[name] = self._elementwise(operation, values)
                elif isinstance(operation, ReductionNode):
                    source_indices = graph.value_indices(operation.input)
                    axes = tuple(source_indices.index(index) for index in operation.reduced_indices)
                    values[name] = (
                        values[operation.input]
                        if not axes
                        else torch.sum(values[operation.input], dim=axes)
                    )
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
        bindings: Mapping[str, Any],
        output: str | None = None,
    ) -> torch.Tensor:
        """Execute one selected or unambiguous symbolic output."""
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

    def _bind_inputs(
        self,
        graph: SymbolicGraph,
        bindings: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        for name in graph.tensors:
            if name not in bindings:
                raise MissingValueError(f"missing PyTorch binding for symbolic tensor {name!r}")
            try:
                supplied = bindings[name]
                if isinstance(supplied, torch.Tensor):
                    value = supplied
                    if self.device is not None or self.dtype is not None:
                        value = value.to(
                            device=self.device if self.device is not None else value.device,
                            dtype=self.dtype if self.dtype is not None else value.dtype,
                        )
                else:
                    value = torch.as_tensor(
                        supplied, device=self.device, dtype=self.dtype
                    )
            except Exception as error:
                raise BackendExecutionError(
                    self.name, name, "SymbolicTensor", (name,), str(error)
                ) from error
            expected = graph.value_shape(name)
            if tuple(value.shape) != expected:
                raise DimensionMismatchError(
                    f"PyTorch binding for tensor {name!r} has shape {tuple(value.shape)}, "
                    f"expected {expected}"
                )
            values[name] = value
        return values

    @staticmethod
    def _equation(
        signatures: tuple[tuple[str, ...], ...],
        output_indices: tuple[str, ...],
    ) -> str:
        names = tuple(dict.fromkeys(index for signature in signatures for index in signature))
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if len(names) > len(alphabet):
            raise ValueError(
                f"PyTorch einsum supports at most {len(alphabet)} distinct named indices"
            )
        labels = {name: alphabet[position] for position, name in enumerate(names)}
        inputs = ",".join(
            "".join(labels[index] for index in signature)
            for signature in signatures
        )
        output = "".join(labels[index] for index in output_indices)
        return f"{inputs}->{output}"

    def _contraction(
        self,
        graph: SymbolicGraph,
        operation: ContractionNode,
        values: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        signatures = tuple(graph.value_indices(name) for name in operation.inputs)
        equation = self._equation(signatures, operation.output_indices)
        return torch.einsum(equation, *(values[name] for name in operation.inputs))

    @staticmethod
    def _elementwise(
        operation: ElementwiseNode,
        values: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        arrays = tuple(values[name] for name in operation.inputs)
        result = arrays[0]
        for array in arrays[1:]:
            result = (
                result + array
                if operation.operation == ElementwiseKind.ADD
                else result * array
            )
        return result
