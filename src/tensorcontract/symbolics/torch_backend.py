"""Sequential PyTorch executor for explicit contraction plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tensorcontract.ir import TensorNetwork
from tensorcontract.planner import ContractionPlan


@dataclass(frozen=True)
class TorchExecution:
    value: float
    device: str
    dtype: str
    peak_memory_bytes: int | None


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _equation(left: tuple[str, ...], right: tuple[str, ...], output: tuple[str, ...]) -> str:
    names = tuple(dict.fromkeys(left + right + output))
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(names) > len(alphabet):
        raise ValueError("too many distinct indices for torch.einsum string notation")
    labels = {name: alphabet[position] for position, name in enumerate(names)}
    return f"{''.join(labels[x] for x in left)},{''.join(labels[x] for x in right)}->{''.join(labels[x] for x in output)}"


def execute_torch(
    network: TensorNetwork,
    plan: ContractionPlan,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    measure_memory: bool = True,
) -> TorchExecution:
    selected = resolve_device(device) if isinstance(device, str) else device
    if selected.type == "cuda" and measure_memory:
        torch.cuda.reset_peak_memory_stats(selected)
    values = {
        name: torch.as_tensor(np.asarray(node.data), device=selected, dtype=dtype)
        for name, node in network.nodes.items()
    }
    indices = {name: node.indices for name, node in network.nodes.items()}
    with torch.no_grad():
        for step in plan.steps:
            left = values.pop(step.left)
            right = values.pop(step.right)
            left_indices = indices.pop(step.left)
            right_indices = indices.pop(step.right)
            values[step.output] = torch.einsum(
                _equation(left_indices, right_indices, step.output_indices), left, right
            )
            indices[step.output] = step.output_indices
    if len(values) != 1:
        raise ValueError("plan did not reduce the network to one tensor")
    result = next(iter(values.values())) * network.scalar
    if result.numel() != 1:
        raise ValueError("symbolic benchmark expects a scalar contraction")
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
        peak = torch.cuda.max_memory_allocated(selected) if measure_memory else None
    else:
        peak = None
    return TorchExecution(float(result.detach().cpu()), str(selected), str(dtype), peak)
