"""Reference NumPy executor for explicit pairwise plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .ir import TensorNetwork

if TYPE_CHECKING:
    from .planner import ContractionPlan


def _einsum_pair(a: NDArray, ai: tuple[str, ...], b: NDArray, bi: tuple[str, ...], out: tuple[str, ...]) -> NDArray:
    labels = {name: n for n, name in enumerate(dict.fromkeys(ai + bi + out))}
    return np.einsum(a, [labels[x] for x in ai], b, [labels[x] for x in bi], [labels[x] for x in out], optimize=False)


def execute_numpy(network: TensorNetwork, plan: "ContractionPlan") -> NDArray:
    values = {name: np.asarray(node.data) for name, node in network.nodes.items()}
    indices = {name: node.indices for name, node in network.nodes.items()}
    for step in plan.steps:
        values[step.output] = _einsum_pair(values.pop(step.left), indices.pop(step.left), values.pop(step.right), indices.pop(step.right), step.output_indices)
        indices[step.output] = step.output_indices
    if not values:
        result = np.asarray(1.0)
    elif len(values) == 1:
        name, result = next(iter(values.items()))
        current = indices[name]
        if current != network.open_indices:
            labels = {x: i for i, x in enumerate(dict.fromkeys(current + network.open_indices))}
            result = np.einsum(result, [labels[x] for x in current], [labels[x] for x in network.open_indices])
    else:
        raise ValueError("plan did not contract all tensors")
    return np.asarray(result * network.scalar)


def contract_numpy(network: TensorNetwork) -> NDArray:
    from .planner import PlanConstraints, plan_contraction
    return execute_numpy(network, plan_contraction(network, PlanConstraints()).selected)
