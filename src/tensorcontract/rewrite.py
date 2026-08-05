"""Deterministic, inspectable semantics-preserving graph rewrites."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable

import numpy as np

from .ir import TensorKind, TensorNetwork, TensorNode


@dataclass(frozen=True)
class RewriteRecord:
    rule: str
    matched: tuple[str, ...]
    preconditions: str
    before_elements: int
    after_elements: int
    transformation: str


def _elements(network: TensorNetwork) -> int:
    return sum(prod(network.indices[i].dimension for i in node.indices) for node in network.nodes.values())


class RewriteEngine:
    """Apply rules to a fixed point in stable rule/name order."""

    def simplify(self, source: TensorNetwork) -> tuple[TensorNetwork, tuple[RewriteRecord, ...]]:
        network = source.copy()
        trace: list[RewriteRecord] = []
        rules: tuple[Callable[[TensorNetwork], RewriteRecord | None], ...] = (
            self._scalar_fold, self._identity_eliminate, self._degree_one_absorb,
        )
        while True:
            for rule in rules:
                record = rule(network)
                if record is not None:
                    trace.append(record)
                    break
            else:
                return network, tuple(trace)

    @staticmethod
    def _scalar_fold(network: TensorNetwork) -> RewriteRecord | None:
        names = sorted(name for name, node in network.nodes.items() if not node.indices)
        if not names:
            return None
        before = _elements(network)
        value = 1.0 + 0j
        for name in names:
            node = network.nodes.pop(name)
            if node.data is None:
                raise ValueError("cannot fold a symbolic scalar")
            value *= complex(np.asarray(node.data))
        network.scalar *= value
        return RewriteRecord("scalar-folding", tuple(names), "rank-zero concrete tensors", before, _elements(network), f"multiply global scalar by {value}")

    @staticmethod
    def _identity_eliminate(network: TensorNetwork) -> RewriteRecord | None:
        incidence = network.incidence()
        for name in sorted(network.nodes):
            node = network.nodes[name]
            if node.kind != TensorKind.IDENTITY or len(node.indices) != 2:
                continue
            left, right = node.indices
            if left in network.open_indices or right in network.open_indices:
                continue
            if network.indices[left].dimension != network.indices[right].dimension:
                continue
            # At least one endpoint must be private to the identity and one
            # neighbour. The other endpoint may be a genuine hyperedge.
            if len(incidence[left]) != 2 and len(incidence[right]) != 2:
                continue
            if node.data is None or not np.array_equal(node.data, np.eye(network.indices[left].dimension)):
                continue
            before = _elements(network)
            if len(incidence[right]) == 2 and len(incidence[left]) != 2:
                left, right = right, left
            del network.nodes[name]
            for other_name, other in list(network.nodes.items()):
                if right in other.indices:
                    network.nodes[other_name] = TensorNode(other.name, tuple(left if i == right else i for i in other.indices), other.kind, other.data, other.metadata)
            del network.indices[right]
            return RewriteRecord("identity-elimination", (name,), "equal-dimensional internal legs, one private endpoint, and exact delta data", before, _elements(network), f"remove {name} and merge index {right} into {left}")
        return None

    @staticmethod
    def _degree_one_absorb(network: TensorNetwork) -> RewriteRecord | None:
        incidence = network.incidence()
        for name in sorted(network.nodes):
            node = network.nodes[name]
            if len(node.indices) != 1 or node.kind == TensorKind.IDENTITY:
                continue
            index = node.indices[0]
            if index in network.open_indices or len(incidence[index]) != 2:
                continue
            target_name = next(n for n in incidence[index] if n != name)
            target = network.nodes[target_name]
            if target.indices.count(index) != 1 or node.data is None or target.data is None:
                continue
            before = _elements(network)
            axis = target.indices.index(index)
            data = np.tensordot(target.data, node.data, axes=([axis], [0]))
            out_indices = target.indices[:axis] + target.indices[axis + 1:]
            # tensordot leaves the target axes in their original relative order.
            network.nodes[target_name] = TensorNode(target.name, out_indices, target.kind, data, target.metadata)
            del network.nodes[name]
            del network.indices[index]
            return RewriteRecord("degree-one-absorption", (name, target_name), "closed degree-two index and concrete rank-one leaf", before, _elements(network), f"contract {name} into {target_name}")
        return None


def format_trace(trace: tuple[RewriteRecord, ...]) -> str:
    if not trace:
        return "(no rewrites)"
    return "\n".join(
        f"{n:02d}. {r.rule}: match={r.matched} elements={r.before_elements}->{r.after_elements}; {r.transformation}; preconditions={r.preconditions}"
        for n, r in enumerate(trace, 1)
    )
