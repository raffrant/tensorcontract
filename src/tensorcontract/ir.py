"""Backend-independent tensor-network intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


class TensorKind(str, Enum):
    DENSE = "dense"
    DIAGONAL = "diagonal"
    SPARSE = "sparse"
    PARITY = "parity"
    COPY = "copy_delta"
    PERMUTATION = "permutation"
    STABILIZER = "stabilizer"
    PAULI_NOISE = "pauli_noise"
    IDENTITY = "identity"
    SCALAR = "scalar"


@dataclass(frozen=True, slots=True)
class Index:
    name: str
    dimension: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.dimension < 1:
            raise ValueError("an index needs a name and positive dimension")


@dataclass(frozen=True, slots=True)
class TensorNode:
    name: str
    indices: tuple[str, ...]
    kind: TensorKind = TensorKind.DENSE
    data: NDArray[np.generic] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_data(self, data: NDArray[np.generic], indices: tuple[str, ...] | None = None) -> "TensorNode":
        return replace(self, data=np.asarray(data), indices=self.indices if indices is None else indices)


@dataclass
class TensorNetwork:
    indices: dict[str, Index] = field(default_factory=dict)
    nodes: dict[str, TensorNode] = field(default_factory=dict)
    open_indices: tuple[str, ...] = ()
    scalar: complex = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "TensorNetwork":
        return TensorNetwork(dict(self.indices), dict(self.nodes), self.open_indices, self.scalar, dict(self.metadata))

    def add_index(self, index: Index) -> None:
        old = self.indices.get(index.name)
        if old is not None and old.dimension != index.dimension:
            raise ValueError(f"index {index.name!r} has conflicting dimensions")
        self.indices[index.name] = index

    def add_node(self, node: TensorNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"duplicate node {node.name!r}")
        missing = set(node.indices) - self.indices.keys()
        if missing:
            raise ValueError(f"node {node.name!r} uses unknown indices {sorted(missing)}")
        if node.data is not None:
            shape = tuple(self.indices[i].dimension for i in node.indices)
            if node.data.shape != shape:
                raise ValueError(f"node {node.name!r} shape {node.data.shape} != {shape}")
        self.nodes[node.name] = node

    def incidence(self) -> dict[str, list[str]]:
        result = {name: [] for name in self.indices}
        for node in self.nodes.values():
            for index in set(node.indices):
                result[index].append(node.name)
        return result

    def validate(self) -> None:
        for name in self.open_indices:
            if name not in self.indices:
                raise ValueError(f"unknown open index {name!r}")
        for node in self.nodes.values():
            if node.data is None:
                raise ValueError(f"node {node.name!r} has no concrete data")
            expected = tuple(self.indices[i].dimension for i in node.indices)
            if node.data.shape != expected:
                raise ValueError(f"invalid shape for {node.name!r}")

    @property
    def hyperedges(self) -> dict[str, tuple[str, ...]]:
        return {key: tuple(value) for key, value in self.incidence().items()}
