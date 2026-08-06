"""SymPy-backed symbolic tensor-network definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import sympy as sp

from tensorcontract.ir import Index, TensorKind, TensorNetwork, TensorNode


@dataclass(frozen=True)
class SymbolicNode:
    name: str
    variables: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    expression: sp.Expr
    metadata: Mapping[str, object]

    def materialize(self, coordinates: np.ndarray) -> np.ndarray:
        """Evaluate this rank-3 factor on a Cartesian coordinate grid."""
        function = sp.lambdify(self.variables, self.expression, modules="numpy")
        grids = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
        values = np.asarray(function(*grids), dtype=np.float64)
        if values.shape == ():
            values = np.full((len(coordinates),) * 3, values, dtype=np.float64)
        return np.broadcast_to(values, (len(coordinates),) * 3).copy()


@dataclass(frozen=True)
class SymbolicNetwork:
    variables: tuple[sp.Symbol, ...]
    nodes: tuple[SymbolicNode, ...]
    dimension: int
    seed: int

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise ValueError("symbolic index dimension must be at least two")
        if len(self.nodes) != 5 or any(len(node.variables) != 3 for node in self.nodes):
            raise ValueError("this benchmark requires five rank-3 symbolic nodes")

    @property
    def interaction_graph(self) -> dict[str, tuple[str, ...]]:
        """Return neighbours induced by at least one shared variable."""
        result: dict[str, tuple[str, ...]] = {}
        for node in self.nodes:
            neighbours = tuple(
                other.name for other in self.nodes
                if other.name != node.name and set(node.variables) & set(other.variables)
            )
            result[node.name] = neighbours
        return result

    @property
    def is_fully_connected(self) -> bool:
        return all(len(neighbours) == len(self.nodes) - 1 for neighbours in self.interaction_graph.values())

    def materialize(self) -> TensorNetwork:
        """Convert symbolic factors to the backend-independent concrete IR."""
        coordinates = np.linspace(-1.0, 1.0, self.dimension, dtype=np.float64)
        network = TensorNetwork(metadata={
            "construction": "complete-five-node-symbolic",
            "symbolic_seed": self.seed,
            "exact": False,
            "note": "SymPy expressions evaluated on a finite coordinate grid",
        })
        for variable in self.variables:
            network.add_index(Index(str(variable), self.dimension, {"coordinates": coordinates.tolist()}))
        for node in self.nodes:
            network.add_node(TensorNode(
                node.name,
                tuple(str(variable) for variable in node.variables),
                TensorKind.DENSE,
                node.materialize(coordinates),
                {"symbolic_expression": sp.srepr(node.expression), **node.metadata},
            ))
        network.validate()
        return network


def _random_expression(rng: np.random.Generator, variables: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Build a bounded, reproducible nonlinear function of all three inputs."""
    x, y, z = variables
    coefficients = rng.uniform(-1.0, 1.0, size=8)
    expression = (
        coefficients[0]
        + coefficients[1] * x
        + coefficients[2] * y
        + coefficients[3] * z
        + coefficients[4] * x * y
        + coefficients[5] * y * z
        + coefficients[6] * z * x
        + sp.sin(coefficients[7] * (x + 2 * y - z))
    )
    return sp.expand(expression)


def build_complete_five_node_network(dimension: int = 8, seed: int = 7) -> SymbolicNetwork:
    """Create five rank-3 factors whose pairwise interaction graph is K5.

    Cyclic triples over five variables ensure every pair of nodes intersects,
    without incorrectly claiming that a rank-3 node has four ordinary legs.
    """
    variables = sp.symbols("x0:5", real=True)
    rng = np.random.default_rng(seed)
    nodes = tuple(
        SymbolicNode(
            f"f{position}",
            tuple(variables[(position + offset) % 5] for offset in range(3)),  # type: ignore[arg-type]
            _random_expression(rng, tuple(variables[(position + offset) % 5] for offset in range(3))),
            {"generator": "random-bounded-polynomial-trigonometric", "position": position},
        )
        for position in range(5)
    )
    network = SymbolicNetwork(tuple(variables), nodes, dimension, seed)
    if not network.is_fully_connected:
        raise AssertionError("cyclic triple construction must induce K5")
    return network
