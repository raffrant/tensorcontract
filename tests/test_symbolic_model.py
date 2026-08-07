"""Tests for the core SymPy model that do not require optional backends."""

import numpy as np
import pytest

from tensorcontract.symbolics import SymbolicNetwork, build_complete_five_node_network


def test_five_rank_three_nodes_induce_complete_interaction_graph() -> None:
    symbolic = build_complete_five_node_network(dimension=3, seed=12)
    assert symbolic.is_fully_connected
    assert all(len(neighbours) == 4 for neighbours in symbolic.interaction_graph.values())
    for node in symbolic.nodes:
        assert len(node.variables) == 3
        assert set(node.variables) <= node.expression.free_symbols


def test_materialization_is_reproducible_and_records_provenance() -> None:
    symbolic = build_complete_five_node_network(3, 4)
    left = symbolic.materialize()
    right = build_complete_five_node_network(3, 4).materialize()
    for name in left.nodes:
        assert np.array_equal(left.nodes[name].data, right.nodes[name].data)
        assert "symbolic_expression" in left.nodes[name].metadata
    assert left.metadata["symbolic_seed"] == 4
    assert left.metadata["exact"] is False


def test_different_seeds_change_at_least_one_materialized_node() -> None:
    left = build_complete_five_node_network(3, 1).materialize()
    right = build_complete_five_node_network(3, 2).materialize()
    assert any(not np.array_equal(left.nodes[name].data, right.nodes[name].data) for name in left.nodes)


def test_symbolic_network_rejects_invalid_dimension() -> None:
    valid = build_complete_five_node_network(2, 1)
    with pytest.raises(ValueError, match="dimension must be at least two"):
        SymbolicNetwork(valid.variables, valid.nodes, 1, valid.seed)


def test_symbolic_network_rejects_wrong_node_count() -> None:
    valid = build_complete_five_node_network(2, 1)
    with pytest.raises(ValueError, match="requires five rank-3"):
        SymbolicNetwork(valid.variables, valid.nodes[:-1], valid.dimension, valid.seed)
