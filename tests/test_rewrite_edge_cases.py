"""Characterize conservative rewrite preconditions and trace formatting."""

import numpy as np
import pytest

from tensorcontract.backend import contract_numpy
from tensorcontract.ir import Index, TensorKind, TensorNetwork, TensorNode
from tensorcontract.rewrite import RewriteEngine, format_trace


def test_empty_trace_format_and_noop_rewrite() -> None:
    network = TensorNetwork(indices={"i": Index("i", 2)}, open_indices=("i",))
    network.add_node(TensorNode("vector", ("i",), data=np.array([1.0, 2.0])))
    simplified, trace = RewriteEngine().simplify(network)
    assert trace == ()
    assert format_trace(trace) == "(no rewrites)"
    assert np.array_equal(contract_numpy(simplified), np.array([1.0, 2.0]))


def test_scalar_folding_multiplies_global_scalar_without_mutating_source() -> None:
    source = TensorNetwork(scalar=5.0)
    source.add_node(TensorNode("a", (), TensorKind.SCALAR, np.asarray(2.0)))
    source.add_node(TensorNode("b", (), TensorKind.SCALAR, np.asarray(3.0)))
    simplified, trace = RewriteEngine().simplify(source)
    assert simplified.scalar == pytest.approx(30.0)
    assert set(source.nodes) == {"a", "b"}
    assert trace[0].rule == "scalar-folding"
    assert trace[0].matched == ("a", "b")


def test_symbolic_scalar_folding_has_clear_failure() -> None:
    network = TensorNetwork(nodes={"symbolic": TensorNode("symbolic", (), TensorKind.SCALAR)})
    with pytest.raises(ValueError, match="cannot fold a symbolic scalar"):
        RewriteEngine().simplify(network)


def test_identity_touching_an_open_index_is_not_eliminated() -> None:
    network = TensorNetwork(open_indices=("left", "right"))
    network.add_index(Index("left", 2))
    network.add_index(Index("right", 2))
    network.add_node(TensorNode("identity", ("left", "right"), TensorKind.IDENTITY, np.eye(2)))
    simplified, trace = RewriteEngine().simplify(network)
    assert trace == ()
    assert "identity" in simplified.nodes
    assert np.array_equal(contract_numpy(simplified), np.eye(2))


def test_nonidentity_data_is_not_removed_by_identity_rule() -> None:
    network = TensorNetwork()
    for name in ("left", "right"):
        network.add_index(Index(name, 2))
    network.add_node(TensorNode("left-leaf", ("left",), data=np.ones(2)))
    network.add_node(TensorNode("claimed-identity", ("left", "right"), TensorKind.IDENTITY, np.ones((2, 2))))
    network.add_node(TensorNode("right-leaf", ("right",), data=np.ones(2)))
    _, trace = RewriteEngine().simplify(network)
    assert all(record.rule != "identity-elimination" for record in trace)


def test_degree_one_leaf_on_open_index_is_not_absorbed() -> None:
    network = TensorNetwork(open_indices=("i",))
    network.add_index(Index("i", 2))
    network.add_node(TensorNode("leaf", ("i",), data=np.array([2.0, 3.0])))
    network.add_node(TensorNode("other", ("i",), data=np.array([5.0, 7.0])))
    simplified, trace = RewriteEngine().simplify(network)
    assert trace == ()
    assert set(simplified.nodes) == {"leaf", "other"}
