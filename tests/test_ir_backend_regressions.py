"""Characterization tests for the existing concrete IR and NumPy backend."""

from __future__ import annotations

import numpy as np
import pytest

from tensorcontract.backend import contract_numpy, execute_numpy
from tensorcontract.ir import Index, TensorNetwork, TensorNode
from tensorcontract.planner import ContractionPlan, plan_contraction, PlanConstraints


def _network(
    dimensions: dict[str, int],
    nodes: tuple[tuple[str, tuple[str, ...], np.ndarray], ...],
    open_indices: tuple[str, ...] = (),
) -> TensorNetwork:
    network = TensorNetwork(open_indices=open_indices)
    for name, dimension in dimensions.items():
        network.add_index(Index(name, dimension))
    for name, indices, data in nodes:
        network.add_node(TensorNode(name, indices, data=np.asarray(data)))
    network.validate()
    return network


def test_scalar_empty_network_and_scalar_node() -> None:
    assert contract_numpy(TensorNetwork(scalar=2.5)) == pytest.approx(2.5)
    network = _network({}, (("value", (), np.asarray(3.0)),))
    network.scalar = 2.0
    assert contract_numpy(network) == pytest.approx(6.0)


def test_matrix_multiplication() -> None:
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    network = _network(
        {"i": 2, "k": 3, "j": 4},
        (("left", ("i", "k"), left), ("right", ("k", "j"), right)),
        ("i", "j"),
    )
    assert np.allclose(contract_numpy(network), left @ right)


def test_batched_matrix_multiplication_preserves_batch_index() -> None:
    left = np.arange(12.0).reshape(2, 2, 3)
    right = np.arange(12.0).reshape(2, 3, 2)
    network = _network(
        {"batch": 2, "i": 2, "k": 3, "j": 2},
        (("left", ("batch", "i", "k"), left), ("right", ("batch", "k", "j"), right)),
        ("batch", "i", "j"),
    )
    assert np.allclose(contract_numpy(network), np.einsum("bik,bkj->bij", left, right))


def test_multiple_contracted_indices() -> None:
    left = np.arange(24.0).reshape(2, 3, 4)
    right = np.arange(60.0).reshape(3, 4, 5)
    network = _network(
        {"i": 2, "k": 3, "l": 4, "j": 5},
        (("left", ("i", "k", "l"), left), ("right", ("k", "l", "j"), right)),
        ("i", "j"),
    )
    assert np.allclose(contract_numpy(network), np.einsum("ikl,klj->ij", left, right))


def test_outer_product_and_requested_free_index_order() -> None:
    left = np.array([1.0, 2.0])
    right = np.array([3.0, 4.0, 5.0])
    network = _network(
        {"i": 2, "j": 3},
        (("left", ("i",), left), ("right", ("j",), right)),
        ("j", "i"),
    )
    assert np.array_equal(contract_numpy(network), np.outer(right, left))


def test_single_tensor_transpose_and_trace_via_repeated_index() -> None:
    matrix = np.arange(6.0).reshape(2, 3)
    transpose = _network({"i": 2, "j": 3}, (("matrix", ("i", "j"), matrix),), ("j", "i"))
    assert np.array_equal(contract_numpy(transpose), matrix.T)

    square = np.arange(9.0).reshape(3, 3)
    trace = _network({"i": 3}, (("matrix", ("i", "i"), square),))
    assert contract_numpy(trace) == pytest.approx(np.trace(square))
    assert trace.incidence()["i"] == ["matrix"]


def test_index_and_network_construction_errors_are_identifiable() -> None:
    with pytest.raises(ValueError, match="positive dimension"):
        Index("bad", 0)
    with pytest.raises(ValueError, match="needs a name"):
        Index("", 2)

    network = TensorNetwork()
    network.add_index(Index("i", 2))
    with pytest.raises(ValueError, match="conflicting dimensions"):
        network.add_index(Index("i", 3))
    with pytest.raises(ValueError, match="unknown indices.*missing"):
        network.add_node(TensorNode("bad", ("missing",), data=np.ones(2)))
    network.add_node(TensorNode("good", ("i",), data=np.ones(2)))
    with pytest.raises(ValueError, match="duplicate node"):
        network.add_node(TensorNode("good", ("i",), data=np.ones(2)))


def test_invalid_shapes_open_indices_and_missing_data() -> None:
    network = TensorNetwork()
    network.add_index(Index("i", 2))
    with pytest.raises(ValueError, match=r"shape \(3,\) != \(2,\)"):
        network.add_node(TensorNode("bad-shape", ("i",), data=np.ones(3)))

    unknown_open = TensorNetwork(open_indices=("missing",))
    with pytest.raises(ValueError, match="unknown open index"):
        unknown_open.validate()

    symbolic = TensorNetwork(indices={"i": Index("i", 2)})
    symbolic.add_node(TensorNode("symbolic", ("i",)))
    with pytest.raises(ValueError, match="has no concrete data"):
        symbolic.validate()


def test_executor_rejects_an_incomplete_plan() -> None:
    network = _network(
        {"i": 2, "j": 2},
        (("left", ("i",), np.ones(2)), ("right", ("j",), np.ones(2))),
    )
    incomplete = ContractionPlan("incomplete", (), 0, 2, 0, 0.0)
    with pytest.raises(ValueError, match="did not contract all tensors"):
        execute_numpy(network, incomplete)


def test_network_copy_preserves_values_and_is_structurally_independent() -> None:
    original = _network({"i": 2}, (("node", ("i",), np.ones(2)),), ("i",))
    copied = original.copy()
    copied.nodes.pop("node")
    copied.open_indices = ()
    assert "node" in original.nodes
    assert original.open_indices == ("i",)
    assert np.array_equal(contract_numpy(original), np.ones(2))
