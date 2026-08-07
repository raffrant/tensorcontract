"""Tests for the minimal fixed-shape symbolic intermediate representation."""

from __future__ import annotations

import numpy as np
import pytest

from tensorcontract.symbolics import (
    ContractionNode,
    DimensionMismatchError,
    ElementwiseKind,
    ElementwiseNode,
    IndexRole,
    InvalidOperationError,
    MissingIndexError,
    MissingValueError,
    ReductionNode,
    RepeatedIndexError,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicTensor,
    TransposeNode,
    execute_symbolic_numpy,
    execute_symbolic_numpy_all,
)


def _matrix_graph(output_indices: tuple[str, ...] = ("i", "j")) -> SymbolicGraph:
    graph = SymbolicGraph(metadata={"purpose": "matrix multiplication"})
    graph.add_index(SymbolicIndex("i", 2, IndexRole.FREE))
    graph.add_index(SymbolicIndex("k", 3, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("j", 4, IndexRole.FREE))
    graph.add_tensor(SymbolicTensor("A", ("i", "k"), (2, 3), {"layout": "row-major"}))
    graph.add_tensor(SymbolicTensor("B", ("k", "j"), (3, 4)))
    graph.add_operation(ContractionNode("C", ("A", "B"), output_indices))
    return graph


def test_matrix_multiplication_matches_numpy() -> None:
    graph = _matrix_graph()
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    actual = execute_symbolic_numpy(graph, {"A": left, "B": right})
    assert np.allclose(actual, left @ right)
    assert graph.contracted_indices("C") == ("k",)
    assert graph.dependencies == {"C": ("A", "B")}
    assert graph.output_names == ("C",)


def test_batched_contraction_matches_numpy() -> None:
    graph = SymbolicGraph()
    for index in (
        SymbolicIndex("batch", 2, IndexRole.BATCH),
        SymbolicIndex("i", 3),
        SymbolicIndex("k", 4, IndexRole.CONTRACTED),
        SymbolicIndex("j", 2),
    ):
        graph.add_index(index)
    graph.add_tensor(SymbolicTensor("A", ("batch", "i", "k"), (2, 3, 4)))
    graph.add_tensor(SymbolicTensor("B", ("batch", "k", "j"), (2, 4, 2)))
    graph.add_operation(
        ContractionNode("C", ("A", "B"), ("batch", "i", "j"))
    )
    left = np.arange(24.0).reshape(2, 3, 4)
    right = np.arange(16.0).reshape(2, 4, 2)
    expected = np.einsum("bik,bkj->bij", left, right)
    assert np.allclose(execute_symbolic_numpy(graph, {"A": left, "B": right}), expected)


def test_multiple_contracted_indices_match_numpy() -> None:
    graph = SymbolicGraph()
    for name, dimension, role in (
        ("i", 2, IndexRole.FREE),
        ("k", 3, IndexRole.CONTRACTED),
        ("l", 4, IndexRole.CONTRACTED),
        ("j", 5, IndexRole.FREE),
    ):
        graph.add_index(SymbolicIndex(name, dimension, role))
    graph.add_tensor(SymbolicTensor("A", ("i", "k", "l"), (2, 3, 4)))
    graph.add_tensor(SymbolicTensor("B", ("k", "l", "j"), (3, 4, 5)))
    graph.add_operation(ContractionNode("C", ("A", "B"), ("i", "j")))
    left = np.arange(24.0).reshape(2, 3, 4)
    right = np.arange(60.0).reshape(3, 4, 5)
    assert graph.contracted_indices("C") == ("k", "l")
    assert np.allclose(
        execute_symbolic_numpy(graph, {"A": left, "B": right}),
        np.einsum("ikl,klj->ij", left, right),
    )


def test_free_index_order_is_explicit() -> None:
    graph = _matrix_graph(("j", "i"))
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    result = execute_symbolic_numpy(graph, {"A": left, "B": right})
    assert result.shape == (4, 2)
    assert np.allclose(result, (left @ right).T)


def test_basic_operations_and_dependency_execution() -> None:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("i", 2, IndexRole.FREE))
    graph.add_index(SymbolicIndex("j", 3, IndexRole.CONTRACTED))
    graph.add_tensor(SymbolicTensor("X", ("i", "j"), (2, 3)))
    graph.add_tensor(SymbolicTensor("Y", ("i", "j"), (2, 3)))
    graph.add_operation(
        ElementwiseNode("sum", ("X", "Y"), ("i", "j"), ElementwiseKind.ADD)
    )
    graph.add_operation(TransposeNode("transposed", "sum", ("j", "i")))
    graph.add_operation(ReductionNode("reduced", "sum", ("j",), ("i",)))
    x = np.arange(6.0).reshape(2, 3)
    y = np.ones((2, 3))
    values = execute_symbolic_numpy_all(graph, {"X": x, "Y": y})
    assert np.array_equal(values["sum"], x + y)
    assert np.array_equal(values["transposed"], (x + y).T)
    assert np.array_equal(values["reduced"], np.sum(x + y, axis=1))
    assert graph.dependencies == {
        "sum": ("X", "Y"),
        "transposed": ("sum",),
        "reduced": ("sum",),
    }


def test_multiply_operation() -> None:
    graph = SymbolicGraph(indices={"i": SymbolicIndex("i", 3)})
    graph.add_tensor(SymbolicTensor("X", ("i",), (3,)))
    graph.add_tensor(SymbolicTensor("Y", ("i",), (3,)))
    graph.add_operation(
        ElementwiseNode("product", ("X", "Y"), ("i",), ElementwiseKind.MULTIPLY)
    )
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    assert np.array_equal(execute_symbolic_numpy(graph, {"X": x, "Y": y}), x * y)


def test_readable_debug_representation_includes_roles_and_dependencies() -> None:
    graph = _matrix_graph()
    rendered = graph.debug_string()
    assert "i:2[free]" in rendered
    assert "k:3[contracted]" in rendered
    assert "tensor A(i, k) shape=2×3" in rendered
    assert "contract C = (A, B) -> (i, j)" in rendered
    assert str(graph) == rendered


def test_incompatible_dimensions_are_rejected() -> None:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("i", 2))
    with pytest.raises(DimensionMismatchError, match="incompatible dimensions 2 and 3"):
        graph.add_index(SymbolicIndex("i", 3))
    with pytest.raises(DimensionMismatchError, match=r"shape \(3,\).+\(2,\)"):
        graph.add_tensor(SymbolicTensor("bad", ("i",), (3,)))


def test_missing_and_repeated_tensor_indices_are_rejected() -> None:
    graph = SymbolicGraph(indices={"i": SymbolicIndex("i", 2)})
    with pytest.raises(MissingIndexError, match="missing indices.*j"):
        graph.add_tensor(SymbolicTensor("missing", ("i", "j"), (2, 2)))
    with pytest.raises(RepeatedIndexError, match="repeats indices.*i"):
        graph.add_tensor(SymbolicTensor("diagonal", ("i", "i"), (2, 2)))


def test_invalid_contraction_dependencies_and_indices_are_rejected() -> None:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("i", 2))
    graph.add_index(SymbolicIndex("j", 3))
    graph.add_index(SymbolicIndex("unused", 5))
    graph.add_tensor(SymbolicTensor("A", ("i",), (2,)))
    graph.add_tensor(SymbolicTensor("B", ("j",), (3,)))
    with pytest.raises(MissingValueError, match="missing value 'missing'"):
        graph.add_operation(ContractionNode("bad-input", ("A", "missing"), ("i",)))
    with pytest.raises(MissingIndexError, match="missing indices.*missing"):
        graph.add_operation(ContractionNode("bad-output", ("A", "B"), ("missing",)))
    with pytest.raises(MissingIndexError, match="unavailable indices.*unused"):
        graph.add_operation(ContractionNode("unavailable-output", ("A", "B"), ("unused",)))
    with pytest.raises(RepeatedIndexError, match="repeats indices.*i"):
        graph.add_operation(ContractionNode("repeated-output", ("A", "B"), ("i", "i")))
    with pytest.raises(InvalidOperationError, match="implicitly reduce unpaired indices"):
        graph.add_operation(ContractionNode("unpaired", ("A", "B"), ("i",)))
    with pytest.raises(InvalidOperationError, match="repeats an input"):
        graph.add_operation(ContractionNode("same-input", ("A", "A"), ("i",)))


def test_invalid_transpose_elementwise_and_reduction_are_rejected() -> None:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("i", 2))
    graph.add_index(SymbolicIndex("j", 3))
    graph.add_tensor(SymbolicTensor("A", ("i", "j"), (2, 3)))
    graph.add_tensor(SymbolicTensor("B", ("j", "i"), (3, 2)))
    with pytest.raises(InvalidOperationError, match="not a permutation"):
        graph.add_operation(TransposeNode("bad-transpose", "A", ("i",)))
    with pytest.raises(InvalidOperationError, match="identical index order"):
        graph.add_operation(
            ElementwiseNode("bad-add", ("A", "B"), ("i", "j"), ElementwiseKind.ADD)
        )
    with pytest.raises(MissingIndexError, match="cannot reduce missing indices"):
        graph.add_operation(ReductionNode("bad-reduce", "A", ("missing",), ("i", "j")))
    with pytest.raises(InvalidOperationError, match="output must be"):
        graph.add_operation(ReductionNode("bad-order", "A", ("j",), ()))


def test_numpy_bindings_and_output_selection_are_validated() -> None:
    graph = _matrix_graph()
    left = np.ones((2, 3))
    right = np.ones((3, 4))
    with pytest.raises(MissingValueError, match="missing NumPy binding.*B"):
        execute_symbolic_numpy(graph, {"A": left})
    with pytest.raises(DimensionMismatchError, match=r"shape \(3, 2\), expected \(2, 3\)"):
        execute_symbolic_numpy(graph, {"A": left.T, "B": right})
    with pytest.raises(MissingValueError, match="unknown requested output"):
        execute_symbolic_numpy(graph, {"A": left, "B": right}, output="missing")


def test_ambiguous_implicit_output_requires_explicit_name() -> None:
    graph = SymbolicGraph(indices={"i": SymbolicIndex("i", 2)})
    graph.add_tensor(SymbolicTensor("A", ("i",), (2,)))
    graph.add_tensor(SymbolicTensor("B", ("i",), (2,)))
    bindings = {"A": np.ones(2), "B": np.ones(2)}
    with pytest.raises(MissingValueError, match="output is ambiguous"):
        execute_symbolic_numpy(graph, bindings)
    assert np.array_equal(execute_symbolic_numpy(graph, bindings, output="A"), np.ones(2))
