"""Regression tests for the conservative NumPy batched-matmul lowering."""

from __future__ import annotations

import numpy as np
import pytest

from tensorcontract.backends import NumPyBackend, get_backend
from tensorcontract.symbolics import (
    ContractionNode,
    IndexRole,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicTensor,
)


def _batched_graph(
    *,
    output_indices: tuple[str, ...] = ("b", "i", "j"),
    batch_role: IndexRole = IndexRole.BATCH,
) -> tuple[SymbolicGraph, ContractionNode]:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("b", 3, batch_role))
    graph.add_index(SymbolicIndex("i", 4))
    graph.add_index(SymbolicIndex("k", 5, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("j", 2))
    graph.add_tensor(SymbolicTensor("left", ("b", "i", "k"), (3, 4, 5)))
    graph.add_tensor(SymbolicTensor("right", ("b", "k", "j"), (3, 5, 2)))
    operation = ContractionNode("result", ("left", "right"), output_indices)
    graph.add_operation(operation)
    return graph, operation


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_canonical_batched_contraction_uses_matmul_and_matches_einsum(dtype: type[np.generic]) -> None:
    graph, operation = _batched_graph()
    rng = np.random.default_rng(7)
    bindings = {
        "left": rng.normal(size=(3, 4, 5)).astype(dtype),
        "right": rng.normal(size=(3, 5, 2)).astype(dtype),
    }

    optimized = NumPyBackend()
    baseline = NumPyBackend(enable_batched_matmul=False)
    assert optimized.contraction_implementation(graph, operation) == "batched-matmul"
    assert baseline.contraction_implementation(graph, operation) == "einsum"
    np.testing.assert_allclose(
        optimized.execute(graph, bindings),
        baseline.execute(graph, bindings),
        rtol=1e-5 if dtype is np.float32 else 1e-12,
        atol=1e-6 if dtype is np.float32 else 1e-12,
    )


def test_canonical_lowering_calls_numpy_matmul(monkeypatch: pytest.MonkeyPatch) -> None:
    graph, _ = _batched_graph()
    bindings = {"left": np.ones((3, 4, 5)), "right": np.ones((3, 5, 2))}
    original = np.matmul
    calls = 0

    def recording_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(np, "matmul", recording_matmul)
    result = NumPyBackend().execute(graph, bindings)
    assert calls == 1
    np.testing.assert_array_equal(result, np.full((3, 4, 2), 5.0))


def test_noncanonical_output_order_falls_back_and_remains_correct() -> None:
    graph, operation = _batched_graph(output_indices=("b", "j", "i"))
    rng = np.random.default_rng(11)
    left = rng.normal(size=(3, 4, 5))
    right = rng.normal(size=(3, 5, 2))
    backend = NumPyBackend()
    assert backend.contraction_implementation(graph, operation) == "einsum"
    np.testing.assert_allclose(
        backend.execute(graph, {"left": left, "right": right}),
        np.matmul(left, right).transpose(0, 2, 1),
    )


def test_incorrect_batch_role_prevents_lowering() -> None:
    graph, operation = _batched_graph(batch_role=IndexRole.FREE)
    assert NumPyBackend().contraction_implementation(graph, operation) == "einsum"


def test_strided_inputs_are_numerically_correct() -> None:
    graph, _ = _batched_graph()
    rng = np.random.default_rng(13)
    left_storage = rng.normal(size=(3, 4, 10))
    right_storage = rng.normal(size=(3, 5, 4))
    left = left_storage[..., ::2]
    right = right_storage[..., ::2]
    assert not left.flags.c_contiguous
    assert not right.flags.c_contiguous
    result = NumPyBackend().execute(graph, {"left": left, "right": right})
    np.testing.assert_allclose(result, np.einsum("bik,bkj->bij", left, right))


def test_backend_registry_exposes_optimization_opt_out() -> None:
    graph, operation = _batched_graph()
    backend = get_backend("numpy", enable_batched_matmul=False)
    assert isinstance(backend, NumPyBackend)
    assert backend.contraction_implementation(graph, operation) == "einsum"
