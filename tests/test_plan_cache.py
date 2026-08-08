"""Tests for shape-specialized in-memory execution-plan caching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tensorcontract.backends import NumPyBackend
from tensorcontract.symbolics import (
    ContractionNode,
    IndexRole,
    InMemoryPlanCache,
    IncompatibleBindingError,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicPlannerOptions,
    SymbolicTensor,
    make_execution_plan_cache_key,
)


def _matrix_graph(
    rows: int = 2,
    inner: int = 3,
    columns: int = 4,
    *,
    symbolic_label: str = "N",
) -> SymbolicGraph:
    graph = SymbolicGraph(metadata={"family": "matmul"})
    graph.add_index(
        SymbolicIndex("i", rows, IndexRole.FREE, {"symbolic_dimension": symbolic_label})
    )
    graph.add_index(SymbolicIndex("k", inner, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("j", columns, IndexRole.FREE))
    graph.add_tensor(SymbolicTensor("A", ("i", "k"), (rows, inner), {"layout": "runtime"}))
    graph.add_tensor(SymbolicTensor("B", ("k", "j"), (inner, columns)))
    graph.add_operation(ContractionNode("C", ("A", "B"), ("i", "j")))
    return graph


def _bindings(
    rows: int = 2,
    inner: int = 3,
    columns: int = 4,
    dtype: np.dtype | type = np.float64,
) -> dict[str, np.ndarray]:
    return {
        "A": np.arange(rows * inner, dtype=dtype).reshape(rows, inner),
        "B": np.arange(inner * columns, dtype=dtype).reshape(inner, columns),
    }


@dataclass
class FakeBinding:
    shape: tuple[int, ...]
    dtype: object
    device: str
    strides: tuple[int, ...]
    requires_grad: bool = False


def test_first_lookup_misses_and_second_compatible_lookup_hits() -> None:
    graph = _matrix_graph()
    bindings = _bindings()
    cache = InMemoryPlanCache()
    first = cache.get_or_plan(graph, bindings, "numpy")
    second = cache.get_or_plan(graph, bindings, "numpy")
    assert not first.hit
    assert second.hit
    assert first.key == second.key
    assert first.key.digest == second.key.digest
    assert first.plan == second.plan
    assert cache.info().hits == 1
    assert cache.info().misses == 1
    assert cache.info().stores == 1
    assert len(cache) == 1


def test_cached_plan_matches_reference_execution() -> None:
    graph = _matrix_graph()
    bindings = _bindings()
    result = InMemoryPlanCache().get_or_plan(graph, bindings, "numpy")
    planned = NumPyBackend().execute(result.plan.graph, bindings)
    expected = bindings["A"] @ bindings["B"]
    assert np.allclose(planned, expected)


def test_shape_specializations_never_share_a_plan() -> None:
    cache = InMemoryPlanCache()
    small = cache.get_or_plan(_matrix_graph(2, 3, 4), _bindings(2, 3, 4), "numpy")
    large = cache.get_or_plan(_matrix_graph(4, 3, 4), _bindings(4, 3, 4), "numpy")
    assert not small.hit and not large.hit
    assert small.key != large.key
    assert small.key.binding_signatures[0].shape == (2, 3)
    assert large.key.binding_signatures[0].shape == (4, 3)
    assert len(cache) == 2


def test_dtype_separates_specializations() -> None:
    graph = _matrix_graph()
    cache = InMemoryPlanCache()
    float32 = cache.get_or_plan(graph, _bindings(dtype=np.float32), "numpy")
    float64 = cache.get_or_plan(graph, _bindings(dtype=np.float64), "numpy")
    assert float32.key != float64.key
    assert float32.key.binding_signatures[0].dtype == "float32"
    assert float64.key.binding_signatures[0].dtype == "float64"
    assert cache.info().misses == 2


def test_device_separates_specializations_without_importing_a_gpu_backend() -> None:
    graph = _matrix_graph()
    cpu = {
        "A": FakeBinding((2, 3), np.dtype("float32"), "cpu", (3, 1)),
        "B": FakeBinding((3, 4), np.dtype("float32"), "cpu", (4, 1)),
    }
    cuda = {
        "A": FakeBinding((2, 3), np.dtype("float32"), "cuda:0", (3, 1)),
        "B": FakeBinding((3, 4), np.dtype("float32"), "cuda:0", (4, 1)),
    }
    cpu_key = make_execution_plan_cache_key(graph, cpu, "torch")
    cuda_key = make_execution_plan_cache_key(graph, cuda, "torch")
    assert cpu_key != cuda_key
    assert cpu_key.binding_signatures[0].device == "cpu"
    assert cuda_key.binding_signatures[0].device == "cuda:0"


def test_backend_separates_specializations() -> None:
    graph = _matrix_graph()
    bindings = _bindings()
    cache = InMemoryPlanCache()
    numpy_result = cache.get_or_plan(graph, bindings, "numpy")
    torch_result = cache.get_or_plan(graph, bindings, "torch")
    assert numpy_result.key != torch_result.key
    assert numpy_result.key.backend_name == "numpy"
    assert torch_result.key.backend_name == "torch"
    assert cache.info().misses == 2


def test_layout_and_strides_separate_specializations() -> None:
    graph = _matrix_graph()
    contiguous = _bindings()
    fortran = dict(contiguous)
    fortran["A"] = np.asfortranarray(contiguous["A"])
    c_key = make_execution_plan_cache_key(graph, contiguous, "numpy")
    f_key = make_execution_plan_cache_key(graph, fortran, "numpy")
    assert c_key != f_key
    assert c_key.binding_signatures[0].layout == "c-contiguous"
    assert f_key.binding_signatures[0].layout == "f-contiguous"
    assert c_key.binding_signatures[0].strides != f_key.binding_signatures[0].strides


def test_symbolic_metadata_and_operation_structure_are_in_key() -> None:
    bindings = _bindings()
    n_key = make_execution_plan_cache_key(_matrix_graph(symbolic_label="N"), bindings, "numpy")
    m_key = make_execution_plan_cache_key(_matrix_graph(symbolic_label="M"), bindings, "numpy")
    assert n_key != m_key

    reversed_graph = _matrix_graph()
    reversed_graph.operations["C"] = ContractionNode("C", ("B", "A"), ("i", "j"))
    reversed_key = make_execution_plan_cache_key(reversed_graph, bindings, "numpy")
    assert n_key != reversed_key


def test_planner_and_backend_options_separate_specializations() -> None:
    graph = _matrix_graph()
    bindings = _bindings()
    default = make_execution_plan_cache_key(graph, bindings, "numpy")
    float32_cost = make_execution_plan_cache_key(
        graph, bindings, "numpy", SymbolicPlannerOptions(precision_bytes=4)
    )
    configured_backend = make_execution_plan_cache_key(
        graph, bindings, "numpy", backend_options={"thread_count": 2}
    )
    assert default != float32_cost
    assert default != configured_backend


def test_requires_grad_separates_specializations() -> None:
    graph = _matrix_graph()
    plain = {
        "A": FakeBinding((2, 3), np.dtype("float32"), "cpu", (3, 1), False),
        "B": FakeBinding((3, 4), np.dtype("float32"), "cpu", (4, 1), False),
    }
    differentiable = dict(plain)
    differentiable["A"] = FakeBinding(
        (2, 3), np.dtype("float32"), "cpu", (3, 1), True
    )
    assert make_execution_plan_cache_key(graph, plain, "torch") != make_execution_plan_cache_key(
        graph, differentiable, "torch"
    )


def test_caching_can_be_disabled_without_storing() -> None:
    cache = InMemoryPlanCache(enabled=False)
    graph = _matrix_graph()
    bindings = _bindings()
    first = cache.get_or_plan(graph, bindings, "numpy")
    second = cache.get_or_plan(graph, bindings, "numpy")
    assert not first.hit and not second.hit
    assert not first.cache_enabled and not second.cache_enabled
    assert cache.info().misses == 2
    assert cache.info().hits == 0
    assert cache.info().stores == 0
    assert len(cache) == 0


def test_cache_returns_mutation_isolated_plan_copies() -> None:
    cache = InMemoryPlanCache()
    graph = _matrix_graph()
    bindings = _bindings()
    first = cache.get_or_plan(graph, bindings, "numpy")
    first.plan.graph.operations.clear()
    second = cache.get_or_plan(graph, bindings, "numpy")
    assert second.hit
    second.plan.validate()
    assert "C" in second.plan.graph.operations


def test_invalidation_clear_and_lru_eviction() -> None:
    cache = InMemoryPlanCache(max_entries=1)
    first = cache.get_or_plan(_matrix_graph(), _bindings(), "numpy")
    cache.get_or_plan(_matrix_graph(4, 3, 4), _bindings(4, 3, 4), "numpy")
    assert cache.info().evictions == 1
    assert len(cache) == 1
    assert not cache.invalidate(first.key)
    current = cache.get_or_plan(_matrix_graph(4, 3, 4), _bindings(4, 3, 4), "numpy")
    assert current.hit
    assert cache.invalidate(current.key)
    assert len(cache) == 0
    cache.clear(reset_statistics=True)
    assert cache.info().hits == cache.info().misses == 0


def test_incompatible_bindings_fail_before_cache_lookup() -> None:
    graph = _matrix_graph()
    cache = InMemoryPlanCache()
    with pytest.raises(IncompatibleBindingError, match="missing binding 'B'"):
        cache.get_or_plan(graph, {"A": np.ones((2, 3))}, "numpy")
    with pytest.raises(IncompatibleBindingError, match=r"shape \(3, 2\), expected \(2, 3\)"):
        cache.get_or_plan(
            graph,
            {"A": np.ones((3, 2)), "B": np.ones((3, 4))},
            "numpy",
        )
    assert cache.info().hits == cache.info().misses == 0
