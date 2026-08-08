"""Tests for the backend-independent interface and NumPy implementation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tensorcontract.backends import (
    BackendExecutionError,
    BackendNotFoundError,
    BackendUnavailableError,
    ExecutionBackend,
    NumPyBackend,
    available_backends,
    get_backend,
    is_backend_available,
)
from tensorcontract.symbolics import (
    ContractionNode,
    DimensionMismatchError,
    ElementwiseKind,
    ElementwiseNode,
    IndexRole,
    ReductionNode,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicTensor,
    TransposeNode,
    execute_symbolic_numpy,
)


def _operation_graph() -> SymbolicGraph:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("i", 2, IndexRole.FREE))
    graph.add_index(SymbolicIndex("k", 3, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("j", 2, IndexRole.FREE))
    graph.add_tensor(SymbolicTensor("A", ("i", "k"), (2, 3)))
    graph.add_tensor(SymbolicTensor("B", ("k", "j"), (3, 2)))
    graph.add_tensor(SymbolicTensor("bias", ("i", "j"), (2, 2)))
    graph.add_operation(ContractionNode("contract", ("A", "B"), ("i", "j")))
    graph.add_operation(
        ElementwiseNode("add", ("contract", "bias"), ("i", "j"), ElementwiseKind.ADD)
    )
    graph.add_operation(
        ElementwiseNode("square", ("add", "add"), ("i", "j"), ElementwiseKind.MULTIPLY)
    )
    graph.add_operation(TransposeNode("transpose", "square", ("j", "i")))
    graph.add_operation(ReductionNode("reduce", "square", ("j",), ("i",)))
    return graph


def test_numpy_backend_satisfies_protocol_and_registry_is_deterministic() -> None:
    backend = get_backend("numpy")
    assert isinstance(backend, ExecutionBackend)
    assert isinstance(backend, NumPyBackend)
    assert backend.name == "numpy"
    assert available_backends() == ("numpy",)
    assert is_backend_available("numpy")
    assert not is_backend_available("missing")
    with pytest.raises(BackendNotFoundError, match="available backends.*numpy"):
        get_backend("missing")


def test_unavailable_optional_backend_fails_without_affecting_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    import tensorcontract.backends as registry

    def unavailable(name: str, package: str | None = None):
        raise ModuleNotFoundError("torch deliberately unavailable")

    monkeypatch.setattr(registry, "import_module", unavailable)
    with pytest.raises(BackendUnavailableError, match=r"install tensorcontract\[torch\]"):
        registry.get_backend("torch")
    assert isinstance(registry.get_backend("numpy"), NumPyBackend)


def test_unavailable_triton_backend_fails_without_affecting_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorcontract.backends as registry

    def unavailable(name: str, package: str | None = None):
        raise ModuleNotFoundError("triton deliberately unavailable")

    monkeypatch.setattr(registry, "import_module", unavailable)
    with pytest.raises(BackendUnavailableError, match=r"tensorcontract\[triton\]"):
        registry.get_backend("triton")
    assert isinstance(registry.get_backend("numpy"), NumPyBackend)


def test_numpy_backend_executes_every_supported_operation() -> None:
    graph = _operation_graph()
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(6.0).reshape(3, 2)
    bias = np.ones((2, 2))
    values = NumPyBackend().execute_all(graph, {"A": left, "B": right, "bias": bias})
    contracted = left @ right
    added = contracted + bias
    squared = added * added
    assert np.allclose(values["contract"], contracted)
    assert np.allclose(values["add"], added)
    assert np.allclose(values["square"], squared)
    assert np.allclose(values["transpose"], squared.T)
    assert np.allclose(values["reduce"], np.sum(squared, axis=1))


def test_backend_and_stage3_function_api_are_numerically_equivalent() -> None:
    graph = _operation_graph()
    bindings = {
        "A": np.arange(6.0).reshape(2, 3),
        "B": np.arange(6.0).reshape(3, 2),
        "bias": np.ones((2, 2)),
    }
    direct = NumPyBackend().execute(graph, bindings, "reduce")
    compatibility = execute_symbolic_numpy(graph, bindings, "reduce")
    assert np.array_equal(direct, compatibility)


def test_runtime_error_names_operation_type_and_dependency_tensors() -> None:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("i", 2))
    graph.add_index(SymbolicIndex("k", 2, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("j", 2))
    graph.add_tensor(SymbolicTensor("left", ("i", "k"), (2, 2)))
    graph.add_tensor(SymbolicTensor("right", ("k", "j"), (2, 2)))
    graph.add_operation(
        ContractionNode("matmul", ("left", "right"), ("i", "j"))
    )
    strings = np.full((2, 2), "not-numeric")
    with pytest.raises(BackendExecutionError) as captured:
        NumPyBackend().execute(graph, {"left": strings, "right": strings})
    error = captured.value
    assert error.backend == "numpy"
    assert error.operation == "matmul"
    assert error.operation_type == "ContractionNode"
    assert error.dependencies == ("left", "right")
    assert "operation 'matmul'" in str(error)
    assert "tensors ['left', 'right']" in str(error)


def test_binding_error_identifies_tensor_name() -> None:
    graph = SymbolicGraph(indices={"i": SymbolicIndex("i", 2)})
    graph.add_tensor(SymbolicTensor("input", ("i",), (2,)))
    with pytest.raises(DimensionMismatchError, match=r"tensor 'input'.*expected \(2,\)"):
        NumPyBackend().execute(graph, {"input": np.ones(3)})


def test_backends_package_imports_when_all_optional_runtimes_are_blocked() -> None:
    script = """
import builtins
real_import = builtins.__import__
blocked = ('torch', 'jax', 'triton', 'cuda')
def guarded(name, *args, **kwargs):
    if any(name == item or name.startswith(item + '.') for item in blocked):
        raise ImportError(name + ' deliberately unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from tensorcontract.backends import NumPyBackend, available_backends
print(NumPyBackend.name, available_backends())
"""
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True, env=environment,
    )
    assert completed.stdout.strip() == "numpy ('numpy',)"


def test_symbolic_expression_example_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "symbolic_numpy_execution.py")],
        check=True, capture_output=True, text=True, env=environment,
    )
    assert "left expression:" in completed.stdout
    assert "contract result = (left, right)" in completed.stdout
    assert "matches NumPy reference: True" in completed.stdout
