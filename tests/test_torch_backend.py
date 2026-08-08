"""Numerical and autograd tests for the optional PyTorch backend."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch backend is optional")

from tensorcontract.backends import BackendExecutionError, BackendUnavailableError, ExecutionBackend, get_backend
from tensorcontract.backends.torch import TorchBackend
from tensorcontract.symbolics import (
    ContractionNode,
    ElementwiseKind,
    ElementwiseNode,
    IndexRole,
    ReductionNode,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicTensor,
    TransposeNode,
    plan_symbolic_contractions,
)


def _matrix_graph(output: tuple[str, ...] = ("i", "j")) -> SymbolicGraph:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("i", 2))
    graph.add_index(SymbolicIndex("k", 3, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("j", 4))
    graph.add_tensor(SymbolicTensor("A", ("i", "k"), (2, 3)))
    graph.add_tensor(SymbolicTensor("B", ("k", "j"), (3, 4)))
    graph.add_operation(ContractionNode("C", ("A", "B"), output))
    return graph


def test_registry_loads_torch_lazily_and_satisfies_protocol() -> None:
    backend = get_backend("torch", device="cpu")
    assert isinstance(backend, TorchBackend)
    assert isinstance(backend, ExecutionBackend)
    assert backend.name == "torch"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matrix_contraction_matches_numpy_with_dtype_tolerance(dtype: torch.dtype) -> None:
    graph = _matrix_graph()
    left = torch.arange(6, dtype=dtype).reshape(2, 3)
    right = torch.arange(12, dtype=dtype).reshape(3, 4)
    result = TorchBackend().execute(graph, {"A": left, "B": right})
    expected = np.arange(6, dtype=np.float64).reshape(2, 3) @ np.arange(
        12, dtype=np.float64
    ).reshape(3, 4)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    assert result.dtype == dtype
    np.testing.assert_allclose(result.detach().numpy(), expected, rtol=tolerance, atol=tolerance)


def test_batched_and_multiple_index_contractions_match_numpy() -> None:
    batch = SymbolicGraph()
    for index in (
        SymbolicIndex("batch", 2, IndexRole.BATCH),
        SymbolicIndex("i", 2),
        SymbolicIndex("k", 3, IndexRole.CONTRACTED),
        SymbolicIndex("j", 2),
    ):
        batch.add_index(index)
    batch.add_tensor(SymbolicTensor("A", ("batch", "i", "k"), (2, 2, 3)))
    batch.add_tensor(SymbolicTensor("B", ("batch", "k", "j"), (2, 3, 2)))
    batch.add_operation(ContractionNode("C", ("A", "B"), ("batch", "i", "j")))
    left = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3)
    right = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)
    actual = TorchBackend().execute(batch, {"A": left, "B": right})
    expected = np.einsum("bik,bkj->bij", left.numpy(), right.numpy())
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)

    multiple = SymbolicGraph()
    for name, dimension, role in (
        ("i", 2, IndexRole.FREE),
        ("k", 2, IndexRole.CONTRACTED),
        ("l", 3, IndexRole.CONTRACTED),
        ("j", 2, IndexRole.FREE),
    ):
        multiple.add_index(SymbolicIndex(name, dimension, role))
    multiple.add_tensor(SymbolicTensor("A", ("i", "k", "l"), (2, 2, 3)))
    multiple.add_tensor(SymbolicTensor("B", ("k", "l", "j"), (2, 3, 2)))
    multiple.add_operation(ContractionNode("C", ("A", "B"), ("i", "j")))
    a = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3)
    b = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)
    result = TorchBackend().execute(multiple, {"A": a, "B": b})
    np.testing.assert_allclose(
        result.numpy(), np.einsum("ikl,klj->ij", a.numpy(), b.numpy()),
        rtol=1e-12, atol=1e-12,
    )


def test_free_index_order_and_all_basic_operations() -> None:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("i", 2))
    graph.add_index(SymbolicIndex("j", 3, IndexRole.CONTRACTED))
    graph.add_tensor(SymbolicTensor("X", ("i", "j"), (2, 3)))
    graph.add_tensor(SymbolicTensor("Y", ("i", "j"), (2, 3)))
    graph.add_operation(ElementwiseNode("add", ("X", "Y"), ("i", "j"), ElementwiseKind.ADD))
    graph.add_operation(ElementwiseNode("product", ("add", "add"), ("i", "j"), ElementwiseKind.MULTIPLY))
    graph.add_operation(TransposeNode("transpose", "product", ("j", "i")))
    graph.add_operation(ReductionNode("reduce", "product", ("j",), ("i",)))
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    y = torch.ones((2, 3), dtype=torch.float64)
    values = TorchBackend().execute_all(graph, {"X": x, "Y": y})
    expected = (x + y) * (x + y)
    assert torch.equal(values["add"], x + y)
    assert torch.equal(values["product"], expected)
    assert torch.equal(values["transpose"], expected.permute(1, 0))
    assert torch.equal(values["reduce"], expected.sum(dim=1))


def test_autograd_is_preserved() -> None:
    graph = _matrix_graph()
    left = torch.arange(6, dtype=torch.float64).reshape(2, 3).requires_grad_()
    right = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    result = TorchBackend().execute(graph, {"A": left, "B": right})
    assert result.requires_grad
    result.sum().backward()
    expected_left_gradient = torch.ones((2, 4), dtype=torch.float64) @ right.detach().T
    expected_right_gradient = left.detach().T @ torch.ones((2, 4), dtype=torch.float64)
    assert torch.allclose(left.grad, expected_left_gradient)
    assert torch.allclose(right.grad, expected_right_gradient)


def test_dtype_device_options_and_planned_graph_execution() -> None:
    graph = _matrix_graph()
    plan = plan_symbolic_contractions(graph)
    left = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    right = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    result = TorchBackend(device="cpu", dtype=torch.float64).execute(
        plan.graph, {"A": left, "B": right}
    )
    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert result.requires_grad
    result.sum().backward()
    assert left.grad is not None and right.grad is not None


def test_torch_runtime_error_identifies_operation_and_tensors() -> None:
    graph = _matrix_graph()
    left = torch.ones((2, 3), dtype=torch.float32)
    right = torch.ones((3, 4), dtype=torch.float64)
    with pytest.raises(BackendExecutionError) as captured:
        TorchBackend().execute(graph, {"A": left, "B": right})
    assert captured.value.operation == "C"
    assert captured.value.operation_type == "ContractionNode"
    assert captured.value.dependencies == ("A", "B")
    assert "tensors ['A', 'B']" in str(captured.value)


def test_cuda_request_is_explicitly_rejected_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this test host")
    with pytest.raises(BackendUnavailableError, match="CUDA is unavailable"):
        TorchBackend(device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_execution_preserves_cuda_device() -> None:
    graph = _matrix_graph()
    left = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
    right = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
    result = TorchBackend().execute(graph, {"A": left, "B": right})
    assert result.device.type == "cuda"
    expected = (left @ right).cpu().numpy()
    np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)
