"""Correctness and fallback tests for the optional Triton backend."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is optional")
pytest.importorskip("triton", reason="Triton is optional")

from tensorcontract.backends import ExecutionBackend, get_backend
from tensorcontract.backends.triton import TritonBackend, TritonSelectionPolicy
from tensorcontract.symbolics import (
    ContractionNode,
    IndexRole,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicTensor,
)


def _graph(
    batch: int,
    rows: int,
    contracted: int,
    columns: int,
    *,
    output: tuple[str, ...] = ("b", "i", "j"),
) -> tuple[SymbolicGraph, ContractionNode]:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("b", batch, IndexRole.BATCH))
    graph.add_index(SymbolicIndex("i", rows))
    graph.add_index(SymbolicIndex("k", contracted, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("j", columns))
    graph.add_tensor(SymbolicTensor("left", ("b", "i", "k"), (batch, rows, contracted)))
    graph.add_tensor(SymbolicTensor("right", ("b", "k", "j"), (batch, contracted, columns)))
    operation = ContractionNode("result", ("left", "right"), output)
    graph.add_operation(operation)
    return graph, operation


def test_registry_loads_triton_lazily_and_default_policy_is_torch() -> None:
    backend = get_backend("triton", device="cpu")
    assert isinstance(backend, TritonBackend)
    assert isinstance(backend, ExecutionBackend)
    assert backend.policy is TritonSelectionPolicy.TORCH


def test_invalid_selection_policy_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unknown Triton selection policy"):
        TritonBackend(policy="fastest-maybe")


def test_invalid_benchmark_approved_shape_fails_clearly() -> None:
    with pytest.raises(ValueError, match="four positive integers"):
        TritonBackend(policy="auto", approved_shapes=[(16, 64, 64)])
    with pytest.raises(ValueError, match="four positive integers"):
        TritonBackend(policy="auto", approved_shapes=[(16, 64, 0, 64)])


def test_cpu_inputs_fall_back_and_match_numpy_and_torch() -> None:
    graph, operation = _graph(3, 4, 5, 2)
    rng = np.random.default_rng(19)
    left_numpy = rng.normal(size=(3, 4, 5)).astype(np.float32)
    right_numpy = rng.normal(size=(3, 5, 2)).astype(np.float32)
    left = torch.from_numpy(left_numpy)
    right = torch.from_numpy(right_numpy)
    values = {"left": left, "right": right}
    backend = TritonBackend(policy="triton")

    selection = backend.contraction_selection(graph, operation, values)
    assert selection.implementation == "torch-matmul"
    assert selection.reason == "Triton kernel requires CUDA tensors"
    actual = backend.execute(graph, values)
    torch_reference = torch.matmul(left, right)
    numpy_reference = np.matmul(left_numpy, right_numpy)
    assert torch.equal(actual, torch_reference)
    np.testing.assert_allclose(actual.numpy(), numpy_reference, rtol=1e-5, atol=1e-6)


def test_noncanonical_shape_uses_general_torch_fallback() -> None:
    graph, operation = _graph(2, 3, 4, 5, output=("b", "j", "i"))
    left = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    right = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
    backend = TritonBackend(policy="triton")
    selection = backend.contraction_selection(
        graph, operation, {"left": left, "right": right}
    )
    assert selection.implementation == "torch-einsum"
    actual = backend.execute(graph, {"left": left, "right": right})
    expected = torch.matmul(left, right).transpose(1, 2)
    assert torch.equal(actual, expected)


def test_default_policy_does_not_silently_select_custom_kernel() -> None:
    backend = TritonBackend()
    assert backend.policy is TritonSelectionPolicy.TORCH
    assert backend.approved_shapes == frozenset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("shape", "dtype", "rtol", "atol"),
    [
        ((2, 17, 19, 23), torch.float32, 2e-5, 2e-5),
        ((3, 33, 35, 29), torch.float16, 3e-3, 3e-3),
    ],
)
def test_triton_kernel_matches_numpy_and_torch_on_cuda(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    batch, rows, contracted, columns = shape
    graph, operation = _graph(*shape)
    generator = torch.Generator(device="cuda").manual_seed(23)
    left = torch.randn(
        (batch, rows, contracted), dtype=dtype, device="cuda", generator=generator
    )
    right = torch.randn(
        (batch, contracted, columns), dtype=dtype, device="cuda", generator=generator
    )
    values = {"left": left, "right": right}
    backend = TritonBackend(policy="triton")
    selection = backend.contraction_selection(graph, operation, values)
    assert selection.implementation == "triton-batched-matmul"
    actual = backend.execute(graph, values)
    torch_reference = torch.matmul(left, right)
    numpy_reference = np.matmul(left.cpu().numpy(), right.cpu().numpy())
    torch.testing.assert_close(actual, torch_reference, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual.cpu().numpy(), numpy_reference, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unsupported_dtype_and_autograd_fall_back_on_cuda() -> None:
    graph, operation = _graph(2, 4, 5, 3)
    backend = TritonBackend(policy="triton")

    left64 = torch.randn((2, 4, 5), dtype=torch.float64, device="cuda")
    right64 = torch.randn((2, 5, 3), dtype=torch.float64, device="cuda")
    selection = backend.contraction_selection(
        graph, operation, {"left": left64, "right": right64}
    )
    assert selection.implementation == "torch-matmul"
    assert "unsupported" in selection.reason
    torch.testing.assert_close(
        backend.execute(graph, {"left": left64, "right": right64}),
        torch.matmul(left64, right64),
    )

    left_grad = left64.float().detach().requires_grad_()
    right_grad = right64.float().detach().requires_grad_()
    selection = backend.contraction_selection(
        graph, operation, {"left": left_grad, "right": right_grad}
    )
    assert selection.implementation == "torch-matmul"
    assert "autograd" in selection.reason
    result = backend.execute(graph, {"left": left_grad, "right": right_grad})
    result.sum().backward()
    assert left_grad.grad is not None
    assert right_grad.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_auto_policy_requires_benchmark_approved_shape() -> None:
    shape = (2, 16, 16, 16)
    graph, operation = _graph(*shape)
    left = torch.randn((2, 16, 16), device="cuda")
    right = torch.randn((2, 16, 16), device="cuda")
    values = {"left": left, "right": right}
    unapproved = TritonBackend(policy="auto")
    approved = TritonBackend(policy="auto", approved_shapes=[shape])
    assert unapproved.contraction_selection(graph, operation, values).implementation == "torch-matmul"
    assert approved.contraction_selection(graph, operation, values).implementation == "triton-batched-matmul"
