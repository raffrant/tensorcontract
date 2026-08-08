"""Optional Triton lowering for one canonical batched matrix multiplication.

This module is loaded lazily. Importing :mod:`tensorcontract` does not import
PyTorch, Triton, CUDA, or this module.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import triton as triton_runtime
import triton.language as tl

from tensorcontract.symbolics.ir import ContractionNode, SymbolicGraph

from .lowering import BatchedMatmulLowering, recognize_batched_matmul
from .torch import TorchBackend


class TritonSelectionPolicy(str, Enum):
    """Policy controlling whether the custom kernel may be selected."""

    TORCH = "torch"
    TRITON = "triton"
    AUTO = "auto"


@dataclass(frozen=True, slots=True)
class KernelSelection:
    """Inspectable execution decision for a contraction."""

    implementation: str
    reason: str


@dataclass(frozen=True, slots=True)
class FusedQECLaunch:
    """Counters and inspectable launch metadata from fused QEC execution."""

    logical_failures: torch.Tensor
    physical_error_bits: torch.Tensor
    kernel_count: int
    registers_per_thread: int | None
    register_spills: int | None
    shared_memory_bytes: int | None
    occupancy: float | None


@triton_runtime.jit
def _batched_matmul_kernel(
    left_pointer,
    right_pointer,
    output_pointer,
    size_m,
    size_n,
    size_k,
    left_stride_batch,
    left_stride_m,
    left_stride_k,
    right_stride_batch,
    right_stride_k,
    right_stride_n,
    output_stride_batch,
    output_stride_m,
    output_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one batch tile, accumulating products in float32."""
    batch = tl.program_id(axis=2)
    block_m = tl.program_id(axis=0)
    block_n = tl.program_id(axis=1)

    offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    left_pointers = (
        left_pointer
        + batch * left_stride_batch
        + offsets_m[:, None] * left_stride_m
        + offsets_k[None, :] * left_stride_k
    )
    right_pointers = (
        right_pointer
        + batch * right_stride_batch
        + offsets_k[:, None] * right_stride_k
        + offsets_n[None, :] * right_stride_n
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(size_k, BLOCK_K)):
        remaining_k = size_k - k_block * BLOCK_K
        left = tl.load(
            left_pointers,
            mask=(offsets_m[:, None] < size_m) & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        right = tl.load(
            right_pointers,
            mask=(offsets_k[:, None] < remaining_k) & (offsets_n[None, :] < size_n),
            other=0.0,
        )
        accumulator += tl.dot(left, right, input_precision="ieee")
        left_pointers += BLOCK_K * left_stride_k
        right_pointers += BLOCK_K * right_stride_k

    output_offsets = (
        batch * output_stride_batch
        + offsets_m[:, None] * output_stride_m
        + offsets_n[None, :] * output_stride_n
    )
    output_mask = (offsets_m[:, None] < size_m) & (offsets_n[None, :] < size_n)
    tl.store(output_pointer + output_offsets, accumulator, mask=output_mask)


@triton_runtime.jit
def _three_qubit_qec_trajectory_kernel(
    logical_failure_counter,
    physical_error_counter,
    number_of_shots,
    base_shot,
    seed,
    local_probability,
    correlated_probability,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse RNG, repetition-code decoding, and block reduction."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < number_of_shots
    global_shots = base_shot + offsets
    random_offsets = global_shots * 4
    correlated = tl.rand(seed, random_offsets) < correlated_probability
    local_1 = tl.rand(seed, random_offsets + 1) < local_probability
    local_2 = tl.rand(seed, random_offsets + 2) < local_probability
    local_3 = tl.rand(seed, random_offsets + 3) < local_probability

    error_1 = correlated != local_1
    error_2 = correlated != local_2
    error_3 = correlated != local_3
    syndrome_1 = error_1 != error_2
    syndrome_2 = error_2 != error_3

    # Recovery table: 10 -> q1, 11 -> q2, 01 -> q3, 00 -> identity.
    recovery_1 = syndrome_1 & (syndrome_2 == 0)
    recovery_2 = syndrome_1 & syndrome_2
    recovery_3 = (syndrome_1 == 0) & syndrome_2
    residual_1 = error_1 != recovery_1
    residual_2 = error_2 != recovery_2
    residual_3 = error_3 != recovery_3
    logical_failure = residual_1 & residual_2 & residual_3

    logical_count = tl.sum(tl.where(active & logical_failure, 1, 0), axis=0)
    physical_count = tl.sum(
        tl.where(active, error_1.to(tl.int32) + error_2.to(tl.int32) + error_3.to(tl.int32), 0),
        axis=0,
    )
    tl.atomic_add(logical_failure_counter, logical_count)
    tl.atomic_add(physical_error_counter, physical_count)


class TritonBackend(TorchBackend):
    """PyTorch execution with an optional, explicit Triton batched-GEMM path.

    The default ``policy="torch"`` never launches the custom kernel. With
    ``policy="auto"``, only shapes explicitly supplied through
    ``approved_shapes`` may use it; applications should populate that set from
    representative benchmarks. ``policy="triton"`` explicitly requests the
    custom kernel. Unsupported operations, devices, dtypes, layouts, and
    autograd inputs fall back to the ordinary PyTorch implementation.
    """

    name = "triton"

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        policy: str | TritonSelectionPolicy = TritonSelectionPolicy.TORCH,
        approved_shapes: Collection[tuple[int, int, int, int]] = (),
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        try:
            self.policy = TritonSelectionPolicy(policy)
        except ValueError as error:
            choices = [item.value for item in TritonSelectionPolicy]
            raise ValueError(
                f"unknown Triton selection policy {policy!r}; expected one of {choices}"
            ) from error
        normalized_shapes: set[tuple[int, int, int, int]] = set()
        for shape in approved_shapes:
            normalized = tuple(shape)
            if len(normalized) != 4 or any(
                not isinstance(dimension, int) or dimension < 1
                for dimension in normalized
            ):
                raise ValueError(
                    "approved Triton shapes must contain four positive integers "
                    "ordered as (batch, M, K, N)"
                )
            normalized_shapes.add(
                (normalized[0], normalized[1], normalized[2], normalized[3])
            )
        self.approved_shapes = frozenset(normalized_shapes)

    @staticmethod
    def _shape(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        return (left.shape[0], left.shape[1], left.shape[2], right.shape[2])

    def contraction_selection(
        self,
        graph: SymbolicGraph,
        operation: ContractionNode,
        values: Mapping[str, torch.Tensor],
    ) -> KernelSelection:
        """Explain which implementation would run and why."""
        lowering = recognize_batched_matmul(graph, operation)
        if lowering is None:
            return KernelSelection("torch-einsum", "operation is not canonical batched matmul")
        left = values[lowering.left]
        right = values[lowering.right]
        unsupported = self._unsupported_reason(left, right)
        if unsupported is not None:
            return KernelSelection("torch-matmul", unsupported)
        if self.policy is TritonSelectionPolicy.TORCH:
            return KernelSelection("torch-matmul", "policy explicitly selects PyTorch")
        shape = self._shape(left, right)
        if self.policy is TritonSelectionPolicy.AUTO and shape not in self.approved_shapes:
            return KernelSelection(
                "torch-matmul",
                f"shape {shape} has no benchmark approval",
            )
        reason = (
            "policy explicitly selects Triton"
            if self.policy is TritonSelectionPolicy.TRITON
            else f"shape {shape} is benchmark-approved"
        )
        return KernelSelection("triton-batched-matmul", reason)

    @staticmethod
    def _unsupported_reason(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> str | None:
        if left.device.type != "cuda" or right.device.type != "cuda":
            return "Triton kernel requires CUDA tensors"
        if left.device != right.device:
            return "input tensors are on different devices"
        if left.dtype != right.dtype:
            return "input tensors have different dtypes"
        if left.dtype not in (torch.float16, torch.float32):
            return f"dtype {left.dtype} is unsupported by the custom kernel"
        if not left.is_contiguous() or not right.is_contiguous():
            return "custom kernel currently requires contiguous inputs"
        if left.requires_grad or right.requires_grad:
            return "custom kernel has no autograd implementation"
        if left.shape[0] > 65_535:
            return "batch dimension exceeds the custom kernel launch-grid limit"
        return None

    def _contraction(
        self,
        graph: SymbolicGraph,
        operation: ContractionNode,
        values: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        lowering = recognize_batched_matmul(graph, operation)
        selection = self.contraction_selection(graph, operation, values)
        if lowering is None:
            return super()._contraction(graph, operation, values)
        left = values[lowering.left]
        right = values[lowering.right]
        if selection.implementation != "triton-batched-matmul":
            return torch.matmul(left, right)
        return self._launch_batched_matmul(left, right)

    @staticmethod
    def _launch_batched_matmul(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        batch, size_m, size_k = left.shape
        size_n = right.shape[2]
        output = torch.empty(
            (batch, size_m, size_n), dtype=left.dtype, device=left.device
        )
        block_m = 32
        block_n = 32
        block_k = 32
        grid = (
            triton_runtime.cdiv(size_m, block_m),
            triton_runtime.cdiv(size_n, block_n),
            batch,
        )
        _batched_matmul_kernel[grid](
            left,
            right,
            output,
            size_m,
            size_n,
            size_k,
            left.stride(0),
            left.stride(1),
            left.stride(2),
            right.stride(0),
            right.stride(1),
            right.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )
        return output

    @staticmethod
    def _qec_launch_metadata(
        device: torch.device,
    ) -> tuple[int | None, int | None, int | None, float | None]:
        """Read public-ish Triton metadata, returning unknowns conservatively."""
        try:
            device_index = torch.cuda.current_device() if device.index is None else device.index
            compiled_entries = _three_qubit_qec_trajectory_kernel.cache[device_index]
            compiled = next(reversed(compiled_entries.values()))
            registers = int(compiled.n_regs)
            spills = int(compiled.n_spills)
            metadata = compiled.metadata
            shared = int(metadata.shared)
            threads_per_block = int(metadata.num_warps) * 32
            properties = torch.cuda.get_device_properties(device_index)
            register_blocks = properties.regs_per_multiprocessor // max(
                1, registers * threads_per_block
            )
            thread_blocks = properties.max_threads_per_multi_processor // threads_per_block
            active_threads = min(register_blocks, thread_blocks) * threads_per_block
            occupancy = min(
                1.0,
                active_threads / properties.max_threads_per_multi_processor,
            )
            return registers, spills, shared, occupancy
        except (AttributeError, IndexError, KeyError, StopIteration, TypeError, ValueError):
            return None, None, None, None

    @staticmethod
    def launch_three_qubit_qec_trajectory(
        *,
        number_of_shots: int,
        base_shot: int,
        seed: int,
        p: float,
        rho: float,
        device: torch.device,
        block_size: int = 256,
        counters: torch.Tensor | None = None,
    ) -> FusedQECLaunch:
        """Launch one fused correlated-noise trajectory kernel.

        Only two scalar counters are written to global memory. Triton's normal
        specialization cache owns the compiled binary; callers additionally
        cache this selected schedule using their execution-plan key.
        """
        if device.type != "cuda":
            raise ValueError("fused QEC trajectory requires a CUDA device")
        if number_of_shots < 0 or base_shot < 0:
            raise ValueError("shot counts and offsets must be nonnegative")
        if block_size not in (64, 128, 256, 512):
            raise ValueError("block_size must be one of 64, 128, 256, or 512")
        if counters is None:
            counters = torch.zeros(2, dtype=torch.int32, device=device)
        elif (
            counters.shape != (2,)
            or counters.dtype != torch.int32
            or counters.device.type != "cuda"
            or (device.index is not None and counters.device.index != device.index)
        ):
            raise ValueError("fused QEC counters must be a two-element int32 CUDA tensor")
        grid = (triton_runtime.cdiv(max(1, number_of_shots), block_size),)
        _three_qubit_qec_trajectory_kernel[grid](
            counters[0:1],
            counters[1:2],
            number_of_shots,
            base_shot,
            seed,
            p,
            rho,
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=1,
        )
        registers, spills, shared, occupancy = TritonBackend._qec_launch_metadata(device)
        return FusedQECLaunch(
            counters[0], counters[1], 1, registers, spills, shared, occupancy
        )

    @classmethod
    def prepare_three_qubit_qec_trajectory(
        cls,
        *,
        device: torch.device,
        block_size: int = 256,
        number_of_shots: int = 65_536,
    ) -> FusedQECLaunch:
        """Compile/warm the fused specialization once for a cached plan."""
        launch = cls.launch_three_qubit_qec_trajectory(
            number_of_shots=max(1, number_of_shots),
            base_shot=0,
            seed=17,
            p=0.25,
            rho=0.25,
            device=device,
            block_size=block_size,
        )
        torch.cuda.synchronize(device)
        return launch
