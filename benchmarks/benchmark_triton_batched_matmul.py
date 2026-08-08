"""Compare NumPy, PyTorch, and the optional Triton batched-GEMM path.

The Triton column is measured only with an available CUDA runtime. Its cold
first-call measurement includes compilation, allocation, and one launch, so it
is a conservative upper bound rather than a pure compiler timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import statistics
import time
from typing import Any, Callable

import numpy as np

from tensorcontract.backends import NumPyBackend, get_backend
from tensorcontract.symbolics import (
    ContractionNode,
    IndexRole,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicTensor,
)


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    rows: int
    contracted: int
    columns: int
    repeats: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.rows, self.contracted, self.columns


def make_graph(workload: Workload) -> SymbolicGraph:
    graph = SymbolicGraph()
    graph.add_index(SymbolicIndex("b", workload.batch, IndexRole.BATCH))
    graph.add_index(SymbolicIndex("i", workload.rows))
    graph.add_index(SymbolicIndex("k", workload.contracted, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("j", workload.columns))
    graph.add_tensor(
        SymbolicTensor(
            "left",
            ("b", "i", "k"),
            (workload.batch, workload.rows, workload.contracted),
        )
    )
    graph.add_tensor(
        SymbolicTensor(
            "right",
            ("b", "k", "j"),
            (workload.batch, workload.contracted, workload.columns),
        )
    )
    graph.add_operation(
        ContractionNode("result", ("left", "right"), ("b", "i", "j"))
    )
    return graph


def cpu_median_ms(function: Callable[[], Any], repeats: int) -> float:
    for _ in range(5):
        function()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples)


def cuda_measurements(
    function: Callable[[], Any],
    repeats: int,
    torch: Any,
) -> tuple[float, float]:
    torch.cuda.synchronize()
    cold_start = time.perf_counter_ns()
    function()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter_ns() - cold_start) / 1_000_000
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return cold_ms, statistics.median(samples)


def main() -> None:
    workloads = (
        Workload("medium", 16, 64, 64, 64, 50),
        Workload("large", 8, 128, 128, 128, 30),
    )
    numpy_backend = NumPyBackend()
    rng = np.random.default_rng(20260909)

    torch_available = find_spec("torch") is not None
    triton_available = find_spec("triton") is not None
    torch = None
    cuda_available = False
    if torch_available:
        import torch as torch_runtime

        torch = torch_runtime
        cuda_available = torch.cuda.is_available()

    print("Operation: canonical (B,M,K) x (B,K,N) -> (B,M,N), float32")
    print("NumPy and PyTorch are eager: compilation_ms=0.000")
    print(
        f"environment: torch_installed={torch_available} "
        f"triton_installed={triton_available} cuda_available={cuda_available}"
    )
    for workload in workloads:
        graph = make_graph(workload)
        left_numpy = rng.normal(
            size=(workload.batch, workload.rows, workload.contracted)
        ).astype(np.float32)
        right_numpy = rng.normal(
            size=(workload.batch, workload.contracted, workload.columns)
        ).astype(np.float32)
        numpy_bindings = {"left": left_numpy, "right": right_numpy}
        reference = numpy_backend.execute(graph, numpy_bindings)
        numpy_ms = cpu_median_ms(
            lambda: numpy_backend.execute(graph, numpy_bindings), workload.repeats
        )
        print(f"\n{workload.name} shape={workload.shape}")
        print(f"  NumPy reference  runtime_ms={numpy_ms:.4f} max_error=0.000e+00")

        if not torch_available or torch is None:
            print("  PyTorch          UNAVAILABLE (optional dependency not installed)")
            print("  Triton           UNAVAILABLE (PyTorch is required)")
            continue

        if not cuda_available:
            torch_backend = get_backend("torch", device="cpu")
            torch_bindings = {
                "left": torch.from_numpy(left_numpy),
                "right": torch.from_numpy(right_numpy),
            }
            torch_result = torch_backend.execute(graph, torch_bindings)
            torch_ms = cpu_median_ms(
                lambda: torch_backend.execute(graph, torch_bindings), workload.repeats
            )
            torch_error = float(
                np.max(np.abs(torch_result.detach().numpy() - reference))
            )
            print(
                f"  PyTorch CPU      runtime_ms={torch_ms:.4f} "
                f"max_error={torch_error:.3e}"
            )
            reason = (
                "CUDA runtime unavailable"
                if triton_available
                else "optional Triton dependency not installed"
            )
            print(f"  Triton           UNAVAILABLE ({reason}); no speedup claim")
            continue

        torch_bindings = {
            "left": torch.from_numpy(left_numpy).cuda(),
            "right": torch.from_numpy(right_numpy).cuda(),
        }
        torch_backend = get_backend("torch", device="cuda")
        torch_cold, torch_ms = cuda_measurements(
            lambda: torch_backend.execute(graph, torch_bindings),
            workload.repeats,
            torch,
        )
        torch_result = torch_backend.execute(graph, torch_bindings)
        torch_error = float(
            np.max(np.abs(torch_result.detach().cpu().numpy() - reference))
        )
        print(
            f"  PyTorch CUDA     runtime_ms={torch_ms:.4f} first_call_ms={torch_cold:.4f} "
            f"compilation_ms=0.000 max_error={torch_error:.3e}"
        )

        if not triton_available:
            print("  Triton           UNAVAILABLE (optional dependency not installed)")
            continue
        triton_backend = get_backend("triton", device="cuda", policy="triton")
        triton_cold, triton_ms = cuda_measurements(
            lambda: triton_backend.execute(graph, torch_bindings),
            workload.repeats,
            torch,
        )
        triton_result = triton_backend.execute(graph, torch_bindings)
        triton_error = float(
            np.max(np.abs(triton_result.detach().cpu().numpy() - reference))
        )
        upper_bound = max(0.0, triton_cold - triton_ms)
        print(
            f"  Triton CUDA      runtime_ms={triton_ms:.4f} first_call_ms={triton_cold:.4f} "
            f"compile_upper_bound_ms={upper_bound:.4f} max_error={triton_error:.3e}"
        )
        ratio = torch_ms / triton_ms
        outcome = "faster" if ratio > 1.0 else "slower"
        print(f"  Triton/PyTorch   ratio={ratio:.3f}x ({outcome} on this workload only)")


if __name__ == "__main__":
    main()
