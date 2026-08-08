"""Benchmark the Stage 8 NumPy batched-matrix lowering.

Run with one BLAS thread for more reproducible comparisons::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python benchmarks/benchmark_batched_matmul.py

The script compares the prior general-einsum path with the optimized backend.
It performs no ahead-of-time or just-in-time compilation, so compilation time
is reported as zero rather than conflated with first-call execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time

import numpy as np

from tensorcontract.backends import NumPyBackend
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
    output_indices: tuple[str, ...] = ("b", "i", "j")


def make_graph(workload: Workload) -> tuple[SymbolicGraph, ContractionNode]:
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
    operation = ContractionNode(
        "result", ("left", "right"), workload.output_indices
    )
    graph.add_operation(operation)
    return graph, operation


def median_runtime_ms(
    backend: NumPyBackend,
    graph: SymbolicGraph,
    bindings: dict[str, np.ndarray],
    repeats: int,
) -> float:
    for _ in range(5):
        backend.execute(graph, bindings)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        backend.execute(graph, bindings)
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples)


def main() -> None:
    workloads = (
        Workload("tiny", 1, 2, 2, 2, 500),
        Workload("small", 4, 8, 8, 8, 300),
        Workload("medium", 16, 64, 64, 64, 40),
        Workload("large", 8, 128, 128, 128, 15),
        # This legal output permutation cannot use the conservative lowering.
        Workload("fallback-transposed", 4, 8, 8, 8, 300, ("b", "j", "i")),
    )
    baseline = NumPyBackend(enable_batched_matmul=False)
    optimized = NumPyBackend(enable_batched_matmul=True)
    rng = np.random.default_rng(20260808)
    print("NumPy eager execution; compilation time: baseline=0.000 ms optimized=0.000 ms")
    print("medians include validation, binding, dispatch, and execution")
    for workload in workloads:
        graph, operation = make_graph(workload)
        bindings = {
            "left": rng.normal(
                size=(workload.batch, workload.rows, workload.contracted)
            ),
            "right": rng.normal(
                size=(workload.batch, workload.contracted, workload.columns)
            ),
        }
        expected = baseline.execute(graph, bindings)
        actual = optimized.execute(graph, bindings)
        error = float(np.max(np.abs(expected - actual)))
        baseline_ms = median_runtime_ms(
            baseline, graph, bindings, workload.repeats
        )
        optimized_ms = median_runtime_ms(
            optimized, graph, bindings, workload.repeats
        )
        ratio = baseline_ms / optimized_ms
        outcome = "REGRESSION" if ratio < 1.0 else "improved"
        selected = optimized.contraction_implementation(graph, operation)
        print(
            f"{workload.name:19s} implementation={selected:14s} "
            f"baseline={baseline_ms:9.4f} ms optimized={optimized_ms:9.4f} ms "
            f"ratio={ratio:6.2f}x error={error:.3e} {outcome}"
        )


if __name__ == "__main__":
    main()
