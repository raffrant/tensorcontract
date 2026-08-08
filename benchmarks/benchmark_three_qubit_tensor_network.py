"""Benchmark direct versus planned exact three-qubit tensor contractions."""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np

from tensorcontract.backends import NumPyBackend
from tensorcontract.quantum import (
    CorrelatedXXXNoise,
    build_correlated_noise_tensor_network,
)


def median_ms(function: Callable[[], object], *, repeats: int) -> float:
    for _ in range(10):
        function()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return statistics.median(samples)


def main() -> None:
    noise = CorrelatedXXXNoise(p=0.17, rho=0.23)
    configurations = (
        ("error", None, "selector"),
        ("syndrome", None, "selector"),
        ("logical-open", None, "selector"),
        ("logical-selector-00", (0, 0), "selector"),
        ("logical-slice-00", (0, 0), "slice"),
    )
    backend = NumPyBackend(enable_batched_matmul=False)
    print("Three-qubit tensor-network NumPy benchmark; p=0.17 rho=0.23")
    print("All paths are eager; compilation_ms=0.000; median of 500 executions")
    for name, syndrome, conditioning in configurations:
        calculation = (
            "error"
            if name == "error"
            else "syndrome"
            if name == "syndrome"
            else "logical_syndrome"
        )
        planning_started = time.perf_counter_ns()
        network = build_correlated_noise_tensor_network(
            noise,
            calculation=calculation,
            fixed_syndrome=syndrome,
            conditioning=conditioning,
        )
        planning_ms = (time.perf_counter_ns() - planning_started) / 1_000_000.0
        direct_call = lambda: backend.execute(
            network.graph, network.bindings, network.output_name
        )
        planned_call = lambda: backend.execute(
            network.plan.graph, network.bindings, network.output_name
        )
        direct = np.asarray(direct_call())
        planned = np.asarray(planned_call())
        difference = float(np.max(np.abs(direct - planned)))
        direct_ms = median_ms(direct_call, repeats=500)
        planned_ms = median_ms(planned_call, repeats=500)
        ratio = direct_ms / planned_ms
        print(
            f"{name:21s} planning_ms={planning_ms:7.3f} "
            f"direct_ms={direct_ms:8.4f} planned_ms={planned_ms:8.4f} "
            f"direct/planned={ratio:6.3f}x error={difference:.3e} "
            f"steps={len(network.plan.steps):2d} flops={network.estimated_flops:3d} "
            f"peak_bytes={network.peak_memory_bytes:4d}"
        )


if __name__ == "__main__":
    main()
