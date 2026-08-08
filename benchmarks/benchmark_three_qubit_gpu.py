"""Benchmark vectorized NumPy against optional high-level PyTorch CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

from tensorcontract.quantum import (
    QuantumExecutionPlanCache,
    export_performance_results_csv,
    is_gpu_available,
    performance_results_to_rows,
    run_monte_carlo,
)


def benchmark_backend(
    *,
    backend: str,
    shots: int,
    p: float,
    rho: float,
    seed: int,
    batch_size: int,
    warm_repetitions: int = 5,
) -> dict[str, object]:
    plan_cache = QuantumExecutionPlanCache()
    started = time.perf_counter()
    cold = run_monte_carlo(
        shots, p, rho, seed=seed, batch_size=batch_size, backend=backend,
        plan_cache=plan_cache,
    )
    cold_total = time.perf_counter() - started
    warm_results = []
    warm_totals = []
    for _ in range(warm_repetitions):
        started = time.perf_counter()
        result = run_monte_carlo(
            shots,
            p,
            rho,
            seed=seed,
            batch_size=batch_size,
            backend=backend,
            plan_cache=plan_cache,
        )
        warm_totals.append(time.perf_counter() - started)
        warm_results.append(result)
    median_position = sorted(range(len(warm_totals)), key=warm_totals.__getitem__)[
        len(warm_totals) // 2
    ]
    representative = warm_results[median_position]
    warm_total = statistics.median(warm_totals)
    row = dict(performance_results_to_rows([representative])[0])
    row.update(
        runtime_seconds=warm_total,
        shots_per_second=shots / warm_total if warm_total > 0.0 else 0.0,
        cold_start_runtime=cold_total,
        warm_runtime=warm_total,
    )
    if cold.logical_failures != representative.logical_failures:
        raise RuntimeError(f"{backend} fixed-seed cold and warm results disagree")
    return row


def print_row(row: dict[str, object]) -> None:
    estimate = float(row["p_logical_estimate"])
    exact = float(row["p_logical_exact"])
    print(
        f"  {str(row['backend']):12s} cold_s={float(row['cold_start_runtime']):9.6f} "
        f"warm_s={float(row['warm_runtime']):9.6f} "
        f"shots_per_s={float(row['shots_per_second']):12.1f} "
        f"rng_s={float(row['random_generation_runtime']):8.6f} "
        f"h2d_s={float(row['host_to_device_runtime']):8.6f} "
        f"kernel_s={float(row['kernel_runtime']):8.6f} "
        f"reduce_s={float(row['reduction_runtime']):8.6f} "
        f"d2h_s={float(row['device_to_host_runtime']):8.6f} "
        f"cache_hit={bool(row['cache_hit'])!s:5s} "
        f"compile_s={float(row['compilation_time']):8.6f} "
        f"plan_s={float(row['planning_time']):8.6f} "
        f"estimate={estimate:.8f} abs_error={abs(estimate-exact):.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shots", nargs="+", type=int, default=[1_000, 10_000, 100_000, 1_000_000]
    )
    parser.add_argument("--p", type=float, default=0.17)
    parser.add_argument("--rho", type=float, default=0.23)
    parser.add_argument("--seed", type=int, default=202605)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--output-csv", type=Path)
    arguments = parser.parse_args()

    gpu_available = is_gpu_available()
    print("Three-qubit batched Monte Carlo backend benchmark")
    print(
        f"gpu_available={gpu_available}; PyTorch uses eager high-level CUDA operations; "
        "dynamic_compilation_per_shot=false"
    )
    records: list[dict[str, object]] = []
    for shots in arguments.shots:
        print(f"\nshots={shots:,} batch_size={arguments.batch_size:,}")
        numpy_row = benchmark_backend(
            backend="numpy",
            shots=shots,
            p=arguments.p,
            rho=arguments.rho,
            seed=arguments.seed,
            batch_size=arguments.batch_size,
        )
        records.append(numpy_row)
        print_row(numpy_row)
        if not gpu_available:
            print("  GPU          UNAVAILABLE; NumPy fallback exists; no GPU speedup claim")
            continue
        gpu_row = benchmark_backend(
            backend="gpu",
            shots=shots,
            p=arguments.p,
            rho=arguments.rho,
            seed=arguments.seed,
            batch_size=arguments.batch_size,
        )
        records.append(gpu_row)
        print_row(gpu_row)
        ratio = float(numpy_row["warm_runtime"]) / float(gpu_row["warm_runtime"])
        outcome = "GPU faster" if ratio > 1.0 else "GPU overhead dominates"
        print(f"  NumPy/GPU warm ratio={ratio:.3f}x; {outcome} for this workload only")
    if arguments.output_csv is not None:
        export_performance_results_csv(records, arguments.output_csv)
        print(f"\nWrote performance data to {arguments.output_csv}")


if __name__ == "__main__":
    main()
