"""Compare NumPy, eager CUDA, and optional fused Triton QEC trajectories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from tensorcontract.backends import is_backend_available
from tensorcontract.quantum import (
    QuantumExecutionPlanCache,
    is_gpu_available,
    run_monte_carlo,
)


@dataclass(frozen=True, slots=True)
class Measurement:
    implementation: str
    shots: int
    phase: str
    kernel_count: int | None
    compilation_time: float
    kernel_execution_time: float
    total_runtime: float
    random_generation_time: float
    reduction_time: float
    shots_per_second: float
    registers_per_thread: int | None
    register_spills: int | None
    shared_memory_bytes: int | None
    occupancy: float | None
    logical_rate: float
    exact_rate: float


def measure(
    implementation: str,
    shots: int,
    p: float,
    rho: float,
    seed: int,
    batch_size: int,
    cache: QuantumExecutionPlanCache,
    phase: str,
) -> Measurement:
    fused = implementation == "fused-gpu"
    backend = "numpy" if implementation == "numpy" else "gpu"
    result = run_monte_carlo(
        shots,
        p,
        rho,
        seed=seed,
        batch_size=batch_size,
        backend=backend,
        fusion_options={"enabled": fused, "block_size": 256},
        plan_cache=cache,
    )
    return Measurement(
        implementation,
        shots,
        phase,
        result.kernel_count,
        result.compilation_time,
        result.kernel_runtime,
        result.total_time,
        result.random_generation_runtime,
        result.reduction_runtime,
        shots / result.total_time if result.total_time else 0.0,
        result.registers_per_thread,
        result.register_spills,
        result.shared_memory_bytes,
        result.occupancy,
        result.p_logical_estimate,
        result.p_logical_exact,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shots", nargs="+", type=int,
        default=[1_000, 10_000, 100_000, 1_000_000],
    )
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--p", type=float, default=0.17)
    parser.add_argument("--rho", type=float, default=0.23)
    parser.add_argument("--seed", type=int, default=202607)
    args = parser.parse_args()
    fused_available = is_gpu_available() and is_backend_available("triton")
    implementations = ["numpy"]
    if is_gpu_available():
        implementations.append("high-level-gpu")
    if fused_available:
        implementations.append("fused-gpu")
    print(f"gpu_available={is_gpu_available()} fused_available={fused_available}")
    print(
        "Fused RNG and reduction costs are included in kernel_execution_time; "
        "they are not separately measurable."
    )

    for shots in args.shots:
        print(f"\nshots={shots:,} batch_size={args.batch_size:,}")
        for implementation in implementations:
            cache = QuantumExecutionPlanCache()
            cold = measure(
                implementation, shots, args.p, args.rho, args.seed,
                args.batch_size, cache, "cold",
            )
            warm = measure(
                implementation, shots, args.p, args.rho, args.seed,
                args.batch_size, cache, "warm",
            )
            if cold.logical_rate != warm.logical_rate:
                raise RuntimeError(f"{implementation} is not fixed-seed reproducible")
            for result in (cold, warm):
                print(
                    f"  {implementation:14s} {result.phase:4s} "
                    f"total_s={result.total_runtime:.6f} "
                    f"compile_s={result.compilation_time:.6f} "
                    f"kernel_s={result.kernel_execution_time:.6f} "
                    f"rng_s={result.random_generation_time:.6f} "
                    f"reduce_s={result.reduction_time:.6f} "
                    f"shots/s={result.shots_per_second:.1f} "
                    f"kernels={result.kernel_count} regs={result.registers_per_thread} "
                    f"spills={result.register_spills} occupancy={result.occupancy} "
                    f"abs_error={abs(result.logical_rate-result.exact_rate):.3e}"
                )


if __name__ == "__main__":
    main()
