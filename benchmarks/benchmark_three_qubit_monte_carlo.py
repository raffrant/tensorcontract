"""Compare Python, vectorized NumPy, and chunked NumPy Monte Carlo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import sqrt
import time

import numpy as np

from tensorcontract.quantum import (
    CorrelatedXXXNoise,
    exact_logical_error_rate,
    majority_vote_recovery,
    residual_after_recovery,
    run_monte_carlo,
    syndrome_for_error,
)


@dataclass(frozen=True, slots=True)
class PythonResult:
    logical_failures: int
    estimate: float
    standard_error: float
    elapsed_time: float


def python_brute_force(
    num_shots: int,
    p: float,
    rho: float,
    seed: int,
) -> PythonResult:
    """Deliberately scalar baseline with RNG and reduction inside the timer."""
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    failures = 0
    for _ in range(num_shots):
        draws = rng.random(4)
        correlated = int(draws[0] < rho)
        local = tuple(int(value < p) for value in draws[1:])
        error = tuple(correlated ^ bit for bit in local)
        syndrome = syndrome_for_error(error)
        recovery = majority_vote_recovery(syndrome)
        residual = residual_after_recovery(error, recovery)
        failures += int(residual.bits == (1, 1, 1))
    estimate = failures / num_shots if num_shots else 0.0
    standard_error = (
        sqrt(estimate * (1.0 - estimate) / num_shots) if num_shots else 0.0
    )
    return PythonResult(failures, estimate, standard_error, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", nargs="+", type=int, default=[1_000, 10_000, 100_000])
    parser.add_argument("--p", type=float, default=0.17)
    parser.add_argument("--rho", type=float, default=0.23)
    parser.add_argument("--seed", type=int, default=202603)
    parser.add_argument("--chunk-size", type=int, default=8_192)
    parser.add_argument(
        "--include-million",
        action="store_true",
        help="also run 1,000,000 shots; the scalar baseline may take time",
    )
    arguments = parser.parse_args()
    shot_counts = list(arguments.shots)
    if arguments.include_million and 1_000_000 not in shot_counts:
        shot_counts.append(1_000_000)
    noise = CorrelatedXXXNoise(arguments.p, arguments.rho)
    exact = exact_logical_error_rate(noise)
    print(
        f"Three-qubit Monte Carlo: p={arguments.p} rho={arguments.rho} "
        f"exact_logical_rate={exact:.12f}"
    )
    print("Times include RNG construction/generation, decoding, and final reduction.")
    for shots in shot_counts:
        python_result = python_brute_force(
            shots, arguments.p, arguments.rho, arguments.seed
        )

        started = time.perf_counter()
        vectorized = run_monte_carlo(
            shots, arguments.p, arguments.rho, seed=arguments.seed
        )
        vectorized_total = time.perf_counter() - started

        started = time.perf_counter()
        chunked = run_monte_carlo(
            shots,
            arguments.p,
            arguments.rho,
            seed=arguments.seed,
            batch_size=arguments.chunk_size,
        )
        chunked_total = time.perf_counter() - started
        if not (
            python_result.logical_failures
            == vectorized.logical_failures
            == chunked.logical_failures
        ):
            raise RuntimeError("fixed-seed benchmark implementations disagree")

        rows = (
            ("python", python_result.elapsed_time, python_result.estimate, python_result.standard_error),
            ("numpy", vectorized_total, vectorized.p_logical_estimate, vectorized.standard_error),
            ("numpy-chunked", chunked_total, chunked.p_logical_estimate, chunked.standard_error),
        )
        print(f"\nshots={shots:,} failures={vectorized.logical_failures:,}")
        for name, elapsed, estimate, standard_error in rows:
            throughput = shots / elapsed if shots and elapsed > 0.0 else 0.0
            print(
                f"  {name:13s} runtime_s={elapsed:9.6f} shots_per_s={throughput:12.1f} "
                f"estimate={estimate:.8f} abs_error={abs(estimate-exact):.3e} "
                f"standard_error={standard_error:.3e}"
            )


if __name__ == "__main__":
    main()
