"""Reproducible microbenchmark for the Stage 1 exhaustive CPU reference.

This benchmark measures eight-pattern enumeration only. Batched Monte Carlo,
tensor-network execution, and accelerator comparisons belong to later stages.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable, TypeVar

from tensorcontract.quantum import (
    CorrelatedXXXNoise,
    Syndrome,
    decode_syndrome,
    exact_logical_diagnostics,
)


T = TypeVar("T")


def median_per_call_us(
    function: Callable[[], T],
    *,
    calls_per_round: int = 1_000,
    rounds: int = 9,
) -> tuple[float, T]:
    """Return median time per call after one untimed warmup."""
    result = function()
    samples: list[float] = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        for _ in range(calls_per_round):
            result = function()
        elapsed = time.perf_counter_ns() - started
        samples.append(elapsed / calls_per_round / 1_000.0)
    return statistics.median(samples), result


def main() -> None:
    noise = CorrelatedXXXNoise(p=0.17, rho=0.23)
    syndromes = (Syndrome(0, 0), Syndrome(0, 1), Syndrome(1, 0), Syndrome(1, 1))
    table_us, table = median_per_call_us(noise.probability_table)
    decode_us, decoded = median_per_call_us(
        lambda: tuple(decode_syndrome(syndrome, noise) for syndrome in syndromes)
    )
    diagnostics_us, diagnostics = median_per_call_us(
        lambda: exact_logical_diagnostics(noise)
    )
    normalization_error = abs(sum(probability for _, probability in table) - 1.0)
    syndrome_error = abs(
        sum(probability for _, probability in diagnostics.syndrome_probabilities) - 1.0
    )
    assert len(decoded) == 4
    print("Three-qubit exact CPU reference; no compilation")
    print("parameters: p=0.17 rho=0.23; 9 rounds x 1000 calls")
    print(f"probability_table_us={table_us:.3f}")
    print(f"decode_all_syndromes_us={decode_us:.3f}")
    print(f"exact_diagnostics_us={diagnostics_us:.3f}")
    print(f"logical_error_rate={diagnostics.logical_error_rate:.12f}")
    print(f"probability_normalization_error={normalization_error:.3e}")
    print(f"syndrome_normalization_error={syndrome_error:.3e}")


if __name__ == "__main__":
    main()
