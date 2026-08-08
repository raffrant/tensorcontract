"""Batched NumPy Monte Carlo for the three-qubit correlated-noise model."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from numbers import Integral
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .three_qubit import CorrelatedXXXNoise, exact_logical_error_rate


_RECOVERY_LOOKUP = np.asarray(
    (
        (0, 0, 0),  # syndrome 00
        (0, 0, 1),  # syndrome 01
        (1, 0, 0),  # syndrome 10
        (0, 1, 0),  # syndrome 11
    ),
    dtype=np.int8,
)
_NORMAL_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class MonteCarloSamples:
    """Optional per-shot trajectories from one vectorized simulation."""

    correlated: NDArray[np.int8]
    local_errors: NDArray[np.int8]
    errors: NDArray[np.int8]
    syndromes: NDArray[np.int8]
    recoveries: NDArray[np.int8]
    residual_errors: NDArray[np.int8]
    logical_failures: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Aggregate logical-error statistics for a batched simulation."""

    num_shots: int
    logical_failures: int
    p_phys: float
    p_phys_estimate: float
    p_logical_estimate: float
    p_logical_exact: float
    standard_error: float
    confidence_interval: tuple[float, float]
    p: float
    rho: float
    backend: str
    batch_size: int | None
    elapsed_time: float
    shots_per_second: float
    samples: MonteCarloSamples | None = None


def _validate_shot_count(num_shots: int) -> int:
    if not isinstance(num_shots, Integral) or isinstance(num_shots, bool):
        raise TypeError("num_shots must be an integer")
    if num_shots < 0:
        raise ValueError("num_shots must be nonnegative")
    return int(num_shots)


def _validate_batch_size(batch_size: int | None) -> int | None:
    if batch_size is None:
        return None
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer or None")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return int(batch_size)


def _allocate_samples(num_shots: int) -> MonteCarloSamples:
    return MonteCarloSamples(
        correlated=np.empty(num_shots, dtype=np.int8),
        local_errors=np.empty((num_shots, 3), dtype=np.int8),
        errors=np.empty((num_shots, 3), dtype=np.int8),
        syndromes=np.empty((num_shots, 2), dtype=np.int8),
        recoveries=np.empty((num_shots, 3), dtype=np.int8),
        residual_errors=np.empty((num_shots, 3), dtype=np.int8),
        logical_failures=np.empty(num_shots, dtype=np.bool_),
    )


def run_monte_carlo(
    num_shots: int,
    p: float,
    rho: float,
    seed: int | None = None,
    batch_size: int | None = None,
    backend: str = "numpy",
    return_samples: bool = False,
) -> MonteCarloResult:
    """Estimate the logical-error rate using vectorized NumPy batches.

    A Python loop is used only to bound memory through chunks. Every operation
    within a chunk—including sampling, XOR, recovery lookup, residual decoding,
    and reduction—is vectorized over all shots in that chunk.

    For zero shots, the estimate, standard error, confidence bounds, and
    throughput are defined as zero. ``p_phys`` is the exact model rate
    ``p + rho - 2*p*rho``; ``p_phys_estimate`` is the sampled bit-error rate.
    """
    shots = _validate_shot_count(num_shots)
    configured_batch = _validate_batch_size(batch_size)
    if backend != "numpy":
        raise ValueError(
            f"unsupported Monte Carlo backend {backend!r}; Stage 3 supports only 'numpy'"
        )
    if not isinstance(return_samples, bool):
        raise TypeError("return_samples must be a bool")
    noise = CorrelatedXXXNoise(p, rho)
    effective_batch = shots if configured_batch is None else configured_batch
    effective_batch = max(1, effective_batch)
    rng = np.random.default_rng(seed)
    retained = _allocate_samples(shots) if return_samples else None

    started = perf_counter()
    logical_failures = 0
    physical_error_bits = 0
    offset = 0
    while offset < shots:
        count = min(effective_batch, shots - offset)
        # One row per shot makes RNG consumption independent of chunk size.
        draws = rng.random((count, 4))
        correlated = (draws[:, 0] < noise.rho).astype(np.int8)
        local = (draws[:, 1:] < noise.p).astype(np.int8)
        errors = np.bitwise_xor(correlated[:, None], local)
        syndromes = np.column_stack(
            (
                np.bitwise_xor(errors[:, 0], errors[:, 1]),
                np.bitwise_xor(errors[:, 1], errors[:, 2]),
            )
        ).astype(np.int8, copy=False)
        syndrome_codes = syndromes[:, 0] * 2 + syndromes[:, 1]
        recoveries = _RECOVERY_LOOKUP[syndrome_codes]
        residuals = np.bitwise_xor(errors, recoveries)
        failures = np.all(residuals == 1, axis=1)
        successes = np.all(residuals == 0, axis=1)
        if not np.all(np.logical_or(failures, successes)):
            raise RuntimeError("vectorized recovery produced a residual outside the code space")

        logical_failures += int(np.count_nonzero(failures))
        physical_error_bits += int(np.count_nonzero(errors))
        if retained is not None:
            destination = slice(offset, offset + count)
            retained.correlated[destination] = correlated
            retained.local_errors[destination] = local
            retained.errors[destination] = errors
            retained.syndromes[destination] = syndromes
            retained.recoveries[destination] = recoveries
            retained.residual_errors[destination] = residuals
            retained.logical_failures[destination] = failures
        offset += count

    logical_estimate = logical_failures / shots if shots else 0.0
    physical_estimate = physical_error_bits / (3 * shots) if shots else 0.0
    standard_error = (
        sqrt(logical_estimate * (1.0 - logical_estimate) / shots)
        if shots
        else 0.0
    )
    confidence_interval = (
        max(0.0, logical_estimate - _NORMAL_95 * standard_error),
        min(1.0, logical_estimate + _NORMAL_95 * standard_error),
    )
    exact = exact_logical_error_rate(noise)
    elapsed = perf_counter() - started
    throughput = shots / elapsed if shots and elapsed > 0.0 else 0.0
    return MonteCarloResult(
        num_shots=shots,
        logical_failures=logical_failures,
        p_phys=noise.physical_error_rate(),
        p_phys_estimate=physical_estimate,
        p_logical_estimate=logical_estimate,
        p_logical_exact=exact,
        standard_error=standard_error,
        confidence_interval=confidence_interval,
        p=noise.p,
        rho=noise.rho,
        backend=backend,
        batch_size=configured_batch,
        elapsed_time=elapsed,
        shots_per_second=throughput,
        samples=retained,
    )


def estimate_logical_error_rate(
    num_shots: int,
    p: float,
    rho: float,
    seed: int | None = None,
    batch_size: int | None = None,
    backend: str = "numpy",
    return_samples: bool = False,
) -> MonteCarloResult:
    """Compatibility alias for :func:`run_monte_carlo`."""
    return run_monte_carlo(
        num_shots,
        p,
        rho,
        seed=seed,
        batch_size=batch_size,
        backend=backend,
        return_samples=return_samples,
    )
