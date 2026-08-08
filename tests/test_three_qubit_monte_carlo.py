"""Stage 3 tests for batched NumPy correlated-noise Monte Carlo."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorcontract.quantum import (
    estimate_logical_error_rate,
    exact_logical_error_rate,
    run_monte_carlo,
)
from tensorcontract.quantum.three_qubit import CorrelatedXXXNoise


def test_deterministic_zero_noise_has_no_errors_or_failures() -> None:
    result = run_monte_carlo(
        257, p=0.0, rho=0.0, seed=4, return_samples=True
    )
    assert result.logical_failures == 0
    assert result.p_phys == 0.0
    assert result.p_phys_estimate == 0.0
    assert result.p_logical_estimate == 0.0
    assert result.p_logical_exact == 0.0
    assert result.standard_error == 0.0
    assert result.confidence_interval == (0.0, 0.0)
    assert result.samples is not None
    assert not np.any(result.samples.correlated)
    assert not np.any(result.samples.local_errors)
    assert not np.any(result.samples.errors)
    assert not np.any(result.samples.logical_failures)


def test_deterministic_correlated_xxx_has_all_logical_failures() -> None:
    result = run_monte_carlo(
        257, p=0.0, rho=1.0, seed=4, batch_size=31, return_samples=True
    )
    assert result.logical_failures == result.num_shots
    assert result.p_phys == 1.0
    assert result.p_phys_estimate == 1.0
    assert result.p_logical_estimate == 1.0
    assert result.p_logical_exact == 1.0
    assert result.standard_error == 0.0
    assert result.confidence_interval == (1.0, 1.0)
    assert result.samples is not None
    assert np.all(result.samples.correlated == 1)
    assert np.all(result.samples.local_errors == 0)
    assert np.all(result.samples.errors == 1)
    assert np.all(result.samples.syndromes == 0)
    assert np.all(result.samples.recoveries == 0)
    assert np.all(result.samples.residual_errors == 1)
    assert np.all(result.samples.logical_failures)


def test_fixed_seed_is_reproducible() -> None:
    first = run_monte_carlo(
        1_003, 0.17, 0.23, seed=71, batch_size=128, return_samples=True
    )
    second = run_monte_carlo(
        1_003, 0.17, 0.23, seed=71, batch_size=128, return_samples=True
    )
    assert first.logical_failures == second.logical_failures
    assert first.p_phys_estimate == second.p_phys_estimate
    assert first.p_logical_estimate == second.p_logical_estimate
    assert first.samples is not None and second.samples is not None
    for field in (
        "correlated",
        "local_errors",
        "errors",
        "syndromes",
        "recoveries",
        "residual_errors",
        "logical_failures",
    ):
        assert np.array_equal(getattr(first.samples, field), getattr(second.samples, field))


def test_batch_size_none_processes_one_vectorized_batch() -> None:
    result = run_monte_carlo(513, 0.12, 0.08, seed=12, batch_size=None)
    assert result.num_shots == 513
    assert result.batch_size is None
    assert result.samples is None
    assert result.backend == "numpy"


def test_chunked_and_unchunked_runs_are_trajectory_identical() -> None:
    unchunked = run_monte_carlo(
        2_051, 0.19, 0.27, seed=99, batch_size=None, return_samples=True
    )
    chunked = run_monte_carlo(
        2_051, 0.19, 0.27, seed=99, batch_size=113, return_samples=True
    )
    assert chunked.batch_size == 113
    assert unchunked.logical_failures == chunked.logical_failures
    assert unchunked.p_phys_estimate == chunked.p_phys_estimate
    assert unchunked.p_logical_estimate == chunked.p_logical_estimate
    assert unchunked.samples is not None and chunked.samples is not None
    for field in (
        "correlated",
        "local_errors",
        "errors",
        "syndromes",
        "recoveries",
        "residual_errors",
        "logical_failures",
    ):
        assert np.array_equal(
            getattr(unchunked.samples, field), getattr(chunked.samples, field)
        )


def test_returned_samples_follow_vectorized_xor_and_recovery_pipeline() -> None:
    result = run_monte_carlo(
        301, 0.31, 0.22, seed=83, batch_size=47, return_samples=True
    )
    samples = result.samples
    assert samples is not None
    assert samples.correlated.shape == (301,)
    assert samples.local_errors.shape == (301, 3)
    assert samples.errors.shape == (301, 3)
    assert samples.syndromes.shape == (301, 2)
    np.testing.assert_array_equal(
        samples.errors,
        np.bitwise_xor(samples.correlated[:, None], samples.local_errors),
    )
    np.testing.assert_array_equal(
        samples.syndromes[:, 0],
        np.bitwise_xor(samples.errors[:, 0], samples.errors[:, 1]),
    )
    np.testing.assert_array_equal(
        samples.syndromes[:, 1],
        np.bitwise_xor(samples.errors[:, 1], samples.errors[:, 2]),
    )
    np.testing.assert_array_equal(
        samples.residual_errors,
        np.bitwise_xor(samples.errors, samples.recoveries),
    )
    np.testing.assert_array_equal(
        samples.logical_failures,
        np.all(samples.residual_errors == 1, axis=1),
    )
    assert result.logical_failures == int(np.count_nonzero(samples.logical_failures))


def test_monte_carlo_agrees_with_exact_rate_within_statistical_tolerance() -> None:
    result = run_monte_carlo(
        200_000, p=0.17, rho=0.23, seed=2026, batch_size=16_384
    )
    discrepancy = abs(result.p_logical_estimate - result.p_logical_exact)
    assert discrepancy <= 5.0 * result.standard_error + 1.0 / result.num_shots
    assert result.confidence_interval[0] <= result.p_logical_estimate
    assert result.p_logical_estimate <= result.confidence_interval[1]
    assert result.p_logical_exact == pytest.approx(
        exact_logical_error_rate(CorrelatedXXXNoise(0.17, 0.23))
    )
    assert result.p_phys == pytest.approx(0.17 + 0.23 - 2 * 0.17 * 0.23)
    assert abs(result.p_phys_estimate - result.p_phys) < 0.01


def test_zero_shots_returns_defined_empty_statistics_and_samples() -> None:
    result = run_monte_carlo(
        0, 0.2, 0.3, seed=5, batch_size=10, return_samples=True
    )
    assert result.num_shots == 0
    assert result.logical_failures == 0
    assert result.p_logical_estimate == 0.0
    assert result.standard_error == 0.0
    assert result.confidence_interval == (0.0, 0.0)
    assert result.shots_per_second == 0.0
    assert result.elapsed_time >= 0.0
    assert result.samples is not None
    assert result.samples.errors.shape == (0, 3)
    assert result.samples.syndromes.shape == (0, 2)


def test_very_small_probability_produces_finite_valid_statistics() -> None:
    result = run_monte_carlo(10_000, p=1e-9, rho=1e-9, seed=7, batch_size=997)
    assert result.logical_failures == 0
    assert result.p_logical_estimate == 0.0
    assert result.standard_error == 0.0
    assert result.confidence_interval == (0.0, 0.0)
    assert result.p_logical_exact > 0.0
    assert all(math.isfinite(value) for value in result.confidence_interval)


def test_default_does_not_retain_trajectories() -> None:
    result = run_monte_carlo(100, 0.1, 0.2, seed=1)
    assert result.samples is None
    assert result.elapsed_time > 0.0
    assert result.shots_per_second > 0.0


def test_estimate_alias_matches_primary_api() -> None:
    primary = run_monte_carlo(777, 0.1, 0.2, seed=13, batch_size=64)
    alias = estimate_logical_error_rate(777, 0.1, 0.2, seed=13, batch_size=64)
    assert primary.logical_failures == alias.logical_failures
    assert primary.p_logical_estimate == alias.p_logical_estimate
    assert primary.standard_error == alias.standard_error
    assert primary.confidence_interval == alias.confidence_interval


@pytest.mark.parametrize("num_shots", [-1, 1.5, True])
def test_invalid_shot_counts_fail_clearly(num_shots: object) -> None:
    error = ValueError if num_shots == -1 else TypeError
    with pytest.raises(error, match="num_shots"):
        run_monte_carlo(num_shots, 0.1, 0.2)  # type: ignore[arg-type]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_sizes_fail_clearly(batch_size: object) -> None:
    error = ValueError if batch_size in (0, -1) else TypeError
    with pytest.raises(error, match="batch_size"):
        run_monte_carlo(10, 0.1, 0.2, batch_size=batch_size)  # type: ignore[arg-type]


def test_unsupported_backend_and_invalid_sample_flag_fail_clearly() -> None:
    with pytest.raises(ValueError, match="supports only 'numpy'"):
        run_monte_carlo(10, 0.1, 0.2, backend="cuda")
    with pytest.raises(TypeError, match="return_samples"):
        run_monte_carlo(10, 0.1, 0.2, return_samples=1)  # type: ignore[arg-type]
