"""Stage 5 tests for optional batched PyTorch CUDA Monte Carlo."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tensorcontract.quantum import (
    PERFORMANCE_COLUMNS,
    QuantumExecutionPlanCache,
    is_gpu_available,
    performance_results_to_rows,
    run_monte_carlo,
)


def test_package_and_gpu_probe_work_without_torch_imports() -> None:
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.') or name == 'triton' or name.startswith('triton.'):
        raise ImportError(name + ' deliberately unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import tensorcontract
from tensorcontract.quantum import is_gpu_available, run_monte_carlo
assert not is_gpu_available()
result = run_monte_carlo(8, 0.0, 0.0, seed=1, backend='gpu')
assert result.backend == 'numpy' and result.fallback_used
print('GPU dependencies remain optional')
"""
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == "GPU dependencies remain optional"


@pytest.mark.parametrize("batch_size", [None, 1, 17, 256])
def test_forced_unavailable_gpu_falls_back_to_identical_numpy(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int | None,
) -> None:
    import tensorcontract.quantum.monte_carlo as monte_carlo

    monkeypatch.setattr(monte_carlo, "is_gpu_available", lambda: False)
    numpy_result = run_monte_carlo(
        513, 0.17, 0.23, seed=31, batch_size=batch_size, return_samples=True
    )
    gpu_request = run_monte_carlo(
        513,
        0.17,
        0.23,
        seed=31,
        batch_size=batch_size,
        backend="gpu",
        return_samples=True,
    )
    assert gpu_request.requested_backend == "gpu"
    assert gpu_request.backend == "numpy"
    assert gpu_request.device == "cpu"
    assert gpu_request.fallback_used
    assert gpu_request.logical_failures == numpy_result.logical_failures
    assert gpu_request.p_logical_estimate == numpy_result.p_logical_estimate
    assert gpu_request.samples is not None and numpy_result.samples is not None
    np.testing.assert_array_equal(gpu_request.samples.errors, numpy_result.samples.errors)
    np.testing.assert_array_equal(
        gpu_request.samples.logical_failures,
        numpy_result.samples.logical_failures,
    )


def test_numpy_result_preserves_schema_and_reports_phase_timings() -> None:
    result = run_monte_carlo(10_000, 0.17, 0.23, seed=3, batch_size=1_024)
    assert result.requested_backend == result.backend == "numpy"
    assert result.device == "cpu"
    assert not result.fallback_used
    assert result.runtime_seconds == result.elapsed_time
    assert result.random_generation_runtime >= 0.0
    assert result.kernel_runtime >= 0.0
    assert result.reduction_runtime >= 0.0
    assert result.host_to_device_runtime == 0.0
    assert result.device_to_host_runtime == 0.0
    assert (
        result.random_generation_runtime
        + result.kernel_runtime
        + result.reduction_runtime
        <= result.elapsed_time
    )


def test_performance_schema_contains_gpu_timing_phases() -> None:
    result = run_monte_carlo(1_000, 0.17, 0.23, seed=5)
    row = performance_results_to_rows([result])[0]
    assert tuple(row) == PERFORMANCE_COLUMNS
    assert row["requested_backend"] == "numpy"
    assert row["device"] == "cpu"
    assert row["random_generation_runtime"] == result.random_generation_runtime
    assert row["kernel_runtime"] == result.kernel_runtime
    assert row["reduction_runtime"] == result.reduction_runtime


CUDA_REQUIRED = pytest.mark.skipif(
    not is_gpu_available(), reason="PyTorch CUDA Monte Carlo is unavailable"
)


@CUDA_REQUIRED
@pytest.mark.parametrize("p,rho,expected", [(0.0, 0.0, 0), (0.0, 1.0, 2_057)])
@pytest.mark.parametrize("batch_size", [None, 127, 1_024])
def test_gpu_deterministic_special_cases(
    p: float,
    rho: float,
    expected: int,
    batch_size: int | None,
) -> None:
    result = run_monte_carlo(
        2_057, p, rho, seed=17, batch_size=batch_size, backend="gpu"
    )
    assert result.backend == "torch-cuda"
    assert result.requested_backend == "gpu"
    assert result.device.startswith("cuda")
    assert not result.fallback_used
    assert result.logical_failures == expected
    assert result.p_logical_estimate in (0.0, 1.0)
    assert 0.0 <= result.p_phys_estimate <= 1.0


@CUDA_REQUIRED
@pytest.mark.parametrize("batch_size", [257, 4_096, None])
def test_gpu_seed_is_reproducible_for_each_batch_configuration(
    batch_size: int | None,
) -> None:
    first = run_monte_carlo(
        20_003, 0.17, 0.23, seed=991, batch_size=batch_size, backend="gpu"
    )
    second = run_monte_carlo(
        20_003, 0.17, 0.23, seed=991, batch_size=batch_size, backend="gpu"
    )
    assert first.logical_failures == second.logical_failures
    assert first.p_phys_estimate == second.p_phys_estimate
    assert first.p_logical_estimate == second.p_logical_estimate


@CUDA_REQUIRED
def test_gpu_statistical_rate_and_failure_count_agree_with_numpy() -> None:
    shots = 300_000
    numpy_result = run_monte_carlo(
        shots, 0.17, 0.23, seed=2027, batch_size=65_536
    )
    gpu_result = run_monte_carlo(
        shots, 0.17, 0.23, seed=2027, batch_size=65_536, backend="gpu"
    )
    combined_standard_error = np.sqrt(
        numpy_result.standard_error**2 + gpu_result.standard_error**2
    )
    assert abs(gpu_result.p_logical_estimate - numpy_result.p_logical_estimate) <= (
        6.0 * combined_standard_error + 1.0 / shots
    )
    assert abs(gpu_result.p_logical_estimate - gpu_result.p_logical_exact) <= (
        6.0 * gpu_result.standard_error + 1.0 / shots
    )
    failure_tolerance = 6.0 * combined_standard_error * shots + 1.0
    assert abs(gpu_result.logical_failures - numpy_result.logical_failures) <= failure_tolerance
    assert 0 <= gpu_result.logical_failures <= shots
    assert 0.0 <= gpu_result.p_logical_estimate <= 1.0
    assert 0.0 <= gpu_result.p_phys_estimate <= 1.0


@CUDA_REQUIRED
def test_gpu_returned_samples_follow_the_same_decoding_invariants() -> None:
    result = run_monte_carlo(
        4_113,
        0.21,
        0.14,
        seed=8,
        batch_size=511,
        backend="gpu",
        return_samples=True,
    )
    samples = result.samples
    assert samples is not None
    np.testing.assert_array_equal(
        samples.errors,
        np.bitwise_xor(samples.correlated[:, None], samples.local_errors),
    )
    np.testing.assert_array_equal(
        samples.residual_errors,
        np.bitwise_xor(samples.errors, samples.recoveries),
    )
    np.testing.assert_array_equal(
        samples.logical_failures,
        np.all(samples.residual_errors == 1, axis=1),
    )


@CUDA_REQUIRED
def test_gpu_backend_plan_is_cached_on_compatible_device_signature() -> None:
    cache = QuantumExecutionPlanCache()
    cold = run_monte_carlo(
        4_096, 0.11, 0.07, seed=77, batch_size=2_048,
        backend="gpu", plan_cache=cache,
    )
    warm = run_monte_carlo(
        4_096, 0.21, 0.17, seed=78, batch_size=2_048,
        backend="gpu", plan_cache=cache,
    )
    assert cold.backend == warm.backend == "torch-cuda"
    assert not cold.cache_hit
    assert warm.cache_hit
    assert cold.plan_key == warm.plan_key
    assert warm.compilation_time == 0.0
    assert cache.info().backend.size == 1
