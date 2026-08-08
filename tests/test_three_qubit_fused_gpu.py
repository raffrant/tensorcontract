"""Stage 7 correctness and fallback tests for optional fused CUDA execution."""

from __future__ import annotations

import numpy as np
import pytest

from tensorcontract.backends import BackendUnavailableError, is_backend_available
from tensorcontract.quantum import (
    QuantumExecutionPlanCache,
    is_gpu_available,
    run_monte_carlo,
)


FUSED_REQUIRED = pytest.mark.skipif(
    not is_gpu_available() or not is_backend_available("triton"),
    reason="Triton CUDA trajectory kernel is unavailable",
)


@FUSED_REQUIRED
@pytest.mark.parametrize("batch_size", [None, 127, 4_096])
@pytest.mark.parametrize("p,rho,expected", [(0.0, 0.0, 0), (0.0, 1.0, 10_003)])
def test_fused_deterministic_cases_and_batch_sizes(
    batch_size: int | None, p: float, rho: float, expected: int
) -> None:
    result = run_monte_carlo(
        10_003,
        p,
        rho,
        seed=31,
        batch_size=batch_size,
        backend="gpu",
        fusion_options={"enabled": True},
    )
    assert result.backend == "triton-cuda"
    assert result.fusion_used
    assert result.logical_failures == expected
    assert result.kernel_count == 1 + int(np.ceil(10_003 / (batch_size or 10_003)))
    assert result.random_generation_fused and result.reduction_fused


@FUSED_REQUIRED
def test_fused_seed_is_reproducible_and_chunk_independent() -> None:
    options = {"enabled": True}
    unchunked = run_monte_carlo(
        40_003, 0.17, 0.23, seed=991, backend="gpu", fusion_options=options
    )
    chunked = run_monte_carlo(
        40_003, 0.17, 0.23, seed=991, batch_size=1_009,
        backend="gpu", fusion_options=options,
    )
    repeated = run_monte_carlo(
        40_003, 0.17, 0.23, seed=991, backend="gpu", fusion_options=options
    )
    assert unchunked.logical_failures == chunked.logical_failures
    assert unchunked.logical_failures == repeated.logical_failures
    assert unchunked.p_phys_estimate == chunked.p_phys_estimate


@FUSED_REQUIRED
def test_fused_statistically_agrees_with_exact_and_numpy() -> None:
    shots = 300_000
    fused = run_monte_carlo(
        shots, 0.17, 0.23, seed=2028, batch_size=65_536,
        backend="gpu", fusion_options={"enabled": True},
    )
    numpy_result = run_monte_carlo(
        shots, 0.17, 0.23, seed=2028, batch_size=65_536
    )
    combined = np.sqrt(fused.standard_error**2 + numpy_result.standard_error**2)
    assert abs(fused.p_logical_estimate - fused.p_logical_exact) <= (
        6.0 * fused.standard_error + 1.0 / shots
    )
    assert abs(fused.p_logical_estimate - numpy_result.p_logical_estimate) <= (
        6.0 * combined + 1.0 / shots
    )
    assert 0.0 <= fused.p_phys_estimate <= 1.0


@FUSED_REQUIRED
def test_fused_compilation_is_reused_by_stage_six_plan_key() -> None:
    cache = QuantumExecutionPlanCache()
    options = {"enabled": True, "block_size": 256}
    cold = run_monte_carlo(
        65_536, 0.1, 0.05, seed=4, batch_size=65_536,
        backend="gpu", fusion_options=options, plan_cache=cache,
    )
    warm = run_monte_carlo(
        65_536, 0.2, 0.15, seed=5, batch_size=65_536,
        backend="gpu", fusion_options=options, plan_cache=cache,
    )
    assert cold.backend == warm.backend == "triton-cuda"
    assert not cold.cache_hit and warm.cache_hit
    assert cold.plan_key == warm.plan_key
    assert warm.compilation_time == 0.0
    assert warm.registers_per_thread is None or warm.registers_per_thread > 0
    assert warm.occupancy is None or 0.0 < warm.occupancy <= 1.0


@FUSED_REQUIRED
def test_samples_and_invalid_schedule_fall_back_to_high_level_gpu() -> None:
    samples = run_monte_carlo(
        100, 0.1, 0.2, seed=1, backend="gpu", return_samples=True,
        fusion_options={"enabled": True},
    )
    unsupported = run_monte_carlo(
        100, 0.1, 0.2, seed=1, backend="gpu",
        fusion_options={"enabled": True, "block_size": 7},
    )
    assert samples.backend == unsupported.backend == "torch-cuda"
    assert samples.samples is not None
    assert not samples.fusion_used and not unsupported.fusion_used


@FUSED_REQUIRED
def test_triton_construction_failure_falls_back_to_high_level_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorcontract.quantum.monte_carlo as monte_carlo

    actual = monte_carlo.get_backend

    def selective_failure(name: str, **options: object):
        if name == "triton":
            raise BackendUnavailableError("synthetic Triton failure")
        return actual(name, **options)

    monkeypatch.setattr(monte_carlo, "get_backend", selective_failure)
    result = run_monte_carlo(
        1_000, 0.1, 0.2, seed=1, backend="gpu",
        fusion_options={"enabled": True},
    )
    assert result.backend == "torch-cuda"
    assert not result.fusion_used


def test_fusion_request_remains_safe_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import tensorcontract.quantum.monte_carlo as monte_carlo

    monkeypatch.setattr(monte_carlo, "is_gpu_available", lambda: False)
    result = run_monte_carlo(
        100, 0.1, 0.2, seed=2, backend="gpu",
        fusion_options={"enabled": True},
    )
    assert result.backend == "numpy"
    assert result.fallback_used
    assert not result.fusion_used
