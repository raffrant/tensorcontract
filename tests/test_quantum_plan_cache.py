"""Stage 6 regression tests for QEC execution-plan caching."""

from __future__ import annotations

import pytest

from tensorcontract.quantum import (
    BackendExecutionPlan,
    CorrelatedXXXNoise,
    InMemoryBackendPlanCache,
    QuantumExecutionPlanCache,
    build_correlated_noise_tensor_network,
    make_backend_plan_cache_key,
    run_monte_carlo,
)


def _key(**changes: object):
    values = {
        "network_topology": ("qec", "xor"),
        "tensor_shapes": ((128, 4), (128, 3)),
        "index_structure": ("batch", "c", "u", "e", "s"),
        "noise_model_structure": ("shared-bernoulli-xor", 3),
        "dtype": "float32",
        "backend": "numpy",
        "batch_size": 128,
        "device": "cpu",
        "fusion_options": {"enabled": False},
        "contraction_options": {"method": "greedy"},
    }
    values.update(changes)
    return make_backend_plan_cache_key(**values)


@pytest.mark.parametrize(
    "change",
    (
        {"tensor_shapes": ((64, 4), (64, 3))},
        {"dtype": "float64"},
        {"backend": "gpu"},
        {"device": "cuda:0"},
        {"batch_size": 64},
        {"network_topology": ("different",)},
        {"index_structure": ("batch", "different")},
        {"fusion_options": {"enabled": True}},
        {"contraction_options": {"method": "left-to-right"}},
    ),
)
def test_backend_key_separates_incompatible_signatures(change: dict[str, object]) -> None:
    assert _key(**change) != _key()
    assert _key(**change).digest != _key().digest


def test_monte_carlo_cache_hit_ignores_runtime_noise_values() -> None:
    cache = QuantumExecutionPlanCache()
    cold = run_monte_carlo(500, 0.05, 0.1, seed=7, batch_size=128, plan_cache=cache)
    warm = run_monte_carlo(500, 0.2, 0.3, seed=8, batch_size=128, plan_cache=cache)

    assert cold.cache_hit is False
    assert warm.cache_hit is True
    assert cold.plan_key == warm.plan_key
    assert cache.info().backend.hits == 1
    assert cache.info().backend.misses == 1
    assert cold.compilation_time >= 0.0
    assert warm.compilation_time == 0.0


def test_batch_structure_change_is_a_cache_miss() -> None:
    cache = QuantumExecutionPlanCache()
    first = run_monte_carlo(100, 0.1, 0.2, batch_size=32, plan_cache=cache)
    changed = run_monte_carlo(100, 0.1, 0.2, batch_size=64, plan_cache=cache)
    assert not first.cache_hit
    assert not changed.cache_hit
    assert first.plan_key != changed.plan_key
    assert cache.info().backend.size == 2


def test_cache_can_be_disabled_and_cleared() -> None:
    cache = QuantumExecutionPlanCache()
    uncached_a = run_monte_carlo(20, 0.1, 0.0, plan_cache=cache, cache_enabled=False)
    uncached_b = run_monte_carlo(20, 0.1, 0.0, plan_cache=cache, cache_enabled=False)
    assert not uncached_a.cache_enabled and not uncached_b.cache_hit
    assert cache.info().backend.size == 0

    run_monte_carlo(20, 0.1, 0.0, plan_cache=cache)
    assert cache.info().backend.size == 1
    assert cache.inspect()["backend_entries"]
    cache.clear(reset_statistics=True)
    assert cache.info().backend.size == 0
    assert cache.info().backend.hits == cache.info().backend.misses == 0
    cache.set_enabled(False)
    disabled = run_monte_carlo(20, 0.1, 0.0, plan_cache=cache)
    assert not disabled.cache_enabled and not disabled.cache_hit
    assert cache.info().backend.size == 0


def test_symbolic_contraction_cache_reuses_plan_for_changed_p_and_rho() -> None:
    cache = QuantumExecutionPlanCache()
    first = build_correlated_noise_tensor_network(
        CorrelatedXXXNoise(0.1, 0.2), plan_cache=cache
    )
    second = build_correlated_noise_tensor_network(
        CorrelatedXXXNoise(0.3, 0.4), plan_cache=cache
    )
    assert not first.cache_hit
    assert second.cache_hit
    assert first.cache_key_digest == second.cache_key_digest
    assert first.contraction_order == second.contraction_order


def test_changed_network_topology_misses_symbolic_cache() -> None:
    cache = QuantumExecutionPlanCache()
    open_network = build_correlated_noise_tensor_network(
        CorrelatedXXXNoise(0.1, 0.2), calculation="syndrome", plan_cache=cache
    )
    fixed_network = build_correlated_noise_tensor_network(
        CorrelatedXXXNoise(0.1, 0.2), calculation="syndrome",
        fixed_syndrome=(0, 0), plan_cache=cache,
    )
    assert not open_network.cache_hit
    assert not fixed_network.cache_hit
    assert open_network.cache_key_digest != fixed_network.cache_key_digest


def test_compilation_failure_falls_back_without_poisoning_cache() -> None:
    cache = InMemoryBackendPlanCache()

    def fail() -> BackendExecutionPlan:
        raise RuntimeError("synthetic compiler failure")

    def fallback(error: Exception) -> BackendExecutionPlan:
        return BackendExecutionPlan(
            "gpu", "numpy", "cpu", "float64", 128, False,
            fallback_used=True, fallback_reason=str(error),
        )

    result = cache.get_or_create(_key(backend="gpu"), fail, fallback_factory=fallback)
    assert result.plan.fallback_used
    assert not result.hit
    assert cache.info().compilation_failures == 1
    assert cache.info().size == 0


def test_gpu_unavailable_uses_safe_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import tensorcontract.quantum.monte_carlo as monte_carlo

    monkeypatch.setattr(monte_carlo, "is_gpu_available", lambda: False)
    result = run_monte_carlo(
        100, 0.1, 0.2, seed=9, backend="gpu",
        plan_cache=QuantumExecutionPlanCache(),
    )
    assert result.requested_backend == "gpu"
    assert result.backend == "numpy"
    assert result.device == "cpu"
    assert result.fallback_used
    assert 0 <= result.logical_failures <= result.num_shots


def test_cache_selection_is_deterministic() -> None:
    cache = QuantumExecutionPlanCache()
    first = build_correlated_noise_tensor_network(
        CorrelatedXXXNoise(0.12, 0.07), plan_cache=cache
    )
    cache.clear()
    second = build_correlated_noise_tensor_network(
        CorrelatedXXXNoise(0.12, 0.07), plan_cache=cache
    )
    assert first.contraction_order == second.contraction_order
    assert first.estimated_flops == second.estimated_flops
