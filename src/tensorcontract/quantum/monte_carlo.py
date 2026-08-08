"""Batched NumPy and optional high-level PyTorch CUDA Monte Carlo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib import import_module
from math import sqrt
from numbers import Integral
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from tensorcontract.backends import (
    BackendUnavailableError,
    get_backend,
    is_backend_available,
)

from .three_qubit import CorrelatedXXXNoise, exact_logical_error_rate
from .cache import (
    BackendExecutionPlan,
    InMemoryBackendPlanCache,
    QuantumExecutionPlanCache,
    make_backend_plan_cache_key,
)


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
class MonteCarloTimings:
    """Measured pipeline phases in seconds; unavailable phases remain zero."""

    random_generation: float = 0.0
    host_to_device_transfer: float = 0.0
    kernel_execution: float = 0.0
    reduction: float = 0.0
    device_to_host_transfer: float = 0.0


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
    requested_backend: str = "numpy"
    device: str = "cpu"
    fallback_used: bool = False
    timings: MonteCarloTimings = field(default_factory=MonteCarloTimings)
    cache_hit: bool = False
    cache_enabled: bool = False
    compilation_time: float = 0.0
    planning_time: float = 0.0
    execution_time: float = 0.0
    total_time: float = 0.0
    plan_key: str | None = None
    fusion_used: bool = False
    kernel_count: int | None = None
    registers_per_thread: int | None = None
    register_spills: int | None = None
    shared_memory_bytes: int | None = None
    occupancy: float | None = None
    random_generation_fused: bool = False
    reduction_fused: bool = False

    @property
    def runtime_seconds(self) -> float:
        """Alias used by visualization and benchmark result schemas."""
        return self.elapsed_time

    @property
    def random_generation_runtime(self) -> float:
        return self.timings.random_generation

    @property
    def host_to_device_runtime(self) -> float:
        return self.timings.host_to_device_transfer

    @property
    def kernel_runtime(self) -> float:
        return self.timings.kernel_execution

    @property
    def reduction_runtime(self) -> float:
        return self.timings.reduction

    @property
    def device_to_host_runtime(self) -> float:
        return self.timings.device_to_host_transfer


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


def is_gpu_available() -> bool:
    """Return whether the optional PyTorch backend has a usable CUDA device."""
    if not is_backend_available("torch"):
        return False
    try:
        torch = import_module("torch")
        return bool(torch.cuda.is_available())
    except (ImportError, OSError, RuntimeError):
        return False


def _statistics(
    *,
    shots: int,
    logical_failures: int,
    physical_error_bits: int,
    noise: CorrelatedXXXNoise,
    backend: str,
    requested_backend: str,
    device: str,
    fallback_used: bool,
    configured_batch: int | None,
    elapsed: float,
    retained: MonteCarloSamples | None,
    timings: MonteCarloTimings,
) -> MonteCarloResult:
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
    throughput = shots / elapsed if shots and elapsed > 0.0 else 0.0
    return MonteCarloResult(
        num_shots=shots,
        logical_failures=logical_failures,
        p_phys=noise.physical_error_rate(),
        p_phys_estimate=physical_estimate,
        p_logical_estimate=logical_estimate,
        p_logical_exact=exact_logical_error_rate(noise),
        standard_error=standard_error,
        confidence_interval=confidence_interval,
        p=noise.p,
        rho=noise.rho,
        backend=backend,
        batch_size=configured_batch,
        elapsed_time=elapsed,
        shots_per_second=throughput,
        samples=retained,
        requested_backend=requested_backend,
        device=device,
        fallback_used=fallback_used,
        timings=timings,
    )


def _run_numpy_monte_carlo(
    shots: int,
    noise: CorrelatedXXXNoise,
    seed: int | None,
    configured_batch: int | None,
    return_samples: bool,
    *,
    requested_backend: str,
    fallback_used: bool,
) -> MonteCarloResult:
    """Execute the existing vectorized NumPy reference pipeline."""
    effective_batch = shots if configured_batch is None else configured_batch
    effective_batch = max(1, effective_batch)
    rng = np.random.default_rng(seed)
    retained = _allocate_samples(shots) if return_samples else None

    started = perf_counter()
    random_generation_time = 0.0
    kernel_execution_time = 0.0
    reduction_time = 0.0
    logical_failures = 0
    physical_error_bits = 0
    offset = 0
    while offset < shots:
        count = min(effective_batch, shots - offset)
        random_started = perf_counter()
        # One row per shot makes RNG consumption independent of chunk size.
        draws = rng.random((count, 4))
        random_generation_time += perf_counter() - random_started

        kernel_started = perf_counter()
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
        kernel_execution_time += perf_counter() - kernel_started

        reduction_started = perf_counter()
        logical_failures += int(np.count_nonzero(failures))
        physical_error_bits += int(np.count_nonzero(errors))
        reduction_time += perf_counter() - reduction_started
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

    elapsed = perf_counter() - started
    return _statistics(
        shots=shots,
        logical_failures=logical_failures,
        physical_error_bits=physical_error_bits,
        noise=noise,
        backend="numpy",
        requested_backend=requested_backend,
        device="cpu",
        fallback_used=fallback_used,
        configured_batch=configured_batch,
        elapsed=elapsed,
        retained=retained,
        timings=MonteCarloTimings(
            random_generation=random_generation_time,
            kernel_execution=kernel_execution_time,
            reduction=reduction_time,
        ),
    )


def _run_gpu_monte_carlo(
    shots: int,
    noise: CorrelatedXXXNoise,
    seed: int | None,
    configured_batch: int | None,
    return_samples: bool,
    backend_plan: BackendExecutionPlan | None = None,
) -> MonteCarloResult:
    """Execute batched high-level PyTorch operations on one CUDA device."""
    # Constructing the existing backend validates optional PyTorch/CUDA setup.
    get_backend("torch", device="cuda")
    torch = import_module("torch")
    device = torch.device("cuda")
    effective_batch = shots if configured_batch is None else configured_batch
    effective_batch = max(1, effective_batch)
    retained = _allocate_samples(shots) if return_samples else None

    started = perf_counter()
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    torch.cuda.synchronize()
    transfer_started = perf_counter()
    cached_lookup = (
        None
        if backend_plan is None or backend_plan.artifacts is None
        else backend_plan.artifacts.get("recovery_lookup")
    )
    recovery_lookup = (
        cached_lookup
        if cached_lookup is not None
        else torch.as_tensor(_RECOVERY_LOOKUP, device=device)
    )
    torch.cuda.synchronize()
    host_to_device_time = (
        0.0 if cached_lookup is not None else perf_counter() - transfer_started
    )
    random_generation_time = 0.0
    kernel_execution_time = 0.0
    reduction_time = 0.0
    device_to_host_time = 0.0
    logical_failures = 0
    physical_error_bits = 0
    offset = 0
    while offset < shots:
        count = min(effective_batch, shots - offset)
        torch.cuda.synchronize()
        phase_started = perf_counter()
        draws = torch.rand((count, 4), generator=generator, device=device)
        torch.cuda.synchronize()
        random_generation_time += perf_counter() - phase_started

        phase_started = perf_counter()
        correlated = (draws[:, 0] < noise.rho).to(torch.int8)
        local = (draws[:, 1:] < noise.p).to(torch.int8)
        errors = torch.bitwise_xor(correlated[:, None], local)
        syndromes = torch.stack(
            (
                torch.bitwise_xor(errors[:, 0], errors[:, 1]),
                torch.bitwise_xor(errors[:, 1], errors[:, 2]),
            ),
            dim=1,
        )
        syndrome_codes = (syndromes[:, 0] * 2 + syndromes[:, 1]).to(torch.int64)
        recoveries = recovery_lookup[syndrome_codes]
        residuals = torch.bitwise_xor(errors, recoveries)
        failures = torch.all(residuals == 1, dim=1)
        successes = torch.all(residuals == 0, dim=1)
        torch.cuda.synchronize()
        kernel_execution_time += perf_counter() - phase_started

        phase_started = perf_counter()
        failure_count = torch.count_nonzero(failures)
        physical_count = torch.count_nonzero(errors)
        residuals_valid = torch.all(torch.logical_or(failures, successes))
        torch.cuda.synchronize()
        reduction_time += perf_counter() - phase_started

        phase_started = perf_counter()
        logical_failures += int(failure_count.item())
        physical_error_bits += int(physical_count.item())
        if not bool(residuals_valid.item()):
            raise RuntimeError("GPU recovery produced a residual outside the code space")
        if retained is not None:
            destination = slice(offset, offset + count)
            retained.correlated[destination] = correlated.cpu().numpy()
            retained.local_errors[destination] = local.cpu().numpy()
            retained.errors[destination] = errors.cpu().numpy()
            retained.syndromes[destination] = syndromes.cpu().numpy()
            retained.recoveries[destination] = recoveries.cpu().numpy()
            retained.residual_errors[destination] = residuals.cpu().numpy()
            retained.logical_failures[destination] = failures.cpu().numpy()
        torch.cuda.synchronize()
        device_to_host_time += perf_counter() - phase_started
        offset += count

    elapsed = perf_counter() - started
    return _statistics(
        shots=shots,
        logical_failures=logical_failures,
        physical_error_bits=physical_error_bits,
        noise=noise,
        backend="torch-cuda",
        requested_backend="gpu",
        device=str(device),
        fallback_used=False,
        configured_batch=configured_batch,
        elapsed=elapsed,
        retained=retained,
        timings=MonteCarloTimings(
            random_generation=random_generation_time,
            host_to_device_transfer=host_to_device_time,
            kernel_execution=kernel_execution_time,
            reduction=reduction_time,
            device_to_host_transfer=device_to_host_time,
        ),
    )


def _run_fused_gpu_monte_carlo(
    shots: int,
    noise: CorrelatedXXXNoise,
    seed: int | None,
    configured_batch: int | None,
    backend_plan: BackendExecutionPlan,
) -> MonteCarloResult:
    """Execute RNG, decoding, and reduction in one Triton kernel per chunk."""
    torch = import_module("torch")
    device = torch.device(backend_plan.device)
    artifacts = backend_plan.artifacts or {}
    triton_backend = artifacts.get("triton_backend")
    if triton_backend is None:
        raise BackendUnavailableError("cached fused plan has no Triton backend")
    block_size = int(artifacts.get("block_size", 256))
    effective_batch = max(1, shots if configured_batch is None else configured_batch)
    normalized_seed = (
        int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        if seed is None
        else int(seed) & 0xFFFFFFFF
    )

    started = perf_counter()
    torch.cuda.synchronize(device)
    counters = torch.zeros(2, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    kernel_time = 0.0
    kernel_count = 1  # counter initialization is one device kernel
    offset = 0
    diagnostics = artifacts.get("diagnostics")
    while offset < shots:
        count = min(effective_batch, shots - offset)
        torch.cuda.synchronize(device)
        phase_started = perf_counter()
        diagnostics = triton_backend.launch_three_qubit_qec_trajectory(
            number_of_shots=count,
            base_shot=offset,
            seed=normalized_seed,
            p=noise.p,
            rho=noise.rho,
            device=device,
            block_size=block_size,
            counters=counters,
        )
        torch.cuda.synchronize(device)
        kernel_time += perf_counter() - phase_started
        kernel_count += 1
        offset += count

    transfer_started = perf_counter()
    host_counts = counters.cpu().numpy()
    torch.cuda.synchronize(device)
    device_to_host_time = perf_counter() - transfer_started
    elapsed = perf_counter() - started
    result = _statistics(
        shots=shots,
        logical_failures=int(host_counts[0]),
        physical_error_bits=int(host_counts[1]),
        noise=noise,
        backend="triton-cuda",
        requested_backend="gpu",
        device=str(device),
        fallback_used=False,
        configured_batch=configured_batch,
        elapsed=elapsed,
        retained=None,
        timings=MonteCarloTimings(
            kernel_execution=kernel_time,
            device_to_host_transfer=device_to_host_time,
        ),
    )
    return replace(
        result,
        fusion_used=True,
        kernel_count=kernel_count,
        registers_per_thread=getattr(diagnostics, "registers_per_thread", None),
        register_spills=getattr(diagnostics, "register_spills", None),
        shared_memory_bytes=getattr(diagnostics, "shared_memory_bytes", None),
        occupancy=getattr(diagnostics, "occupancy", None),
        random_generation_fused=True,
        reduction_fused=True,
    )


def run_monte_carlo(
    num_shots: int,
    p: float,
    rho: float,
    seed: int | None = None,
    batch_size: int | None = None,
    backend: str = "numpy",
    return_samples: bool = False,
    plan_cache: QuantumExecutionPlanCache | InMemoryBackendPlanCache | None = None,
    cache_enabled: bool = True,
    fusion_options: Mapping[str, object] | None = None,
    contraction_options: Mapping[str, object] | None = None,
) -> MonteCarloResult:
    """Estimate the logical-error rate using vectorized NumPy or CUDA batches.

    A Python loop is used only to bound memory through chunks. Every operation
    within a chunk—including sampling, XOR, recovery lookup, residual decoding,
    and reduction—is vectorized over all shots in that chunk.

    For zero shots, the estimate, standard error, confidence bounds, and
    throughput are defined as zero. ``p_phys`` is the exact model rate
    ``p + rho - 2*p*rho``; ``p_phys_estimate`` is the sampled bit-error rate.
    """
    total_started = perf_counter()
    shots = _validate_shot_count(num_shots)
    configured_batch = _validate_batch_size(batch_size)
    if backend not in ("numpy", "gpu"):
        raise ValueError(
            f"unsupported Monte Carlo backend {backend!r}; Monte Carlo supports "
            "only 'numpy' or 'gpu'"
        )
    if not isinstance(return_samples, bool):
        raise TypeError("return_samples must be a bool")
    noise = CorrelatedXXXNoise(p, rho)
    fusion_requested = bool((fusion_options or {}).get("enabled", False))

    effective_batch = max(1, shots if configured_batch is None else configured_batch)
    gpu_available = backend == "gpu" and is_gpu_available()
    key = make_backend_plan_cache_key(
        network_topology=("three-qubit-repetition", "correlated-xxx-monte-carlo", 1),
        tensor_shapes=(("draws", (effective_batch, 4)), ("errors", (effective_batch, 3))),
        index_structure=("batch", "c", "u1", "u2", "u3", "e1", "e2", "e3", "s1", "s2"),
        noise_model_structure=("bernoulli-shared-xor", 1, 3),
        dtype="torch.float32" if gpu_available else "numpy.float64",
        backend=backend,
        batch_size=effective_batch,
        device="cuda" if gpu_available else ("cpu-fallback" if backend == "gpu" else "cpu"),
        fusion_options={
            **dict(fusion_options or {}),
            "return_samples": return_samples,
        },
        contraction_options=contraction_options,
    )

    def create_plan() -> BackendExecutionPlan:
        if not gpu_available:
            return BackendExecutionPlan(
                backend, "numpy", "cpu", "numpy.float64", effective_batch,
                bool((fusion_options or {}).get("enabled", False)),
                fallback_used=backend == "gpu",
                fallback_reason="CUDA GPU unavailable" if backend == "gpu" else None,
            )
        torch = import_module("torch")
        device = torch.device("cuda")
        if fusion_requested and not return_samples and shots <= (2**31 - 1) // 3:
            triton_backend = get_backend(
                "triton", device="cuda", policy="triton"
            )
            block_size = int((fusion_options or {}).get("block_size", 256))
            diagnostics = triton_backend.prepare_three_qubit_qec_trajectory(
                device=device,
                block_size=block_size,
                number_of_shots=effective_batch,
            )
            return BackendExecutionPlan(
                backend,
                "triton-cuda",
                str(device),
                "triton-rand-fp32",
                effective_batch,
                True,
                artifacts={
                    "triton_backend": triton_backend,
                    "block_size": block_size,
                    "diagnostics": diagnostics,
                },
            )
        get_backend("torch", device="cuda")
        recovery_lookup = torch.as_tensor(_RECOVERY_LOOKUP, device=device)
        torch.cuda.synchronize()
        return BackendExecutionPlan(
            backend, "torch-cuda", str(device), "torch.float32", effective_batch,
            False,
            artifacts={"recovery_lookup": recovery_lookup},
        )

    def fallback_plan(error: Exception) -> BackendExecutionPlan:
        if gpu_available:
            torch = import_module("torch")
            device = torch.device("cuda")
            recovery_lookup = torch.as_tensor(_RECOVERY_LOOKUP, device=device)
            torch.cuda.synchronize()
            return BackendExecutionPlan(
                backend,
                "torch-cuda",
                str(device),
                "torch.float32",
                effective_batch,
                False,
                fallback_used=True,
                fallback_reason=(
                    f"fused backend plan construction failed: {type(error).__name__}"
                ),
                artifacts={"recovery_lookup": recovery_lookup},
            )
        return BackendExecutionPlan(
            backend, "numpy", "cpu", "numpy.float64", effective_batch,
            False, fallback_used=True,
            fallback_reason=f"backend plan construction failed: {type(error).__name__}",
        )

    cache_store = plan_cache.backend if isinstance(plan_cache, QuantumExecutionPlanCache) else plan_cache
    if cache_store is None:
        cache_store = InMemoryBackendPlanCache(enabled=False)
    if cache_enabled:
        lookup = cache_store.get_or_create(key, create_plan, fallback_factory=fallback_plan)
    else:
        disabled_store = InMemoryBackendPlanCache(enabled=False)
        lookup = disabled_store.get_or_create(key, create_plan, fallback_factory=fallback_plan)

    execution_started = perf_counter()
    if lookup.plan.backend == "triton-cuda":
        try:
            result = _run_fused_gpu_monte_carlo(
                shots, noise, seed, configured_batch, lookup.plan
            )
        except (BackendUnavailableError, ImportError, OSError, RuntimeError):
            # Runtime device failures retain the already-correct CUDA fallback.
            result = _run_gpu_monte_carlo(
                shots, noise, seed, configured_batch, return_samples
            )
            result = replace(result, fallback_used=True)
    elif lookup.plan.backend == "torch-cuda":
        try:
            result = _run_gpu_monte_carlo(
                shots, noise, seed, configured_batch, return_samples, lookup.plan
            )
        except (BackendUnavailableError, ImportError, OSError):
            # Availability can change between probing and backend construction.
            result = _run_numpy_monte_carlo(
                shots, noise, seed, configured_batch, return_samples,
                requested_backend=backend, fallback_used=True,
            )
    else:
        result = _run_numpy_monte_carlo(
            shots, noise, seed, configured_batch, return_samples,
            requested_backend=backend, fallback_used=lookup.plan.fallback_used,
        )
    execution_time = perf_counter() - execution_started
    return replace(
        result,
        cache_hit=lookup.hit,
        cache_enabled=lookup.cache_enabled and cache_enabled,
        compilation_time=lookup.compilation_time,
        planning_time=lookup.planning_time,
        execution_time=execution_time,
        total_time=perf_counter() - total_started,
        plan_key=lookup.key.digest,
    )


def estimate_logical_error_rate(
    num_shots: int,
    p: float,
    rho: float,
    seed: int | None = None,
    batch_size: int | None = None,
    backend: str = "numpy",
    return_samples: bool = False,
    plan_cache: QuantumExecutionPlanCache | InMemoryBackendPlanCache | None = None,
    cache_enabled: bool = True,
    fusion_options: Mapping[str, object] | None = None,
    contraction_options: Mapping[str, object] | None = None,
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
        plan_cache=plan_cache,
        cache_enabled=cache_enabled,
        fusion_options=fusion_options,
        contraction_options=contraction_options,
    )
