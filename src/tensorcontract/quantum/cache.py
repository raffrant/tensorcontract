"""In-memory plan caches for the three-qubit QEC workflows.

The symbolic contraction cache is reused directly.  This module adds the
corresponding backend-plan cache and a small facade so callers can inspect or
clear both stores together.  Noise *values* are deliberately absent from the
backend key: changing ``p`` or ``rho`` changes bindings, not the executable
topology.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from time import perf_counter
from typing import Any, Hashable

from tensorcontract.symbolics import InMemoryPlanCache, PlanCacheInfo


def _freeze(value: Any) -> Hashable:
    """Convert public cache-key inputs to deterministic, hashable values."""
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, Mapping):
        return tuple(sorted(((_freeze(k), _freeze(v)) for k, v in value.items()), key=repr))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return (type(value).__module__, type(value).__qualname__, repr(value))


@dataclass(frozen=True, slots=True)
class BackendPlanCacheKey:
    """Complete specialization signature for a batched backend plan."""

    schema_version: int
    network_topology: Hashable
    tensor_shapes: Hashable
    index_structure: Hashable
    noise_model_structure: Hashable
    dtype: str
    backend: str
    batch_size: int
    device: str
    fusion_options: Hashable
    contraction_options: Hashable

    @property
    def digest(self) -> str:
        """Return a stable, compact identifier for diagnostics."""
        return sha256(repr(self).encode("utf-8")).hexdigest()[:16]


def make_backend_plan_cache_key(
    *,
    network_topology: object,
    tensor_shapes: object,
    index_structure: object,
    noise_model_structure: object,
    dtype: object,
    backend: str,
    batch_size: int,
    device: str,
    fusion_options: Mapping[str, object] | None = None,
    contraction_options: Mapping[str, object] | None = None,
) -> BackendPlanCacheKey:
    """Build a stable backend key; runtime probabilities are not accepted."""
    if batch_size < 1:
        raise ValueError("batch_size in a backend plan key must be positive")
    return BackendPlanCacheKey(
        1,
        _freeze(network_topology),
        _freeze(tensor_shapes),
        _freeze(index_structure),
        _freeze(noise_model_structure),
        str(dtype),
        str(backend),
        int(batch_size),
        str(device),
        _freeze(dict(fusion_options or {})),
        _freeze(dict(contraction_options or {})),
    )


@dataclass(frozen=True, slots=True)
class BackendExecutionPlan:
    """Reusable backend choice and immutable device-side artifacts."""

    requested_backend: str
    backend: str
    device: str
    dtype: str
    batch_size: int
    fusion_enabled: bool
    fallback_used: bool = False
    fallback_reason: str | None = None
    artifacts: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BackendPlanResult:
    """One backend plan lookup with cache and construction timings."""

    plan: BackendExecutionPlan
    key: BackendPlanCacheKey
    hit: bool
    cache_enabled: bool
    compilation_time: float
    planning_time: float


@dataclass(frozen=True, slots=True)
class BackendPlanCacheInfo:
    """Inspectable backend cache counters."""

    enabled: bool
    hits: int
    misses: int
    stores: int
    compilation_failures: int
    size: int


class InMemoryBackendPlanCache:
    """Thread-safe deterministic cache for reusable backend plans."""

    def __init__(self, *, enabled: bool = True, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.enabled = enabled
        self.max_entries = max_entries
        self._entries: OrderedDict[BackendPlanCacheKey, BackendExecutionPlan] = OrderedDict()
        self._hits = self._misses = self._stores = self._compilation_failures = 0
        self._lock = RLock()

    def get_or_create(
        self,
        key: BackendPlanCacheKey,
        factory: Callable[[], BackendExecutionPlan],
        *,
        fallback_factory: Callable[[Exception], BackendExecutionPlan] | None = None,
    ) -> BackendPlanResult:
        """Return a compatible plan, falling back safely after build failure."""
        planning_started = perf_counter()
        with self._lock:
            if self.enabled and key in self._entries:
                self._hits += 1
                self._entries.move_to_end(key)
                return BackendPlanResult(
                    self._entries[key], key, True, True, 0.0,
                    perf_counter() - planning_started,
                )
            self._misses += 1

        compile_started = perf_counter()
        try:
            plan = factory()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            with self._lock:
                self._compilation_failures += 1
            if fallback_factory is None:
                raise
            plan = fallback_factory(error)
        compilation_time = perf_counter() - compile_started

        # A failed device compilation is intentionally not cached: a transient
        # CUDA failure must not permanently shadow a subsequently usable GPU.
        failed = plan.fallback_reason is not None and plan.requested_backend != plan.backend
        with self._lock:
            if self.enabled and not failed:
                self._entries[key] = plan
                self._entries.move_to_end(key)
                self._stores += 1
                if self.max_entries is not None and len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
        return BackendPlanResult(
            plan, key, False, self.enabled, compilation_time,
            perf_counter() - planning_started,
        )

    def inspect(self) -> tuple[tuple[str, BackendExecutionPlan], ...]:
        """Return cache digests and plans in least-to-most-recent order."""
        with self._lock:
            return tuple((key.digest, plan) for key, plan in self._entries.items())

    def clear(self, *, reset_statistics: bool = False) -> None:
        with self._lock:
            self._entries.clear()
            if reset_statistics:
                self._hits = self._misses = self._stores = self._compilation_failures = 0

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = bool(enabled)

    def info(self) -> BackendPlanCacheInfo:
        with self._lock:
            return BackendPlanCacheInfo(
                self.enabled, self._hits, self._misses, self._stores,
                self._compilation_failures, len(self._entries),
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


@dataclass(frozen=True, slots=True)
class QuantumCacheInfo:
    """Combined inspection snapshot for contraction and backend caches."""

    contraction: PlanCacheInfo
    backend: BackendPlanCacheInfo


class QuantumExecutionPlanCache:
    """Facade owning both symbolic-contraction and backend-plan stores."""

    def __init__(self, *, enabled: bool = True, max_entries: int | None = None) -> None:
        self.contraction = InMemoryPlanCache(enabled=enabled, max_entries=max_entries)
        self.backend = InMemoryBackendPlanCache(enabled=enabled, max_entries=max_entries)

    def clear(self, *, reset_statistics: bool = False) -> None:
        self.contraction.clear(reset_statistics=reset_statistics)
        self.backend.clear(reset_statistics=reset_statistics)

    def set_enabled(self, enabled: bool) -> None:
        self.contraction.set_enabled(enabled)
        self.backend.set_enabled(enabled)

    def info(self) -> QuantumCacheInfo:
        return QuantumCacheInfo(self.contraction.info(), self.backend.info())

    def inspect(self) -> dict[str, object]:
        """Return public cache state without exposing mutable internals."""
        return {"info": self.info(), "backend_entries": self.backend.inspect()}
