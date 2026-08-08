"""Shape-specialized in-memory caching for symbolic execution plans."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from threading import RLock
from typing import Any, Hashable

from .ir import SymbolicGraph
from .planner import (
    SymbolicExecutionPlan,
    SymbolicPlannerOptions,
    plan_symbolic_contractions,
)


class PlanCacheError(ValueError):
    """Base class for invalid plan-cache usage."""


class IncompatibleBindingError(PlanCacheError):
    """Raised when runtime bindings do not match the symbolic graph."""


def _freeze(value: Any) -> Hashable:
    """Convert configuration and metadata into a deterministic hashable form."""
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, Enum):
        return (type(value).__module__, type(value).__qualname__, _freeze(value.value))
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((_freeze(key), _freeze(item)) for key, item in value.items()),
                key=repr,
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple((field.name, _freeze(getattr(value, field.name))) for field in fields(value)),
        )
    return (type(value).__module__, type(value).__qualname__, repr(value))


@dataclass(frozen=True, slots=True)
class TensorBindingSignature:
    """Correctness- and layout-relevant properties of one runtime binding."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    strides: tuple[int, ...] | None
    stride_unit: str
    layout: str
    storage_offset: int | None
    requires_grad: bool


@dataclass(frozen=True, slots=True)
class ExecutionPlanCacheKey:
    """Stable specialization key for one graph/backend/runtime signature."""

    schema_version: int
    graph_signature: Hashable
    binding_signatures: tuple[TensorBindingSignature, ...]
    backend_name: str
    backend_configuration: Hashable
    planner_configuration: Hashable

    @property
    def digest(self) -> str:
        """Return a short stable identifier useful in cache diagnostics."""
        return sha256(repr(self).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class PlanCacheResult:
    """A planned result together with hit/miss reporting."""

    plan: SymbolicExecutionPlan
    key: ExecutionPlanCacheKey
    hit: bool
    cache_enabled: bool


@dataclass(frozen=True, slots=True)
class PlanCacheInfo:
    """Snapshot of in-memory cache state and counters."""

    enabled: bool
    hits: int
    misses: int
    stores: int
    evictions: int
    size: int
    max_entries: int | None


def _graph_signature(graph: SymbolicGraph) -> Hashable:
    return (
        "SymbolicGraph-v1",
        tuple(
            (
                index.name,
                _freeze(index.dimension),
                index.role.value,
                _freeze(index.metadata),
            )
            for index in graph.indices.values()
        ),
        tuple(
            (
                tensor.name,
                tensor.indices,
                tensor.shape,
                _freeze(tensor.metadata),
            )
            for tensor in graph.tensors.values()
        ),
        tuple(_freeze(operation) for operation in graph.operations.values()),
        _freeze(graph.metadata),
    )


def _binding_signature(name: str, value: Any) -> TensorBindingSignature:
    if not hasattr(value, "shape"):
        raise IncompatibleBindingError(
            f"binding {name!r} has no shape and cannot specialize a plan"
        )
    try:
        shape = tuple(int(dimension) for dimension in value.shape)
    except (TypeError, ValueError) as error:
        raise IncompatibleBindingError(
            f"binding {name!r} has an invalid shape {getattr(value, 'shape', None)!r}"
        ) from error

    dtype_object = getattr(value, "dtype", None)
    dtype = str(dtype_object) if dtype_object is not None else type(value).__qualname__
    device = str(getattr(value, "device", "cpu"))
    module = type(value).__module__

    raw_strides: Any = None
    stride_unit = "unknown"
    if module.startswith("numpy"):
        raw_strides = getattr(value, "strides", None)
        itemsize = int(getattr(dtype_object, "itemsize", 1) or 1)
        strides = (
            None
            if raw_strides is None
            else tuple(int(stride) // itemsize for stride in raw_strides)
        )
        stride_unit = "elements"
    else:
        stride_attribute = getattr(value, "stride", None)
        if callable(stride_attribute):
            raw_strides = stride_attribute()
            stride_unit = "elements"
        elif hasattr(value, "strides"):
            raw_strides = value.strides
            stride_unit = "native"
        strides = (
            None if raw_strides is None else tuple(int(stride) for stride in raw_strides)
        )

    layout = "unknown"
    flags = getattr(value, "flags", None)
    if flags is not None:
        if bool(getattr(flags, "c_contiguous", False)):
            layout = "c-contiguous"
        elif bool(getattr(flags, "f_contiguous", False)):
            layout = "f-contiguous"
        else:
            layout = "strided"
    else:
        contiguous = getattr(value, "is_contiguous", None)
        if callable(contiguous):
            layout = "contiguous" if bool(contiguous()) else "strided"
        elif strides is not None:
            layout = "strided"

    offset_attribute = getattr(value, "storage_offset", None)
    storage_offset = int(offset_attribute()) if callable(offset_attribute) else None
    return TensorBindingSignature(
        name,
        shape,
        dtype,
        device,
        strides,
        stride_unit,
        layout,
        storage_offset,
        bool(getattr(value, "requires_grad", False)),
    )


def make_execution_plan_cache_key(
    graph: SymbolicGraph,
    bindings: Mapping[str, Any],
    backend: str | Any,
    planner_options: SymbolicPlannerOptions | None = None,
    backend_options: Mapping[str, Any] | None = None,
) -> ExecutionPlanCacheKey:
    """Build a conservative specialization key without importing backends."""
    graph.validate()
    signatures: list[TensorBindingSignature] = []
    for name in graph.tensors:
        if name not in bindings:
            raise IncompatibleBindingError(
                f"missing binding {name!r} while constructing a plan-cache key"
            )
        signature = _binding_signature(name, bindings[name])
        expected = graph.value_shape(name)
        if signature.shape != expected:
            raise IncompatibleBindingError(
                f"binding {name!r} has shape {signature.shape}, expected {expected}"
            )
        signatures.append(signature)

    if isinstance(backend, str):
        backend_name = backend
        implicit_configuration: Mapping[str, Any] = {}
    else:
        backend_name = str(getattr(backend, "name", type(backend).__qualname__))
        implicit_configuration = {
            key: value
            for key, value in vars(backend).items()
            if not key.startswith("_")
        }
    combined_configuration = {
        "instance": implicit_configuration,
        "options": dict(backend_options or {}),
    }
    configured = planner_options or SymbolicPlannerOptions()
    return ExecutionPlanCacheKey(
        1,
        _graph_signature(graph),
        tuple(signatures),
        backend_name,
        _freeze(combined_configuration),
        _freeze(configured),
    )


class InMemoryPlanCache:
    """Thread-safe LRU cache for shape-specialized symbolic execution plans."""

    def __init__(self, *, enabled: bool = True, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries < 1:
            raise PlanCacheError("max_entries must be positive")
        self.enabled = enabled
        self.max_entries = max_entries
        self._entries: OrderedDict[ExecutionPlanCacheKey, SymbolicExecutionPlan] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0
        self._lock = RLock()

    def get_or_plan(
        self,
        graph: SymbolicGraph,
        bindings: Mapping[str, Any],
        backend: str | Any,
        planner_options: SymbolicPlannerOptions | None = None,
        backend_options: Mapping[str, Any] | None = None,
    ) -> PlanCacheResult:
        """Return a compatible cached plan or deterministically create one."""
        key = make_execution_plan_cache_key(
            graph, bindings, backend, planner_options, backend_options
        )
        with self._lock:
            if self.enabled and key in self._entries:
                self._hits += 1
                self._entries.move_to_end(key)
                return PlanCacheResult(deepcopy(self._entries[key]), key, True, True)

            self._misses += 1
            plan = plan_symbolic_contractions(graph, planner_options)
            if self.enabled:
                self._entries[key] = deepcopy(plan)
                self._entries.move_to_end(key)
                self._stores += 1
                if self.max_entries is not None and len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
                    self._evictions += 1
            return PlanCacheResult(deepcopy(plan), key, False, self.enabled)

    def invalidate(self, key: ExecutionPlanCacheKey) -> bool:
        """Remove one specialization, returning whether it existed."""
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self, *, reset_statistics: bool = False) -> None:
        """Remove all cached plans and optionally reset counters."""
        with self._lock:
            self._entries.clear()
            if reset_statistics:
                self._hits = self._misses = self._stores = self._evictions = 0

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable reuse without deleting existing entries."""
        with self._lock:
            self.enabled = enabled

    def info(self) -> PlanCacheInfo:
        """Return hit, miss, storage, eviction, and size statistics."""
        with self._lock:
            return PlanCacheInfo(
                self.enabled,
                self._hits,
                self._misses,
                self._stores,
                self._evictions,
                len(self._entries),
                self.max_entries,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
