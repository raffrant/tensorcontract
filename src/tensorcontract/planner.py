"""Explainable beam-search contraction planner with hardware-aware estimates."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import prod
from time import perf_counter

from .ir import TensorNetwork


@dataclass(frozen=True)
class HardwareProfile:
    name: str = "generic-cpu"
    memory_bytes: int = 8 * 2**30
    bandwidth_bytes_s: float = 50e9
    peak_flops_s: float = 200e9


@dataclass(frozen=True)
class PlanConstraints:
    max_memory_bytes: int | None = None
    max_intermediate_elements: int | None = None
    precision_bytes: int = 8
    beam_width: int = 32
    objective_flops_weight: float = 1.0
    objective_memory_weight: float = 1.0
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    exact: bool = True


@dataclass(frozen=True)
class ContractionStep:
    left: str
    right: str
    output: str
    output_indices: tuple[str, ...]
    flops: int
    output_elements: int
    bytes_moved: int
    arithmetic_intensity: float
    shape_class: str


@dataclass(frozen=True)
class ContractionPlan:
    name: str
    steps: tuple[ContractionStep, ...]
    total_flops: int
    peak_elements: int
    estimated_bytes_moved: int
    score: float
    exact: bool = True

    def report(self) -> str:
        lines = [f"plan={self.name} score={self.score:.3g} flops={self.total_flops} peak_elements={self.peak_elements} bytes={self.estimated_bytes_moved}"]
        lines += [f"  {s.left} x {s.right} -> {s.output} {s.output_indices} flops={s.flops} size={s.output_elements} AI={s.arithmetic_intensity:.2f} ({s.shape_class})" for s in self.steps]
        return "\n".join(lines)


@dataclass(frozen=True)
class PlanningResult:
    selected: ContractionPlan
    candidates: tuple[ContractionPlan, ...]
    planning_seconds: float


def _step(network: TensorNetwork, live: dict[str, tuple[str, ...]], left: str, right: str, number: int, precision: int) -> ContractionStep:
    ai, bi = live[left], live[right]
    others = set(network.open_indices)
    for key, inds in live.items():
        if key not in (left, right):
            others.update(inds)
    union = tuple(dict.fromkeys(ai + bi))
    # Keep precisely labels that are externally live or explicitly open. This
    # also sums a closed degree-one label as soon as its tensor participates.
    out = tuple(i for i in union if i in others)
    dims = network.indices
    contracted = [i for i in union if i not in out]
    flops = max(1, prod(dims[i].dimension for i in union))
    output_elements = max(1, prod(dims[i].dimension for i in out))
    input_elements = prod(dims[i].dimension for i in dict.fromkeys(ai)) + prod(dims[i].dimension for i in dict.fromkeys(bi))
    moved = precision * (input_elements + output_elements)
    shared = [i for i in ai if i in bi]
    if not contracted:
        shape_class = "outer-product/elementwise"
    elif output_elements > flops // 2:
        shape_class = "output-dominated"
    elif shared and min((prod(dims[i].dimension for i in ai if i not in shared) or 1), (prod(dims[i].dimension for i in bi if i not in shared) or 1)) <= 2:
        shape_class = "skinny-GEMM"
    else:
        shape_class = "GEMM-like"
    return ContractionStep(left, right, f"_t{number}", out, flops, output_elements, moved, flops / moved, shape_class)


def _greedy(network: TensorNetwork, constraints: PlanConstraints, mode: str) -> ContractionPlan:
    live = {name: node.indices for name, node in network.nodes.items()}
    steps: list[ContractionStep] = []
    peak = max((prod(network.indices[i].dimension for i in set(v)) for v in live.values()), default=1)
    while len(live) > 1:
        options = [_step(network, live, a, b, len(steps), constraints.precision_bytes) for a, b in combinations(sorted(live), 2)]
        feasible = [s for s in options if (constraints.max_intermediate_elements is None or s.output_elements <= constraints.max_intermediate_elements) and (constraints.max_memory_bytes is None or s.output_elements * constraints.precision_bytes <= constraints.max_memory_bytes)]
        if not feasible:
            raise MemoryError("no contraction satisfies the configured limits")
        if mode == "flops":
            chosen = min(feasible, key=lambda s: (s.flops, s.output_elements, s.left, s.right))
        elif mode == "memory":
            best_size = min(s.output_elements for s in feasible)
            size_ties = [s for s in feasible if s.output_elements == best_size]
            best_flops = min(s.flops for s in size_ties)
            # Deliberately take the reverse stable tie to diversify candidates
            # without weakening the memory-first objective.
            chosen = max((s for s in size_ties if s.flops == best_flops), key=lambda s: (s.left, s.right))
        else:
            chosen = min(feasible, key=lambda s: (s.flops * constraints.objective_flops_weight + s.output_elements * constraints.objective_memory_weight, s.left, s.right))
        del live[chosen.left], live[chosen.right]
        live[chosen.output] = chosen.output_indices
        steps.append(chosen)
        peak = max(peak, chosen.output_elements)
    flops = sum(s.flops for s in steps)
    moved = sum(s.bytes_moved for s in steps)
    score = constraints.objective_flops_weight * flops + constraints.objective_memory_weight * peak
    return ContractionPlan(mode, tuple(steps), flops, peak, moved, score, constraints.exact)


def plan_contraction(network: TensorNetwork, constraints: PlanConstraints) -> PlanningResult:
    started = perf_counter()
    candidates = tuple(_greedy(network, constraints, mode) for mode in ("flops", "memory", "balanced"))
    selected = min(candidates, key=lambda p: (p.score, p.total_flops, p.peak_elements, p.name))
    return PlanningResult(selected, candidates, perf_counter() - started)
