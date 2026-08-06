"""Benchmark contraction ordering for materialized symbolic networks."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np
import torch

from tensorcontract.backend import execute_numpy
from tensorcontract.planner import PlanConstraints, build_ordered_plan, plan_contraction

from .model import SymbolicNetwork
from .torch_backend import execute_torch, resolve_device


@dataclass(frozen=True)
class OrderingBenchmark:
    ordering: str
    step_order: tuple[str, ...]
    median_seconds: float
    minimum_seconds: float
    estimated_flops: int
    peak_intermediate_elements: int
    estimated_bytes_moved: int
    measured_peak_device_bytes: int | None
    absolute_error_vs_numpy: float
    result: float


@dataclass(frozen=True)
class SymbolicBenchmarkReport:
    device: str
    cuda_available: bool
    torch_version: str
    dimension: int
    seed: int
    warmup: int
    repeats: int
    symbolic_expressions: dict[str, str]
    orderings: tuple[OrderingBenchmark, ...]

def benchmark_orderings(
    symbolic: SymbolicNetwork,
    device: str = "auto",
    warmup: int = 3,
    repeats: int = 20,
) -> SymbolicBenchmarkReport:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    concrete = symbolic.materialize()
    planning = plan_contraction(concrete, PlanConstraints())
    adverse = build_ordered_plan(
        concrete,
        (("f0", "f2"), ("f1", "f3"), ("_t0", "f4"), ("_t1", "_t2")),
        name="adverse",
    )
    # Keep distinct step sequences only, while always including the adverse
    # order that exposes the cost of forming a five-index intermediate early.
    plans = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for plan in (*planning.candidates, adverse):
        signature = tuple((step.left, step.right) for step in plan.steps)
        if signature not in seen:
            seen.add(signature)
            plans.append(plan)
    selected_device = resolve_device(device)
    records: list[OrderingBenchmark] = []
    for plan in plans:  # Intentionally benchmark one ordering at a time.
        reference = float(np.real_if_close(execute_numpy(concrete, plan)))
        for _ in range(warmup):
            execute_torch(concrete, plan, selected_device, measure_memory=False)
        samples: list[float] = []
        measured_peak: int | None = None
        execution = None
        for _ in range(repeats):
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)
            started = perf_counter()
            execution = execute_torch(concrete, plan, selected_device, measure_memory=True)
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)
            samples.append(perf_counter() - started)
            if execution.peak_memory_bytes is not None:
                measured_peak = max(measured_peak or 0, execution.peak_memory_bytes)
        assert execution is not None
        records.append(OrderingBenchmark(
            plan.name,
            tuple(f"{step.left}×{step.right}" for step in plan.steps),
            median(samples),
            min(samples),
            plan.total_flops,
            plan.peak_elements,
            plan.estimated_bytes_moved,
            measured_peak,
            abs(execution.value - reference),
            execution.value,
        ))
    return SymbolicBenchmarkReport(
        str(selected_device), torch.cuda.is_available(), torch.__version__, symbolic.dimension,
        symbolic.seed, warmup, repeats,
        {node.name: str(node.expression) for node in symbolic.nodes}, tuple(records),
    )
