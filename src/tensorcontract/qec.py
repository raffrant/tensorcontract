"""Exact small-code QEC constructions and independent Pauli-X decoding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Iterable

import numpy as np

from .backend import execute_numpy
from .ir import Index, TensorKind, TensorNetwork, TensorNode
from .planner import PlanConstraints, PlanningResult, plan_contraction
from .rewrite import RewriteEngine, RewriteRecord


@dataclass(frozen=True)
class IndependentPauliNoise:
    """Single-qubit Pauli channel; the vertical slice decodes its X component."""
    p_i: float
    p_x: float
    p_y: float = 0.0
    p_z: float = 0.0

    def __post_init__(self) -> None:
        probabilities = (self.p_i, self.p_x, self.p_y, self.p_z)
        if any(p < 0 for p in probabilities) or not np.isclose(sum(probabilities), 1.0):
            raise ValueError("Pauli probabilities must be nonnegative and sum to one")

    @classmethod
    def bit_flip(cls, probability: float) -> "IndependentPauliNoise":
        return cls(1.0 - probability, probability)

    @property
    def x_component_probability(self) -> float:
        return self.p_x + self.p_y


@dataclass(frozen=True)
class RepetitionCode:
    length: int

    def __post_init__(self) -> None:
        if self.length < 2:
            raise ValueError("repetition code length must be at least two")

    @property
    def parity_checks(self) -> tuple[tuple[int, int], ...]:
        return tuple((i, i + 1) for i in range(self.length - 1))

    def syndrome(self, error: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(error[a] ^ error[b] for a, b in self.parity_checks)

    def logical_sector(self, error: tuple[int, ...]) -> int:
        # Errors with the same syndrome differ by either I or the all-X logical.
        return error[0]


def _parity_data(rank: int, target: int) -> np.ndarray:
    data = np.zeros((2,) * rank, dtype=np.float64)
    for bits in product((0, 1), repeat=rank):
        data[bits] = float(sum(bits) % 2 == target)
    return data


def build_repetition_decoding_network(code: RepetitionCode, syndrome: tuple[int, ...], noise: IndependentPauliNoise, logical_sector: int) -> TensorNetwork:
    if len(syndrome) != code.length - 1 or logical_sector not in (0, 1):
        raise ValueError("invalid syndrome or logical sector")
    net = TensorNetwork(metadata={"construction": "repetition-decoder", "exact": True})
    for q in range(code.length):
        net.add_index(Index(f"raw{q}", 2, {"role": "error"}))
        net.add_index(Index(f"e{q}", 2, {"role": "error-hyperedge"}))
    net.add_index(Index("logical", 2, {"role": "logical-sector"}))
    p = noise.x_component_probability
    for q in range(code.length):
        net.add_node(TensorNode(f"noise{q}", (f"raw{q}",), TensorKind.PAULI_NOISE, np.array([1-p, p])))
        net.add_node(TensorNode(f"wire{q}", (f"raw{q}", f"e{q}"), TensorKind.IDENTITY, np.eye(2)))
    for c, ((a, b), observed) in enumerate(zip(code.parity_checks, syndrome)):
        net.add_node(TensorNode(f"check{c}", (f"e{a}", f"e{b}"), TensorKind.PARITY, _parity_data(2, observed), {"syndrome": observed}))
    logical_indices = tuple(f"e{q}" for q in range(code.length)) + ("logical",)
    logical_data = np.zeros((2,) * (code.length + 1))
    for bits in product((0, 1), repeat=code.length):
        logical_data[bits + (bits[0],)] = 1.0
    net.add_node(TensorNode("logical-map", logical_indices, TensorKind.PARITY, logical_data))
    selector = np.array([1.0, 0.0]) if logical_sector == 0 else np.array([0.0, 1.0])
    net.add_node(TensorNode("logical-selector", ("logical",), TensorKind.DENSE, selector))
    net.add_node(TensorNode("normalization", (), TensorKind.SCALAR, np.asarray(1.0)))
    net.validate()
    return net


def exhaustive_coset_probabilities(code: RepetitionCode, syndrome: tuple[int, ...], noise: IndependentPauliNoise) -> tuple[float, float]:
    p = noise.x_component_probability
    totals = [0.0, 0.0]
    for error in product((0, 1), repeat=code.length):
        if code.syndrome(error) == syndrome:
            weight = sum(error)
            totals[code.logical_sector(error)] += p**weight * (1-p)**(code.length-weight)
    return totals[0], totals[1]


@dataclass(frozen=True)
class DecodeResult:
    probabilities: tuple[float, float]
    selected_sector: int
    exhaustive_probabilities: tuple[float, float]
    verified: bool
    exact: bool
    rewrite_traces: tuple[tuple[RewriteRecord, ...], tuple[RewriteRecord, ...]]
    plans: tuple[PlanningResult, PlanningResult]
    runtime_seconds: float


def decode_repetition(code: RepetitionCode, syndrome: tuple[int, ...], noise: IndependentPauliNoise, constraints: PlanConstraints | None = None) -> DecodeResult:
    started = perf_counter()
    probabilities: list[float] = []
    traces: list[tuple[RewriteRecord, ...]] = []
    plans: list[PlanningResult] = []
    for sector in (0, 1):
        simplified, trace = RewriteEngine().simplify(build_repetition_decoding_network(code, syndrome, noise, sector))
        planning = plan_contraction(simplified, constraints or PlanConstraints())
        probabilities.append(float(np.real_if_close(execute_numpy(simplified, planning.selected))))
        traces.append(trace)
        plans.append(planning)
    exhaustive = exhaustive_coset_probabilities(code, syndrome, noise)
    return DecodeResult(tuple(probabilities), int(probabilities[1] > probabilities[0]), exhaustive, bool(np.allclose(probabilities, exhaustive)), True, (traces[0], traces[1]), (plans[0], plans[1]), perf_counter() - started)


@dataclass(frozen=True)
class BatchMetrics:
    syndromes: int
    verified: int
    runtime_seconds: float
    peak_intermediate_elements: int
    estimated_flops: int
    kernel_count: int


def benchmark_syndromes(code: RepetitionCode, noise: IndependentPauliNoise, syndromes: Iterable[tuple[int, ...]]) -> BatchMetrics:
    started = perf_counter()
    results = [decode_repetition(code, syndrome, noise) for syndrome in syndromes]
    selected = [planning.selected for result in results for planning in result.plans]
    return BatchMetrics(len(results), sum(r.verified for r in results), perf_counter() - started, max((p.peak_elements for p in selected), default=0), sum(p.total_flops for p in selected), sum(len(p.steps) for p in selected))
