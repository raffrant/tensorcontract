"""Symbolic tensor networks for exact three-qubit correlated-noise decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tensorcontract.backends import NumPyBackend
from tensorcontract.symbolics import (
    ContractionNode,
    IndexRole,
    SymbolicExecutionPlan,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicPlannerOptions,
    SymbolicTensor,
    plan_symbolic_contractions,
)

from .three_qubit import (
    ERROR_PATTERNS,
    CorrelatedXXXNoise,
    ErrorPattern,
    LogicalClass,
    Syndrome,
    decode_syndrome,
    logical_failure,
)


CalculationKind = Literal["error", "syndrome", "logical_syndrome"]
SyndromeConditioning = Literal["selector", "slice"]

_BINARY_INDICES = ("c", "u1", "u2", "u3", "e1", "e2", "e3", "s1", "s2", "L")


def _xor_factor() -> NDArray[np.float64]:
    factor = np.zeros((2, 2, 2), dtype=np.float64)
    for output in (0, 1):
        for left in (0, 1):
            for right in (0, 1):
                factor[output, left, right] = float(output == (left ^ right))
    return factor


def _fixed_parity_factor(observed: int) -> NDArray[np.float64]:
    factor = np.zeros((2, 2), dtype=np.float64)
    for left in (0, 1):
        for right in (0, 1):
            factor[left, right] = float(observed == (left ^ right))
    return factor


def _logical_class_factor() -> NDArray[np.float64]:
    factor = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for error in ERROR_PATTERNS:
        logical_class = int(logical_failure(error))
        factor[(logical_class,) + error.bits] = 1.0
    return factor


@dataclass(frozen=True, slots=True)
class CorrelatedNoiseTensorNetwork:
    """A symbolic graph, its numeric factor bindings, and deterministic plan."""

    graph: SymbolicGraph
    bindings: dict[str, NDArray[np.float64]]
    plan: SymbolicExecutionPlan
    calculation: CalculationKind
    output_name: str
    fixed_syndrome: Syndrome | None
    conditioning: SyndromeConditioning | None

    @property
    def index_names(self) -> tuple[str, ...]:
        """Return named binary indices in deterministic graph order."""
        return tuple(self.graph.indices)

    @property
    def contraction_order(self) -> tuple[tuple[str, ...], ...]:
        """Return the dependencies contracted by each planned step."""
        return tuple(step.dependencies for step in self.plan.steps)

    @property
    def estimated_flops(self) -> int:
        return self.plan.cost.total_flops

    @property
    def intermediate_tensor_sizes(self) -> tuple[tuple[str, tuple[int, ...], int], ...]:
        """Return each planned result's name, shape, and element count."""
        return tuple(
            (step.name, step.output_shape, step.output_elements)
            for step in self.plan.steps
        )

    @property
    def peak_memory_bytes(self) -> int:
        return self.plan.cost.peak_memory_bytes

    def debug_string(self) -> str:
        fixed = "open" if self.fixed_syndrome is None else str(self.fixed_syndrome)
        return (
            f"CorrelatedNoiseTensorNetwork(calculation={self.calculation!r}, "
            f"syndrome={fixed!r}, conditioning={self.conditioning!r})\n"
            f"{self.graph.debug_string()}\n{self.plan.debug_string()}"
        )


@dataclass(frozen=True, slots=True)
class TensorNetworkContraction:
    """Direct and planned NumPy contraction results for one network."""

    network: CorrelatedNoiseTensorNetwork
    values: NDArray[np.float64]
    reference_values: NDArray[np.float64]
    max_abs_difference: float

    @property
    def contraction_order(self) -> tuple[tuple[str, ...], ...]:
        return self.network.contraction_order

    @property
    def estimated_flops(self) -> int:
        return self.network.estimated_flops

    @property
    def intermediate_tensor_sizes(self) -> tuple[tuple[str, tuple[int, ...], int], ...]:
        return self.network.intermediate_tensor_sizes

    @property
    def peak_memory_bytes(self) -> int:
        return self.network.peak_memory_bytes


@dataclass(frozen=True, slots=True)
class LogicalPosteriorTensorResult:
    """Exact tensor-network logical probabilities conditioned on one syndrome."""

    syndrome: Syndrome
    joint_probabilities: tuple[float, float]
    normalized_probabilities: tuple[float, float]
    syndrome_probability: float
    conditional_probability_defined: bool
    contraction: TensorNetworkContraction

    def probability(self, logical_class: LogicalClass) -> float:
        """Return one normalized logical-class probability."""
        index = 0 if logical_class is LogicalClass.I_L else 1
        return self.normalized_probabilities[index]


def _output_indices(
    calculation: CalculationKind,
    fixed_syndrome: Syndrome | None,
) -> tuple[str, ...]:
    if calculation == "error":
        return ("e1", "e2", "e3")
    prefix = ("L",) if calculation == "logical_syndrome" else ()
    return prefix if fixed_syndrome is not None else prefix + ("s1", "s2")


def build_correlated_noise_tensor_network(
    noise_model: CorrelatedXXXNoise,
    *,
    calculation: CalculationKind = "logical_syndrome",
    fixed_syndrome: Syndrome | tuple[int, int] | None = None,
    conditioning: SyndromeConditioning = "selector",
    planner_options: SymbolicPlannerOptions | None = None,
) -> CorrelatedNoiseTensorNetwork:
    """Build and plan one exact correlated-noise symbolic tensor network.

    ``conditioning="selector"`` retains syndrome indices and contracts them
    against one-hot factors. ``conditioning="slice"`` binds the parity factors
    before graph construction, removing the syndrome indices entirely.
    """
    if not isinstance(noise_model, CorrelatedXXXNoise):
        raise TypeError("noise_model must be CorrelatedXXXNoise")
    if calculation not in ("error", "syndrome", "logical_syndrome"):
        raise ValueError(f"unknown tensor-network calculation {calculation!r}")
    if conditioning not in ("selector", "slice"):
        raise ValueError(f"unknown syndrome conditioning mode {conditioning!r}")
    measured = None if fixed_syndrome is None else Syndrome.from_value(fixed_syndrome)
    if calculation == "error" and measured is not None:
        raise ValueError("the error-distribution network does not accept a fixed syndrome")

    output_indices = _output_indices(calculation, measured)
    graph = SymbolicGraph(
        metadata={
            "construction": "three-qubit-correlated-xxx",
            "calculation": calculation,
            "fixed_syndrome": None if measured is None else measured.bits,
            "conditioning": None if measured is None else conditioning,
            "exact": True,
        }
    )
    required_indices = {"c", "u1", "u2", "u3", "e1", "e2", "e3"}
    if calculation != "error" and (measured is None or conditioning == "selector"):
        required_indices.update(("s1", "s2"))
    if calculation == "logical_syndrome":
        required_indices.add("L")
    for name in _BINARY_INDICES:
        if name not in required_indices:
            continue
        role = IndexRole.FREE if name in output_indices else IndexRole.CONTRACTED
        graph.add_index(
            SymbolicIndex(name, 2, role, {"domain": "binary", "semantic_name": name})
        )

    tensors: list[SymbolicTensor] = [
        SymbolicTensor("p_c", ("c",), (2,), {"factor": "bernoulli-correlated"}),
        SymbolicTensor("p_u1", ("u1",), (2,), {"factor": "bernoulli-local"}),
        SymbolicTensor("p_u2", ("u2",), (2,), {"factor": "bernoulli-local"}),
        SymbolicTensor("p_u3", ("u3",), (2,), {"factor": "bernoulli-local"}),
        SymbolicTensor("xor_e1", ("e1", "c", "u1"), (2, 2, 2), {"factor": "xor"}),
        SymbolicTensor("xor_e2", ("e2", "c", "u2"), (2, 2, 2), {"factor": "xor"}),
        SymbolicTensor("xor_e3", ("e3", "c", "u3"), (2, 2, 2), {"factor": "xor"}),
    ]
    if calculation != "error":
        if measured is not None and conditioning == "slice":
            tensors.extend(
                (
                    SymbolicTensor("xor_s1", ("e1", "e2"), (2, 2), {"factor": "xor-slice", "value": measured.s1}),
                    SymbolicTensor("xor_s2", ("e2", "e3"), (2, 2), {"factor": "xor-slice", "value": measured.s2}),
                )
            )
        else:
            tensors.extend(
                (
                    SymbolicTensor("xor_s1", ("s1", "e1", "e2"), (2, 2, 2), {"factor": "xor"}),
                    SymbolicTensor("xor_s2", ("s2", "e2", "e3"), (2, 2, 2), {"factor": "xor"}),
                )
            )
            if measured is not None:
                tensors.extend(
                    (
                        SymbolicTensor("select_s1", ("s1",), (2,), {"factor": "selector", "value": measured.s1}),
                        SymbolicTensor("select_s2", ("s2",), (2,), {"factor": "selector", "value": measured.s2}),
                    )
                )
    if calculation == "logical_syndrome":
        tensors.append(
            SymbolicTensor(
                "logical_class",
                ("L", "e1", "e2", "e3"),
                (2, 2, 2, 2),
                {"factor": "post-recovery-logical-class", "classes": ("I_L", "X_L")},
            )
        )
    for tensor in tensors:
        graph.add_tensor(tensor)
    graph.add_operation(
        ContractionNode(
            "probabilities",
            tuple(tensor.name for tensor in tensors),
            output_indices,
            {"exact": True, "calculation": calculation},
        )
    )
    graph.validate()

    local_probability = np.array([1.0 - noise_model.p, noise_model.p], dtype=np.float64)
    xor = _xor_factor()
    bindings: dict[str, NDArray[np.float64]] = {
        "p_c": np.array([1.0 - noise_model.rho, noise_model.rho], dtype=np.float64),
        "p_u1": local_probability.copy(),
        "p_u2": local_probability.copy(),
        "p_u3": local_probability.copy(),
        "xor_e1": xor.copy(),
        "xor_e2": xor.copy(),
        "xor_e3": xor.copy(),
    }
    if calculation != "error":
        if measured is not None and conditioning == "slice":
            bindings["xor_s1"] = _fixed_parity_factor(measured.s1)
            bindings["xor_s2"] = _fixed_parity_factor(measured.s2)
        else:
            bindings["xor_s1"] = xor.copy()
            bindings["xor_s2"] = xor.copy()
            if measured is not None:
                bindings["select_s1"] = np.eye(2, dtype=np.float64)[measured.s1]
                bindings["select_s2"] = np.eye(2, dtype=np.float64)[measured.s2]
    if calculation == "logical_syndrome":
        bindings["logical_class"] = _logical_class_factor()

    plan = plan_symbolic_contractions(graph, planner_options)
    return CorrelatedNoiseTensorNetwork(
        graph,
        bindings,
        plan,
        calculation,
        "probabilities",
        measured,
        None if measured is None else conditioning,
    )


def contract_correlated_noise_tensor_network(
    network: CorrelatedNoiseTensorNetwork,
    *,
    tolerance: float = 1e-12,
) -> TensorNetworkContraction:
    """Execute both the general reference graph and its planned pairwise graph."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    backend = NumPyBackend(enable_batched_matmul=False)
    reference = np.asarray(
        backend.execute(network.graph, network.bindings, network.output_name),
        dtype=np.float64,
    )
    planned = np.asarray(
        backend.execute(network.plan.graph, network.bindings, network.output_name),
        dtype=np.float64,
    )
    difference = float(np.max(np.abs(reference - planned))) if reference.size else 0.0
    if difference > tolerance:
        raise RuntimeError(
            f"planned contraction differs from direct reference by {difference}, "
            f"exceeding tolerance {tolerance}"
        )
    if float(np.min(planned)) < -tolerance:
        raise RuntimeError(
            f"tensor-network probability is negative: minimum={float(np.min(planned))}"
        )
    return TensorNetworkContraction(network, planned, reference, difference)


def tensor_error_pattern_probabilities(
    noise_model: CorrelatedXXXNoise,
) -> TensorNetworkContraction:
    """Contract the exact final-error distribution ``P(e1,e2,e3)``."""
    network = build_correlated_noise_tensor_network(noise_model, calculation="error")
    return contract_correlated_noise_tensor_network(network)


def tensor_syndrome_probabilities(
    noise_model: CorrelatedXXXNoise,
) -> TensorNetworkContraction:
    """Contract the exact open-syndrome distribution ``P(s1,s2)``."""
    network = build_correlated_noise_tensor_network(noise_model, calculation="syndrome")
    return contract_correlated_noise_tensor_network(network)


def tensor_logical_syndrome_probabilities(
    noise_model: CorrelatedXXXNoise,
) -> TensorNetworkContraction:
    """Contract joint ``P(L,s1,s2)`` with ``L=(I_L,X_L)``."""
    network = build_correlated_noise_tensor_network(
        noise_model, calculation="logical_syndrome"
    )
    return contract_correlated_noise_tensor_network(network)


def tensor_fixed_syndrome_probability(
    noise_model: CorrelatedXXXNoise,
    syndrome: Syndrome | tuple[int, int],
    *,
    conditioning: SyndromeConditioning = "selector",
) -> TensorNetworkContraction:
    """Contract scalar ``P(s)`` using selector or sliced conditioning."""
    network = build_correlated_noise_tensor_network(
        noise_model,
        calculation="syndrome",
        fixed_syndrome=syndrome,
        conditioning=conditioning,
    )
    return contract_correlated_noise_tensor_network(network)


def tensor_logical_posteriors(
    noise_model: CorrelatedXXXNoise,
    syndrome: Syndrome | tuple[int, int],
    *,
    conditioning: SyndromeConditioning = "selector",
) -> LogicalPosteriorTensorResult:
    """Contract ``P(L,s)`` and normalize only after contraction."""
    measured = Syndrome.from_value(syndrome)
    network = build_correlated_noise_tensor_network(
        noise_model,
        calculation="logical_syndrome",
        fixed_syndrome=measured,
        conditioning=conditioning,
    )
    contraction = contract_correlated_noise_tensor_network(network)
    joint = np.asarray(contraction.values, dtype=np.float64)
    syndrome_probability = float(np.sum(joint))
    conditional_defined = syndrome_probability > 0.0
    normalized = (
        joint / syndrome_probability
        if conditional_defined
        else np.zeros(2, dtype=np.float64)
    )
    reference = decode_syndrome(measured, noise_model)
    expected_joint = np.asarray(
        reference.diagnostic_information.logical_joint_probabilities,
        dtype=np.float64,
    )
    if not np.allclose(joint, expected_joint, rtol=1e-12, atol=1e-14):
        raise RuntimeError(
            "tensor-network logical probabilities disagree with exhaustive decoding"
        )
    return LogicalPosteriorTensorResult(
        measured,
        (float(joint[0]), float(joint[1])),
        (float(normalized[0]), float(normalized[1])),
        syndrome_probability,
        conditional_defined,
        contraction,
    )
