"""Stage 2 tests for exact correlated-noise tensor-network contraction."""

from __future__ import annotations

import numpy as np
import pytest

from tensorcontract.quantum import (
    ERROR_PATTERNS,
    CorrelatedXXXNoise,
    ErrorPattern,
    LogicalClass,
    Syndrome,
    build_correlated_noise_tensor_network,
    decode_syndrome,
    exact_logical_diagnostics,
    exact_logical_error_rate,
    tensor_error_pattern_probabilities,
    tensor_fixed_syndrome_probability,
    tensor_logical_posteriors,
    tensor_logical_syndrome_probabilities,
    tensor_syndrome_probabilities,
)
from tensorcontract.symbolics import ContractionNode, SymbolicGraph


SYNDROMES = (Syndrome(0, 0), Syndrome(0, 1), Syndrome(1, 0), Syndrome(1, 1))


def test_decoder_network_reuses_symbolic_graph_and_contains_required_factors() -> None:
    network = build_correlated_noise_tensor_network(
        CorrelatedXXXNoise(0.17, 0.23), calculation="logical_syndrome"
    )
    assert isinstance(network.graph, SymbolicGraph)
    assert set(network.index_names) == {
        "c", "u1", "u2", "u3", "e1", "e2", "e3", "s1", "s2", "L"
    }
    assert set(network.graph.tensors) == {
        "p_c",
        "p_u1",
        "p_u2",
        "p_u3",
        "xor_e1",
        "xor_e2",
        "xor_e3",
        "xor_s1",
        "xor_s2",
        "logical_class",
    }
    operation = network.graph.operations[network.output_name]
    assert isinstance(operation, ContractionNode)
    assert operation.output_indices == ("L", "s1", "s2")
    assert network.graph.metadata["exact"] is True


def test_xor_factors_encode_binary_constraints_exactly() -> None:
    network = build_correlated_noise_tensor_network(
        CorrelatedXXXNoise(0.2, 0.3), calculation="syndrome"
    )
    for name in ("xor_e1", "xor_e2", "xor_e3", "xor_s1", "xor_s2"):
        factor = network.bindings[name]
        assert factor.shape == (2, 2, 2)
        assert np.count_nonzero(factor) == 4
        for output in (0, 1):
            for left in (0, 1):
                for right in (0, 1):
                    assert factor[output, left, right] == float(output == (left ^ right))


@pytest.mark.parametrize(
    ("p", "rho"),
    [(0.0, 0.0), (0.0, 1.0), (0.13, 0.0), (0.17, 0.23), (0.8, 0.6)],
)
def test_tensor_error_probabilities_match_stage1_brute_force(p: float, rho: float) -> None:
    noise = CorrelatedXXXNoise(p, rho)
    result = tensor_error_pattern_probabilities(noise)
    assert result.values.shape == (2, 2, 2)
    for error in ERROR_PATTERNS:
        assert result.values[error.bits] == pytest.approx(
            noise.error_probability(error), rel=1e-12, abs=1e-14
        )
    assert float(np.sum(result.values)) == pytest.approx(1.0, abs=1e-14)
    assert float(np.min(result.values)) >= -1e-14


@pytest.mark.parametrize("p,rho", [(0.0, 0.0), (0.0, 1.0), (0.17, 0.23)])
def test_open_syndrome_probabilities_match_exhaustive_reference(
    p: float,
    rho: float,
) -> None:
    noise = CorrelatedXXXNoise(p, rho)
    result = tensor_syndrome_probabilities(noise)
    expected = dict(exact_logical_diagnostics(noise).syndrome_probabilities)
    assert result.values.shape == (2, 2)
    for syndrome in SYNDROMES:
        assert result.values[syndrome.bits] == pytest.approx(
            expected[syndrome], rel=1e-12, abs=1e-14
        )
    assert float(np.sum(result.values)) == pytest.approx(1.0, abs=1e-14)


def test_open_logical_syndrome_probabilities_match_every_brute_force_coset() -> None:
    noise = CorrelatedXXXNoise(0.17, 0.23)
    result = tensor_logical_syndrome_probabilities(noise)
    assert result.values.shape == (2, 2, 2)
    for syndrome in SYNDROMES:
        expected = decode_syndrome(
            syndrome, noise
        ).diagnostic_information.logical_joint_probabilities
        np.testing.assert_allclose(
            result.values[:, syndrome.s1, syndrome.s2],
            expected,
            rtol=1e-12,
            atol=1e-14,
        )
    assert float(np.sum(result.values[1])) == pytest.approx(
        exact_logical_error_rate(noise), rel=1e-12, abs=1e-14
    )
    assert float(np.sum(result.values)) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("conditioning", ["selector", "slice"])
def test_fixed_syndrome_logical_posteriors_match_reference_and_normalize(
    conditioning: str,
) -> None:
    noise = CorrelatedXXXNoise(0.17, 0.23)
    for syndrome in SYNDROMES:
        result = tensor_logical_posteriors(
            noise, syndrome, conditioning=conditioning  # type: ignore[arg-type]
        )
        reference = decode_syndrome(syndrome, noise)
        np.testing.assert_allclose(
            result.joint_probabilities,
            reference.diagnostic_information.logical_joint_probabilities,
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result.normalized_probabilities,
            reference.diagnostic_information.logical_posteriors,
            rtol=1e-12,
            atol=1e-14,
        )
        assert sum(result.normalized_probabilities) == pytest.approx(1.0, abs=1e-14)
        assert min(result.normalized_probabilities) >= -1e-14


def test_selector_and_slice_conditioning_are_equivalent_and_structurally_distinct() -> None:
    noise = CorrelatedXXXNoise(0.21, 0.31)
    syndrome = Syndrome(1, 1)
    selector = tensor_fixed_syndrome_probability(
        noise, syndrome, conditioning="selector"
    )
    sliced = tensor_fixed_syndrome_probability(noise, syndrome, conditioning="slice")
    assert float(selector.values) == pytest.approx(float(sliced.values), abs=1e-14)
    assert {"s1", "s2"} <= set(selector.network.index_names)
    assert {"select_s1", "select_s2"} <= set(selector.network.graph.tensors)
    assert "s1" not in sliced.network.index_names
    assert "s2" not in sliced.network.index_names
    assert "select_s1" not in sliced.network.graph.tensors
    assert sliced.network.graph.tensors["xor_s1"].indices == ("e1", "e2")


def test_correlated_xxx_certain_case_is_exact_in_tensor_network() -> None:
    noise = CorrelatedXXXNoise(p=0.0, rho=1.0)
    errors = tensor_error_pattern_probabilities(noise).values
    assert errors[ErrorPattern(1, 1, 1).bits] == pytest.approx(1.0)
    assert np.count_nonzero(errors) == 1
    posterior = tensor_logical_posteriors(noise, Syndrome(0, 0))
    assert posterior.joint_probabilities == pytest.approx((0.0, 1.0))
    assert posterior.normalized_probabilities == pytest.approx((0.0, 1.0))
    assert posterior.probability(LogicalClass.X_L) == pytest.approx(1.0)


def test_impossible_syndrome_has_explicit_undefined_posterior() -> None:
    result = tensor_logical_posteriors(
        CorrelatedXXXNoise(0.0, 0.0), Syndrome(0, 1)
    )
    assert result.syndrome_probability == 0.0
    assert not result.conditional_probability_defined
    assert result.normalized_probabilities == (0.0, 0.0)


def test_direct_and_planned_contractions_agree_for_all_calculations() -> None:
    noise = CorrelatedXXXNoise(0.29, 0.11)
    results = (
        tensor_error_pattern_probabilities(noise),
        tensor_syndrome_probabilities(noise),
        tensor_logical_syndrome_probabilities(noise),
        tensor_fixed_syndrome_probability(noise, (1, 0), conditioning="selector"),
        tensor_fixed_syndrome_probability(noise, (1, 0), conditioning="slice"),
    )
    for result in results:
        np.testing.assert_allclose(
            result.values, result.reference_values, rtol=1e-12, atol=1e-14
        )
        assert result.max_abs_difference <= 1e-14


def test_plan_is_deterministic_valid_and_exposes_costs() -> None:
    noise = CorrelatedXXXNoise(0.17, 0.23)
    first = build_correlated_noise_tensor_network(noise)
    second = build_correlated_noise_tensor_network(noise)
    first.plan.validate()
    assert first.contraction_order == second.contraction_order
    assert first.plan.debug_string() == second.plan.debug_string()
    assert len(first.contraction_order) == len(first.graph.tensors) - 1
    assert first.estimated_flops == sum(step.estimated_flops for step in first.plan.steps)
    assert first.peak_memory_bytes == first.plan.cost.peak_live_elements * 8
    assert first.intermediate_tensor_sizes[-1][0] == "probabilities"
    assert all(elements > 0 for _, _, elements in first.intermediate_tensor_sizes)


def test_noise_values_change_bindings_but_not_topology_or_plan() -> None:
    first = build_correlated_noise_tensor_network(CorrelatedXXXNoise(0.1, 0.2))
    second = build_correlated_noise_tensor_network(CorrelatedXXXNoise(0.4, 0.3))
    assert first.graph.debug_string() == second.graph.debug_string()
    assert first.plan.debug_string() == second.plan.debug_string()
    assert not np.array_equal(first.bindings["p_c"], second.bindings["p_c"])
    assert not np.array_equal(first.bindings["p_u1"], second.bindings["p_u1"])


def test_builder_rejects_invalid_calculation_and_conditioning() -> None:
    noise = CorrelatedXXXNoise(0.1, 0.2)
    with pytest.raises(ValueError, match="unknown tensor-network calculation"):
        build_correlated_noise_tensor_network(noise, calculation="threshold")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="conditioning mode"):
        build_correlated_noise_tensor_network(noise, conditioning="project")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not accept a fixed syndrome"):
        build_correlated_noise_tensor_network(
            noise, calculation="error", fixed_syndrome=(0, 0)
        )
