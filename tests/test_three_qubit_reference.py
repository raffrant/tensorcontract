"""Stage 1 tests for exact three-qubit correlated-noise decoding."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import comb

import numpy as np
import pytest

from tensorcontract.quantum import (
    ERROR_PATTERNS,
    CorrelatedXXXNoise,
    ErrorPattern,
    LogicalClass,
    LogicalErrorType,
    Syndrome,
    decode_syndrome,
    exact_logical_diagnostics,
    exact_logical_error_rate,
    logical_failure,
    majority_vote_recovery,
    residual_after_recovery,
    syndrome_for_error,
)


EXPECTED_SYNDROMES = {
    (0, 0, 0): (0, 0),
    (1, 0, 0): (1, 0),
    (0, 1, 0): (1, 1),
    (0, 0, 1): (0, 1),
    (1, 1, 0): (0, 1),
    (1, 0, 1): (1, 1),
    (0, 1, 1): (1, 0),
    (1, 1, 1): (0, 0),
}


def test_syndrome_mapping_matches_specification() -> None:
    actual = {
        error: syndrome_for_error(error).bits for error in EXPECTED_SYNDROMES
    }
    assert actual == EXPECTED_SYNDROMES


def test_syndrome_and_error_patterns_are_immutable_and_binary() -> None:
    syndrome = Syndrome(1, 0)
    with pytest.raises(FrozenInstanceError):
        syndrome.s1 = 0  # type: ignore[misc]
    with pytest.raises(ValueError, match="binary"):
        Syndrome(2, 0)
    with pytest.raises(ValueError, match="length-two"):
        Syndrome.from_value((0, 0, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="binary"):
        ErrorPattern(0, -1, 0)
    with pytest.raises(ValueError, match="length-three"):
        ErrorPattern.from_value((0, 1))  # type: ignore[arg-type]


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_noise_rejects_invalid_local_probability(p: float) -> None:
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        CorrelatedXXXNoise(p, 0.2)


@pytest.mark.parametrize("rho", [-0.1, 1.1, float("nan")])
def test_noise_rejects_invalid_correlated_probability(rho: float) -> None:
    with pytest.raises(ValueError, match="0 <= rho <= 1"):
        CorrelatedXXXNoise(0.2, rho)


@pytest.mark.parametrize(
    ("noise", "certain_error"),
    [
        (CorrelatedXXXNoise(0.0, 0.0), ErrorPattern(0, 0, 0)),
        (CorrelatedXXXNoise(0.0, 1.0), ErrorPattern(1, 1, 1)),
    ],
)
def test_deterministic_noise_special_cases(
    noise: CorrelatedXXXNoise,
    certain_error: ErrorPattern,
) -> None:
    table = dict(noise.probability_table())
    assert table[certain_error] == pytest.approx(1.0)
    assert sum(value > 0.0 for value in table.values()) == 1


@pytest.mark.parametrize(
    ("p", "rho"),
    [(0.0, 0.0), (0.0, 1.0), (0.2, 0.0), (0.17, 0.23), (1.0, 0.4)],
)
def test_error_pattern_probabilities_sum_to_one(p: float, rho: float) -> None:
    noise = CorrelatedXXXNoise(p, rho)
    assert len(noise.probability_table()) == 8
    assert sum(probability for _, probability in noise.probability_table()) == pytest.approx(1.0)


def test_independent_limit_matches_binomial_local_noise() -> None:
    p = 0.19
    noise = CorrelatedXXXNoise(p, 0.0)
    for error, probability in noise.probability_table():
        expected = p**error.weight * (1.0 - p) ** (3 - error.weight)
        assert probability == pytest.approx(expected)
    assert noise.error_weight_probabilities() == pytest.approx(
        tuple(comb(3, weight) * p**weight * (1.0 - p) ** (3 - weight) for weight in range(4))
    )


def test_intermediate_noise_is_sum_of_both_latent_sectors() -> None:
    p = 0.17
    rho = 0.23
    noise = CorrelatedXXXNoise(p, rho)
    for error, probability in noise.probability_table():
        weight = error.weight
        expected = (
            (1.0 - rho) * p**weight * (1.0 - p) ** (3 - weight)
            + rho * p ** (3 - weight) * (1.0 - p) ** weight
        )
        assert probability == pytest.approx(expected)


@pytest.mark.parametrize("p,rho", [(0.0, 0.0), (0.1, 0.3), (0.7, 0.2), (1.0, 1.0)])
def test_physical_rate_and_weight_diagnostics(p: float, rho: float) -> None:
    noise = CorrelatedXXXNoise(p, rho)
    diagnostics = noise.diagnostics()
    expected_rate = p + rho - 2.0 * p * rho
    assert noise.physical_error_rate() == pytest.approx(expected_rate)
    assert diagnostics.physical_error_rate == pytest.approx(expected_rate)
    assert diagnostics.average_physical_error_weight == pytest.approx(3.0 * expected_rate)
    assert sum(diagnostics.weight_probabilities) == pytest.approx(1.0)
    assert diagnostics.zero_error_probability == pytest.approx(
        noise.error_probability((0, 0, 0))
    )
    assert diagnostics.exactly_one_error_probability == pytest.approx(
        diagnostics.weight_probabilities[1]
    )
    assert diagnostics.two_or_more_error_probability == pytest.approx(
        diagnostics.weight_probabilities[2] + diagnostics.weight_probabilities[3]
    )
    assert diagnostics.xxx_probability == pytest.approx(noise.error_probability((1, 1, 1)))


def test_majority_vote_recovery_table() -> None:
    assert majority_vote_recovery((0, 0)) == ErrorPattern(0, 0, 0)
    assert majority_vote_recovery((1, 0)) == ErrorPattern(1, 0, 0)
    assert majority_vote_recovery((1, 1)) == ErrorPattern(0, 1, 0)
    assert majority_vote_recovery((0, 1)) == ErrorPattern(0, 0, 1)


@pytest.mark.parametrize("error", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_zero_and_single_errors_are_corrected(error: tuple[int, int, int]) -> None:
    assert residual_after_recovery(error) == ErrorPattern(0, 0, 0)
    assert not logical_failure(error)


@pytest.mark.parametrize("error", [(1, 1, 0), (1, 0, 1), (0, 1, 1)])
def test_double_errors_cause_logical_failure_after_recovery(
    error: tuple[int, int, int],
) -> None:
    assert residual_after_recovery(error) == ErrorPattern(1, 1, 1)
    assert logical_failure(error)


def test_xxx_is_a_logical_x_failure_after_identity_recovery() -> None:
    error = ErrorPattern(1, 1, 1)
    assert syndrome_for_error(error) == Syndrome(0, 0)
    assert majority_vote_recovery((0, 0)) == ErrorPattern(0, 0, 0)
    assert residual_after_recovery(error) == error
    assert logical_failure(error)


def test_all_eight_patterns_appear_once_in_exhaustive_decoder_candidates() -> None:
    noise = CorrelatedXXXNoise(0.16, 0.27)
    decoded_candidates = tuple(
        candidate.error
        for syndrome in (Syndrome(0, 0), Syndrome(0, 1), Syndrome(1, 0), Syndrome(1, 1))
        for candidate in decode_syndrome(syndrome, noise).diagnostic_information.candidates
    )
    assert len(decoded_candidates) == 8
    assert set(decoded_candidates) == set(ERROR_PATTERNS)


def test_decode_result_contains_joint_and_normalized_logical_probabilities() -> None:
    noise = CorrelatedXXXNoise(0.12, 0.18)
    result = decode_syndrome((1, 0), noise)
    diagnostics = result.diagnostic_information
    assert result.syndrome == Syndrome(1, 0)
    assert result.selected_recovery == ErrorPattern(1, 0, 0)
    assert len(result.candidate_error_probabilities) == 2
    assert diagnostics.conditional_probability_defined
    assert sum(diagnostics.logical_joint_probabilities) == pytest.approx(
        diagnostics.syndrome_probability
    )
    assert sum(diagnostics.logical_posteriors) == pytest.approx(1.0)
    assert result.posterior_probability == pytest.approx(
        max(diagnostics.logical_posteriors)
    )


def test_correlated_xxx_certain_event_selects_logical_x_for_zero_syndrome() -> None:
    noise = CorrelatedXXXNoise(p=0.0, rho=1.0)
    result = decode_syndrome((0, 0), noise)
    assert result.selected_logical_class is LogicalClass.X_L
    assert result.logical_error_flag
    assert result.logical_error_type is LogicalErrorType.LOGICAL_X
    assert result.posterior_probability == pytest.approx(1.0)
    assert exact_logical_error_rate(noise) == pytest.approx(1.0)


def test_impossible_syndrome_is_returned_without_invalid_normalization() -> None:
    result = decode_syndrome((0, 1), CorrelatedXXXNoise(0.0, 0.0))
    assert not result.diagnostic_information.conditional_probability_defined
    assert result.diagnostic_information.logical_posteriors == (0.0, 0.0)
    assert result.posterior_probability == 0.0
    assert result.selected_logical_class is LogicalClass.I_L


def test_exact_logical_rate_matches_independent_repetition_formula() -> None:
    p = 0.13
    expected = 3.0 * p**2 * (1.0 - p) + p**3
    assert exact_logical_error_rate(CorrelatedXXXNoise(p, 0.0)) == pytest.approx(expected)


def test_exact_diagnostics_are_complete_and_normalized() -> None:
    diagnostics = exact_logical_diagnostics(CorrelatedXXXNoise(0.21, 0.14))
    assert len(diagnostics.syndrome_probabilities) == 4
    assert len(diagnostics.logical_coset_probabilities) == 4
    assert sum(probability for _, probability in diagnostics.syndrome_probabilities) == pytest.approx(1.0)
    assert sum(
        sum(classes) for _, classes in diagnostics.logical_coset_probabilities
    ) == pytest.approx(1.0)
    assert diagnostics.logical_error_rate == pytest.approx(
        sum(classes[1] for _, classes in diagnostics.logical_coset_probabilities)
    )


def test_fixed_seed_batch_sampling_is_reproducible_and_vectorized() -> None:
    noise = CorrelatedXXXNoise(0.2, 0.3)
    first = noise.sample_batch(512, seed=91)
    second = noise.sample_batch(512, seed=91)
    assert first.shape == (512, 3)
    assert first.dtype == np.int8
    assert np.array_equal(first, second)
    assert set(np.unique(first)).issubset({0, 1})
    assert noise.sample_batch(0, seed=91).shape == (0, 3)


def test_single_sample_is_reproducible_with_caller_owned_rng() -> None:
    noise = CorrelatedXXXNoise(0.2, 0.3)
    first = noise.sample_error(np.random.default_rng(44))
    second = noise.sample_error(np.random.default_rng(44))
    assert first == second
    with pytest.raises(TypeError, match="Generator"):
        noise.sample_error(object())  # type: ignore[arg-type]


def test_batch_sampling_rejects_invalid_shot_counts() -> None:
    noise = CorrelatedXXXNoise(0.2, 0.3)
    with pytest.raises(ValueError, match="nonnegative"):
        noise.sample_batch(-1)
    with pytest.raises(TypeError, match="integer"):
        noise.sample_batch(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        noise.sample_batch(True)
