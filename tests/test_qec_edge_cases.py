"""Boundary and invalid-input tests for the existing repetition-code API."""

from itertools import product

import numpy as np
import pytest

from tensorcontract.qec import (
    IndependentPauliNoise,
    RepetitionCode,
    benchmark_syndromes,
    build_repetition_decoding_network,
    decode_repetition,
    exhaustive_coset_probabilities,
)


@pytest.mark.parametrize(
    "probabilities",
    [
        (-0.1, 1.1, 0.0, 0.0),
        (0.5, 0.4, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 0.0),
    ],
)
def test_invalid_pauli_channels_are_rejected(probabilities: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="sum to one"):
        IndependentPauliNoise(*probabilities)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_invalid_bit_flip_probabilities_are_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="sum to one"):
        IndependentPauliNoise.bit_flip(probability)


def test_repetition_code_rejects_invalid_length() -> None:
    with pytest.raises(ValueError, match="at least two"):
        RepetitionCode(1)


@pytest.mark.parametrize(
    ("syndrome", "sector"),
    [((0,), 0), ((0, 0, 0), 0), ((0, 0), -1), ((0, 0), 2)],
)
def test_decoding_network_rejects_invalid_conditioning(syndrome: tuple[int, ...], sector: int) -> None:
    with pytest.raises(ValueError, match="invalid syndrome or logical sector"):
        build_repetition_decoding_network(
            RepetitionCode(3), syndrome, IndependentPauliNoise.bit_flip(0.1), sector,
        )


def test_exhaustive_sector_mass_matches_direct_error_probability() -> None:
    code = RepetitionCode(3)
    noise = IndependentPauliNoise.bit_flip(0.2)
    total = 0.0
    for syndrome in product((0, 1), repeat=2):
        sectors = exhaustive_coset_probabilities(code, syndrome, noise)
        total += sum(sectors)
    assert total == pytest.approx(1.0)


def test_zero_noise_decoder_selects_trivial_sector_for_zero_syndrome() -> None:
    result = decode_repetition(
        RepetitionCode(3), (0, 0), IndependentPauliNoise.bit_flip(0.0),
    )
    assert result.probabilities == pytest.approx((1.0, 0.0))
    assert result.selected_sector == 0
    assert result.verified


def test_empty_syndrome_batch_has_zero_metrics() -> None:
    metrics = benchmark_syndromes(
        RepetitionCode(3), IndependentPauliNoise.bit_flip(0.1), (),
    )
    assert metrics.syndromes == 0
    assert metrics.verified == 0
    assert metrics.peak_intermediate_elements == 0
    assert metrics.estimated_flops == 0
    assert metrics.kernel_count == 0
    assert np.isfinite(metrics.runtime_seconds)
