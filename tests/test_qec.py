from itertools import product

import numpy as np

from tensorcontract.qec import (
    IndependentPauliNoise, RepetitionCode, benchmark_syndromes,
    build_repetition_decoding_network, decode_repetition,
)


def test_all_length_three_syndromes_match_exhaustive_enumeration() -> None:
    code = RepetitionCode(3)
    noise = IndependentPauliNoise.bit_flip(0.13)
    for syndrome in product((0, 1), repeat=2):
        result = decode_repetition(code, syndrome, noise)
        assert result.exact
        assert result.verified
        assert np.allclose(result.probabilities, result.exhaustive_probabilities)
        assert result.selected_sector == int(result.probabilities[1] > result.probabilities[0])


def test_network_contains_hyperedges_and_symbolic_kinds() -> None:
    network = build_repetition_decoding_network(RepetitionCode(3), (0, 1), IndependentPauliNoise.bit_flip(0.1), 0)
    assert len(network.hyperedges["e1"]) >= 3
    assert network.metadata["exact"] is True
    assert {node.kind.value for node in network.nodes.values()} >= {"parity", "pauli_noise", "identity"}


def test_batch_metrics() -> None:
    metrics = benchmark_syndromes(RepetitionCode(3), IndependentPauliNoise.bit_flip(0.05), product((0, 1), repeat=2))
    assert metrics.syndromes == metrics.verified == 4
    assert metrics.estimated_flops > 0
    assert metrics.kernel_count > 0
    assert metrics.peak_intermediate_elements > 0
