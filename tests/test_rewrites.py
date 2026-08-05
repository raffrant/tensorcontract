from itertools import product

import numpy as np

from tensorcontract.backend import contract_numpy
from tensorcontract.qec import IndependentPauliNoise, RepetitionCode, build_repetition_decoding_network
from tensorcontract.rewrite import RewriteEngine


def test_rewrites_preserve_all_small_decoder_values() -> None:
    code = RepetitionCode(4)
    noise = IndependentPauliNoise.bit_flip(0.21)
    for syndrome in product((0, 1), repeat=3):
        for sector in (0, 1):
            original = build_repetition_decoding_network(code, syndrome, noise, sector)
            before = contract_numpy(original)
            simplified, trace = RewriteEngine().simplify(original)
            after = contract_numpy(simplified)
            assert np.allclose(before, after)
            assert {record.rule for record in trace} >= {
                "scalar-folding", "identity-elimination", "degree-one-absorption"
            }


def test_rewrite_is_deterministic() -> None:
    network = build_repetition_decoding_network(RepetitionCode(3), (1, 1), IndependentPauliNoise.bit_flip(0.2), 1)
    _, left = RewriteEngine().simplify(network)
    _, right = RewriteEngine().simplify(network)
    assert left == right
