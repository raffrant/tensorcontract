"""Run: PYTHONPATH=src python3 examples/repetition_decode.py"""

from itertools import product

from tensorcontract.qec import IndependentPauliNoise, RepetitionCode, benchmark_syndromes, decode_repetition
from tensorcontract.rewrite import format_trace


code = RepetitionCode(3)
noise = IndependentPauliNoise.bit_flip(0.1)
result = decode_repetition(code, (1, 0), noise)

print("Tensor probabilities:", result.probabilities)
print("Exhaustive probabilities:", result.exhaustive_probabilities)
print("Selected logical sector:", result.selected_sector)
print("Exact verification:", result.verified)
print("\nRewrite trace (sector 0):")
print(format_trace(result.rewrite_traces[0]))
print("\nCandidate plans (sector 0):")
for candidate in result.plans[0].candidates:
    print(candidate.report())
print("\nSelected plan:", result.plans[0].selected.name)

syndromes = list(product((0, 1), repeat=code.length - 1))
print("\nBatch metrics:", benchmark_syndromes(code, noise, syndromes))
