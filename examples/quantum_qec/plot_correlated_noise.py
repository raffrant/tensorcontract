"""Generate correlated-noise QEC figures and full-precision CSV data.

Run from the repository root::

    PYTHONPATH=src python3 examples/quantum_qec/plot_correlated_noise.py
"""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tensorcontract.quantum import (
    CorrelatedXXXNoise,
    ErrorWeightSeries,
    LogicalPosteriorPlotResult,
    Syndrome,
    SyndromeDistributionResult,
    decode_syndrome,
    exact_logical_diagnostics,
    export_results_csv,
    plot_backend_performance,
    plot_error_weight_distribution,
    plot_logical_posteriors,
    plot_monte_carlo_convergence,
    plot_physical_vs_logical_rate,
    plot_syndrome_distribution,
    run_monte_carlo,
)


SYNDROMES = (Syndrome(0, 0), Syndrome(0, 1), Syndrome(1, 0), Syndrome(1, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("qec_correlated_noise_outputs"))
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=202604)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    rate_results = []
    for rho in (0.0, 0.05, 0.20):
        for p in (0.0, 0.01, 0.03, 0.06, 0.10, 0.17):
            rate_results.append(
                run_monte_carlo(
                    arguments.shots,
                    p,
                    rho,
                    seed=arguments.seed + len(rate_results),
                    batch_size=8_192,
                )
            )

    independent = CorrelatedXXXNoise(0.10, 0.0)
    correlated = CorrelatedXXXNoise(0.10, 0.20)
    weight_series = tuple(
        ErrorWeightSeries(
            label,
            noise.error_weight_probabilities(),
            tuple((str(error), probability) for error, probability in noise.probability_table()),
        )
        for label, noise in (("independent rho=0", independent), ("correlated rho=0.2", correlated))
    )

    representative = run_monte_carlo(
        max(arguments.shots, 10_000),
        correlated.p,
        correlated.rho,
        seed=arguments.seed,
        batch_size=8_192,
        return_samples=True,
    )
    assert representative.samples is not None
    syndrome_codes = (
        representative.samples.syndromes[:, 0] * 2
        + representative.samples.syndromes[:, 1]
    )
    counts = np.bincount(syndrome_codes, minlength=4)
    frequencies = counts / representative.num_shots
    syndrome_errors = tuple(
        sqrt(value * (1.0 - value) / representative.num_shots)
        for value in frequencies
    )
    exact = exact_logical_diagnostics(correlated)
    exact_syndromes = tuple(probability for _, probability in exact.syndrome_probabilities)
    syndrome_result = SyndromeDistributionResult(
        exact_syndromes, tuple(frequencies), syndrome_errors
    )

    decoded = tuple(decode_syndrome(syndrome, correlated) for syndrome in SYNDROMES)
    posteriors = tuple(result.diagnostic_information.logical_posteriors for result in decoded)
    probability_00 = decoded[0].diagnostic_information.syndrome_probability
    xxx_contribution = (
        correlated.error_probability((1, 1, 1)) / probability_00
        if probability_00 > 0.0
        else 0.0
    )
    logical_result = LogicalPosteriorPlotResult(
        posteriors,  # type: ignore[arg-type]
        xxx_contribution,
    )

    convergence = tuple(
        run_monte_carlo(
            shots,
            correlated.p,
            correlated.rho,
            seed=arguments.seed,
            batch_size=8_192,
        )
        for shots in (1_000, 10_000, 100_000)
    )

    plots = {
        "physical_vs_logical.png": plot_physical_vs_logical_rate(rate_results),
        "error_weights.png": plot_error_weight_distribution(weight_series),
        "syndromes.png": plot_syndrome_distribution(syndrome_result),
        "logical_posteriors.png": plot_logical_posteriors(logical_result),
        "monte_carlo_convergence.png": plot_monte_carlo_convergence(convergence),
        "backend_performance.png": plot_backend_performance(convergence),
    }
    for filename, axis in plots.items():
        axis.figure.tight_layout()
        axis.figure.savefig(arguments.output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(axis.figure)
    csv_path = export_results_csv(rate_results, arguments.output_dir / "rate_sweep.csv")
    print(f"Saved {len(plots)} figures and {csv_path} to {arguments.output_dir}")


if __name__ == "__main__":
    main()
