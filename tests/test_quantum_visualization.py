"""Headless plotting and export tests for precomputed QEC results."""

from __future__ import annotations

import copy
import csv

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="Matplotlib is optional")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from tensorcontract.quantum.visualization import (
    EXPORT_COLUMNS,
    ErrorWeightSeries,
    LogicalPosteriorPlotResult,
    SyndromeDistributionResult,
    export_results_csv,
    plot_backend_performance,
    plot_error_weight_distribution,
    plot_logical_posteriors,
    plot_monte_carlo_convergence,
    plot_physical_vs_logical_rate,
    plot_syndrome_distribution,
    results_to_dataframe,
    results_to_rows,
)


def _rate_row(
    p: float,
    rho: float,
    shots: int,
    estimate: float,
    exact: float,
    *,
    backend: str = "numpy",
) -> dict[str, object]:
    standard_error = np.sqrt(estimate * (1.0 - estimate) / shots) if shots else 0.0
    return {
        "p": p,
        "rho": rho,
        "p_phys": p + rho - 2 * p * rho,
        "p_logical_exact": exact,
        "p_logical_estimate": estimate,
        "logical_failures": round(estimate * shots),
        "num_shots": shots,
        "standard_error": standard_error,
        "confidence_interval": (
            max(0.0, estimate - 1.96 * standard_error),
            min(1.0, estimate + 1.96 * standard_error),
        ),
        "backend": backend,
        "elapsed_time": 0.01 + shots / 10_000_000,
        "shots_per_second": shots / (0.01 + shots / 10_000_000) if shots else 0.0,
    }


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


def test_physical_vs_logical_plot_returns_axis_and_supports_existing_axis() -> None:
    rows = [
        _rate_row(0.01, 0.0, 10_000, 0.0004, 0.000298),
        _rate_row(0.05, 0.0, 10_000, 0.0075, 0.00725),
        _rate_row(0.01, 0.1, 10_000, 0.10, 0.098),
        _rate_row(0.05, 0.1, 10_000, 0.11, 0.108),
    ]
    axis = plot_physical_vs_logical_rate(rows)
    assert isinstance(axis, Axes)
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert "Physical" in axis.get_xlabel()
    figure, supplied = plt.subplots()
    assert plot_physical_vs_logical_rate(rows, ax=supplied) is supplied
    assert supplied.figure is figure


def test_zero_rates_do_not_crash_log_scale_plots() -> None:
    rows = [
        _rate_row(0.0, 0.0, 0, 0.0, 0.0),
        _rate_row(0.01, 0.0, 1_000, 0.0, 0.000298),
    ]
    physical_axis = plot_physical_vs_logical_rate(rows)
    convergence_axis = plot_monte_carlo_convergence(rows)
    physical_axis.figure.canvas.draw()
    convergence_axis.figure.canvas.draw()
    assert physical_axis.get_xscale() == "log"
    assert convergence_axis.get_xscale() == "log"


def test_error_weight_plot_supports_independent_correlated_and_patterns() -> None:
    series = (
        ErrorWeightSeries("independent", (0.729, 0.243, 0.027, 0.001)),
        ErrorWeightSeries(
            "correlated",
            (0.60, 0.15, 0.10, 0.15),
            (("000", 0.60), ("111", 0.15)),
        ),
    )
    axis = plot_error_weight_distribution(series)
    assert isinstance(axis, Axes)
    assert len(axis.patches) == 8
    assert len(axis.collections) == 2
    assert "weight" in axis.get_xlabel().lower()


def test_syndrome_distribution_plot_has_exact_and_sampled_bars() -> None:
    result = SyndromeDistributionResult(
        (0.7, 0.1, 0.1, 0.1),
        (0.69, 0.11, 0.10, 0.10),
        (0.01, 0.006, 0.006, 0.006),
    )
    axis = plot_syndrome_distribution(result)
    assert isinstance(axis, Axes)
    assert len(axis.patches) == 8
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["00", "01", "10", "11"]


def test_logical_posteriors_show_xxx_contribution() -> None:
    result = LogicalPosteriorPlotResult(
        ((0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.55, 0.45)),
        xxx_syndrome_00_contribution=0.2,
    )
    axis = plot_logical_posteriors(result)
    assert isinstance(axis, Axes)
    assert len(axis.patches) == 9
    labels = axis.get_legend_handles_labels()[1]
    assert any("XXX" in label for label in labels)


def test_convergence_plot_contains_estimate_exact_and_confidence_band() -> None:
    rows = [
        _rate_row(0.1, 0.2, 100, 0.25, 0.27),
        _rate_row(0.1, 0.2, 1_000, 0.275, 0.27),
        _rate_row(0.1, 0.2, 10_000, 0.269, 0.27),
    ]
    axis = plot_monte_carlo_convergence(rows)
    assert isinstance(axis, Axes)
    assert axis.get_xscale() == "log"
    assert len(axis.lines) >= 2
    assert len(axis.collections) >= 1


def test_backend_plot_includes_total_cold_and_warm_runtime() -> None:
    rows = [
        {
            **_rate_row(0.1, 0.2, 10_000, 0.27, 0.27, backend="numpy"),
            "cold_start_runtime": 0.02,
            "warm_runtime": 0.01,
        },
        {
            **_rate_row(0.1, 0.2, 10_000, 0.27, 0.27, backend="gpu"),
            "cold_start_runtime": 0.20,
            "warm_runtime": 0.002,
        },
    ]
    axis = plot_backend_performance(rows)
    assert isinstance(axis, Axes)
    assert len(axis.patches) == 2
    assert len(axis.figure.axes) == 2
    runtime_axis = axis.figure.axes[1]
    assert len(runtime_axis.lines) == 3


def test_csv_export_uses_documented_columns_and_full_values(tmp_path) -> None:
    rows = [_rate_row(0.123456789, 0.2, 12_345, 0.234567891, 0.23)]
    destination = export_results_csv(rows, tmp_path / "results.csv")
    with destination.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        exported = list(reader)
    assert tuple(reader.fieldnames or ()) == EXPORT_COLUMNS
    assert len(exported) == 1
    assert float(exported[0]["p"]) == rows[0]["p"]
    assert float(exported[0]["p_logical_estimate"]) == rows[0]["p_logical_estimate"]
    assert exported[0]["backend"] == "numpy"


def test_row_export_accepts_elapsed_time_and_confidence_tuple() -> None:
    source = _rate_row(0.1, 0.2, 100, 0.3, 0.27)
    row = results_to_rows([source])[0]
    assert tuple(row) == EXPORT_COLUMNS
    assert row["runtime_seconds"] == source["elapsed_time"]
    assert row["ci_low"] == source["confidence_interval"][0]
    assert row["ci_high"] == source["confidence_interval"][1]


def test_dataframe_export_has_documented_columns_when_pandas_is_installed() -> None:
    pytest.importorskip("pandas", reason="pandas is optional")
    frame = results_to_dataframe([_rate_row(0.1, 0.2, 100, 0.3, 0.27)])
    assert tuple(frame.columns) == EXPORT_COLUMNS
    assert len(frame) == 1


def test_plotting_does_not_modify_precomputed_numerical_results(monkeypatch) -> None:
    rows = [_rate_row(0.1, 0.2, 1_000, 0.3, 0.27)]
    original = copy.deepcopy(rows)

    def forbidden(*args, **kwargs):
        raise AssertionError("plotting must not run Monte Carlo")

    import tensorcontract.quantum.monte_carlo as monte_carlo

    monkeypatch.setattr(monte_carlo, "run_monte_carlo", forbidden)
    plot_physical_vs_logical_rate(rows)
    plot_monte_carlo_convergence(rows)
    plot_backend_performance(rows)
    assert rows == original
