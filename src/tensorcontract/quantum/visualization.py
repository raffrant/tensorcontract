"""Optional plots and result export for precomputed three-qubit QEC data.

This module performs no simulation. Matplotlib and pandas are imported lazily,
so importing :mod:`tensorcontract` or :mod:`tensorcontract.quantum` requires
neither optional dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPORT_COLUMNS = (
    "p",
    "rho",
    "p_phys",
    "p_logical_exact",
    "p_logical_estimate",
    "logical_failures",
    "num_shots",
    "standard_error",
    "ci_low",
    "ci_high",
    "backend",
    "runtime_seconds",
    "shots_per_second",
)

PERFORMANCE_COLUMNS = EXPORT_COLUMNS + (
    "requested_backend",
    "device",
    "fallback_used",
    "cold_start_runtime",
    "warm_runtime",
    "random_generation_runtime",
    "host_to_device_runtime",
    "kernel_runtime",
    "reduction_runtime",
    "device_to_host_runtime",
    "cache_hit",
    "cache_enabled",
    "compilation_time",
    "planning_time",
    "execution_time",
    "total_time",
    "plan_key",
    "fusion_used",
    "kernel_count",
    "registers_per_thread",
    "register_spills",
    "shared_memory_bytes",
    "occupancy",
    "random_generation_fused",
    "reduction_fused",
)

SYNDROME_LABELS = ("00", "01", "10", "11")


@dataclass(frozen=True, slots=True)
class ErrorWeightSeries:
    """One precomputed weight distribution for plotting."""

    label: str
    weight_probabilities: tuple[float, float, float, float]
    pattern_probabilities: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class SyndromeDistributionResult:
    """Exact and optional sampled syndrome probabilities in 00,01,10,11 order."""

    exact_probabilities: tuple[float, float, float, float]
    monte_carlo_frequencies: tuple[float, float, float, float] | None = None
    monte_carlo_standard_errors: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class LogicalPosteriorPlotResult:
    """Logical posterior pairs grouped in 00,01,10,11 syndrome order."""

    posteriors: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    xxx_syndrome_00_contribution: float | None = None


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as error:
        raise ImportError(
            "plotting requires optional matplotlib; install "
            "tensorcontract[visualization]"
        ) from error
    return plt


def _axis(ax: Any, *, figsize: tuple[float, float] = (7.2, 4.4)) -> Any:
    if ax is not None:
        return ax
    _, created = _pyplot().subplots(figsize=figsize)
    return created


def _get(result: object, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _log_floor(values: Iterable[float]) -> float:
    positive = [float(value) for value in values if np.isfinite(value) and value > 0.0]
    return min(positive) * 0.1 if positive else 1e-12


def _safe_log_values(values: Sequence[float], floor: float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(array) & (array > 0.0), array, floor)


def results_to_rows(results: Iterable[object]) -> tuple[dict[str, object], ...]:
    """Normalize result objects or mappings to the stable export schema."""
    rows: list[dict[str, object]] = []
    for result in results:
        interval = _get(result, "confidence_interval", (None, None))
        if interval is None:
            interval = (None, None)
        runtime = _get(result, "runtime_seconds", None)
        if runtime is None:
            runtime = _get(result, "elapsed_time", None)
        row = {
            "p": _get(result, "p", None),
            "rho": _get(result, "rho", None),
            "p_phys": _get(result, "p_phys", None),
            "p_logical_exact": _get(result, "p_logical_exact", None),
            "p_logical_estimate": _get(result, "p_logical_estimate", None),
            "logical_failures": _get(result, "logical_failures", None),
            "num_shots": _get(result, "num_shots", None),
            "standard_error": _get(result, "standard_error", None),
            "ci_low": _get(result, "ci_low", interval[0]),
            "ci_high": _get(result, "ci_high", interval[1]),
            "backend": _get(result, "backend", None),
            "runtime_seconds": runtime,
            "shots_per_second": _get(result, "shots_per_second", None),
        }
        rows.append({column: row[column] for column in EXPORT_COLUMNS})
    return tuple(rows)


def export_results_csv(
    results: Iterable[object],
    path: str | Path,
) -> Path:
    """Write full-precision precomputed results using the documented columns."""
    destination = Path(path)
    rows = results_to_rows(results)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def performance_results_to_rows(
    results: Iterable[object],
) -> tuple[dict[str, object], ...]:
    """Normalize backend results including optional GPU phase timings."""
    raw = tuple(results)
    base_rows = results_to_rows(raw)
    rows: list[dict[str, object]] = []
    for result, base in zip(raw, base_rows):
        extended = dict(base)
        for column in PERFORMANCE_COLUMNS[len(EXPORT_COLUMNS):]:
            extended[column] = _get(result, column, None)
        rows.append({column: extended[column] for column in PERFORMANCE_COLUMNS})
    return tuple(rows)


def export_performance_results_csv(
    results: Iterable[object],
    path: str | Path,
) -> Path:
    """Export aggregate statistics plus optional accelerator timing phases."""
    destination = Path(path)
    rows = performance_results_to_rows(results)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PERFORMANCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def results_to_dataframe(results: Iterable[object]) -> Any:
    """Return a pandas DataFrame when pandas is installed."""
    try:
        import pandas as pd
    except (ImportError, OSError) as error:
        raise ImportError("DataFrame export requires optional pandas") from error
    return pd.DataFrame(results_to_rows(results), columns=EXPORT_COLUMNS)


def plot_physical_vs_logical_rate(results: Iterable[object], ax: Any = None) -> Any:
    """Plot exact curves and Monte Carlo confidence intervals by ``rho``."""
    axis = _axis(ax)
    rows = results_to_rows(results)
    if not rows:
        raise ValueError("physical-versus-logical plot requires at least one result")
    all_values = [
        float(value)
        for row in rows
        for value in (
            row["p_phys"],
            row["p_logical_exact"],
            row["p_logical_estimate"],
        )
        if value is not None
    ]
    floor = _log_floor(all_values)
    rho_values = sorted({float(row["rho"]) for row in rows if row["rho"] is not None})
    for rho in rho_values:
        group = sorted(
            (row for row in rows if float(row["rho"]) == rho),
            key=lambda row: float(row["p_phys"]),
        )
        physical = [float(row["p_phys"]) for row in group]
        exact = [float(row["p_logical_exact"]) for row in group]
        estimate = [float(row["p_logical_estimate"]) for row in group]
        x = _safe_log_values(physical, floor)
        exact_y = _safe_log_values(exact, floor)
        estimate_y = _safe_log_values(estimate, floor)
        line = axis.plot(x, exact_y, marker="o", label=f"rho={rho:g} exact")[0]
        lower = np.asarray(
            [max(0.0, estimate[index] - float(group[index]["ci_low"])) for index in range(len(group))]
        )
        upper = np.asarray(
            [max(0.0, float(group[index]["ci_high"]) - estimate[index]) for index in range(len(group))]
        )
        axis.errorbar(
            x,
            estimate_y,
            yerr=np.vstack((lower, upper)),
            fmt="s",
            linestyle="none",
            color=line.get_color(),
            capsize=3,
            label=f"rho={rho:g} Monte Carlo (95% CI)",
        )
    bounds = _safe_log_values(all_values, floor)
    low, high = float(np.min(bounds)), float(np.max(bounds))
    if low == high:
        low, high = low * 0.5, high * 2.0
    axis.plot((low, high), (low, high), "--", color="0.45", label="y = x")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Physical error rate per qubit, p_phys")
    axis.set_ylabel("Logical error rate per correction cycle")
    axis.set_title("Three-qubit physical versus logical error rate")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    return axis


def plot_error_weight_distribution(
    result: ErrorWeightSeries | Sequence[ErrorWeightSeries],
    ax: Any = None,
) -> Any:
    """Compare precomputed probabilities for physical error weights zero to three."""
    axis = _axis(ax)
    series = (result,) if isinstance(result, ErrorWeightSeries) else tuple(result)
    if not series:
        raise ValueError("error-weight plot requires at least one series")
    x = np.arange(4, dtype=np.float64)
    width = 0.8 / len(series)
    for number, item in enumerate(series):
        offset = (number - (len(series) - 1) / 2.0) * width
        axis.bar(x + offset, item.weight_probabilities, width, label=item.label)
        for pattern, probability in item.pattern_probabilities:
            weight = pattern.count("1")
            axis.scatter(
                weight + offset,
                probability,
                marker="x",
                s=28,
                label=f"{item.label} {pattern}",
            )
    axis.set_xticks(x, ("0", "1", "2", "3"))
    axis.set_xlabel("Physical X-error weight (qubits)")
    axis.set_ylabel("Probability")
    axis.set_title("Physical error-weight distribution")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return axis


def plot_syndrome_distribution(
    result: SyndromeDistributionResult,
    ax: Any = None,
) -> Any:
    """Compare exact and sampled syndrome probabilities."""
    axis = _axis(ax)
    x = np.arange(4)
    if result.monte_carlo_frequencies is None:
        axis.bar(x, result.exact_probabilities, 0.65, label="Exact")
    else:
        width = 0.36
        axis.bar(x - width / 2, result.exact_probabilities, width, label="Exact")
        errors = result.monte_carlo_standard_errors
        axis.bar(
            x + width / 2,
            result.monte_carlo_frequencies,
            width,
            yerr=errors,
            capsize=3,
            label="Monte Carlo",
        )
    axis.set_xticks(x, SYNDROME_LABELS)
    axis.set_xlabel("Measured syndrome (s1 s2)")
    axis.set_ylabel("Probability")
    axis.set_title("Syndrome distribution")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return axis


def plot_logical_posteriors(
    result: LogicalPosteriorPlotResult,
    ax: Any = None,
) -> Any:
    """Plot ``P(I_L|s)`` and ``P(X_L|s)`` with the XXX/00 contribution."""
    axis = _axis(ax)
    values = np.asarray(result.posteriors, dtype=np.float64)
    if values.shape != (4, 2):
        raise ValueError("logical posteriors must have shape (4 syndromes, 2 classes)")
    x = np.arange(4)
    width = 0.36
    axis.bar(x - width / 2, values[:, 0], width, label="I_L")
    axis.bar(x + width / 2, values[:, 1], width, label="X_L")
    if result.xxx_syndrome_00_contribution is not None:
        axis.bar(
            x[0] + width / 2,
            result.xxx_syndrome_00_contribution,
            width,
            facecolor="none",
            edgecolor="black",
            hatch="///",
            linewidth=1.4,
            label="XXX contribution for syndrome 00",
        )
    axis.set_xticks(x, SYNDROME_LABELS)
    axis.set_xlabel("Measured syndrome (s1 s2)")
    axis.set_ylabel("Posterior probability")
    axis.set_title("Post-recovery logical-class probabilities")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return axis


def plot_monte_carlo_convergence(results: Iterable[object], ax: Any = None) -> Any:
    """Plot logical-rate estimates and confidence bands against shot count."""
    axis = _axis(ax)
    rows = results_to_rows(results)
    if not rows:
        raise ValueError("convergence plot requires at least one result")
    groups: dict[tuple[str, float, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["backend"]), float(row["p"]), float(row["rho"]))
        groups.setdefault(key, []).append(row)
    shot_values = [float(row["num_shots"]) for row in rows]
    x_floor = _log_floor(shot_values)
    for (backend, p, rho), group in sorted(groups.items()):
        group.sort(key=lambda row: float(row["num_shots"]))
        shots = _safe_log_values([float(row["num_shots"]) for row in group], x_floor)
        estimates = np.asarray([float(row["p_logical_estimate"]) for row in group])
        lows = np.asarray([float(row["ci_low"]) for row in group])
        highs = np.asarray([float(row["ci_high"]) for row in group])
        label = f"{backend}, p={p:g}, rho={rho:g}"
        line = axis.plot(shots, estimates, marker="o", label=f"{label} estimate")[0]
        axis.fill_between(shots, lows, highs, color=line.get_color(), alpha=0.18, label="95% CI")
        exact = float(group[0]["p_logical_exact"])
        axis.axhline(exact, color=line.get_color(), linestyle="--", alpha=0.8, label=f"{label} exact")
    axis.set_xscale("log")
    axis.set_xlabel("Monte Carlo shots")
    axis.set_ylabel("Estimated logical error rate")
    axis.set_title("Monte Carlo convergence")
    axis.set_ylim(bottom=0.0)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    return axis


def plot_backend_performance(results: Iterable[object], ax: Any = None) -> Any:
    """Plot throughput and available total/cold/warm runtime measurements."""
    axis = _axis(ax, figsize=(8.2, 4.6))
    raw = tuple(results)
    rows = performance_results_to_rows(raw)
    if not rows:
        raise ValueError("backend-performance plot requires at least one result")
    x = np.arange(len(rows))
    def cache_label(row: Mapping[str, object]) -> str:
        if not bool(row.get("cache_enabled")):
            return "uncached"
        return "warm cache" if bool(row.get("cache_hit")) else "cold cache"

    labels = [
        f"{row['backend']} ({cache_label(row)})\n{int(row['num_shots']):,} shots"
        for row in rows
    ]
    throughput = [float(row["shots_per_second"]) for row in rows]
    bars = axis.bar(x, throughput, 0.62, label="Throughput")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Backend and workload")
    axis.set_ylabel("Throughput (shots/s)")
    axis.set_title("Monte Carlo backend performance")
    axis.grid(axis="y", alpha=0.25)

    runtime_axis = axis.twinx()
    runtime_axis.plot(
        x,
        [float(row["runtime_seconds"]) for row in rows],
        marker="o",
        color="tab:red",
        label="Total runtime",
    )
    for field, label, style in (
        ("cold_start_runtime", "Cold-start runtime", ":"),
        ("warm_runtime", "Warm runtime", "--"),
        ("random_generation_runtime", "Random generation", "-."),
        ("host_to_device_runtime", "Host-to-device", ":"),
        ("kernel_runtime", "Vectorized kernels", "-"),
        ("reduction_runtime", "Reduction", "--"),
        ("device_to_host_runtime", "Device-to-host", "-."),
        ("compilation_time", "Plan compilation", ":"),
        ("planning_time", "Plan lookup/planning", "--"),
        ("execution_time", "Execution", "-"),
        ("total_time", "End-to-end total", "-."),
    ):
        values = [_float_or_none(row[field]) for row in rows]
        if any(value is not None for value in values):
            runtime_axis.plot(
                x,
                [np.nan if value is None else value for value in values],
                marker="o",
                linestyle=style,
                label=label,
            )
    runtime_axis.set_ylabel("Runtime (seconds)")
    handles = [bars] + runtime_axis.get_lines()
    axis.legend(handles, [handle.get_label() for handle in handles], loc="best")
    return axis
