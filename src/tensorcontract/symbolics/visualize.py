"""Plot measured and estimated symbolic contraction-order costs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .benchmark import SymbolicBenchmarkReport


def plot_benchmark(report: SymbolicBenchmarkReport, *, show: bool = True) -> Figure:
    """Build the benchmark figure and optionally show it without saving."""
    names = [record.ordering for record in report.orderings]
    x = np.arange(len(names))
    times_ms = [record.median_seconds * 1e3 for record in report.orderings]
    flops = [record.estimated_flops for record in report.orderings]
    peaks = [record.peak_intermediate_elements for record in report.orderings]
    dark, panel, text, muted = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
    cyan, violet, amber = "#39c5cf", "#a371f7", "#e3b341"
    figure = plt.figure(figsize=(12.2, 7.0))
    grid = figure.add_gridspec(2, 2, height_ratios=(2.1, 1.25))
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1])
    sequence_axis = figure.add_subplot(grid[1, :])
    figure.patch.set_facecolor(dark)
    left.bar(x, times_ms, color=cyan, width=0.62)
    left.set_xticks(x, names)
    left.set_ylabel("Median sequential execution (ms)")
    left.set_title(f"Measured on {report.device.upper()} · d={report.dimension}")
    for position, (name, value) in enumerate(zip(names, times_ms)):
        speedup = report.speedup(name)
        left.text(position, value, f"{value:.3f} ms\n{speedup:.2f}× vs adverse",
                  ha="center", va="bottom", color=text, fontsize=9)
    right.bar(x - 0.18, flops, 0.36, label="estimated FLOPs", color=violet)
    right.bar(x + 0.18, peaks, 0.36, label="peak elements", color=amber)
    right.set_xticks(x, names)
    right.set_ylabel("Analytic plan estimate")
    right.set_title("Ordering-dependent contraction cost")
    right.legend(frameon=False, labelcolor=text)
    sequence_axis.set_title("Same K₅ interaction graph, different pairwise contraction sequences")
    sequence_axis.set_xlim(0, 1)
    sequence_axis.set_ylim(0, len(report.orderings) + 0.5)
    sequence_axis.set_xticks([])
    sequence_axis.set_yticks([])
    for row, record in enumerate(report.orderings):
        y = len(report.orderings) - row
        sequence = "  →  ".join(
            f"{step} [{size:,}]" for step, size in zip(record.step_order, record.intermediate_elements)
        )
        sequence_axis.text(0.02, y, f"{record.ordering:8s}", color=(amber if record.ordering == "adverse" else cyan),
                           fontsize=10, fontweight="bold", va="center", family="monospace")
        sequence_axis.text(0.15, y, sequence, color=text, fontsize=9, va="center", family="monospace")
    sequence_axis.text(
        0.02, 0.25,
        "[n] is the output size of each step. The adverse order contracts factors sharing only one variable first,\n"
        "creating a five-index intermediate; planner orders begin with two-variable overlaps and keep rank lower.",
        color=muted, fontsize=9, va="bottom",
    )
    for axis in (left, right, sequence_axis):
        axis.set_facecolor(panel)
        axis.tick_params(colors=muted)
        axis.xaxis.label.set_color(text)
        axis.yaxis.label.set_color(text)
        axis.title.set_color(text)
        axis.grid(axis="y", color="#30363d", alpha=0.55)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#30363d")
    suffix = "CUDA synchronized; peak device allocation measured" if report.device.startswith("cuda") else "CUDA unavailable; explicitly labeled CPU fallback"
    figure.suptitle(f"Five-node symbolic tensor network · {suffix}", color=text, fontsize=13)
    figure.tight_layout()
    if show:
        plt.show()
    return figure
