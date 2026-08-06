"""Plot measured and estimated symbolic contraction-order costs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .benchmark import SymbolicBenchmarkReport


def plot_benchmark(report: SymbolicBenchmarkReport, path: str | Path) -> None:
    names = [record.ordering for record in report.orderings]
    x = np.arange(len(names))
    times_ms = [record.median_seconds * 1e3 for record in report.orderings]
    flops = [record.estimated_flops for record in report.orderings]
    peaks = [record.peak_intermediate_elements for record in report.orderings]
    dark, panel, text, muted = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
    cyan, violet, amber = "#39c5cf", "#a371f7", "#e3b341"
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    figure.patch.set_facecolor(dark)
    left.bar(x, times_ms, color=cyan, width=0.62)
    left.set_xticks(x, names)
    left.set_ylabel("Median sequential execution (ms)")
    left.set_title(f"Measured on {report.device.upper()} · d={report.dimension}")
    for position, value in enumerate(times_ms):
        left.text(position, value, f"{value:.3f}", ha="center", va="bottom", color=text, fontsize=9)
    right.bar(x - 0.18, flops, 0.36, label="estimated FLOPs", color=violet)
    right.bar(x + 0.18, peaks, 0.36, label="peak elements", color=amber)
    right.set_xticks(x, names)
    right.set_ylabel("Analytic plan estimate")
    right.set_title("Ordering-dependent contraction cost")
    right.legend(frameon=False, labelcolor=text)
    for axis in (left, right):
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
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=dark)
    plt.close(figure)
