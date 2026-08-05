"""Generate reproducible project charts for the README and GitHub profile."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from tensorcontract.qec import IndependentPauliNoise, RepetitionCode, decode_repetition


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

NAVY = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
MUTED = "#8b949e"
CYAN = "#39c5cf"
VIOLET = "#a371f7"
AMBER = "#e3b341"


def style(figure: plt.Figure) -> None:
    figure.patch.set_facecolor(NAVY)
    for axis in figure.axes:
        axis.set_facecolor(PANEL)
        axis.tick_params(colors=MUTED)
        axis.xaxis.label.set_color(TEXT)
        axis.yaxis.label.set_color(TEXT)
        axis.title.set_color(TEXT)
        for spine in axis.spines.values():
            spine.set_color("#30363d")
        axis.grid(axis="y", color="#30363d", alpha=0.55, linewidth=0.7)


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(ASSETS / name, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def coset_probabilities() -> None:
    code = RepetitionCode(3)
    noise = IndependentPauliNoise.bit_flip(0.1)
    syndromes = list(product((0, 1), repeat=2))
    results = [decode_repetition(code, syndrome, noise) for syndrome in syndromes]
    values = np.asarray([result.probabilities for result in results])
    x = np.arange(len(syndromes))
    figure, axis = plt.subplots(figsize=(8.4, 3.8))
    width = 0.36
    axis.bar(x - width / 2, values[:, 0], width, label="logical sector 0", color=CYAN)
    axis.bar(x + width / 2, values[:, 1], width, label="logical sector 1", color=VIOLET)
    axis.set_xticks(x, ["".join(map(str, syndrome)) for syndrome in syndromes])
    axis.set_xlabel("Measured syndrome")
    axis.set_ylabel("Exact joint probability")
    axis.set_title("Exact repetition-code coset probabilities · p(X) = 0.10")
    axis.legend(frameon=False, labelcolor=TEXT)
    axis.text(0.58, 0.72, "tensor contraction = exhaustive enumeration", transform=axis.transAxes,
              ha="center", va="center", color=AMBER, fontsize=9)
    style(figure)
    save(figure, "qec-coset-probabilities.png")


def rewrite_and_plans() -> None:
    result = decode_repetition(RepetitionCode(5), (1, 0, 1, 0), IndependentPauliNoise.bit_flip(0.1))
    trace = result.rewrite_traces[0]
    labels = ["input"] + [record.rule.replace("-", "\n") for record in trace]
    stored = [trace[0].before_elements] + [record.after_elements for record in trace]
    candidates = result.plans[0].candidates

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.1))
    left.plot(range(len(stored)), stored, marker="o", color=CYAN, linewidth=2.2)
    left.fill_between(range(len(stored)), stored, alpha=0.12, color=CYAN)
    left.set_xticks(range(len(labels)), labels, fontsize=7)
    left.set_ylabel("Stored tensor elements")
    left.set_title("Deterministic rewrite reduction")
    left.annotate(f"{stored[0]} → {stored[-1]}", xy=(len(stored)-1, stored[-1]),
                  xytext=(-55, 26), textcoords="offset points", color=AMBER,
                  arrowprops={"arrowstyle": "->", "color": AMBER})

    names = [candidate.name for candidate in candidates]
    x = np.arange(len(names))
    right.bar(x - 0.18, [candidate.total_flops for candidate in candidates], 0.36,
              label="estimated FLOPs", color=VIOLET)
    right.bar(x + 0.18, [candidate.peak_elements for candidate in candidates], 0.36,
              label="peak elements", color=AMBER)
    right.set_xticks(x, names)
    right.set_ylabel("Analytic cost estimate")
    right.set_title("Explainable contraction candidates")
    right.legend(frameon=False, labelcolor=TEXT)
    style(figure)
    figure.suptitle("Rewrite + multiobjective planning pipeline", color=TEXT, fontsize=14, y=1.02)
    save(figure, "rewrite-planner-report.png")


def scaling() -> None:
    noise = IndependentPauliNoise.bit_flip(0.1)
    lengths = list(range(3, 10))
    runtimes: list[float] = []
    flops: list[int] = []
    peaks: list[int] = []
    for length in lengths:
        syndrome = (0,) * (length - 1)
        samples = []
        last = None
        for _ in range(5):
            started = perf_counter()
            last = decode_repetition(RepetitionCode(length), syndrome, noise)
            samples.append((perf_counter() - started) * 1e3)
        assert last is not None and last.verified
        runtimes.append(median(samples))
        flops.append(sum(plan.selected.total_flops for plan in last.plans))
        peaks.append(max(plan.selected.peak_elements for plan in last.plans))

    figure, left = plt.subplots(figsize=(8.4, 3.9))
    right = left.twinx()
    left.plot(lengths, runtimes, marker="o", color=CYAN, label="median decode time")
    right.plot(lengths, flops, marker="s", color=VIOLET, label="estimated FLOPs")
    right.plot(lengths, peaks, marker="^", color=AMBER, label="peak elements")
    left.set_xlabel("Repetition-code length")
    left.set_ylabel("Median end-to-end time (ms)", color=CYAN)
    right.set_ylabel("Planner estimate", color=VIOLET)
    left.set_title("Verified small-code scaling · five local runs per point")
    handles = left.get_lines() + right.get_lines()
    left.legend(handles, [line.get_label() for line in handles], frameon=False, labelcolor=TEXT)
    style(figure)
    right.set_facecolor("none")
    right.tick_params(colors=MUTED)
    right.spines["right"].set_color("#30363d")
    save(figure, "verified-scaling.png")


if __name__ == "__main__":
    coset_probabilities()
    rewrite_and_plans()
    scaling()
    print(f"Wrote GitHub visualizations to {ASSETS}")
