"""Benchmark a complete five-node symbolic network on PyTorch CPU/CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorcontract.symbolics.benchmark import benchmark_orderings
from tensorcontract.symbolics.model import build_complete_five_node_network
from tensorcontract.symbolics.visualize import plot_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=8, help="dimension of each of the five shared indices")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/symbolics"))
    arguments = parser.parse_args()

    symbolic = build_complete_five_node_network(arguments.dimension, arguments.seed)
    report = benchmark_orderings(symbolic, arguments.device, arguments.warmup, arguments.repeats)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report.write_json(arguments.output_dir / "five-node-orderings.json")
    plot_benchmark(report, arguments.output_dir / "five-node-orderings.png")

    print(f"device={report.device} cuda_available={report.cuda_available}")
    print("All five symbolic nodes pairwise interact:", symbolic.is_fully_connected)
    for node in symbolic.nodes:
        print(f"{node.name}{tuple(map(str, node.variables))} = {node.expression}")
    print("\nSequential contraction benchmarks:")
    for record in report.orderings:
        peak = "n/a (CPU)" if record.measured_peak_device_bytes is None else str(record.measured_peak_device_bytes)
        print(
            f"{record.ordering:9s} median={record.median_seconds * 1e3:9.4f} ms "
            f"flops={record.estimated_flops:9d} peak_elements={record.peak_intermediate_elements:8d} "
            f"GPU_peak_bytes={peak} error={record.absolute_error_vs_numpy:.3e}"
        )
        print("  order:", " -> ".join(record.step_order))
    print(f"\nWrote {arguments.output_dir / 'five-node-orderings.json'}")
    print(f"Wrote {arguments.output_dir / 'five-node-orderings.png'}")


if __name__ == "__main__":
    main()
