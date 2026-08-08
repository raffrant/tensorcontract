"""Reproducible cold-cache, warm-cache, and uncached QEC benchmark."""

from __future__ import annotations

import argparse

from tensorcontract.quantum import QuantumExecutionPlanCache, run_monte_carlo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--backend", choices=("numpy", "gpu"), default="numpy")
    parser.add_argument("--p", type=float, default=0.17)
    parser.add_argument("--rho", type=float, default=0.23)
    parser.add_argument("--seed", type=int, default=202606)
    args = parser.parse_args()
    cache = QuantumExecutionPlanCache()

    runs = (
        ("cold-cache", True),
        ("warm-cache", True),
        ("uncached", False),
    )
    print("mode,cache_hit,compilation_time,planning_time,execution_time,total_time")
    for label, enabled in runs:
        result = run_monte_carlo(
            args.shots, args.p, args.rho, seed=args.seed,
            batch_size=args.batch_size, backend=args.backend,
            plan_cache=cache, cache_enabled=enabled,
        )
        print(
            f"{label},{result.cache_hit},{result.compilation_time:.9f},"
            f"{result.planning_time:.9f},{result.execution_time:.9f},"
            f"{result.total_time:.9f}"
        )


if __name__ == "__main__":
    main()
