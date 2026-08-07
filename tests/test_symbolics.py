import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch backend is optional")

from tensorcontract.backend import execute_numpy
from tensorcontract.planner import build_ordered_plan
from tensorcontract.symbolics.benchmark import benchmark_orderings
from tensorcontract.symbolics.model import build_complete_five_node_network
from tensorcontract.symbolics.torch_backend import execute_torch


def test_all_benchmark_orders_match_numpy_on_cpu() -> None:
    report = benchmark_orderings(build_complete_five_node_network(3, 2), device="cpu", warmup=0, repeats=1)
    assert len(report.orderings) >= 2
    assert all(record.absolute_error_vs_numpy < 1e-9 for record in report.orderings)
    adverse = next(record for record in report.orderings if record.ordering == "adverse")
    best_peak = min(record.peak_intermediate_elements for record in report.orderings)
    assert adverse.peak_intermediate_elements > best_peak
    assert report.speedup("adverse") == pytest.approx(1.0)
    assert report.speedup("flops") == pytest.approx(adverse.median_seconds / report.orderings[0].median_seconds)


def test_explicit_order_validation_and_torch_execution() -> None:
    network = build_complete_five_node_network(3, 1).materialize()
    plan = build_ordered_plan(network, (("f0", "f1"), ("_t0", "f2"), ("_t1", "f3"), ("_t2", "f4")))
    expected = execute_numpy(network, plan)
    actual = execute_torch(network, plan, "cpu")
    assert np.allclose(actual.value, expected)
    with pytest.raises(ValueError):
        build_ordered_plan(network, (("missing", "f0"),))


def test_plot_can_be_built_without_writing_or_showing() -> None:
    pytest.importorskip("matplotlib", reason="visualization support is optional")
    from tensorcontract.symbolics.visualize import plot_benchmark

    report = benchmark_orderings(build_complete_five_node_network(2, 3), "cpu", warmup=0, repeats=1)
    figure = plot_benchmark(report, show=False)
    assert len(figure.axes) == 3


def test_cuda_request_fails_clearly_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from tensorcontract.symbolics.torch_backend import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")
