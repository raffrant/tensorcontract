import numpy as np
import pytest

from tensorcontract.backend import execute_numpy
from tensorcontract.planner import PlanConstraints, build_ordered_plan, plan_contraction
from tensorcontract.symbolics.benchmark import benchmark_orderings
from tensorcontract.symbolics.model import build_complete_five_node_network
from tensorcontract.symbolics.torch_backend import execute_torch


def test_five_rank_three_nodes_induce_complete_interaction_graph() -> None:
    symbolic = build_complete_five_node_network(dimension=3, seed=12)
    assert symbolic.is_fully_connected
    assert all(len(neighbours) == 4 for neighbours in symbolic.interaction_graph.values())
    for node in symbolic.nodes:
        assert len(node.variables) == 3
        assert set(node.variables) <= node.expression.free_symbols


def test_materialization_is_reproducible() -> None:
    left = build_complete_five_node_network(3, 4).materialize()
    right = build_complete_five_node_network(3, 4).materialize()
    for name in left.nodes:
        assert np.array_equal(left.nodes[name].data, right.nodes[name].data)


def test_all_benchmark_orders_match_numpy_on_cpu() -> None:
    report = benchmark_orderings(build_complete_five_node_network(3, 2), device="cpu", warmup=0, repeats=1)
    assert len(report.orderings) >= 2
    assert all(record.absolute_error_vs_numpy < 1e-9 for record in report.orderings)
    adverse = next(record for record in report.orderings if record.ordering == "adverse")
    best_peak = min(record.peak_intermediate_elements for record in report.orderings)
    assert adverse.peak_intermediate_elements > best_peak


def test_explicit_order_validation_and_torch_execution() -> None:
    network = build_complete_five_node_network(3, 1).materialize()
    plan = build_ordered_plan(network, (("f0", "f1"), ("_t0", "f2"), ("_t1", "f3"), ("_t2", "f4")))
    expected = execute_numpy(network, plan)
    actual = execute_torch(network, plan, "cpu")
    assert np.allclose(actual.value, expected)
    with pytest.raises(ValueError):
        build_ordered_plan(network, (("missing", "f0"),))
