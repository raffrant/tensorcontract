"""Regression tests for deterministic planning and current cost estimates."""

import numpy as np
import pytest

from tensorcontract.ir import Index, TensorNetwork, TensorNode
from tensorcontract.planner import PlanConstraints, build_ordered_plan, plan_contraction


def _matrix_network() -> TensorNetwork:
    network = TensorNetwork(open_indices=("i", "j"))
    for name, dimension in (("i", 2), ("k", 3), ("j", 4)):
        network.add_index(Index(name, dimension))
    network.add_node(TensorNode("left", ("i", "k"), data=np.ones((2, 3))))
    network.add_node(TensorNode("right", ("k", "j"), data=np.ones((3, 4))))
    return network


def test_planning_is_deterministic_except_for_elapsed_time() -> None:
    network = _matrix_network()
    first = plan_contraction(network, PlanConstraints())
    second = plan_contraction(network, PlanConstraints())
    assert first.selected == second.selected
    assert first.candidates == second.candidates


def test_known_matrix_cost_estimates_and_precision_bytes() -> None:
    network = _matrix_network()
    plan64 = plan_contraction(network, PlanConstraints(precision_bytes=8)).selected
    step64 = plan64.steps[0]
    assert step64.flops == 24
    assert step64.output_elements == 8
    assert step64.bytes_moved == 208
    assert plan64.peak_elements == 12

    step32 = plan_contraction(network, PlanConstraints(precision_bytes=4)).selected.steps[0]
    assert step32.bytes_moved == 104


def test_planner_rejects_impossible_intermediate_limit() -> None:
    with pytest.raises(MemoryError, match="no contraction"):
        plan_contraction(_matrix_network(), PlanConstraints(max_intermediate_elements=7))


def test_explicit_plan_rejects_invalid_or_incomplete_orders() -> None:
    network = _matrix_network()
    with pytest.raises(ValueError, match="invalid explicit pair"):
        build_ordered_plan(network, (("left", "left"),))
    with pytest.raises(ValueError, match="invalid explicit pair"):
        build_ordered_plan(network, (("left", "missing"),))
    with pytest.raises(ValueError, match="explicit order is incomplete"):
        build_ordered_plan(network, ())


def test_explicit_plan_obeys_limits_and_has_stable_intermediate_names() -> None:
    network = _matrix_network()
    plan = build_ordered_plan(network, (("left", "right"),), constraints=PlanConstraints(max_intermediate_elements=8))
    assert plan.steps[0].output == "_t0"
    with pytest.raises(MemoryError, match="max_intermediate_elements"):
        build_ordered_plan(network, (("left", "right"),), constraints=PlanConstraints(max_intermediate_elements=7))
