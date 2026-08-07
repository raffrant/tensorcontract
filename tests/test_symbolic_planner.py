"""Tests for deterministic greedy symbolic contraction planning."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tensorcontract.backends import NumPyBackend
from tensorcontract.symbolics import (
    ContractionNode,
    IndexRole,
    InvalidSymbolicGraphError,
    PlanningUnavailableError,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicPlannerOptions,
    SymbolicPlanningError,
    SymbolicTensor,
    plan_symbolic_contractions,
)


def _matrix_chain() -> SymbolicGraph:
    graph = SymbolicGraph(metadata={"workload": "matrix-chain"})
    for name, dimension, role in (
        ("i", 2, IndexRole.FREE),
        ("a", 3, IndexRole.CONTRACTED),
        ("b", 4, IndexRole.CONTRACTED),
        ("j", 5, IndexRole.FREE),
    ):
        graph.add_index(SymbolicIndex(name, dimension, role))
    graph.add_tensor(SymbolicTensor("A", ("i", "a"), (2, 3)))
    graph.add_tensor(SymbolicTensor("B", ("a", "b"), (3, 4)))
    graph.add_tensor(SymbolicTensor("C", ("b", "j"), (4, 5)))
    graph.add_operation(ContractionNode("result", ("A", "B", "C"), ("i", "j")))
    return graph


def _adverse_input_order_graph() -> SymbolicGraph:
    graph = SymbolicGraph(metadata={"workload": "ordering-sensitive"})
    graph.add_index(SymbolicIndex("x", 100, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("a", 2, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("b", 2, IndexRole.CONTRACTED))
    graph.add_tensor(SymbolicTensor("A", ("x", "a"), (100, 2)))
    graph.add_tensor(SymbolicTensor("C", ("a", "b"), (2, 2)))
    graph.add_tensor(SymbolicTensor("B", ("x", "b"), (100, 2)))
    graph.add_operation(ContractionNode("result", ("A", "C", "B"), ()))
    return graph


def test_plan_is_valid_pairwise_and_debuggable() -> None:
    plan = plan_symbolic_contractions(_matrix_chain())
    plan.validate()
    assert len(plan.steps) == 2
    assert all(
        len(operation.inputs) == 2
        for operation in plan.graph.operations.values()
        if isinstance(operation, ContractionNode)
    )
    assert plan.steps[-1].name == "result"
    assert plan.steps[-1].source_operation == "result"
    report = plan.debug_string()
    assert "SymbolicExecutionPlan(strategy='greedy'" in report
    assert "peak_elements=" in report
    assert "temporary=" in report


def test_planning_is_deterministic() -> None:
    graph = _matrix_chain()
    first = plan_symbolic_contractions(graph)
    second = plan_symbolic_contractions(graph)
    assert first == second
    assert first.debug_string() == second.debug_string()


def test_planned_result_matches_original_reference_backend() -> None:
    graph = _matrix_chain()
    bindings = {
        "A": np.arange(6.0).reshape(2, 3),
        "B": np.arange(12.0).reshape(3, 4),
        "C": np.arange(20.0).reshape(4, 5),
    }
    backend = NumPyBackend()
    reference = backend.execute(graph, bindings)
    plan = plan_symbolic_contractions(graph)
    planned = backend.execute(plan.graph, bindings)
    trusted = np.einsum("ia,ab,bj->ij", bindings["A"], bindings["B"], bindings["C"])
    assert np.allclose(reference, trusted)
    assert np.allclose(planned, trusted)


def test_cost_estimates_are_internally_consistent() -> None:
    plan = plan_symbolic_contractions(
        _matrix_chain(), SymbolicPlannerOptions(precision_bytes=4)
    )
    assert plan.cost.total_flops == sum(step.estimated_flops for step in plan.steps)
    assert plan.cost.total_materialized_elements == sum(
        step.materialized_elements for step in plan.steps
    )
    assert plan.cost.temporary_materialization_elements == sum(
        step.materialized_elements for step in plan.steps[:-1]
    )
    assert plan.cost.peak_live_elements >= max(step.output_elements for step in plan.steps)
    assert plan.cost.peak_memory_bytes == plan.cost.peak_live_elements * 4


def test_plan_validation_detects_corrupted_aggregate_cost() -> None:
    plan = plan_symbolic_contractions(_matrix_chain())
    corrupted = replace(
        plan,
        cost=replace(plan.cost, total_flops=plan.cost.total_flops + 1),
    )
    with pytest.raises(SymbolicPlanningError, match="FLOP estimate is inconsistent"):
        corrupted.validate()


def test_greedy_prefers_smaller_intermediates_than_adverse_input_order() -> None:
    graph = _adverse_input_order_graph()
    greedy = plan_symbolic_contractions(graph)
    input_order = plan_symbolic_contractions(
        graph, SymbolicPlannerOptions(strategy="input-order")
    )
    assert greedy.steps[0].dependencies == ("A", "B")
    assert greedy.steps[0].output_elements == 4
    assert input_order.steps[0].dependencies == ("A", "C")
    assert input_order.steps[0].output_elements == 200
    assert greedy.cost.total_flops < input_order.cost.total_flops
    assert greedy.cost.peak_live_elements < input_order.cost.peak_live_elements
    assert (
        greedy.cost.temporary_materialization_elements
        < input_order.cost.temporary_materialization_elements
    )


def test_unavailable_strategy_falls_back_safely() -> None:
    graph = _matrix_chain()
    fallback = plan_symbolic_contractions(
        graph, SymbolicPlannerOptions(strategy="not-installed", allow_fallback=True)
    )
    explicit = plan_symbolic_contractions(
        graph, SymbolicPlannerOptions(strategy="input-order")
    )
    assert fallback.fallback_used
    assert fallback.strategy == "input-order"
    assert "not-installed" in (fallback.fallback_reason or "")
    assert fallback.steps == explicit.steps
    with pytest.raises(PlanningUnavailableError, match="not-installed"):
        plan_symbolic_contractions(
            graph,
            SymbolicPlannerOptions(strategy="not-installed", allow_fallback=False),
        )


def test_intermediate_limit_is_enforced() -> None:
    graph = _adverse_input_order_graph()
    constrained = plan_symbolic_contractions(
        graph, SymbolicPlannerOptions(max_intermediate_elements=4)
    )
    assert all(step.output_elements <= 4 for step in constrained.steps)
    with pytest.raises(SymbolicPlanningError, match="max_intermediate_elements"):
        plan_symbolic_contractions(
            graph, SymbolicPlannerOptions(max_intermediate_elements=3)
        )


def test_invalid_graph_fails_with_operation_and_dependency_context() -> None:
    graph = _matrix_chain()
    graph.operations["result"] = ContractionNode(
        "result", ("A", "missing", "C"), ("i", "j")
    )
    with pytest.raises(InvalidSymbolicGraphError) as captured:
        plan_symbolic_contractions(graph)
    assert "invalid symbolic graph" in str(captured.value)
    assert "result" in str(captured.value)
    assert "missing" in str(captured.value)


@pytest.mark.parametrize(
    "options",
    [
        SymbolicPlannerOptions(precision_bytes=1),
        SymbolicPlannerOptions(strategy="input-order"),
    ],
)
def test_valid_options_produce_valid_plans(options: SymbolicPlannerOptions) -> None:
    plan_symbolic_contractions(_matrix_chain(), options).validate()


def test_invalid_planner_options_fail_clearly() -> None:
    with pytest.raises(SymbolicPlanningError, match="precision_bytes"):
        SymbolicPlannerOptions(precision_bytes=0)
    with pytest.raises(SymbolicPlanningError, match="max_intermediate_elements"):
        SymbolicPlannerOptions(max_intermediate_elements=0)
