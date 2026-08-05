from tensorcontract.qec import IndependentPauliNoise, RepetitionCode, build_repetition_decoding_network
from tensorcontract.planner import PlanConstraints, plan_contraction
from tensorcontract.rewrite import RewriteEngine


def test_multiple_explainable_plans_and_multiobjective_selection() -> None:
    network, _ = RewriteEngine().simplify(
        build_repetition_decoding_network(RepetitionCode(5), (1, 0, 1, 0), IndependentPauliNoise.bit_flip(0.1), 0)
    )
    result = plan_contraction(network, PlanConstraints(objective_flops_weight=1.0, objective_memory_weight=3.0))
    assert {plan.name for plan in result.candidates} == {"flops", "memory", "balanced"}
    assert len({tuple((s.left, s.right) for s in plan.steps) for plan in result.candidates}) >= 2
    assert result.selected in result.candidates
    assert result.selected.score == min(plan.score for plan in result.candidates)
    assert all(step.shape_class for step in result.selected.steps)
    assert "peak_elements=" in result.selected.report()


def test_intermediate_constraint_is_obeyed() -> None:
    network, _ = RewriteEngine().simplify(
        build_repetition_decoding_network(RepetitionCode(3), (0, 0), IndependentPauliNoise.bit_flip(0.1), 0)
    )
    result = plan_contraction(network, PlanConstraints(max_intermediate_elements=8))
    assert all(step.output_elements <= 8 for step in result.selected.steps)
