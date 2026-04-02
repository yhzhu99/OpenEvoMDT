from evomdt.evolution import apply_feedback, initial_state
from evomdt.models import EvaluationResult


def test_feedback_updates_weights_and_refinements(test_config):
    state = initial_state(test_config)
    evaluation = EvaluationResult(
        metrics={"composite_dimension_mean": 3.1},
        dimension_scores={"safety_ethics": 3.0},
        feedback_tags=["safety_gap", "completeness_gap"],
        summary="needs refinement",
    )
    updated = apply_feedback(test_config, state, evaluation)
    assert updated.role_weights["safety"] > state.role_weights["safety"]
    assert "safety_gap" in updated.prompt_refinements["safety"]
    assert updated.history
