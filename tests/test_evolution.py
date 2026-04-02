import json

from evomdt.evolution import apply_feedback, initial_state, load_state
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


def test_load_state_drops_legacy_coordinator_weight(test_config):
    path = test_config.evolution_state_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "role_weights": {
                    "diagnostic": 1.0,
                    "treatment": 1.0,
                    "safety": 1.0,
                    "monitoring": 1.0,
                    "coordinator": 1.1,
                },
                "prompt_refinements": {
                    "diagnostic": [],
                    "treatment": [],
                    "safety": [],
                    "monitoring": [],
                    "coordinator": ["efficiency_gap"],
                },
                "history": [
                    {
                        "timestamp": "2026-04-03T00:00:00Z",
                        "feedback_tags": ["safety_gap"],
                        "weight_updates": {"coordinator": 1.1, "safety": 1.05},
                        "prompt_updates": {"coordinator": ["efficiency_gap"], "safety": ["safety_gap"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state = load_state(test_config)

    assert "coordinator" not in state.role_weights
    assert "coordinator" not in state.prompt_refinements
    assert state.history[0].weight_updates == {"safety": 1.05}
