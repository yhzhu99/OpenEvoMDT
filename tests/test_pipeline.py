import asyncio
import json

from evomdt.pipeline import EvoMDTSystem

from .conftest import FakeProvider


def test_run_case_generates_artifacts_and_evaluation(sample_case, test_config):
    system = EvoMDTSystem(test_config, provider_factory=lambda model_key: FakeProvider())
    result = asyncio.run(system.run_case(sample_case))
    assert result.coordinator_decision.final_plan
    assert any(trace.role == "coordinator" for trace in result.agent_traces)
    assert "Laparoscopic liver resection" in result.coordinator_decision.accepted_actions
    assert "Major hepatectomy" in result.coordinator_decision.rejected_actions
    assert result.evaluation is not None
    assert result.evaluation.metrics["weighted_plan_concordance"] == 1.0
    assert result.evaluation.metrics["guideline_concordance"] == 1.0
    assert result.evaluation.metrics["safety_violation_rate"] == 0.0
    assert result.evaluation.metrics["evidence_traceability_coverage"] == 0.0


def test_run_case_uses_baseline_state_by_default(sample_case, test_config):
    state_path = test_config.evolution_state_path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "role_weights": {
                    "diagnostic": 1.15,
                    "treatment": 1.15,
                    "safety": 1.15,
                    "monitoring": 1.15,
                },
                "prompt_refinements": {
                    "diagnostic": ["evidence_gap"],
                    "treatment": ["treatment_gap"],
                    "safety": ["safety_gap"],
                    "monitoring": ["monitoring_gap"],
                },
                "history": [],
            }
        ),
        encoding="utf-8",
    )

    system = EvoMDTSystem(test_config, provider_factory=lambda model_key: FakeProvider())
    result = asyncio.run(system.run_case(sample_case, persist=False))

    assert result.coordinator_decision.role_weights == {
        "diagnostic": 1.0,
        "treatment": 1.0,
        "safety": 1.15,
        "monitoring": 1.0,
    }
