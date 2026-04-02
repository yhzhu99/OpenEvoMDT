import asyncio

from evomdt.pipeline import EvoMDTSystem

from .conftest import FakeProvider


def test_run_case_generates_artifacts_and_evaluation(sample_case, test_config):
    system = EvoMDTSystem(test_config, provider_factory=lambda model_key: FakeProvider())
    result = asyncio.run(system.run_case(sample_case))
    assert result.coordinator_decision.final_plan
    assert "Laparoscopic liver resection" in result.coordinator_decision.accepted_actions
    assert "Major hepatectomy" in result.coordinator_decision.rejected_actions
    assert result.evaluation is not None
    assert result.evaluation.metrics["composite_dimension_mean"] >= 3.5
