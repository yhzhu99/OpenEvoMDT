import asyncio
from pathlib import Path

from evomdt.io import iter_dataset_tasks, load_case
from evomdt.pipeline import EvoMDTSystem
from evomdt.prompts import build_coordinator_system_prompt, build_system_prompt

from .conftest import BiomedicalFakeProvider


def test_biomedical_prompts_include_output_contract():
    case = load_case("data/samples/cases/sample_biomedical_hyperpyrexia_case.json")

    specialist_prompt = build_system_prompt(case, "treatment", [])
    coordinator_prompt = build_coordinator_system_prompt(case, ["format_gap"])

    assert "biomedical qa workflow" in specialist_prompt.lower()
    assert "allowed final_answer values: a, b." in specialist_prompt.lower()
    assert "final_answer must contain the exact final answer string" in coordinator_prompt
    assert "follow the output contract exactly" in coordinator_prompt.lower()


def test_biomedical_run_case_generates_enum_answer_and_metrics(test_config):
    case = load_case("data/samples/cases/sample_biomedical_hyperpyrexia_case.json")
    system = EvoMDTSystem(test_config, provider_factory=lambda model_key: BiomedicalFakeProvider())

    result = asyncio.run(system.run_case(case))

    assert result.coordinator_decision.final_answer == "A"
    assert "Final Answer: A" in result.coordinator_decision.response_text
    assert Path(result.artifacts_dir, "coordinator.json").exists()
    assert result.evaluation is not None
    assert result.evaluation.metrics["format_compliance"] == 1.0
    assert result.evaluation.metrics["accuracy"] == 1.0
    assert result.evaluation.metrics["traceability_score"] == 0.75
    assert result.evaluation.feedback_tags == []


def test_biomedical_generation_evaluation_detects_contract_failures(test_config):
    case = list(iter_dataset_tasks("data/samples/benchmarks/sample_biomedical_benchmark.jsonl"))[1]
    system = EvoMDTSystem(test_config, provider_factory=lambda model_key: BiomedicalFakeProvider())

    result = asyncio.run(system.run_case(case, persist=False))

    assert result.coordinator_decision.final_answer == "Maybe B"
    assert result.evaluation is not None
    assert result.evaluation.metrics["format_compliance"] == 0.0
    assert result.evaluation.metrics["accuracy"] == 0.0
    assert result.evaluation.metrics["traceability_score"] == 0.625
    assert result.evaluation.feedback_tags == ["format_gap", "accuracy_gap", "traceability_gap"]


def test_biomedical_evolve_records_feedback_and_coordinator_refinement(test_config):
    system = EvoMDTSystem(test_config, provider_factory=lambda model_key: BiomedicalFakeProvider())

    summary = asyncio.run(system.evolve("data/samples/benchmarks/sample_biomedical_benchmark.jsonl"))

    assert summary["updated_cases"] == 2
    assert summary["role_weights"]["diagnostic"] == 1.075
    assert summary["role_weights"]["treatment"] == 1.075
    assert summary["prompt_refinements"]["coordinator"] == ["format_gap"]
    assert summary["prompt_refinements"]["diagnostic"] == ["accuracy_gap", "traceability_gap"]
