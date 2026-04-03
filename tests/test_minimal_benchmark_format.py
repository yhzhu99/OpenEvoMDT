import asyncio
import json
from pathlib import Path

from evomdt.io import canonicalize_case_record
from evomdt.models import CaseDossier
from evomdt.pipeline import EvoMDTSystem
from evomdt.prompts import build_user_prompt

from .conftest import MinimalTaskFakeProvider


def test_canonicalize_case_record_supports_qid_task_answer_format():
    record = canonicalize_case_record(
        {
            "qid": 7,
            "task": "Among the following, choose the correct answer. (A) one (B) two (C) three",
            "answer": "C",
        },
        line_number=1,
    )

    assert record["case_id"] == "7"
    assert record["domain"] == "biomedical_qa"
    assert record["task_type"] == "generation"
    assert record["question"].startswith("Among the following")
    assert record["options"] == ["A", "B", "C"]
    assert record["reference_answer"] == "C"
    assert record["metadata"]["output_contract"]["allowed_values"] == ["A", "B", "C"]


def test_biomedical_prompt_uses_compact_payload_without_reference_answer():
    record = canonicalize_case_record(
        {
            "qid": 3,
            "task": "Which option is correct? (A) alpha (B) beta (C) gamma",
            "answer": "B",
        }
    )

    case = CaseDossier.model_validate(record)
    prompt = build_user_prompt(case, "treatment")

    assert '"question": "Which option is correct? (A) alpha (B) beta (C) gamma"' in prompt
    assert '"options": [' in prompt
    assert "reference_answer" not in prompt
    assert "patient_context" not in prompt
    assert "source_record" not in prompt


def test_run_benchmark_accepts_minimal_generation_records(tmp_path, test_config):
    dataset_path = tmp_path / "minimal_benchmark.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"qid": 1, "task": "Pick the best answer. (A) first (B) second (C) third", "answer": "C"}),
                json.dumps({"qid": 2, "task": "Choose correctly. (A) wrong (B) right", "answer": "B"}),
            ]
        ),
        encoding="utf-8",
    )

    system = EvoMDTSystem(test_config, provider_factory=lambda model_key: MinimalTaskFakeProvider())
    output_path = Path(test_config.runtime.artifacts_dir) / "minimal-summary.json"

    summary = asyncio.run(system.run_benchmark(dataset_path, output_path=output_path))

    assert summary["cases"] == 2
    assert output_path.exists()
    assert summary["aggregate_metrics"]["accuracy"] == 1.0
