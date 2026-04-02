from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .io import ensure_directory, read_json, write_json
from .models import CaseRunResult


def run_directory(config: AppConfig, run_id: str) -> Path:
    return ensure_directory(config.artifacts_dir / "runs" / run_id)


def save_case_run(config: AppConfig, result: CaseRunResult) -> Path:
    directory = run_directory(config, result.run_id)
    write_json(directory / "case.json", result.case.model_dump())
    write_json(directory / "coordinator.json", result.coordinator_decision.model_dump())
    if result.evaluation:
        write_json(directory / "evaluation.json", result.evaluation.model_dump())
    for trace in result.agent_traces:
        write_json(
            directory / f"{trace.role}.json",
            {
                "role": trace.role,
                "model_name": trace.model_name,
                "raw_content": trace.raw_content,
                "usage": trace.usage,
                "parsed_output": trace.parsed_output.model_dump(),
            },
        )
    write_json(directory / "run_summary.json", result.model_dump())
    return directory


def inspect_run(config: AppConfig, run_id: str) -> dict:
    directory = config.artifacts_dir / "runs" / run_id
    return read_json(directory / "run_summary.json")
