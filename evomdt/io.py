from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from .models import CaseDossier

CASE_ID_KEYS: tuple[str, ...] = ("case_id", "qid", "id", "question_id")
QUESTION_KEYS: tuple[str, ...] = ("question", "task", "prompt", "query", "input")
ANSWER_KEYS: tuple[str, ...] = ("reference_answer", "answer", "gold_answer", "label", "target")
OPTIONS_KEYS: tuple[str, ...] = ("options", "choices", "answer_choices", "candidates")
OPTION_LABEL_PATTERN = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=\n))\(([A-Z])\)\s")


def _first_present(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    return None


def _normalize_case_id(value: Any, line_number: int | None = None) -> str:
    if value not in (None, ""):
        return str(value)
    if line_number is not None:
        return f"case-{line_number}"
    return "case-unknown"


def _normalize_options(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        labels = [str(key).strip() for key in value if str(key).strip()]
        return labels or None
    if isinstance(value, list):
        options = [str(item).strip() for item in value if str(item).strip()]
        return options or None
    text = str(value).strip()
    return [text] if text else None


def _infer_option_labels(question: str) -> list[str] | None:
    labels: list[str] = []
    for match in OPTION_LABEL_PATTERN.finditer(question):
        label = match.group(1)
        if label not in labels:
            labels.append(label)
    return labels or None


def _infer_output_contract(reference_answer: str | None, options: list[str] | None) -> dict[str, Any] | None:
    if not reference_answer or not options:
        return None
    normalized_options = {option.upper() for option in options}
    if reference_answer.strip().upper() not in normalized_options:
        return None
    return {
        "type": "enum_choice",
        "allowed_values": options,
        "final_answer_only": True,
    }


def _looks_like_minimal_task(raw: dict[str, Any]) -> bool:
    return _first_present(raw, QUESTION_KEYS) is not None and (
        _first_present(raw, CASE_ID_KEYS) is not None or _first_present(raw, ANSWER_KEYS) is not None
    )


def canonicalize_case_record(raw: dict[str, Any], *, line_number: int | None = None) -> dict[str, Any]:
    if "case_id" in raw and "question" in raw:
        canonical = dict(raw)
        if "domain" not in canonical and canonical.get("cancer_type") is None:
            canonical["domain"] = "biomedical_qa"
        if "task_type" not in canonical and canonical.get("cancer_type") is None:
            canonical["task_type"] = "generation"
        return canonical

    if not _looks_like_minimal_task(raw):
        required = "case_id/question or compatible aliases like qid/task"
        raise ValueError(f"Dataset record is missing required task identifiers: expected {required}")

    question = str(_first_present(raw, QUESTION_KEYS)).strip()
    case_id = _normalize_case_id(_first_present(raw, CASE_ID_KEYS), line_number=line_number)
    reference_answer_value = _first_present(raw, ANSWER_KEYS)
    reference_answer = str(reference_answer_value).strip() if reference_answer_value not in (None, "") else None
    options = _normalize_options(_first_present(raw, OPTIONS_KEYS))
    if options is None:
        options = _infer_option_labels(question)

    metadata = dict(raw.get("metadata", {})) if isinstance(raw.get("metadata"), dict) else {}
    metadata.setdefault("source_record", raw)
    if "output_contract" not in metadata:
        inferred_contract = _infer_output_contract(reference_answer, options)
        if inferred_contract is not None:
            metadata["output_contract"] = inferred_contract

    return {
        "case_id": case_id,
        "domain": raw.get("domain", "biomedical_qa"),
        "task_type": raw.get("task_type", "generation"),
        "question": question,
        "options": options,
        "reference_answer": reference_answer,
        "metadata": metadata,
    }


def load_case(path: str | Path) -> CaseDossier:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return CaseDossier.model_validate(canonicalize_case_record(raw))


def load_dataset(path: str | Path) -> list[CaseDossier]:
    return list(iter_dataset_tasks(path))


def iter_dataset_tasks(path: str | Path) -> Iterator[CaseDossier]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}") from exc
            yield CaseDossier.model_validate(canonicalize_case_record(raw, line_number=line_number))


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
