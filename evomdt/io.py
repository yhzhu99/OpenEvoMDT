from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .models import CaseDossier


def load_case(path: str | Path) -> CaseDossier:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return CaseDossier.model_validate(raw)


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
            if "case_id" not in raw or "question" not in raw:
                raise ValueError(
                    f"Line {line_number} of {dataset_path} is missing required fields: case_id and question"
                )
            yield CaseDossier.model_validate(raw)


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
