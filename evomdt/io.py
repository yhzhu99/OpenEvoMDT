from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CaseDossier


def load_case(path: str | Path) -> CaseDossier:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return CaseDossier.model_validate(raw)


def load_dataset(path: str | Path) -> list[CaseDossier]:
    items: list[CaseDossier] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        items.append(CaseDossier.model_validate_json(stripped))
    return items


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
