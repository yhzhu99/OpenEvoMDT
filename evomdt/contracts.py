from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import CaseDossier


class OutputContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    allowed_values: list[str] = Field(default_factory=list)
    final_answer_only: bool = False


def resolve_output_contract(case: CaseDossier) -> OutputContract | None:
    raw_contract = case.metadata.get("output_contract")
    if not isinstance(raw_contract, dict):
        return None
    return OutputContract.model_validate(raw_contract)


def output_contract_instructions(case: CaseDossier) -> str | None:
    contract = resolve_output_contract(case)
    if contract is None:
        return None

    parts: list[str] = [
        "This task has an explicit output contract.",
        f"Contract type: {contract.type}.",
    ]
    if contract.allowed_values:
        allowed = ", ".join(contract.allowed_values)
        parts.append(f"Allowed final_answer values: {allowed}.")
    if contract.final_answer_only:
        parts.append("The final answer must be only the enum choice with no extra text.")
    else:
        parts.append("Return final_answer as the constrained value, and keep any explanation in summary or rationale fields.")
    return " ".join(parts)


def normalize_answer(answer: str) -> str:
    return str(answer).strip().upper()


def is_allowed_answer(case: CaseDossier, answer: str) -> bool:
    contract = resolve_output_contract(case)
    if contract is None or not contract.allowed_values:
        return True
    return normalize_answer(answer) in {normalize_answer(value) for value in contract.allowed_values}
