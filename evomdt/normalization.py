from __future__ import annotations

import json
import re
from typing import Any

from .models import CaseDossier, RoleName, SpecialistRoleName

ROLE_NAMES: set[str] = {"diagnostic", "treatment", "safety", "monitoring", "coordinator"}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_stringify(item) for item in value if _stringify(item)]
    if isinstance(value, dict):
        items: list[str] = []
        for key, item_value in value.items():
            rendered = _stringify(item_value)
            if rendered:
                items.append(f"{key}: {rendered}")
        return items
    rendered = _stringify(value)
    return [rendered] if rendered else []


def _coerce_priority(value: Any) -> int:
    if isinstance(value, bool):
        return 3
    if isinstance(value, int):
        return _clamp_int(value, 1, 5)
    if isinstance(value, float):
        return _clamp_int(round(value), 1, 5)
    text = _stringify(value).lower()
    if not text:
        return 3
    match = re.search(r"\d+", text)
    if match:
        return _clamp_int(int(match.group()), 1, 5)
    if any(token in text for token in ["critical", "urgent", "highest", "immediate", "top"]):
        return 1
    if any(token in text for token in ["high", "major"]):
        return 2
    if any(token in text for token in ["moderate", "medium", "standard"]):
        return 3
    if any(token in text for token in ["low", "minor"]):
        return 4
    if any(token in text for token in ["routine", "optional", "defer"]):
        return 5
    return 3


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.5
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1.0:
            numeric = numeric / 100.0 if numeric <= 100.0 else 1.0
        return round(min(max(numeric, 0.0), 1.0), 4)
    text = _stringify(value).lower()
    if not text:
        return 0.5
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        numeric = float(match.group(1))
        if "%" in text or numeric > 1.0:
            numeric = numeric / 100.0 if numeric <= 100.0 else 1.0
        return round(min(max(numeric, 0.0), 1.0), 4)
    if "very high" in text:
        return 0.95
    if "high" in text:
        return 0.85
    if any(token in text for token in ["moderate", "medium", "intermediate"]):
        return 0.65
    if any(token in text for token in ["low", "uncertain", "limited"]):
        return 0.35
    return 0.5


def _coerce_stance(value: Any) -> str:
    text = _stringify(value).lower()
    if not text:
        return "consider"
    if any(token in text for token in ["avoid", "contra", "unsafe", "not recommend", "do not", "withhold"]):
        return "avoid"
    if any(token in text for token in ["monitor", "observe", "follow-up", "watch", "surveil"]):
        return "monitor"
    if any(token in text for token in ["consider", "optional", "possible", "maybe"]):
        return "consider"
    if any(token in text for token in ["recommend", "indicated", "strongly", "preferred", "choose", "select"]):
        return "recommend"
    return "consider"


def _coerce_severity(value: Any) -> str:
    text = _stringify(value).lower()
    if not text:
        return "info"
    if any(token in text for token in ["absolute", "fatal", "life-threatening"]):
        return "absolute"
    if any(token in text for token in ["critical", "major", "severe", "high"]):
        return "major"
    if "moderate" in text:
        return "moderate"
    if any(token in text for token in ["minor", "low"]):
        return "minor"
    return "info"


def _coerce_citations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    citations: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        citations.append(
            {
                "source_id": item.get("source_id"),
                "anchor": item.get("anchor"),
                "version": item.get("version"),
                "evidence_grade": item.get("evidence_grade"),
                "year": item.get("year"),
                "source_type": item.get("source_type"),
            }
        )
    return citations


def _coerce_supporting_facts(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        facts: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                facts.append(
                    {
                        "field_path": _stringify(item.get("field_path") or item.get("field") or "summary"),
                        "value": _stringify(item.get("value") or item.get("fact") or item.get("content")),
                        "note": _stringify(item.get("note") or item.get("rationale") or "Normalized supporting fact."),
                        "citations": _coerce_citations(item.get("citations")),
                    }
                )
                continue
            rendered = _stringify(item)
            if rendered:
                facts.append(
                    {
                        "field_path": "summary",
                        "value": rendered,
                        "note": "Normalized supporting fact.",
                        "citations": [],
                    }
                )
        return facts
    if isinstance(value, dict):
        return [
            {
                "field_path": _stringify(key),
                "value": _stringify(item_value),
                "note": "Normalized supporting fact.",
                "citations": [],
            }
            for key, item_value in value.items()
            if _stringify(item_value)
        ]
    rendered = _stringify(value)
    return (
        [
            {
                "field_path": "summary",
                "value": rendered,
                "note": "Normalized supporting fact.",
                "citations": [],
            }
        ]
        if rendered
        else []
    )


def _coerce_candidate_actions(value: Any) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in _coerce_list(value):
        if isinstance(item, dict):
            action_text = _stringify(item.get("action") or item.get("recommendation") or item.get("name") or item.get("title"))
            if not action_text:
                continue
            actions.append(
                {
                    "action": action_text,
                    "stance": _coerce_stance(item.get("stance")),
                    "rationale": _stringify(item.get("rationale") or item.get("reason") or item.get("justification")),
                    "priority": _coerce_priority(item.get("priority")),
                    "citations": _coerce_citations(item.get("citations")),
                }
            )
            continue
        rendered = _stringify(item)
        if rendered:
            actions.append(
                {
                    "action": rendered,
                    "stance": "consider",
                    "rationale": "Normalized from free-text action.",
                    "priority": 3,
                    "citations": [],
                }
            )
    return actions


def _coerce_risks_or_alerts(value: Any) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for item in _coerce_list(value):
        if isinstance(item, dict):
            concern = _stringify(item.get("concern") or item.get("risk") or item.get("issue") or item.get("summary"))
            if not concern:
                continue
            alerts.append(
                {
                    "severity": _coerce_severity(item.get("severity")),
                    "concern": concern,
                    "mitigation": _stringify(item.get("mitigation") or item.get("action") or "Review clinically before acting."),
                    "related_actions": _coerce_string_list(item.get("related_actions")),
                }
            )
            continue
        rendered = _stringify(item)
        if rendered:
            alerts.append(
                {
                    "severity": "info",
                    "concern": rendered,
                    "mitigation": "Review clinically before acting.",
                    "related_actions": [],
                }
            )
    return alerts


def _coerce_role(value: Any, default: RoleName) -> RoleName:
    text = _stringify(value).lower()
    return text if text in ROLE_NAMES else default


def _coerce_plan_items(value: Any) -> list[dict[str, Any]]:
    plan_items: list[dict[str, Any]] = []
    for item in _coerce_list(value):
        if isinstance(item, dict):
            action_text = _stringify(item.get("action") or item.get("recommendation") or item.get("name") or item.get("title"))
            if not action_text:
                continue
            plan_items.append(
                {
                    "action": action_text,
                    "owner_role": _coerce_role(item.get("owner_role"), "coordinator"),
                    "rationale": _stringify(item.get("rationale") or item.get("reason") or "Normalized plan item."),
                    "priority": _coerce_priority(item.get("priority")),
                    "score": _coerce_confidence(item.get("score") or item.get("confidence") or 0.5),
                    "citations": _coerce_citations(item.get("citations")),
                }
            )
            continue
        rendered = _stringify(item)
        if rendered:
            plan_items.append(
                {
                    "action": rendered,
                    "owner_role": "coordinator",
                    "rationale": "Normalized plan item.",
                    "priority": 3,
                    "score": 0.5,
                    "citations": [],
                }
            )
    return plan_items


def _coerce_conflicts(value: Any) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for item in _coerce_list(value):
        if not isinstance(item, dict):
            continue
        action_text = _stringify(item.get("action") or item.get("topic") or item.get("subject"))
        if not action_text:
            continue
        outcome = _stringify(item.get("outcome")).lower()
        if outcome not in {"accepted", "rejected", "flagged"}:
            outcome = "flagged"
        conflict_type = _stringify(item.get("conflict_type")).lower()
        if conflict_type not in {"semantic", "risk", "implementation"}:
            conflict_type = "semantic"
        conflicts.append(
            {
                "action": action_text,
                "recommenders": [_coerce_role(role, "coordinator") for role in _coerce_string_list(item.get("recommenders"))],
                "objectors": [_coerce_role(role, "coordinator") for role in _coerce_string_list(item.get("objectors"))],
                "severity": _coerce_severity(item.get("severity")),
                "conflict_type": conflict_type,
                "outcome": outcome,
                "rationale": _stringify(item.get("rationale") or item.get("reason") or "Normalized conflict."),
            }
        )
    return conflicts


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_agent_output_payload(payload: Any, role: SpecialistRoleName) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    candidate_actions = _coerce_candidate_actions(raw.get("candidate_actions"))
    summary = _stringify(raw.get("summary") or raw.get("answer") or raw.get("conclusion"))
    if not summary and candidate_actions:
        summary = candidate_actions[0]["action"]

    return {
        "role": role,
        "summary": summary or f"Normalized {role} output.",
        "structured_findings": _coerce_string_list(raw.get("structured_findings")),
        "candidate_actions": candidate_actions,
        "risks_or_alerts": _coerce_risks_or_alerts(raw.get("risks_or_alerts")),
        "confidence": _coerce_confidence(raw.get("confidence")),
        "supporting_facts": _coerce_supporting_facts(raw.get("supporting_facts")),
    }


def normalize_coordinator_output_payload(payload: Any, case: CaseDossier) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    final_plan = _coerce_plan_items(raw.get("final_plan"))
    accepted_actions = _coerce_string_list(raw.get("accepted_actions"))
    if not accepted_actions and final_plan:
        accepted_actions = [item["action"] for item in final_plan]

    return {
        "role": "coordinator",
        "summary": _stringify(raw.get("summary") or raw.get("conclusion") or "Normalized coordinator output."),
        "final_answer": _stringify(raw.get("final_answer") or raw.get("answer") or raw.get("decision")),
        "final_plan": final_plan,
        "accepted_actions": accepted_actions,
        "rejected_actions": _coerce_string_list(raw.get("rejected_actions")),
        "conflicts": _coerce_conflicts(raw.get("conflicts")),
        "decision_rationale": _stringify(raw.get("decision_rationale") or raw.get("rationale") or raw.get("summary")),
        "audit_trace": _coerce_string_list(raw.get("audit_trace") or raw.get("reasoning_trace")),
        "final_confidence": _coerce_confidence(raw.get("final_confidence") or raw.get("confidence")),
        "supporting_facts": _coerce_supporting_facts(raw.get("supporting_facts")),
    }
