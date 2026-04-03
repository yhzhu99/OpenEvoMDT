from __future__ import annotations

import math
from statistics import mean
from typing import Any

from .config import AppConfig
from .consensus import normalize_action
from .contracts import is_allowed_answer, normalize_answer, resolve_output_contract
from .models import AgentOutput, CaseDossier, CitationProvenance, CoordinatorDecision, EvaluationResult

PRIORITY_WEIGHTS: dict[int, float] = {
    1: 1.0,
    2: 0.8,
    3: 0.6,
    4: 0.4,
    5: 0.2,
}


def _priority_weight(priority: int) -> float:
    return PRIORITY_WEIGHTS.get(priority, 0.2)


def _reference_plan(case: CaseDossier) -> list[dict[str, Any]]:
    raw_items = case.metadata.get("reference_plan", [])
    items: list[dict[str, Any]] = []
    for raw_item in raw_items:
        if isinstance(raw_item, str):
            items.append({"action": raw_item, "priority": 3})
            continue
        if not isinstance(raw_item, dict):
            continue
        action = raw_item.get("action")
        if not action:
            continue
        items.append(
            {
                "action": str(action),
                "priority": int(raw_item.get("priority", 3)),
            }
        )
    return items


def weighted_plan_concordance(case: CaseDossier, decision: CoordinatorDecision) -> float | None:
    reference_items = _reference_plan(case)
    if not reference_items:
        return None

    predicted_actions = {normalize_action(item.action) for item in decision.final_plan}
    reference_weight_total = sum(_priority_weight(item["priority"]) for item in reference_items)
    if reference_weight_total == 0:
        return None

    matched_weight = 0.0
    for item in reference_items:
        if normalize_action(item["action"]) in predicted_actions:
            matched_weight += _priority_weight(item["priority"])
    return round(matched_weight / reference_weight_total, 4)


def _normalized_response_text(decision: CoordinatorDecision) -> str:
    return normalize_action(decision.response_text)


def guideline_concordance(case: CaseDossier, decision: CoordinatorDecision) -> float | None:
    guideline_tags = case.metadata.get("guideline_tags") or []
    required_constraints = case.metadata.get("required_constraints") or []
    forbidden_actions = case.metadata.get("forbidden_actions") or []
    if not guideline_tags and not required_constraints and not forbidden_actions:
        return None

    checks: list[float] = []
    normalized_text = _normalized_response_text(decision)
    accepted_actions = {normalize_action(action) for action in decision.accepted_actions}

    for tag in guideline_tags:
        tag_key = normalize_action(str(tag))
        checks.append(1.0 if tag_key and tag_key in normalized_text else 0.0)

    for constraint in required_constraints:
        constraint_key = normalize_action(str(constraint))
        if not constraint_key:
            continue
        checks.append(1.0 if constraint_key in normalized_text else 0.0)

    for action in forbidden_actions:
        action_key = normalize_action(str(action))
        if not action_key:
            continue
        checks.append(1.0 if action_key not in accepted_actions else 0.0)

    if not checks:
        return None
    return round(sum(checks) / len(checks), 4)


def safety_violation_rate(outputs: list[AgentOutput], decision: CoordinatorDecision) -> float:
    accepted_actions = [normalize_action(action) for action in decision.accepted_actions]
    if not accepted_actions:
        return 0.0

    outputs_by_role = {output.role: output for output in outputs}
    safety_output = outputs_by_role["safety"]
    vetoed_actions: set[str] = set()
    for alert in safety_output.risks_or_alerts:
        if alert.severity not in {"absolute", "major"}:
            continue
        vetoed_actions.update(normalize_action(action) for action in alert.related_actions)
    for action in safety_output.candidate_actions:
        if action.stance == "avoid":
            vetoed_actions.add(normalize_action(action.action))

    violations = sum(1 for action in accepted_actions if action in vetoed_actions)
    return round(violations / len(accepted_actions), 4)


def _has_provenance(citation: CitationProvenance) -> bool:
    return any(
        value is not None and value != ""
        for value in [
            citation.source_id,
            citation.anchor,
            citation.version,
            citation.evidence_grade,
            citation.year,
            citation.source_type,
        ]
    )


def evidence_traceability_coverage(decision: CoordinatorDecision) -> float:
    if not decision.final_plan:
        return 0.0
    covered_items = 0
    for item in decision.final_plan:
        if any(_has_provenance(citation) for citation in item.citations):
            covered_items += 1
    return round(covered_items / len(decision.final_plan), 4)


def format_compliance(case: CaseDossier, decision: CoordinatorDecision) -> float | None:
    contract = resolve_output_contract(case)
    if contract is None:
        return None
    return float(is_allowed_answer(case, decision.final_answer))


def supporting_fact_coverage(outputs: list[AgentOutput]) -> float:
    if not outputs:
        return 0.0
    covered_roles = sum(1 for output in outputs if output.supporting_facts)
    return round(covered_roles / len(outputs), 4)


def audit_trace_coverage(decision: CoordinatorDecision) -> float:
    return 1.0 if decision.audit_trace else 0.0


def safety_escalation_coverage(outputs: list[AgentOutput], decision: CoordinatorDecision) -> float | None:
    outputs_by_role = {output.role: output for output in outputs}
    safety_output = outputs_by_role.get("safety")
    if safety_output is None:
        return None

    major_alerts = [alert for alert in safety_output.risks_or_alerts if alert.severity in {"absolute", "major"}]
    if not major_alerts:
        return None

    decision_text = normalize_action(
        " ".join(
            [decision.final_answer, decision.decision_rationale, decision.response_text, *decision.audit_trace]
        )
    )
    hits = 0
    for alert in major_alerts:
        related_actions = [normalize_action(action) for action in alert.related_actions if action]
        concern = normalize_action(alert.concern)
        if concern and concern in decision_text:
            hits += 1
            continue
        if any(action and action in decision_text for action in related_actions):
            hits += 1
    return round(hits / len(major_alerts), 4)


def _maybe_bertscore(reference: str, prediction: str, config: AppConfig) -> float | None:
    if not config.evaluation.enable_bertscore:
        return None
    try:
        from bert_score import score as bert_score
    except ImportError:
        return None
    _, _, f1 = bert_score(
        [prediction],
        [reference],
        lang=config.evaluation.bertscore_lang,
        model_type=config.evaluation.bertscore_model_type,
    )
    return round(float(f1.mean().item()), 4)


def _render_summary(metrics: dict[str, float | None], feedback_tags: list[str]) -> str:
    rendered_metrics = ", ".join(
        f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, float) and not math.isnan(value)
    )
    return f"Metrics: {rendered_metrics or 'none'}. Feedback tags: {', '.join(feedback_tags) if feedback_tags else 'none'}."


def _legacy_evaluate_case(
    case: CaseDossier,
    outputs: list[AgentOutput],
    decision: CoordinatorDecision,
    config: AppConfig,
) -> EvaluationResult:
    metrics: dict[str, float | None] = {}

    if case.task_type == "mcq" and case.reference_answer:
        metrics["accuracy"] = float(normalize_answer(decision.final_answer) == normalize_answer(case.reference_answer))

    if case.task_type == "generation" and case.reference_answer:
        metrics["bertscore_f1"] = _maybe_bertscore(case.reference_answer, decision.response_text, config)

    if case.task_type == "plan_eval":
        metrics["weighted_plan_concordance"] = weighted_plan_concordance(case, decision)
        metrics["guideline_concordance"] = guideline_concordance(case, decision)
        metrics["safety_violation_rate"] = safety_violation_rate(outputs, decision)
        metrics["evidence_traceability_coverage"] = evidence_traceability_coverage(decision)

    feedback_tags: list[str] = []
    wpc = metrics.get("weighted_plan_concordance")
    if isinstance(wpc, float) and wpc < 0.6:
        feedback_tags.append("treatment_gap")

    concordance = metrics.get("guideline_concordance")
    if isinstance(concordance, float) and concordance < 0.75:
        feedback_tags.append("completeness_gap")

    safety_rate = metrics.get("safety_violation_rate")
    if isinstance(safety_rate, float) and safety_rate > 0.0:
        feedback_tags.append("safety_gap")

    traceability = metrics.get("evidence_traceability_coverage")
    if isinstance(traceability, float) and traceability < 0.5:
        feedback_tags.append("evidence_gap")

    if not any(output.role == "monitoring" and output.candidate_actions for output in outputs):
        feedback_tags.append("monitoring_gap")

    bertscore = metrics.get("bertscore_f1")
    if case.task_type == "generation" and isinstance(bertscore, float) and bertscore < 0.6:
        feedback_tags.append("treatment_gap")

    accuracy = metrics.get("accuracy")
    if accuracy == 0.0:
        feedback_tags.append("treatment_gap")

    feedback_tags = list(dict.fromkeys(feedback_tags))
    return EvaluationResult(
        metrics=metrics,
        dimension_scores={},
        feedback_tags=feedback_tags,
        summary=_render_summary(metrics, feedback_tags),
    )


def _biomedical_generation_evaluate_case(
    case: CaseDossier,
    outputs: list[AgentOutput],
    decision: CoordinatorDecision,
    config: AppConfig,
) -> EvaluationResult:
    del config

    metrics: dict[str, float | None] = {
        "format_compliance": format_compliance(case, decision),
        "supporting_fact_coverage": supporting_fact_coverage(outputs),
        "audit_trace_coverage": audit_trace_coverage(decision),
        "safety_escalation_coverage": safety_escalation_coverage(outputs, decision),
    }
    if case.reference_answer:
        metrics["accuracy"] = float(normalize_answer(decision.final_answer) == normalize_answer(case.reference_answer))

    traceability_components = [
        value
        for value in [metrics["supporting_fact_coverage"], metrics["audit_trace_coverage"]]
        if isinstance(value, float)
    ]
    if traceability_components:
        metrics["traceability_score"] = round(sum(traceability_components) / len(traceability_components), 4)

    feedback_tags: list[str] = []
    format_value = metrics.get("format_compliance")
    if format_value == 0.0:
        feedback_tags.append("format_gap")

    accuracy = metrics.get("accuracy")
    if accuracy == 0.0:
        feedback_tags.append("accuracy_gap")

    traceability = metrics.get("traceability_score")
    if isinstance(traceability, float) and traceability < 0.75:
        feedback_tags.append("traceability_gap")

    safety_coverage = metrics.get("safety_escalation_coverage")
    if isinstance(safety_coverage, float) and safety_coverage < 1.0:
        feedback_tags.append("safety_gap")

    feedback_tags = list(dict.fromkeys(feedback_tags))
    return EvaluationResult(
        metrics=metrics,
        dimension_scores={},
        feedback_tags=feedback_tags,
        summary=_render_summary(metrics, feedback_tags),
    )


def evaluate_case(
    case: CaseDossier,
    outputs: list[AgentOutput],
    decision: CoordinatorDecision,
    config: AppConfig,
) -> EvaluationResult:
    if case.task_type == "generation" and (case.domain == "biomedical_qa" or resolve_output_contract(case) is not None):
        return _biomedical_generation_evaluate_case(case, outputs, decision, config)
    return _legacy_evaluate_case(case, outputs, decision, config)


def aggregate_benchmark_results(results: list[EvaluationResult]) -> dict[str, Any]:
    aggregate: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.metrics.items():
            if value is None or isinstance(value, bool) or math.isnan(value):
                continue
            aggregate.setdefault(key, []).append(float(value))
    return {metric: round(mean(values), 4) for metric, values in aggregate.items() if values}
