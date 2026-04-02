from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean

from .models import (
    ActionRecommendation,
    AgentOutput,
    CaseDossier,
    ConflictRecord,
    CoordinatorDecision,
    PlanItem,
    RiskAlert,
    RoleName,
)


def normalize_action(action: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", action.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def case_role_modifiers(case: CaseDossier) -> dict[RoleName, float]:
    modifiers: dict[RoleName, float] = {
        "diagnostic": 1.0,
        "treatment": 1.0,
        "safety": 1.0,
        "monitoring": 1.0,
        "coordinator": 1.0,
    }
    if len(case.lesions) > 1 or len(case.biomarkers) > 2:
        modifiers["diagnostic"] += 0.15
    if case.prior_treatments:
        modifiers["treatment"] += 0.15
    if case.patient_context.performance_status and any(
        marker in case.patient_context.performance_status.lower() for marker in ["2", "3", "4", "poor"]
    ):
        modifiers["safety"] += 0.2
    if any("child" in value.lower() or "bilirubin" in key.lower() for key, value in case.organ_function.items()):
        modifiers["safety"] += 0.15
    if "met" in case.question.lower() or "follow-up" in case.question.lower():
        modifiers["monitoring"] += 0.1
    return modifiers


def _safety_vetoes(safety_output: AgentOutput) -> dict[str, RiskAlert]:
    vetoes: dict[str, RiskAlert] = {}
    for alert in safety_output.risks_or_alerts:
        if alert.severity not in {"absolute", "major"}:
            continue
        for related_action in alert.related_actions:
            vetoes[normalize_action(related_action)] = alert
    for action in safety_output.candidate_actions:
        if action.stance == "avoid":
            vetoes.setdefault(
                normalize_action(action.action),
                RiskAlert(
                    severity="major",
                    concern=action.rationale,
                    mitigation="Avoid this action unless new data changes the risk profile.",
                    related_actions=[action.action],
                ),
            )
    return vetoes


def deterministic_coordinator(
    case: CaseDossier,
    specialist_outputs: list[AgentOutput],
    role_weights: dict[RoleName, float],
) -> CoordinatorDecision:
    outputs_by_role = {output.role: output for output in specialist_outputs}
    modifiers = case_role_modifiers(case)
    effective_weights = {
        role: round(role_weights[role] * modifiers[role], 4)
        for role in role_weights
    }
    safety_output = outputs_by_role["safety"]
    vetoes = _safety_vetoes(safety_output)

    vote_scores: dict[str, float] = defaultdict(float)
    vote_reasons: dict[str, list[str]] = defaultdict(list)
    vote_roles: dict[str, list[RoleName]] = defaultdict(list)
    vote_actions: dict[str, ActionRecommendation] = {}
    objections: dict[str, list[RoleName]] = defaultdict(list)
    conflicts: list[ConflictRecord] = []

    for output in specialist_outputs:
        role_weight = effective_weights[output.role] * output.confidence
        for action in output.candidate_actions:
            key = normalize_action(action.action)
            vote_actions.setdefault(key, action)
            vote_reasons[key].append(f"{output.role}: {action.rationale}")
            if action.stance in {"recommend", "consider", "monitor"}:
                vote_scores[key] += role_weight * (1.0 if action.stance == "recommend" else 0.7)
                vote_roles[key].append(output.role)
            elif action.stance == "avoid":
                vote_scores[key] -= role_weight
                objections[key].append(output.role)

    final_plan: list[PlanItem] = []
    accepted_actions: list[str] = []
    rejected_actions: list[str] = []
    audit_trace: list[str] = []

    for key, score in sorted(vote_scores.items(), key=lambda item: item[1], reverse=True):
        action = vote_actions[key]
        veto = vetoes.get(key)
        objectors = objections.get(key, [])

        if veto is not None:
            rejected_actions.append(action.action)
            conflicts.append(
                ConflictRecord(
                    action=action.action,
                    recommenders=vote_roles[key],
                    objectors=["safety"],
                    severity=veto.severity,
                    outcome="rejected",
                    rationale=f"Safety-first veto: {veto.concern}",
                )
            )
            audit_trace.append(f"Rejected '{action.action}' because Safety flagged {veto.severity} risk.")
            continue

        if score <= 0:
            rejected_actions.append(action.action)
            if objectors:
                conflicts.append(
                    ConflictRecord(
                        action=action.action,
                        recommenders=vote_roles[key],
                        objectors=objectors,
                        severity="moderate",
                        outcome="rejected",
                        rationale="Negative weighted consensus after objections.",
                    )
                )
            audit_trace.append(f"Rejected '{action.action}' because the weighted consensus score was {score:.2f}.")
            continue

        if objectors:
            conflicts.append(
                ConflictRecord(
                    action=action.action,
                    recommenders=vote_roles[key],
                    objectors=objectors,
                    severity="major" if "safety" in objectors else "moderate",
                    outcome="accepted",
                    rationale="Accepted after positive weighted consensus despite objections.",
                )
            )
        accepted_actions.append(action.action)
        final_plan.append(
            PlanItem(
                action=action.action,
                owner_role=vote_roles[key][0] if vote_roles[key] else "coordinator",
                rationale=" | ".join(vote_reasons[key]),
                priority=action.priority,
                score=round(score, 4),
            )
        )
        audit_trace.append(f"Accepted '{action.action}' with weighted consensus score {score:.2f}.")

    final_plan.sort(key=lambda item: (-item.score, item.priority))
    final_answer = accepted_actions[0] if accepted_actions else safety_output.summary
    confidence_values = [output.confidence for output in specialist_outputs]
    final_confidence = round(mean(confidence_values), 4) if confidence_values else 0.0

    response_lines = [
        f"Case: {case.case_id} ({case.cancer_type})",
        f"Question: {case.question}",
        "",
        "Final MDT Recommendation:",
    ]
    if final_plan:
        for item in final_plan:
            response_lines.append(f"- {item.action} [{item.owner_role}, score={item.score:.2f}]")
    else:
        response_lines.append("- No safe consensus action was accepted.")
    if safety_output.risks_or_alerts:
        response_lines.append("")
        response_lines.append("Safety Signals:")
        for alert in safety_output.risks_or_alerts:
            response_lines.append(f"- {alert.severity.upper()}: {alert.concern} | Mitigation: {alert.mitigation}")
    response_lines.append("")
    response_lines.append("Audit Trace:")
    response_lines.extend(f"- {entry}" for entry in audit_trace)

    decision_rationale = (
        "Deterministic coordinator combined specialist confidence with role weights and case modifiers, "
        "while enforcing safety-first vetoes before ranking the surviving actions."
    )
    return CoordinatorDecision(
        final_plan=final_plan,
        final_answer=final_answer,
        accepted_actions=accepted_actions,
        rejected_actions=rejected_actions,
        conflicts=conflicts,
        role_weights=effective_weights,
        decision_rationale=decision_rationale,
        audit_trace=audit_trace,
        response_text="\n".join(response_lines),
        final_confidence=final_confidence,
    )
