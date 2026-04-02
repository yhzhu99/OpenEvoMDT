from __future__ import annotations

import re

from .models import (
    AgentOutput,
    CaseDossier,
    ConflictRecord,
    CoordinatorDecision,
    CoordinatorOutput,
    PlanItem,
    RiskAlert,
    RoleName,
    SpecialistRoleName,
)


def normalize_action(action: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", action.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def case_role_modifiers(case: CaseDossier) -> dict[SpecialistRoleName, float]:
    modifiers: dict[SpecialistRoleName, float] = {
        "diagnostic": 1.0,
        "treatment": 1.0,
        "safety": 1.0,
        "monitoring": 1.0,
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


def _role_weights_with_modifiers(
    case: CaseDossier,
    role_weights: dict[SpecialistRoleName, float],
) -> dict[SpecialistRoleName, float]:
    modifiers = case_role_modifiers(case)
    return {role: round(role_weights[role] * modifiers[role], 4) for role in role_weights}


def apply_safety_guardrail(
    case: CaseDossier,
    specialist_outputs: list[AgentOutput],
    coordinator_output: CoordinatorOutput,
    role_weights: dict[SpecialistRoleName, float],
) -> CoordinatorDecision:
    outputs_by_role = {output.role: output for output in specialist_outputs}
    safety_output = outputs_by_role["safety"]
    vetoes = _safety_vetoes(safety_output)

    final_plan: list[PlanItem] = []
    accepted_actions: list[str] = []
    rejected_actions = list(dict.fromkeys(coordinator_output.rejected_actions))
    conflicts = list(coordinator_output.conflicts)
    audit_trace = list(coordinator_output.audit_trace)

    for item in coordinator_output.final_plan:
        key = normalize_action(item.action)
        veto = vetoes.get(key)
        if veto is None:
            final_plan.append(item)
            accepted_actions.append(item.action)
            continue

        rejected_actions.append(item.action)
        conflicts.append(
            ConflictRecord(
                action=item.action,
                recommenders=[item.owner_role],
                objectors=["safety"],
                severity=veto.severity,
                conflict_type="risk",
                outcome="rejected",
                rationale=f"Safety guardrail vetoed coordinator plan item: {veto.concern}",
            )
        )
        audit_trace.append(f"Guardrail removed '{item.action}' because Safety flagged {veto.severity} risk.")

    if not accepted_actions:
        for action in coordinator_output.accepted_actions:
            if normalize_action(action) not in vetoes:
                accepted_actions.append(action)
            else:
                rejected_actions.append(action)
                audit_trace.append(f"Guardrail removed accepted action '{action}' because Safety vetoed it.")

    final_plan.sort(key=lambda item: (-item.score, item.priority, item.action))
    accepted_actions = list(dict.fromkeys(accepted_actions))
    rejected_actions = list(dict.fromkeys(rejected_actions))
    effective_weights = _role_weights_with_modifiers(case, role_weights)

    top_action = accepted_actions[0] if accepted_actions else safety_output.summary
    final_answer = coordinator_output.summary if final_plan else top_action

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

    return CoordinatorDecision(
        final_plan=final_plan,
        final_answer=final_answer,
        accepted_actions=accepted_actions,
        rejected_actions=rejected_actions,
        conflicts=conflicts,
        role_weights=effective_weights,
        decision_rationale=coordinator_output.decision_rationale,
        audit_trace=audit_trace,
        response_text="\n".join(response_lines),
        final_confidence=coordinator_output.final_confidence,
    )


def deterministic_coordinator(
    case: CaseDossier,
    specialist_outputs: list[AgentOutput],
    role_weights: dict[SpecialistRoleName, float],
) -> CoordinatorDecision:
    outputs_by_role = {output.role: output for output in specialist_outputs}
    safety_output = outputs_by_role["safety"]
    effective_weights = _role_weights_with_modifiers(case, role_weights)

    plan_items: list[PlanItem] = []
    conflicts: list[ConflictRecord] = []
    audit_trace: list[str] = []

    for output in specialist_outputs:
        for action in output.candidate_actions:
            if action.stance == "avoid":
                conflicts.append(
                    ConflictRecord(
                        action=action.action,
                        recommenders=[],
                        objectors=[output.role],
                        severity="major",
                        conflict_type="risk",
                        outcome="flagged",
                        rationale=action.rationale,
                    )
                )
                continue
            score = round(effective_weights[output.role] * output.confidence, 4)
            plan_items.append(
                PlanItem(
                    action=action.action,
                    owner_role=output.role,
                    rationale=action.rationale,
                    priority=action.priority,
                    score=score,
                    citations=action.citations,
                )
            )
            audit_trace.append(f"Fallback coordinator kept '{action.action}' from {output.role}.")

    fallback = CoordinatorOutput(
        summary="Fallback coordinator summary generated from specialist actions.",
        final_plan=plan_items,
        accepted_actions=[item.action for item in plan_items],
        rejected_actions=[],
        conflicts=conflicts,
        decision_rationale="Fallback deterministic coordinator used because LLM coordinator output was unavailable.",
        audit_trace=audit_trace,
        final_confidence=max((output.confidence for output in specialist_outputs), default=0.0),
        supporting_facts=[],
    )
    return apply_safety_guardrail(case, specialist_outputs, fallback, role_weights)
