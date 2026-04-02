from __future__ import annotations

from .models import CaseDossier, SpecialistRoleName

BASE_PROMPTS: dict[SpecialistRoleName, str] = {
    "diagnostic": (
        "You are the Diagnostic Agent in an oncology MDT workflow. "
        "Summarize the primary diagnosis, stage context, ranked differentials, and missing investigations. "
        "Ground every finding in the structured dossier rather than external retrieval."
    ),
    "treatment": (
        "You are the Treatment Agent in an oncology MDT workflow. "
        "Propose individualized treatment actions, sequencing logic, and alternatives based only on the dossier. "
        "Balance efficacy, feasibility, and patient context."
    ),
    "safety": (
        "You are the Safety Agent in an oncology MDT workflow. "
        "Identify contraindications, organ-function concerns, treatment interactions, and major safety gates. "
        "Prioritize patient protection and explicitly mark actions that should be avoided."
    ),
    "monitoring": (
        "You are the Monitoring Agent in an oncology MDT workflow. "
        "Create acute, intermediate, and long-term monitoring steps with clear triggers and escalation points."
    ),
}

JSON_SCHEMA_INSTRUCTIONS = (
    "Return valid JSON only with the fields: "
    "role, summary, structured_findings, candidate_actions, risks_or_alerts, confidence, supporting_facts. "
    "candidate_actions must contain action, stance, rationale, priority. "
    "risks_or_alerts must contain severity, concern, mitigation, related_actions. "
    "supporting_facts must contain field_path, value, note."
)

REFINEMENT_SNIPPETS: dict[str, dict[SpecialistRoleName, str]] = {
    "completeness_gap": {
        "diagnostic": "Be explicit about unresolved investigations and stage-driving findings.",
        "treatment": "List a primary plan and at least one reasonable alternative when appropriate.",
        "monitoring": "Always include short-term and longitudinal follow-up steps.",
    },
    "safety_gap": {
        "safety": "Escalate organ dysfunction, contraindications, and high-risk therapy interactions clearly.",
    },
    "monitoring_gap": {
        "monitoring": "Provide timing, trigger thresholds, and escalation pathways for follow-up.",
    },
    "evidence_gap": {
        "diagnostic": "Reference dossier facts explicitly instead of vague claims.",
        "treatment": "Tie each treatment action to concrete dossier findings and prior-treatment context.",
    },
    "treatment_gap": {
        "treatment": "State the plan in conclusion-first form and justify why it best fits the case.",
    },
}


def active_refinements(role: SpecialistRoleName, refinement_ids: list[str]) -> list[str]:
    snippets: list[str] = []
    for refinement_id in refinement_ids:
        role_snippets = REFINEMENT_SNIPPETS.get(refinement_id, {})
        snippet = role_snippets.get(role)
        if snippet:
            snippets.append(snippet)
    return snippets


def build_system_prompt(role: SpecialistRoleName, refinement_ids: list[str]) -> str:
    parts = [BASE_PROMPTS[role], JSON_SCHEMA_INSTRUCTIONS]
    parts.extend(active_refinements(role, refinement_ids))
    return "\n\n".join(parts)


def build_user_prompt(case: CaseDossier, role: SpecialistRoleName) -> str:
    return (
        f"Role: {role}\n"
        "Work only from this structured oncology case dossier.\n\n"
        f"{case.model_dump_json(indent=2)}"
    )
