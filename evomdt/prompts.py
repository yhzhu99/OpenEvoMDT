from __future__ import annotations

import json

from .models import AgentOutput, CaseDossier, SpecialistRoleName

BASE_PROMPTS: dict[SpecialistRoleName, str] = {
    "diagnostic": (
        "You are the Diagnostic Agent in an oncology MDT workflow. "
        "Produce a lesion-level diagnostic summary with the primary diagnosis, stage context, ranked differentials, "
        "key missing investigations, and calibrated confidence. "
        "Use only the dossier. Do not claim retrieval, guidelines, or literature that you did not receive."
    ),
    "treatment": (
        "You are the Treatment Agent in an oncology MDT workflow. "
        "Produce an individualized treatment plan with sequencing logic, alternatives, line-of-therapy awareness, "
        "and explicit fit to stage, biomarkers, organ reserve, and patient preferences. "
        "Use only the dossier. Do not invent external citations."
    ),
    "safety": (
        "You are the Safety Agent in an oncology MDT workflow. "
        "Identify contraindications, organ-function concerns, treatment interactions, and major safety gates. "
        "Prioritize patient protection. Any action that is clearly unsafe should be marked as avoid with a major or absolute risk alert."
    ),
    "monitoring": (
        "You are the Monitoring Agent in an oncology MDT workflow. "
        "Create acute, intermediate, and long-term monitoring steps with clear timing, trigger thresholds, and escalation points."
    ),
}

COORDINATOR_PROMPT = (
    "You are the Coordinator Agent in an oncology MDT workflow. "
    "Synthesize the Diagnostic, Treatment, Safety, and Monitoring outputs into a single MDT-style recommendation. "
    "Resolve semantic, risk, and implementation conflicts explicitly. Respect the Safety agent's major concerns, "
    "favor implementable plans, and keep uncertainty calibrated. "
    "Use only the dossier and specialist outputs provided. Do not invent retrieval results or citations. "
    "If provenance is not available, leave citation lists empty."
)

SPECIALIST_JSON_SCHEMA_INSTRUCTIONS = (
    "Return valid JSON only with the fields: "
    "role, summary, structured_findings, candidate_actions, risks_or_alerts, confidence, supporting_facts. "
    "candidate_actions must contain action, stance, rationale, priority, citations. "
    "risks_or_alerts must contain severity, concern, mitigation, related_actions. "
    "supporting_facts must contain field_path, value, note, citations. "
    "Each citation object must contain source_id, anchor, version, evidence_grade, year, source_type. "
    "When you do not have provenance, return citations as an empty list."
)

COORDINATOR_JSON_SCHEMA_INSTRUCTIONS = (
    "Return valid JSON only with the fields: "
    "role, summary, final_plan, accepted_actions, rejected_actions, conflicts, decision_rationale, audit_trace, final_confidence, supporting_facts. "
    "final_plan items must contain action, owner_role, rationale, priority, score, citations. "
    "conflicts must contain action, recommenders, objectors, severity, conflict_type, outcome, rationale. "
    "supporting_facts must contain field_path, value, note, citations. "
    "Each citation object must contain source_id, anchor, version, evidence_grade, year, source_type. "
    "Do not include markdown fences."
)

NO_RAG_INSTRUCTIONS = (
    "This run is no-RAG. You must not claim to have retrieved documents, guidelines, or papers. "
    "You may reason from the structured dossier and prior agent outputs only."
)

REFINEMENT_SNIPPETS: dict[str, dict[SpecialistRoleName, str]] = {
    "completeness_gap": {
        "diagnostic": "Be explicit about unresolved investigations, stage-driving findings, and missing data that block certainty.",
        "treatment": "List a primary plan and at least one reasonable alternative when appropriate.",
        "monitoring": "Always include near-term and longitudinal follow-up steps.",
    },
    "safety_gap": {
        "safety": "Escalate organ dysfunction, contraindications, and high-risk interactions clearly, with explicit avoid actions.",
    },
    "monitoring_gap": {
        "monitoring": "Provide timing, trigger thresholds, and escalation pathways for follow-up.",
    },
    "evidence_gap": {
        "diagnostic": "Tie each key claim to explicit dossier fields and leave citations empty instead of inventing them.",
        "treatment": "Tie each treatment action to concrete dossier findings and leave citations empty unless provenance was provided upstream.",
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
    parts = [BASE_PROMPTS[role], SPECIALIST_JSON_SCHEMA_INSTRUCTIONS, NO_RAG_INSTRUCTIONS]
    parts.extend(active_refinements(role, refinement_ids))
    return "\n\n".join(parts)


def build_user_prompt(case: CaseDossier, role: SpecialistRoleName) -> str:
    return (
        f"Role: {role}\n"
        "Work only from this structured oncology case dossier.\n\n"
        f"{case.model_dump_json(indent=2)}"
    )


def build_coordinator_system_prompt() -> str:
    return "\n\n".join([COORDINATOR_PROMPT, COORDINATOR_JSON_SCHEMA_INSTRUCTIONS, NO_RAG_INSTRUCTIONS])


def build_coordinator_user_prompt(case: CaseDossier, specialist_outputs: list[AgentOutput]) -> str:
    payload = {
        "case": case.model_dump(mode="json"),
        "specialist_outputs": [output.model_dump(mode="json") for output in specialist_outputs],
    }
    return (
        "Role: coordinator\n"
        "Synthesize the specialist outputs into a final MDT-style plan.\n"
        "Use only this payload.\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )
