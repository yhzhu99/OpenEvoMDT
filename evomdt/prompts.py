from __future__ import annotations

import json
from dataclasses import dataclass

from .contracts import output_contract_instructions
from .models import AgentOutput, CaseDossier, RoleName, SpecialistRoleName


@dataclass(frozen=True)
class PromptProfile:
    specialist_prompts: dict[SpecialistRoleName, str]
    coordinator_prompt: str
    dossier_label: str


PROMPT_PROFILES: dict[str, PromptProfile] = {
    "oncology": PromptProfile(
        specialist_prompts={
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
        },
        coordinator_prompt=(
            "You are the Coordinator Agent in an oncology MDT workflow. "
            "Synthesize the Diagnostic, Treatment, Safety, and Monitoring outputs into a single MDT-style recommendation. "
            "Resolve semantic, risk, and implementation conflicts explicitly. Respect the Safety agent's major concerns, "
            "favor implementable plans, and keep uncertainty calibrated. "
            "Use only the dossier and specialist outputs provided. Do not invent retrieval results or citations. "
            "If provenance is not available, leave citation lists empty."
        ),
        dossier_label="structured oncology case dossier",
    ),
    "biomedical_qa": PromptProfile(
        specialist_prompts={
            "diagnostic": (
                "You are the Diagnostic Agent in a biomedical QA workflow. "
                "Extract the key stated facts, identify missing context, and separate observed evidence from assumptions. "
                "Your goal is to ground the later answer in the dossier rather than to make unsupported clinical claims."
            ),
            "treatment": (
                "You are the Treatment Agent in a biomedical QA workflow. "
                "Produce the strongest answer candidate allowed by the dossier, explain why it fits the facts, and note competing interpretations. "
                "Use only the dossier and do not invent outside evidence."
            ),
            "safety": (
                "You are the Safety Agent in a biomedical QA workflow. "
                "Challenge overconfident reasoning, identify dangerous assumptions, and flag when the dossier is too weak for a strong conclusion. "
                "Any conclusion that could cause harmful overstatement should be marked as avoid with a major or absolute risk alert."
            ),
            "monitoring": (
                "You are the Monitoring Agent in a biomedical QA workflow. "
                "Identify what follow-up information, observation windows, and escalation triggers would matter for a safer answer. "
                "If urgent action or observation is warranted, express it as candidate actions."
            ),
        },
        coordinator_prompt=(
            "You are the Coordinator Agent in a biomedical QA workflow. "
            "Synthesize the Diagnostic, Treatment, Safety, and Monitoring outputs into one constrained final answer. "
            "Resolve conflicts explicitly, preserve uncertainty, and follow any output contract exactly. "
            "Use only the dossier and specialist outputs provided. Do not invent retrieval results or citations."
        ),
        dossier_label="structured biomedical task dossier",
    ),
}

SPECIALIST_JSON_SCHEMA_INSTRUCTIONS = (
    "Return valid JSON only with the fields: "
    "role, summary, structured_findings, candidate_actions, risks_or_alerts, confidence, supporting_facts. "
    "structured_findings must be a JSON array of strings. "
    "candidate_actions must contain action, stance, rationale, priority, citations. "
    "Use stance exactly as one of recommend, consider, avoid, monitor. "
    "Use priority as an integer from 1 to 5. "
    "risks_or_alerts must contain severity, concern, mitigation, related_actions. "
    "Use severity exactly as one of absolute, major, moderate, minor, info. "
    "supporting_facts must contain field_path, value, note, citations. "
    "Use confidence as a numeric value from 0.0 to 1.0. "
    "Each citation object must contain source_id, anchor, version, evidence_grade, year, source_type. "
    "When you do not have provenance, return citations as an empty list."
)

COORDINATOR_JSON_SCHEMA_INSTRUCTIONS = (
    "Return valid JSON only with the fields: "
    "role, summary, final_answer, final_plan, accepted_actions, rejected_actions, conflicts, decision_rationale, audit_trace, final_confidence, supporting_facts. "
    "final_answer must contain the exact final answer string. "
    "final_plan items must contain action, owner_role, rationale, priority, score, citations. "
    "Use owner_role exactly as one of diagnostic, treatment, safety, monitoring, coordinator. "
    "Use priority as an integer from 1 to 5 and score as a numeric value from 0.0 to 1.0. "
    "conflicts must contain action, recommenders, objectors, severity, conflict_type, outcome, rationale. "
    "supporting_facts must contain field_path, value, note, citations. "
    "Use final_confidence as a numeric value from 0.0 to 1.0. "
    "Each citation object must contain source_id, anchor, version, evidence_grade, year, source_type. "
    "Do not include markdown fences."
)

NO_RAG_INSTRUCTIONS = (
    "This run is no-RAG. You must not claim to have retrieved documents, guidelines, or papers. "
    "You may reason from the structured dossier and prior agent outputs only."
)

REFINEMENT_SNIPPETS: dict[str, dict[RoleName, str]] = {
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
    "accuracy_gap": {
        "diagnostic": "Distinguish direct dossier facts from inferred claims so the final answer can be defended cleanly.",
        "treatment": "Make the answer candidate explicit, concise, and tightly tied to the stated facts.",
    },
    "traceability_gap": {
        "diagnostic": "Enumerate the dossier fields that most directly support or weaken the answer.",
        "treatment": "Anchor the answer rationale to the strongest supporting facts and surface key missing data.",
    },
    "format_gap": {
        "coordinator": "Follow the output contract exactly and set final_answer to one allowed value with no paraphrase.",
    },
}


def prompt_profile_for(case: CaseDossier) -> PromptProfile:
    return PROMPT_PROFILES.get(case.domain, PROMPT_PROFILES["oncology"])


def active_refinements(role: RoleName, refinement_ids: list[str]) -> list[str]:
    snippets: list[str] = []
    for refinement_id in refinement_ids:
        role_snippets = REFINEMENT_SNIPPETS.get(refinement_id, {})
        snippet = role_snippets.get(role)
        if snippet:
            snippets.append(snippet)
    return snippets


def _contract_parts(case: CaseDossier) -> list[str]:
    contract_text = output_contract_instructions(case)
    return [contract_text] if contract_text else []


def build_system_prompt(case: CaseDossier, role: SpecialistRoleName, refinement_ids: list[str]) -> str:
    profile = prompt_profile_for(case)
    parts = [profile.specialist_prompts[role], SPECIALIST_JSON_SCHEMA_INSTRUCTIONS, NO_RAG_INSTRUCTIONS]
    parts.extend(_contract_parts(case))
    parts.extend(active_refinements(role, refinement_ids))
    return "\n\n".join(parts)


def build_user_prompt(case: CaseDossier, role: SpecialistRoleName) -> str:
    profile = prompt_profile_for(case)
    contract_text = output_contract_instructions(case)
    prompt = (
        f"Role: {role}\n"
        f"Work only from this {profile.dossier_label}.\n\n"
        f"{case.model_dump_json(indent=2)}"
    )
    if not contract_text:
        return prompt
    return f"{prompt}\n\nTask output contract:\n{contract_text}"


def build_coordinator_system_prompt(case: CaseDossier, refinement_ids: list[str]) -> str:
    profile = prompt_profile_for(case)
    parts = [profile.coordinator_prompt, COORDINATOR_JSON_SCHEMA_INSTRUCTIONS, NO_RAG_INSTRUCTIONS]
    parts.extend(_contract_parts(case))
    parts.extend(active_refinements("coordinator", refinement_ids))
    return "\n\n".join(parts)


def build_coordinator_user_prompt(case: CaseDossier, specialist_outputs: list[AgentOutput]) -> str:
    payload = {
        "case": case.model_dump(mode="json"),
        "specialist_outputs": [output.model_dump(mode="json") for output in specialist_outputs],
    }
    contract_text = output_contract_instructions(case)
    body = (
        "Role: coordinator\n"
        "Synthesize the specialist outputs into a final answer and supporting plan.\n"
        "Use only this payload.\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )
    if not contract_text:
        return body
    return f"{body}\n\nTask output contract:\n{contract_text}"
