from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .io import read_json, write_json
from .models import (
    EvaluationResult,
    EvolutionEvent,
    EvolutionState,
    RoleName,
    SPECIALIST_ROLES,
    SpecialistRoleName,
    utc_timestamp,
)
from .prompts import REFINEMENT_SNIPPETS

WEIGHT_UPDATE_RULES: dict[str, dict[SpecialistRoleName, float]] = {
    "completeness_gap": {"diagnostic": 1.0, "treatment": 1.0, "monitoring": 1.0},
    "safety_gap": {"safety": 1.0},
    "monitoring_gap": {"monitoring": 1.0},
    "evidence_gap": {"diagnostic": 0.5, "treatment": 0.5},
    "treatment_gap": {"treatment": 1.0},
    "accuracy_gap": {"diagnostic": 1.0, "treatment": 1.0},
    "traceability_gap": {"diagnostic": 0.5, "treatment": 0.5},
}
PROMPT_REFINEMENT_ROLES: tuple[RoleName, ...] = (*SPECIALIST_ROLES, "coordinator")


def initial_state(config: AppConfig) -> EvolutionState:
    return EvolutionState(
        role_weights={
            role: config.agents.roles[role].weight
            for role in SPECIALIST_ROLES
            if config.agents.roles[role].weight is not None
        },
        prompt_refinements={role: [] for role in PROMPT_REFINEMENT_ROLES},
        history=[],
    )


def load_state(config: AppConfig) -> EvolutionState:
    path = config.evolution_state_path
    if not path.exists():
        return initial_state(config)
    raw_state = read_json(path)
    raw_state["role_weights"] = {
        role: value
        for role, value in raw_state.get("role_weights", {}).items()
        if role in SPECIALIST_ROLES
    }
    raw_state["prompt_refinements"] = {
        role: value
        for role, value in raw_state.get("prompt_refinements", {}).items()
        if role in PROMPT_REFINEMENT_ROLES
    }
    for event in raw_state.get("history", []):
        event["weight_updates"] = {
            role: value
            for role, value in event.get("weight_updates", {}).items()
            if role in SPECIALIST_ROLES
        }
        event["prompt_updates"] = {
            role: value
            for role, value in event.get("prompt_updates", {}).items()
            if role in PROMPT_REFINEMENT_ROLES
        }
    state = EvolutionState.model_validate(raw_state)
    baseline = initial_state(config)
    for role in SPECIALIST_ROLES:
        state.role_weights.setdefault(role, baseline.role_weights[role])
    for role in PROMPT_REFINEMENT_ROLES:
        state.prompt_refinements.setdefault(role, [])
    return state


def save_state(config: AppConfig, state: EvolutionState) -> Path:
    return write_json(config.evolution_state_path, state.model_dump())


def apply_feedback(config: AppConfig, state: EvolutionState, evaluation: EvaluationResult) -> EvolutionState:
    if not config.evolution.enabled or not evaluation.feedback_tags:
        return state

    updated = state.model_copy(deep=True)
    weight_updates: dict[SpecialistRoleName, float] = {}
    prompt_updates: dict[RoleName, list[str]] = {}

    for tag in evaluation.feedback_tags:
        for role, factor in WEIGHT_UPDATE_RULES.get(tag, {}).items():
            delta = round(config.evolution.weight_step * factor, 4)
            current = updated.role_weights[role]
            baseline = config.agents.roles[role].weight
            upper = baseline + config.evolution.max_weight_delta
            updated.role_weights[role] = min(upper, round(current + delta, 4))
            weight_updates[role] = updated.role_weights[role]

        role_snippets = REFINEMENT_SNIPPETS.get(tag, {})
        for role in role_snippets:
            if role not in PROMPT_REFINEMENT_ROLES:
                continue
            prompts = updated.prompt_refinements.setdefault(role, [])
            if tag not in prompts:
                prompts.append(tag)
                prompt_updates.setdefault(role, []).append(tag)

    updated.history.append(
        EvolutionEvent(
            timestamp=utc_timestamp(),
            feedback_tags=evaluation.feedback_tags,
            weight_updates=weight_updates,
            prompt_updates=prompt_updates,
        )
    )
    updated.history = updated.history[-config.evolution.max_history :]
    return updated
