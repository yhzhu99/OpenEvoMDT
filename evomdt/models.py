from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

RoleName = Literal["diagnostic", "treatment", "safety", "monitoring", "coordinator"]
SpecialistRoleName = Literal["diagnostic", "treatment", "safety", "monitoring"]
TaskType = Literal["plan_eval", "generation", "mcq"]
Severity = Literal["absolute", "major", "moderate", "minor", "info"]
Stance = Literal["recommend", "consider", "avoid", "monitor"]
SPECIALIST_ROLES: tuple[SpecialistRoleName, ...] = ("diagnostic", "treatment", "safety", "monitoring")


class PatientContext(BaseModel):
    age: int | None = None
    sex: str | None = None
    performance_status: str | None = None
    summary: str | None = None


class Lesion(BaseModel):
    site: str
    size_cm: float | None = None
    description: str | None = None
    stage_context: str | None = None


class Biomarker(BaseModel):
    name: str
    value: str


class CaseDossier(BaseModel):
    case_id: str
    cancer_type: str
    task_type: TaskType = "plan_eval"
    patient_context: PatientContext = Field(default_factory=PatientContext)
    lesions: list[Lesion] = Field(default_factory=list)
    biomarkers: list[Biomarker] = Field(default_factory=list)
    organ_function: dict[str, str] = Field(default_factory=dict)
    comorbidities: list[str] = Field(default_factory=list)
    prior_treatments: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    question: str
    options: list[str] | None = None
    reference_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportingFact(BaseModel):
    field_path: str
    value: str
    note: str


class ActionRecommendation(BaseModel):
    action: str
    stance: Stance
    rationale: str
    priority: int = Field(default=3, ge=1, le=5)


class RiskAlert(BaseModel):
    severity: Severity
    concern: str
    mitigation: str
    related_actions: list[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    role: RoleName
    summary: str
    structured_findings: list[str] = Field(default_factory=list)
    candidate_actions: list[ActionRecommendation] = Field(default_factory=list)
    risks_or_alerts: list[RiskAlert] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_facts: list[SupportingFact] = Field(default_factory=list)


class PlanItem(BaseModel):
    action: str
    owner_role: RoleName
    rationale: str
    priority: int
    score: float


class ConflictRecord(BaseModel):
    action: str
    recommenders: list[RoleName] = Field(default_factory=list)
    objectors: list[RoleName] = Field(default_factory=list)
    severity: Severity
    outcome: Literal["accepted", "rejected", "flagged"]
    rationale: str


class CoordinatorDecision(BaseModel):
    final_plan: list[PlanItem] = Field(default_factory=list)
    final_answer: str
    accepted_actions: list[str] = Field(default_factory=list)
    rejected_actions: list[str] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    role_weights: dict[SpecialistRoleName, float]
    decision_rationale: str
    audit_trace: list[str] = Field(default_factory=list)
    response_text: str
    final_confidence: float = Field(ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    metrics: dict[str, float | None] = Field(default_factory=dict)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    feedback_tags: list[str] = Field(default_factory=list)
    summary: str


class EvolutionEvent(BaseModel):
    timestamp: str
    feedback_tags: list[str]
    weight_updates: dict[SpecialistRoleName, float] = Field(default_factory=dict)
    prompt_updates: dict[SpecialistRoleName, list[str]] = Field(default_factory=dict)


class EvolutionState(BaseModel):
    role_weights: dict[SpecialistRoleName, float]
    prompt_refinements: dict[SpecialistRoleName, list[str]]
    history: list[EvolutionEvent] = Field(default_factory=list)


class AgentExecutionTrace(BaseModel):
    role: RoleName
    model_name: str
    raw_content: str
    usage: dict[str, Any] = Field(default_factory=dict)
    parsed_output: AgentOutput


class CaseRunResult(BaseModel):
    run_id: str
    case: CaseDossier
    agent_traces: list[AgentExecutionTrace]
    coordinator_decision: CoordinatorDecision
    evaluation: EvaluationResult | None = None
    duration_seconds: float
    artifacts_dir: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
