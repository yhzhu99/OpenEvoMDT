from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

RoleName = Literal["diagnostic", "treatment", "safety", "monitoring", "coordinator"]
SpecialistRoleName = Literal["diagnostic", "treatment", "safety", "monitoring"]
TaskType = Literal["plan_eval", "generation", "mcq"]
Severity = Literal["absolute", "major", "moderate", "minor", "info"]
Stance = Literal["recommend", "consider", "avoid", "monitor"]
SourceType = Literal["guideline", "literature", "dossier", "other"]
ConflictType = Literal["semantic", "risk", "implementation"]
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


class CitationProvenance(BaseModel):
    source_id: str | None = None
    anchor: str | None = None
    version: str | None = None
    evidence_grade: str | None = None
    year: int | None = None
    source_type: SourceType | None = None


class CaseDossier(BaseModel):
    case_id: str
    domain: str = "oncology"
    cancer_type: str | None = None
    task_type: TaskType = "plan_eval"
    patient_context: PatientContext = Field(default_factory=PatientContext)
    stage: str | None = None
    stage_system: str | None = None
    pathology_summary: str | None = None
    lesions: list[Lesion] = Field(default_factory=list)
    biomarkers: list[Biomarker] = Field(default_factory=list)
    organ_function: dict[str, str] = Field(default_factory=dict)
    comorbidities: list[str] = Field(default_factory=list)
    prior_treatments: list[str] = Field(default_factory=list)
    line_of_therapy: str | None = None
    treatment_intent: str | None = None
    preferences: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    monitoring_context: dict[str, str] = Field(default_factory=dict)
    question: str
    options: list[str] | None = None
    reference_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportingFact(BaseModel):
    field_path: str
    value: str
    note: str
    citations: list[CitationProvenance] = Field(default_factory=list)


class ActionRecommendation(BaseModel):
    action: str
    stance: Stance
    rationale: str
    priority: int = Field(default=3, ge=1, le=5)
    citations: list[CitationProvenance] = Field(default_factory=list)


class RiskAlert(BaseModel):
    severity: Severity
    concern: str
    mitigation: str
    related_actions: list[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    role: SpecialistRoleName
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
    citations: list[CitationProvenance] = Field(default_factory=list)


class ConflictRecord(BaseModel):
    action: str
    recommenders: list[RoleName] = Field(default_factory=list)
    objectors: list[RoleName] = Field(default_factory=list)
    severity: Severity
    conflict_type: ConflictType = "semantic"
    outcome: Literal["accepted", "rejected", "flagged"]
    rationale: str


class CoordinatorOutput(BaseModel):
    role: Literal["coordinator"] = "coordinator"
    summary: str
    final_answer: str = ""
    final_plan: list[PlanItem] = Field(default_factory=list)
    accepted_actions: list[str] = Field(default_factory=list)
    rejected_actions: list[str] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    decision_rationale: str
    audit_trace: list[str] = Field(default_factory=list)
    final_confidence: float = Field(ge=0.0, le=1.0)
    supporting_facts: list[SupportingFact] = Field(default_factory=list)


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
    prompt_updates: dict[RoleName, list[str]] = Field(default_factory=dict)


class EvolutionState(BaseModel):
    role_weights: dict[SpecialistRoleName, float]
    prompt_refinements: dict[RoleName, list[str]]
    history: list[EvolutionEvent] = Field(default_factory=list)


class AgentExecutionTrace(BaseModel):
    role: RoleName
    model_name: str
    raw_content: str
    usage: dict[str, Any] = Field(default_factory=dict)
    parsed_output: AgentOutput | CoordinatorOutput


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
