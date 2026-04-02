from __future__ import annotations

import math
import re
from statistics import mean
from typing import Any

from rouge_score import rouge_scorer

from .config import AppConfig
from .models import AgentOutput, CaseDossier, CoordinatorDecision, EvaluationResult

DIMENSION_NAMES = [
    "clinical_appropriateness",
    "response_completeness",
    "response_efficiency",
    "evidence_utilization",
    "patient_centeredness",
    "safety_ethics",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_f1(reference: str, prediction: str) -> float:
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = sum(min(ref_counts.get(token, 0), pred_counts.get(token, 0)) for token in set(ref_counts) | set(pred_counts))
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _clip_score(value: float, *, low: float = 1.0, high: float = 5.0) -> float:
    return round(max(low, min(high, value)), 2)


def _structure_score(case: CaseDossier, outputs: list[AgentOutput], decision: CoordinatorDecision) -> dict[str, float]:
    output_by_role = {output.role: output for output in outputs}
    completeness_components = [
        bool(output_by_role["diagnostic"].structured_findings),
        bool(decision.final_plan),
        bool(output_by_role["safety"].risks_or_alerts or output_by_role["safety"].candidate_actions),
        bool(output_by_role["monitoring"].candidate_actions),
        bool(decision.audit_trace),
    ]
    completeness = 1.0 + 4.0 * (sum(completeness_components) / len(completeness_components))

    evidence_count = sum(len(output.supporting_facts) for output in outputs)
    evidence_utilization = 1.0 + min(4.0, evidence_count / 3.0)

    preference_hits = 0
    response_text = decision.response_text.lower()
    for preference in case.preferences + case.comorbidities:
        if preference.lower() in response_text:
            preference_hits += 1
    patient_centeredness = 1.0 + min(4.0, preference_hits + (1.0 if case.preferences else 0.0))

    safety_penalty = 0.0
    if any(conflict.outcome == "accepted" and conflict.severity in {"absolute", "major"} for conflict in decision.conflicts):
        safety_penalty += 2.0
    safety_ethics = _clip_score(4.8 - safety_penalty)

    response_length = len(decision.response_text)
    response_efficiency = 4.5
    if response_length < 250:
        response_efficiency = 2.5
    elif response_length > 3000:
        response_efficiency = 3.0

    clinical_appropriateness = 3.2
    if case.reference_answer:
        clinical_appropriateness = 1.0 + 4.0 * lexical_f1(case.reference_answer, decision.response_text)
    elif decision.final_plan:
        clinical_appropriateness = 4.0

    return {
        "clinical_appropriateness": _clip_score(clinical_appropriateness),
        "response_completeness": _clip_score(completeness),
        "response_efficiency": _clip_score(response_efficiency),
        "evidence_utilization": _clip_score(evidence_utilization),
        "patient_centeredness": _clip_score(patient_centeredness),
        "safety_ethics": _clip_score(safety_ethics),
    }


def _maybe_bertscore(reference: str, prediction: str, enabled: bool) -> float | None:
    if not enabled:
        return None
    try:
        from bert_score import score as bert_score
    except ImportError:
        return None
    precision, recall, f1 = bert_score([prediction], [reference], lang="en", model_type="distilbert-base-uncased")
    return round(float(f1.mean().item()), 4)


def evaluate_case(
    case: CaseDossier,
    outputs: list[AgentOutput],
    decision: CoordinatorDecision,
    config: AppConfig,
) -> EvaluationResult:
    metrics: dict[str, float | None] = {}
    if case.task_type == "mcq" and case.reference_answer:
        metrics["accuracy"] = float(decision.final_answer.strip().lower() == case.reference_answer.strip().lower())
    if case.reference_answer:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rouge = scorer.score(case.reference_answer, decision.response_text)
        metrics["rouge1_f1"] = round(rouge["rouge1"].fmeasure, 4)
        metrics["rougeL_f1"] = round(rouge["rougeL"].fmeasure, 4)
        metrics["lexical_f1"] = round(lexical_f1(case.reference_answer, decision.response_text), 4)
        metrics["bertscore_f1"] = _maybe_bertscore(case.reference_answer, decision.response_text, config.evaluation.enable_bertscore)

    dimension_scores = _structure_score(case, outputs, decision)
    metrics["composite_dimension_mean"] = round(mean(dimension_scores.values()), 4)

    feedback_tags: list[str] = []
    if dimension_scores["response_completeness"] < 3.5:
        feedback_tags.append("completeness_gap")
    if dimension_scores["safety_ethics"] < 3.5:
        feedback_tags.append("safety_gap")
    if dimension_scores["evidence_utilization"] < 3.5:
        feedback_tags.append("evidence_gap")
    if dimension_scores["response_efficiency"] < 3.5:
        feedback_tags.append("efficiency_gap")
    if not outputs[3].candidate_actions:
        feedback_tags.append("monitoring_gap")
    if dimension_scores["clinical_appropriateness"] < 3.5:
        feedback_tags.append("treatment_gap")
    if metrics.get("accuracy") == 0.0:
        feedback_tags.append("treatment_gap")

    feedback_tags = list(dict.fromkeys(feedback_tags))
    summary = (
        f"Composite dimension mean: {metrics['composite_dimension_mean']:.2f}. "
        f"Feedback tags: {', '.join(feedback_tags) if feedback_tags else 'none'}."
    )
    return EvaluationResult(
        metrics=metrics,
        dimension_scores=dimension_scores,
        feedback_tags=feedback_tags,
        summary=summary,
    )


def aggregate_benchmark_results(results: list[EvaluationResult]) -> dict[str, Any]:
    aggregate: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.metrics.items():
            if value is None or isinstance(value, bool) or math.isnan(value):
                continue
            aggregate.setdefault(key, []).append(float(value))
    return {
        metric: round(mean(values), 4)
        for metric, values in aggregate.items()
        if values
    }
