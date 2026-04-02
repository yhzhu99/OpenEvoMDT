from evomdt.evaluation import evidence_traceability_coverage, guideline_concordance, weighted_plan_concordance
from evomdt.models import CaseDossier, CoordinatorDecision, PlanItem


def test_guideline_concordance_returns_none_without_explicit_metadata():
    case = CaseDossier(
        case_id="case-1",
        cancer_type="breast_cancer",
        task_type="plan_eval",
        question="Summarize the treatment plan.",
    )
    decision = CoordinatorDecision(
        final_plan=[],
        final_answer="No decision",
        accepted_actions=[],
        rejected_actions=[],
        conflicts=[],
        role_weights={"diagnostic": 1.0, "treatment": 1.0, "safety": 1.0, "monitoring": 1.0},
        decision_rationale="No metadata-driven guideline evaluation available.",
        audit_trace=[],
        response_text="No explicit guideline metadata.",
        final_confidence=0.5,
    )

    assert guideline_concordance(case, decision) is None
    assert weighted_plan_concordance(case, decision) is None


def test_evidence_traceability_coverage_counts_plan_item_citations():
    decision = CoordinatorDecision(
        final_plan=[
            PlanItem(
                action="Action 1",
                owner_role="coordinator",
                rationale="Rationale 1",
                priority=1,
                score=0.9,
                citations=[{"source_id": "guideline-1", "source_type": "guideline"}],
            ),
            PlanItem(
                action="Action 2",
                owner_role="coordinator",
                rationale="Rationale 2",
                priority=2,
                score=0.8,
                citations=[],
            ),
        ],
        final_answer="Action 1",
        accepted_actions=["Action 1", "Action 2"],
        rejected_actions=[],
        conflicts=[],
        role_weights={"diagnostic": 1.0, "treatment": 1.0, "safety": 1.0, "monitoring": 1.0},
        decision_rationale="Traceability test.",
        audit_trace=[],
        response_text="Action 1 and Action 2.",
        final_confidence=0.7,
    )

    assert evidence_traceability_coverage(decision) == 0.5
