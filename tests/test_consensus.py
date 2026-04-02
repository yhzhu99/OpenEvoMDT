from evomdt.consensus import apply_safety_guardrail
from evomdt.models import ActionRecommendation, AgentOutput, CaseDossier, CoordinatorOutput, PatientContext, PlanItem, RiskAlert


def test_safety_veto_rejects_conflicting_action():
    case = CaseDossier(
        case_id="case-1",
        cancer_type="hepatocellular_carcinoma",
        patient_context=PatientContext(performance_status="ECOG 1"),
        question="Recommend treatment.",
    )
    outputs = [
        AgentOutput(
            role="diagnostic",
            summary="Diag",
            structured_findings=["finding"],
            candidate_actions=[ActionRecommendation(action="Major hepatectomy", stance="recommend", rationale="diag", priority=1, citations=[])],
            risks_or_alerts=[],
            confidence=0.8,
            supporting_facts=[],
        ),
        AgentOutput(
            role="treatment",
            summary="Tx",
            structured_findings=["finding"],
            candidate_actions=[ActionRecommendation(action="Major hepatectomy", stance="recommend", rationale="tx", priority=1, citations=[])],
            risks_or_alerts=[],
            confidence=0.8,
            supporting_facts=[],
        ),
        AgentOutput(
            role="safety",
            summary="Safety",
            structured_findings=["finding"],
            candidate_actions=[ActionRecommendation(action="Major hepatectomy", stance="avoid", rationale="unsafe", priority=1, citations=[])],
            risks_or_alerts=[
                RiskAlert(
                    severity="major",
                    concern="unsafe in cirrhosis",
                    mitigation="avoid",
                    related_actions=["Major hepatectomy"],
                )
            ],
            confidence=0.9,
            supporting_facts=[],
        ),
        AgentOutput(
            role="monitoring",
            summary="Monitor",
            structured_findings=["finding"],
            candidate_actions=[],
            risks_or_alerts=[],
            confidence=0.7,
            supporting_facts=[],
        ),
    ]
    coordinator_output = CoordinatorOutput(
        summary="Propose surgery.",
        final_plan=[
            PlanItem(
                action="Major hepatectomy",
                owner_role="coordinator",
                rationale="Coordinator kept the aggressive plan.",
                priority=1,
                score=0.9,
                citations=[],
            )
        ],
        accepted_actions=["Major hepatectomy"],
        rejected_actions=[],
        conflicts=[],
        decision_rationale="Aggressive surgery plan.",
        audit_trace=[],
        final_confidence=0.9,
        supporting_facts=[],
    )
    decision = apply_safety_guardrail(
        case,
        outputs,
        coordinator_output,
        {"diagnostic": 1.0, "treatment": 1.0, "safety": 1.0, "monitoring": 1.0},
    )
    assert "Major hepatectomy" in decision.rejected_actions
    assert not decision.final_plan
