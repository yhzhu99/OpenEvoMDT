from evomdt.models import CaseDossier
from evomdt.normalization import normalize_agent_output_payload, normalize_coordinator_output_payload


def test_normalize_agent_output_payload_coerces_common_schema_deviations():
    normalized = normalize_agent_output_payload(
        {
            "summary": "High-risk synthetic answer candidate.",
            "structured_findings": {"vitals": {"temperature_c": "42.0"}},
            "candidate_actions": [
                {
                    "action": "Escalate urgently",
                    "stance": "Strongly indicated",
                    "rationale": "Potentially life-threatening physiology.",
                    "priority": "Critical",
                }
            ],
            "risks_or_alerts": [
                {
                    "severity": "Critical",
                    "concern": "False reassurance would be unsafe.",
                    "mitigation": "Urgent escalation.",
                }
            ],
            "confidence": "High",
        },
        "safety",
    )

    assert normalized["role"] == "safety"
    assert normalized["structured_findings"] == ['vitals: {"temperature_c": "42.0"}']
    assert normalized["candidate_actions"][0]["stance"] == "recommend"
    assert normalized["candidate_actions"][0]["priority"] == 1
    assert normalized["risks_or_alerts"][0]["severity"] == "major"
    assert normalized["confidence"] == 0.85


def test_normalize_coordinator_output_payload_fills_missing_lists():
    case = CaseDossier(
        case_id="bio-1",
        domain="biomedical_qa",
        task_type="generation",
        question="Answer A or B.",
        metadata={
            "output_contract": {
                "type": "enum_choice",
                "allowed_values": ["A", "B"],
                "final_answer_only": True,
            }
        },
    )

    normalized = normalize_coordinator_output_payload(
        {
            "summary": "Choose A under the synthetic benchmark rule.",
            "final_answer": "A",
            "final_plan": [{"action": "Escalate urgently", "priority": "High", "confidence": "80%"}],
            "audit_trace": "Integrated specialist outputs.",
            "final_confidence": "High",
        },
        case,
    )

    assert normalized["accepted_actions"] == ["Escalate urgently"]
    assert normalized["final_plan"][0]["owner_role"] == "coordinator"
    assert normalized["final_plan"][0]["priority"] == 2
    assert normalized["final_plan"][0]["score"] == 0.8
    assert normalized["audit_trace"] == ["Integrated specialist outputs."]
    assert normalized["final_confidence"] == 0.85
