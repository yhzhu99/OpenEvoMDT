from __future__ import annotations

import json
from pathlib import Path

import pytest

from evomdt.config import AppConfig
from evomdt.io import load_case
from evomdt.llm import LLMTrace


class FakeProvider:
    async def generate_structured(self, messages, parser):
        system_prompt = messages[0].content.lower()
        user_prompt = messages[1].content
        if "coordinator agent" in system_prompt:
            payload = {
                "role": "coordinator",
                "summary": "Recommend limited curative-intent resection, antiviral therapy, and structured surveillance.",
                "final_answer": "Recommend limited curative-intent resection, antiviral therapy, and structured surveillance.",
                "final_plan": [
                    {
                        "action": "Laparoscopic liver resection",
                        "owner_role": "treatment",
                        "rationale": "Curative local therapy remains feasible after integrating stage and liver reserve.",
                        "priority": 1,
                        "score": 0.91,
                        "citations": [],
                    },
                    {
                        "action": "Entecavir antiviral therapy",
                        "owner_role": "treatment",
                        "rationale": "HBV suppression should accompany definitive therapy.",
                        "priority": 2,
                        "score": 0.82,
                        "citations": [],
                    },
                    {
                        "action": "CT or MRI plus AFP every 3 months in year 1",
                        "owner_role": "monitoring",
                        "rationale": "The case requires structured surveillance after treatment.",
                        "priority": 2,
                        "score": 0.78,
                        "citations": [],
                    },
                ],
                "accepted_actions": [
                    "Laparoscopic liver resection",
                    "Entecavir antiviral therapy",
                    "CT or MRI plus AFP every 3 months in year 1",
                ],
                "rejected_actions": ["Major hepatectomy"],
                "conflicts": [
                    {
                        "action": "Major hepatectomy",
                        "recommenders": ["treatment"],
                        "objectors": ["safety"],
                        "severity": "major",
                        "conflict_type": "risk",
                        "outcome": "rejected",
                        "rationale": "Safety concerns outweigh any broader resection argument.",
                    }
                ],
                "decision_rationale": "Coordinator prioritized a curative but liver-sparing plan and excluded unsafe escalation.",
                "audit_trace": [
                    "Integrated specialist outputs into a single MDT plan.",
                    "Rejected major hepatectomy because the Safety agent identified major hepatic risk.",
                ],
                "final_confidence": 0.88,
                "supporting_facts": [
                    {"field_path": "stage", "value": "BCLC A", "note": "Early-stage disease supports curative therapy.", "citations": []},
                    {"field_path": "organ_function.Child-Pugh", "value": "A", "note": "Preserved reserve supports limited resection.", "citations": []},
                ],
            }
        elif "diagnostic agent" in system_prompt:
            payload = {
                "role": "diagnostic",
                "summary": "Solitary early-stage HCC in a resectable location.",
                "structured_findings": ["BCLC A lesion", "Preserved liver function", "HBV-related cirrhosis"],
                "candidate_actions": [
                    {
                        "action": "Laparoscopic liver resection",
                        "stance": "recommend",
                        "rationale": "Solitary lesion with preserved liver reserve supports curative local therapy.",
                        "priority": 1,
                        "citations": [],
                    }
                ],
                "risks_or_alerts": [],
                "confidence": 0.86,
                "supporting_facts": [
                    {"field_path": "lesions[0].stage_context", "value": "BCLC A", "note": "Early-stage disease.", "citations": []},
                    {"field_path": "organ_function.Child-Pugh", "value": "A", "note": "Liver function is preserved.", "citations": []},
                ],
            }
        elif "treatment agent" in system_prompt:
            payload = {
                "role": "treatment",
                "summary": "Primary curative-intent therapy with HBV management.",
                "structured_findings": ["Curative-intent treatment remains feasible."],
                "candidate_actions": [
                    {
                        "action": "Laparoscopic liver resection",
                        "stance": "recommend",
                        "rationale": "Offers curative-intent local control for a solitary lesion.",
                        "priority": 1,
                        "citations": [],
                    },
                    {
                        "action": "Entecavir antiviral therapy",
                        "stance": "recommend",
                        "rationale": "HBV suppression reduces hepatic risk during cancer treatment.",
                        "priority": 2,
                        "citations": [],
                    },
                    {
                        "action": "Major hepatectomy",
                        "stance": "consider",
                        "rationale": "Escalated surgery could be considered in a broader resection strategy.",
                        "priority": 4,
                        "citations": [],
                    },
                ],
                "risks_or_alerts": [],
                "confidence": 0.82,
                "supporting_facts": [
                    {"field_path": "biomarkers[1].value", "value": "positive", "note": "Ongoing HBV replication.", "citations": []},
                    {"field_path": "preferences[0]", "value": "wants curative-intent treatment if feasible", "note": "Supports surgery-first planning.", "citations": []},
                ],
            }
        elif "safety agent" in system_prompt:
            payload = {
                "role": "safety",
                "summary": "Proceed with liver-sparing therapy and avoid unnecessary hepatic risk.",
                "structured_findings": ["Cirrhosis increases surgical risk if liver reserve is overextended."],
                "candidate_actions": [
                    {
                        "action": "Major hepatectomy",
                        "stance": "avoid",
                        "rationale": "Cirrhosis makes large-volume resection unnecessarily risky.",
                        "priority": 1,
                        "citations": [],
                    },
                    {
                        "action": "Pre-operative surgical fitness assessment",
                        "stance": "recommend",
                        "rationale": "Confirms candidacy for limited resection.",
                        "priority": 1,
                        "citations": [],
                    },
                ],
                "risks_or_alerts": [
                    {
                        "severity": "major",
                        "concern": "Major hepatectomy increases decompensation risk in cirrhosis.",
                        "mitigation": "Prefer limited resection and confirm reserve pre-operatively.",
                        "related_actions": ["Major hepatectomy"],
                    }
                ],
                "confidence": 0.91,
                "supporting_facts": [
                    {"field_path": "comorbidities[0]", "value": "HBV-related cirrhosis", "note": "Raises hepatic risk.", "citations": []},
                    {"field_path": "organ_function.bilirubin", "value": "1.0 mg/dL", "note": "Current reserve is preserved but should be protected.", "citations": []},
                ],
            }
        elif "monitoring agent" in system_prompt:
            payload = {
                "role": "monitoring",
                "summary": "Structured post-treatment monitoring is required.",
                "structured_findings": ["Need early post-operative review and longitudinal surveillance."],
                "candidate_actions": [
                    {
                        "action": "CT or MRI plus AFP every 3 months in year 1",
                        "stance": "monitor",
                        "rationale": "Tracks recurrence early after definitive therapy.",
                        "priority": 1,
                        "citations": [],
                    }
                ],
                "risks_or_alerts": [],
                "confidence": 0.79,
                "supporting_facts": [
                    {"field_path": "question", "value": "Provide an MDT-style recommendation for initial management and surveillance.", "note": "Surveillance is explicitly requested.", "citations": []},
                    {"field_path": "preferences[1]", "value": "prefers outpatient follow-up when safe", "note": "Favors structured outpatient monitoring.", "citations": []},
                ],
            }
        else:
            raise AssertionError(f"Unexpected system prompt for FakeProvider: {system_prompt}")
        raw_content = json.dumps(payload)
        parsed = parser(raw_content)
        return LLMTrace(raw_content=raw_content, parsed=parsed, model_name="fake-model", usage={"total_tokens": 1})


class BiomedicalFakeProvider:
    async def generate_structured(self, messages, parser):
        system_prompt = messages[0].content.lower()
        user_prompt = messages[1].content

        if '"case_id": "bio-hot-fever-001"' in user_prompt:
            case_name = "hot"
        elif '"case_id": "bio-low-fever-002"' in user_prompt:
            case_name = "low"
        else:
            raise AssertionError(f"Unexpected biomedical case in prompt: {user_prompt}")

        if "coordinator agent" in system_prompt:
            payload = {
                "role": "coordinator",
                "summary": "Synthetic benchmark answer based on the dossier and specialist debate.",
                "final_answer": "A" if case_name == "hot" else "Maybe B",
                "final_plan": [
                    {
                        "action": "Escalate to urgent clinical assessment",
                        "owner_role": "monitoring",
                        "rationale": "High-risk physiology requires immediate evaluation rather than reassurance.",
                        "priority": 1,
                        "score": 0.9,
                        "citations": [],
                    }
                ],
                "accepted_actions": ["Escalate to urgent clinical assessment"],
                "rejected_actions": ["Provide reassurance only"] if case_name == "hot" else [],
                "conflicts": [],
                "decision_rationale": "Coordinator integrated fact extraction, answer selection, and safety objections.",
                "audit_trace": ["Integrated specialist outputs.", "Applied the explicit A/B output contract."],
                "final_confidence": 0.74,
                "supporting_facts": [],
            }
        elif "diagnostic agent" in system_prompt:
            payload = {
                "role": "diagnostic",
                "summary": "Extracted the key vital-sign facts relevant to the binary task.",
                "structured_findings": ["Temperature is the primary abnormal datum in the dossier."],
                "candidate_actions": [
                    {
                        "action": "Support answer A" if case_name == "hot" else "Support answer B",
                        "stance": "recommend",
                        "rationale": "The observed temperature trend is the main signal available.",
                        "priority": 1,
                        "citations": [],
                    }
                ],
                "risks_or_alerts": [],
                "confidence": 0.8,
                "supporting_facts": (
                    [
                        {
                            "field_path": "patient_context.summary",
                            "value": "Synthetic emergency-fever benchmark case.",
                            "note": "The task is constrained to the provided facts.",
                            "citations": [],
                        }
                    ]
                    if case_name == "hot"
                    else []
                ),
            }
        elif "treatment agent" in system_prompt:
            payload = {
                "role": "treatment",
                "summary": "Produced the main answer candidate for the synthetic task.",
                "structured_findings": ["The answer must follow the A/B output contract."],
                "candidate_actions": [
                    {
                        "action": "Choose A" if case_name == "hot" else "Choose B",
                        "stance": "recommend",
                        "rationale": "This answer candidate best matches the simplified benchmark framing.",
                        "priority": 1,
                        "citations": [],
                    }
                ],
                "risks_or_alerts": [],
                "confidence": 0.76,
                "supporting_facts": [],
            }
        elif "safety agent" in system_prompt:
            payload = {
                "role": "safety",
                "summary": "Flagged overconfidence risk in a minimal synthetic dossier.",
                "structured_findings": ["Avoid presenting the answer as real-world medical certainty."],
                "candidate_actions": [
                    {
                        "action": "Provide reassurance only",
                        "stance": "avoid" if case_name == "hot" else "consider",
                        "rationale": "Minimal data should not be used to downplay risk in a severe-fever scenario.",
                        "priority": 1,
                        "citations": [],
                    }
                ],
                "risks_or_alerts": (
                    [
                        {
                            "severity": "major",
                            "concern": "Severe hyperpyrexia can deteriorate rapidly without immediate evaluation.",
                            "mitigation": "Escalate urgently and avoid false reassurance.",
                            "related_actions": ["Provide reassurance only"],
                        }
                    ]
                    if case_name == "hot"
                    else []
                ),
                "confidence": 0.88,
                "supporting_facts": [
                    {
                        "field_path": "question",
                        "value": "Binary prognosis-style benchmark question.",
                        "note": "The framing itself needs conservative handling.",
                        "citations": [],
                    }
                ],
            }
        elif "monitoring agent" in system_prompt:
            payload = {
                "role": "monitoring",
                "summary": "Specified what observation or escalation would make the answer safer.",
                "structured_findings": ["Observation and urgent reassessment are the key next steps."],
                "candidate_actions": [
                    {
                        "action": "Escalate to urgent clinical assessment",
                        "stance": "monitor",
                        "rationale": "This is the safest next step in the synthetic benchmark framing.",
                        "priority": 1,
                        "citations": [],
                    }
                ],
                "risks_or_alerts": [],
                "confidence": 0.72,
                "supporting_facts": [],
            }
        else:
            raise AssertionError(f"Unexpected system prompt for BiomedicalFakeProvider: {system_prompt}")

        raw_content = json.dumps(payload)
        parsed = parser(raw_content)
        return LLMTrace(raw_content=raw_content, parsed=parsed, model_name="biomedical-fake-model", usage={"total_tokens": 1})


class MinimalTaskFakeProvider:
    async def generate_structured(self, messages, parser):
        system_prompt = messages[0].content.lower()
        user_prompt = messages[1].content

        if '"case_id": "1"' in user_prompt:
            final_answer = "C"
        elif '"case_id": "2"' in user_prompt:
            final_answer = "B"
        else:
            raise AssertionError(f"Unexpected minimal task prompt: {user_prompt}")

        if "coordinator agent" in system_prompt:
            payload = {
                "role": "coordinator",
                "summary": "Selected the best benchmark option from the provided task text.",
                "final_answer": final_answer,
                "final_plan": [],
                "accepted_actions": [f"Return {final_answer}"],
                "rejected_actions": [],
                "conflicts": [],
                "decision_rationale": "Coordinator used the constrained benchmark task text and specialist recommendations.",
                "audit_trace": ["Integrated specialist outputs for the minimal benchmark task."],
                "final_confidence": 0.77,
                "supporting_facts": [
                    {
                        "field_path": "question",
                        "value": "Minimal benchmark task text",
                        "note": "The task text contains all required information.",
                        "citations": [],
                    }
                ],
            }
        else:
            payload = {
                "role": (
                    "diagnostic"
                    if "diagnostic agent" in system_prompt
                    else "treatment"
                    if "treatment agent" in system_prompt
                    else "safety"
                    if "safety agent" in system_prompt
                    else "monitoring"
                ),
                "summary": "Reviewed the minimal benchmark task.",
                "structured_findings": ["Task text includes the answer options inline."],
                "candidate_actions": [
                    {
                        "action": f"Support answer {final_answer}",
                        "stance": "recommend" if "safety agent" not in system_prompt else "consider",
                        "rationale": "Minimal benchmark tasks should still produce a single constrained answer.",
                        "priority": 1,
                        "citations": [],
                    }
                ],
                "risks_or_alerts": [],
                "confidence": 0.7,
                "supporting_facts": [
                    {
                        "field_path": "question",
                        "value": "Minimal benchmark task text",
                        "note": "The raw task text is sufficient for answer selection.",
                        "citations": [],
                    }
                ],
            }

        raw_content = json.dumps(payload)
        parsed = parser(raw_content)
        return LLMTrace(raw_content=raw_content, parsed=parsed, model_name="minimal-task-fake-model", usage={"total_tokens": 1})


@pytest.fixture
def sample_case():
    return load_case(Path("data/samples/cases/sample_hcc_case.json"))


@pytest.fixture
def test_config(tmp_path):
    config = AppConfig.load("config.toml")
    config = config.model_copy(deep=True)
    config.runtime.artifacts_dir = str(tmp_path / "artifacts")
    config.runtime.evolution_state_path = str(tmp_path / "artifacts" / "evolution_state.json")
    return config
