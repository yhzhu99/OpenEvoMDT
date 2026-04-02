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
        if "diagnostic agent" in system_prompt:
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
                    }
                ],
                "risks_or_alerts": [],
                "confidence": 0.86,
                "supporting_facts": [
                    {"field_path": "lesions[0].stage_context", "value": "BCLC A", "note": "Early-stage disease."},
                    {"field_path": "organ_function.Child-Pugh", "value": "A", "note": "Liver function is preserved."},
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
                    },
                    {
                        "action": "Entecavir antiviral therapy",
                        "stance": "recommend",
                        "rationale": "HBV suppression reduces hepatic risk during cancer treatment.",
                        "priority": 2,
                    },
                    {
                        "action": "Major hepatectomy",
                        "stance": "consider",
                        "rationale": "Escalated surgery could be considered in a broader resection strategy.",
                        "priority": 4,
                    },
                ],
                "risks_or_alerts": [],
                "confidence": 0.82,
                "supporting_facts": [
                    {"field_path": "biomarkers[1].value", "value": "positive", "note": "Ongoing HBV replication."},
                    {"field_path": "preferences[0]", "value": "wants curative-intent treatment if feasible", "note": "Supports surgery-first planning."},
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
                    },
                    {
                        "action": "Pre-operative surgical fitness assessment",
                        "stance": "recommend",
                        "rationale": "Confirms candidacy for limited resection.",
                        "priority": 1,
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
                    {"field_path": "comorbidities[0]", "value": "HBV-related cirrhosis", "note": "Raises hepatic risk."},
                    {"field_path": "organ_function.bilirubin", "value": "1.0 mg/dL", "note": "Current reserve is preserved but should be protected."},
                ],
            }
        else:
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
                    }
                ],
                "risks_or_alerts": [],
                "confidence": 0.79,
                "supporting_facts": [
                    {"field_path": "question", "value": "Provide an MDT-style recommendation for initial management and surveillance.", "note": "Surveillance is explicitly requested."},
                    {"field_path": "preferences[1]", "value": "prefers outpatient follow-up when safe", "note": "Favors structured outpatient monitoring."},
                ],
            }
        raw_content = json.dumps(payload)
        parsed = parser(raw_content)
        return LLMTrace(raw_content=raw_content, parsed=parsed, model_name="fake-model", usage={"total_tokens": 1})


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
