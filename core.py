"""Shared policy, schema, and prompt helpers for the pre-op triage scripts."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

# -------------------------
# Policy + system prompt
# -------------------------

BASELINE_SYSTEM_PROMPT = """
You are a clinical operations assistant for pre-op scheduling triage.
Use only the policy below. Do not use outside medical knowledge.

Cadence Surgical Center Pre-Operative Scheduling Policy (effective Jan 1, 2026)

Output exactly one status:
- READY
- NEEDS_FOLLOW_UP
- NOT_CLEARED

Rule 1: Required documentation
- History and Physical (H&P) must exist and be completed within 30 days of procedure date.
- Signed Surgical Consent must exist.
If documentation is missing/outdated -> NEEDS_FOLLOW_UP.

Rule 2: Required testing by procedure risk
- LOW or MODERATE risk: CBC within 30 days of procedure date.
- HIGH risk: CBC within 14 days and CMP within 14 days.
Use only the most recent result for each required test.
If a required test is missing or outside window -> NEEDS_FOLLOW_UP.

Rule 3: Anticoagulation management
If the patient is currently taking an anticoagulant, a perioperative anticoagulation plan must be documented and clear.
If no clear plan is documented -> NEEDS_FOLLOW_UP.

Rule 4: Acute safety exclusions
If any of the following are present at review time -> NOT_CLEARED:
- Systolic BP >= 180 mmHg
- Diastolic BP >= 110 mmHg
- Temperature > 100.4 F
Use the most recent relevant vital.

Final determination
- READY only if all required criteria are satisfied and no exclusions are present.
- If a required field needed to evaluate a rule is missing/unknown -> NEEDS_FOLLOW_UP.

Output requirements
- Return exactly one JSON object.
""".strip()

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

# -------------------------
# Schemas
# -------------------------

class PatientName(BaseModel):

    given: str | None = None
    family: str | None = None

class PatientInfo(BaseModel):

    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None

class ProcedureInfo(BaseModel):

    case_id: str | None = None
    procedure_type: str | None = None
    procedure_risk: ProcedureRisk | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None

class BloodPressureVital(BaseModel):

    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    date: str | None = None
    source: str | None = None

class TemperatureVital(BaseModel):

    type: str | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None

class GenericVital(BaseModel):

    type: str | None = None
    date: str | None = None
    source: str | None = None

Vital = BloodPressureVital | TemperatureVital | GenericVital

class LabResult(BaseModel):

    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None

class Medication(BaseModel):

    name: str | None = None
    active: bool | None = None

class Condition(BaseModel):

    name: str | None = None
    active: bool | None = None

class Document(BaseModel):

    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None

class SubmissionMetadata(BaseModel):

    submission_received_at: str | None = None
    source_system: str | None = None

class PatientSubmission(BaseModel):
    """Single submission package shape from the take-home prompt."""

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None

class TriageIssueEvidence(BaseModel):

    source: str
    details: str

class TriageIssue(BaseModel):

    category: IssueCategory
    description: str
    evidence: TriageIssueEvidence

class TriageOutput(BaseModel):
    """Structured output contract for triage responses."""

    decision: Decision
    issues: list[TriageIssue] = Field(validation_alias=AliasChoices("issues"))
    explanation: str


class PreparedPatientCase(BaseModel):
    """Serialized eval case with submission payload and expected oracle output."""

    case_id: str
    submission: PatientSubmission
    expected_output: TriageOutput


def triage_output_json_schema() -> dict[str, object]:
    """Return the JSON schema used for structured model outputs."""

    schema = TriageOutput.model_json_schema()
    return schema


# -------------------------
# Prompt helpers
# -------------------------


def build_user_prompt(submission: dict[str, object]) -> str:
    """Format a single submission package as user input."""

    sections = [
        "Evaluate this patient package for pre-op scheduling readiness using the provided policy.",
        "Return your response as JSON.",
        "Submission JSON:",
        json.dumps(submission, sort_keys=True),
    ]
    return "\n".join(sections)


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
) -> TriageOutput:
    """Naive baseline implementation: single LLM call with JSON response output."""

    # Import lazily so core utilities remain usable without OpenAI installed.
    from openai import OpenAI

    if isinstance(submission, PatientSubmission):
        submission_payload = submission.model_dump()
    else:
        submission_payload = PatientSubmission.model_validate(submission).model_dump()

    client = OpenAI()
    request_kwargs: dict[str, object] = {
        "model": model,
        "instructions": BASELINE_SYSTEM_PROMPT,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_user_prompt(submission_payload),
                    }
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "preop_triage_output",
                "schema": triage_output_json_schema(),
                "strict": False,
            }
        },
    }

    response = client.responses.create(**request_kwargs)
    return TriageOutput.model_validate_json(response.output_text)
