"""Shared policy, schema, and prompt helpers for the pre-op triage scripts."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import json
import re
from typing import Literal
from openai import OpenAI

from pydantic import AliasChoices, BaseModel, Field

# -------------------------
# Policy + system prompt
# -------------------------

ANTICOAGULATION_MANAGEMENT_SYSTEM_PROMPT = """
Cadence Surgical Center pre-op triage (policy 2026-01-01). Use only the patient JSON; no outside knowledge.

Anticoagulant patients need a documented perioperative anticoagulation plan (how the drug is managed before and after surgery).

CLEAR = that plan is documented and unambiguous for before+after. UNCLEAR = absent, incomplete, or ambiguous.

Output one JSON object only: {"plan":"CLEAR"|"UNCLEAR","index":N}. N is the 0-based index in documents[] of the single best supporting document, or null if none.
""".strip()

ANTICOAGULANT_PATTERN = re.compile(r"perioperative|anticoag|medication\s+plan",re.I)

ANTICOAGULANT_NAME_FRAGMENTS = (
    "apixaban",
    "eliquis",
    "warfarin",
    "coumadin",
    "jantoven",
    "rivaroxaban",
    "xarelto",
    "dabigatran",
    "pradaxa",
    "edoxaban",
    "savaysa",
    "betrixaban",
    "bevyxxa",
    "heparin",
    "enoxaparin",
    "lovenox",
    "dalteparin",
    "fragmin",
    "fondaparinux",
    "arixtra",
)

HISTORY_AND_PHYSICAL_PATTERNS = [
    re.compile(r"History.*Physical", re.I),
    re.compile(r"H.?&.?P", re.I),
    re.compile(r"H.?and.?P", re.I),
]

# -------------------------
# Schemas
# -------------------------

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

class AnticoagulationManagementOutput(BaseModel):
    plan: Literal["CLEAR", "UNCLEAR"]
    index: int | None = Field(
        default=None,
        description=(
            "0-based index into submission.documents for the document that is the "
            "perioperative anticoagulation / medication management plan, or null if none."
        ),
    )

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


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
) -> TriageOutput:
    """Triage a single patient submission package."""

    client = OpenAI()

    if isinstance(submission, PatientSubmission):
        patient_submission = submission
    else:
        patient_submission = PatientSubmission.model_validate(submission)

    output = TriageOutput(decision="READY", issues=[], explanation="")
    output = check_required_documentation(patient_submission, output)
    output = check_required_testing(patient_submission, output)
    output = check_anticoagulation_management(client, model, patient_submission, output)
    output = check_acute_safety_exclusions(patient_submission, output)
    output = build_explanation(output)
    return output

def check_required_documentation(
    submission: PatientSubmission,
    output: TriageOutput,
) -> TriageOutput:
    """Apply Rule 1 (required documentation)."""

    # Find the most recent history and physical document
    most_recent_history_and_physical_index = None
    for index, document in enumerate(submission.documents):
        if (document.type and any(pattern.search(document.type) for pattern in HISTORY_AND_PHYSICAL_PATTERNS)) or (document.text and any(pattern.search(document.text) for pattern in HISTORY_AND_PHYSICAL_PATTERNS)):
            if most_recent_history_and_physical_index is None:
                most_recent_history_and_physical_index = index
            elif document.date and submission.documents[most_recent_history_and_physical_index].date is None:
                most_recent_history_and_physical_index = index
            elif document.date and submission.documents[most_recent_history_and_physical_index].date and document.date > submission.documents[most_recent_history_and_physical_index].date:
                most_recent_history_and_physical_index = index
    
    # Check if a procedure date is present
    has_procedure_date = submission.procedure and submission.procedure.procedure_date

    # Check if the most recent history and physical document is within 30 days of the procedure date
    has_history_and_physical_within_30_days = False
    if most_recent_history_and_physical_index is not None:
        document = submission.documents[most_recent_history_and_physical_index]
        if document.date and has_procedure_date and date_from_string(document.date) > (date_from_string(submission.procedure.procedure_date) - timedelta(days=30)):
            has_history_and_physical_within_30_days = True

    # Find any doc that looks like a signed surgical consent document
    surgical_consent_index = None
    signed_surgical_consent_index = None
    for index, document in enumerate(submission.documents):
        if document_looks_like_surgical_consent(document):
            surgical_consent_index = index
            if document_consent_text_looks_signed(document):
                signed_surgical_consent_index = index
                break

    # Now return the output with any issues
    if surgical_consent_index is None:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(output, TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            description="Surgical consent is not present",
            evidence=TriageIssueEvidence(
                source="documents",
                details="No surgical consent document found",
            ),
        ))
    if surgical_consent_index is not None and signed_surgical_consent_index is None:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(output, TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            description="Surgical consent is not signed",
            evidence=TriageIssueEvidence(
                source=f"documents[{surgical_consent_index}]",
                details="No signed surgical consent document found",
            ),
        ))
    if not has_procedure_date:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(output, TriageIssue(
            category="MISSING_REQUIRED_DATA",
            description="Missing procedure date",
            evidence=TriageIssueEvidence(
                source="procedure.procedure_date",
                details="procedure.procedure_date is null",
            ),
        ))
    if most_recent_history_and_physical_index is None:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(output, TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            description="History and physical document missing",
            evidence=TriageIssueEvidence(
                source="documents",
                details="No History and Physical document with valid date found",
            ),
        ))
    if has_procedure_date and most_recent_history_and_physical_index is not None and not has_history_and_physical_within_30_days:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(output, TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            description=f"History and physical is not within 30 days of the procedure date: {submission.procedure.procedure_date}",
            evidence=TriageIssueEvidence(
                source=f"documents[{most_recent_history_and_physical_index}]",
                details=f"{submission.documents[most_recent_history_and_physical_index].date} not within 30 days of procedure date: {submission.procedure.procedure_date}",
            ),
        ))
    
    return output


def check_required_testing(
    submission: PatientSubmission,
    output: TriageOutput,
) -> TriageOutput:
    """Apply Rule 2 (required pre-operative testing by procedure risk)."""

    if submission.procedure is None:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(output, TriageIssue(
            category="MISSING_REQUIRED_DATA",
            description="Missing procedure",
            evidence=TriageIssueEvidence(
                source="procedure",
                details="procedure is null",
            ),
        ))
        return output

    if submission.procedure.procedure_risk is None:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(
            output,
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                description="Missing procedure risk",
                evidence=TriageIssueEvidence(
                    source="procedure.procedure_risk",
                    details="procedure.procedure_risk is null",
                ),
            ),
        )
        return output

    if submission.procedure.procedure_date is None:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(output, TriageIssue(
            category="MISSING_REQUIRED_DATA",
            description="Missing procedure date",
            evidence=TriageIssueEvidence(
                source="procedure.procedure_date",
                details="procedure.procedure_date is null",
            ),
        ))
        return output
    
    # Find the most recent CBC lab.
    most_recent_cbc_index = None
    for index, lab in enumerate(submission.labs):
        if not lab_looks_like_cbc(lab):
            continue
        if most_recent_cbc_index is None:
            most_recent_cbc_index = index
            continue
        if lab.effective_at is not None and lab.effective_at > submission.labs[most_recent_cbc_index].effective_at:
            most_recent_cbc_index = index
    
    # Find the most recent CMP lab.
    most_recent_cmp_index = None
    for index, lab in enumerate(submission.labs):
        if not lab_looks_like_cmp(lab):
            continue
        if most_recent_cmp_index is None:
            most_recent_cmp_index = index
            continue
        if lab.effective_at is not None and lab.effective_at > submission.labs[most_recent_cmp_index].effective_at:
            most_recent_cmp_index = index
    
    # Check if the most recent CBC lab is missing or has a null effective_at
    if most_recent_cbc_index is None:
            output.decision = "NEEDS_FOLLOW_UP"
            append_triage_issue(output, TriageIssue(
                category="REQUIRED_TESTING",
                description="CBC missing",
                evidence=TriageIssueEvidence(
                    source="labs",
                    details="No CBC result with valid effective_at found",
                ),
            ))
    elif submission.labs[most_recent_cbc_index].effective_at is None:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(output, TriageIssue(
            category="REQUIRED_TESTING",
            description="CBC effective_at is null",
            evidence=TriageIssueEvidence(
                source=f"labs[{most_recent_cbc_index}]",
                details="CBC effective_at is null",
            ),
        ))
    
    # Figure out how recent the labs need to be for this procedure risk.
    lab_recency_in_days = 14 if submission.procedure.procedure_risk == "HIGH" else 30

    # Check if the most recent CBC lab is not within the recency window.
    if most_recent_cbc_index is not None and datetime_from_string(submission.labs[most_recent_cbc_index].effective_at).date() < datetime_from_string(submission.procedure.procedure_date).date() - timedelta(days=lab_recency_in_days):
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(output, TriageIssue(
            category="REQUIRED_TESTING",
            description="CBC effective_at is not within 30 days of the procedure date",
            evidence=TriageIssueEvidence(
                source=f"labs[{most_recent_cbc_index}].effective_at",
                details=f"{submission.labs[most_recent_cbc_index].effective_at} is not within 30 days of {submission.procedure.procedure_date}",
            ),
        ))
    
    if submission.procedure.procedure_risk == "HIGH":
        if most_recent_cmp_index is None:
            output.decision = "NEEDS_FOLLOW_UP"
            append_triage_issue(output, TriageIssue(
                category="REQUIRED_TESTING",
                description="CMP missing",
                evidence=TriageIssueEvidence(
                    source="labs",
                    details="No CMP result with valid effective_at found",
                ),
            ))
        elif submission.labs[most_recent_cmp_index].effective_at is None:
            output.decision = "NEEDS_FOLLOW_UP"
            append_triage_issue(output, TriageIssue(
                category="REQUIRED_TESTING",
                description="CMP effective_at is null",
                evidence=TriageIssueEvidence(
                    source=f"labs[{most_recent_cmp_index}]",
                    details="CMP effective_at is null",
                ),
            ))
        elif datetime_from_string(submission.labs[most_recent_cmp_index].effective_at).date() < datetime_from_string(submission.procedure.procedure_date).date() - timedelta(days=lab_recency_in_days):
            output.decision = "NEEDS_FOLLOW_UP"
            append_triage_issue(output, TriageIssue(
                category="REQUIRED_TESTING",
                description="CMP effective_at is not within 14 days of the procedure date",
                evidence=TriageIssueEvidence(
                    source=f"labs[{most_recent_cmp_index}]",
                    details="CMP effective_at is not within 14 days of the procedure date",
                ),
            ))

    return output

def check_anticoagulation_management(
    client: OpenAI,
    model: str,
    submission: PatientSubmission,
    output: TriageOutput,
) -> TriageOutput:
    """Apply Rule 3 (anticoagulation management)."""

    # Figure out if the patient is taking an anticoagulant medication.
    active_anticoagulant_medication_index = None
    unknown_anticoagulant_medication_index = None
    for index, medication in enumerate(submission.medications):
        if medication_looks_like_anticoagulant(medication):
            if medication.active is True:
                active_anticoagulant_medication_index = index
            elif medication.active is None:
                unknown_anticoagulant_medication_index = index

    # If the patient is not taking an anticoagulant medication, return the output.
    if active_anticoagulant_medication_index is None and unknown_anticoagulant_medication_index is None:
        return output

    # If the patient is taking an anticoagulant medication, check the anticoagulation management with the LLM.
    submission_payload = submission.model_dump()
    request_kwargs: dict[str, object] = {
        "model": model,
        "instructions": ANTICOAGULATION_MANAGEMENT_SYSTEM_PROMPT,
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
                "name": "anticoagulation_management_output",
                "schema": AnticoagulationManagementOutput.model_json_schema(),
                "strict": False,
            }
        },
    }
    response = AnticoagulationManagementOutput.model_validate_json(client.responses.create(**request_kwargs).output_text)
    
    # If the anticoagulation management is unclear, return the output with an issue.
    if response.plan == "UNCLEAR":
        output.decision = "NEEDS_FOLLOW_UP"
        if active_anticoagulant_medication_index is not None:
            append_triage_issue(output, TriageIssue(
                category="ANTICOAGULATION_MANAGEMENT",
                description="Missing perioperative anticoagulation management plan",
                evidence=TriageIssueEvidence(
                    source=f"documents[{response.index}]",
                    details="Missing perioperative anticoagulation management plan",
                ),
            ))
        else:
            append_triage_issue(output, TriageIssue(
                category="MISSING_REQUIRED_DATA",
                description="Unknown anticoagulant active status",
                evidence=TriageIssueEvidence(
                    source=f"medications[{unknown_anticoagulant_medication_index}]",
                    details="Medication has active=null; cannot determine if currently taking",
                ),
            ))
    return output

def check_acute_safety_exclusions(
    submission: PatientSubmission,
    output: TriageOutput,
) -> TriageOutput:
    """Apply Rule 4 (acute safety exclusion vitals)."""

    def _parse_vital_date(d: str | None) -> datetime | None:
        if not d:
            return None
        try:
            return datetime_from_string(d)
        except (TypeError, ValueError):
            return None

    most_recent_bp_index: int | None = None
    most_recent_bp_dt: datetime | None = None
    for index, vital in enumerate(submission.vitals):
        if not isinstance(vital, BloodPressureVital):
            continue
        dt = _parse_vital_date(vital.date)
        if dt is None:
            continue
        if most_recent_bp_dt is None or dt > most_recent_bp_dt:
            most_recent_bp_dt = dt
            most_recent_bp_index = index

    most_recent_temp_index: int | None = None
    most_recent_temp_dt: datetime | None = None
    for index, vital in enumerate(submission.vitals):
        if not isinstance(vital, TemperatureVital):
            continue
        dt = _parse_vital_date(vital.date)
        if dt is None:
            continue
        if most_recent_temp_dt is None or dt > most_recent_temp_dt:
            most_recent_temp_dt = dt
            most_recent_temp_index = index

    excluded = False

    if most_recent_bp_index is not None:
        bp = submission.vitals[most_recent_bp_index]
        if isinstance(bp, BloodPressureVital):
            sys_v = bp.systolic
            dia_v = bp.diastolic
            if sys_v is not None and dia_v is not None:
                systolic = float(sys_v)
                diastolic = float(dia_v)
                if systolic >= 180:
                    excluded = True
                    output.decision = "NOT_CLEARED"
                    append_triage_issue(
                        output,
                        TriageIssue(
                            category="ACUTE_SAFETY_EXCLUSION",
                            description="Systolic blood pressure is at or above 180 mmHg",
                            evidence=TriageIssueEvidence(
                                source=f"vitals[{most_recent_bp_index}]",
                                details=f"Most recent BP {systolic:.0f}/{diastolic:.0f} mmHg on {bp.date}",
                            ),
                        ),
                    )
                elif diastolic >= 110:
                    excluded = True
                    output.decision = "NOT_CLEARED"
                    append_triage_issue(
                        output,
                        TriageIssue(
                            category="ACUTE_SAFETY_EXCLUSION",
                            description="Diastolic blood pressure is at or above 110 mmHg",
                            evidence=TriageIssueEvidence(
                                source=f"vitals[{most_recent_bp_index}]",
                                details=f"Most recent BP {systolic:.0f}/{diastolic:.0f} mmHg on {bp.date}",
                            ),
                        ),
                    )

    if most_recent_temp_index is not None:
        tv = submission.vitals[most_recent_temp_index]
        if isinstance(tv, TemperatureVital) and tv.value_f is not None:
            temp_f = float(tv.value_f)
            if temp_f > 100.4:
                excluded = True
                output.decision = "NOT_CLEARED"
                append_triage_issue(
                    output,
                    TriageIssue(
                        category="ACUTE_SAFETY_EXCLUSION",
                        description="Temperature is above 100.4°F",
                        evidence=TriageIssueEvidence(
                            source=f"vitals[{most_recent_temp_index}]",
                            details=f"Most recent temperature {temp_f}°F on {tv.date}",
                        ),
                    ),
                )

    if excluded:
        return output

    if most_recent_bp_index is None:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(
            output,
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                description="Missing latest blood pressure",
                evidence=TriageIssueEvidence(
                    source="vitals",
                    details="Cannot determine most recent blood pressure",
                ),
            ),
        )
    else:
        bp = submission.vitals[most_recent_bp_index]
        if isinstance(bp, BloodPressureVital) and (
            bp.systolic is None or bp.diastolic is None
        ):
            output.decision = "NEEDS_FOLLOW_UP"
            append_triage_issue(
                output,
                TriageIssue(
                    category="MISSING_REQUIRED_DATA",
                    description="Most recent blood pressure vital is missing systolic or diastolic",
                    evidence=TriageIssueEvidence(
                        source=f"vitals[{most_recent_bp_index}]",
                        details="systolic or diastolic is null",
                    ),
                ),
            )

    if most_recent_temp_index is None:
        output.decision = "NEEDS_FOLLOW_UP"
        append_triage_issue(
            output,
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                description="Missing latest temperature",
                evidence=TriageIssueEvidence(
                    source="vitals",
                    details="Cannot determine most recent temperature",
                ),
            ),
        )
    else:
        tv = submission.vitals[most_recent_temp_index]
        if isinstance(tv, TemperatureVital) and tv.value_f is None:
            output.decision = "NEEDS_FOLLOW_UP"
            append_triage_issue(
                output,
                TriageIssue(
                    category="MISSING_REQUIRED_DATA",
                    description="Most recent temperature vital is missing value_f",
                    evidence=TriageIssueEvidence(
                        source=f"vitals[{most_recent_temp_index}]",
                        details="value_f is null",
                    ),
                ),
            )

    return output


def build_explanation(output: TriageOutput) -> TriageOutput:
    """Summarize issues as ``CATEGORY: description`` segments joined by `` | ``."""

    output.explanation = " | ".join(
        f"{issue.category}: {issue.description}" for issue in output.issues
    )
    return output


def append_triage_issue(output: TriageOutput, issue: TriageIssue) -> TriageOutput:
    """Append ``issue`` when no existing issue has the same category and evidence source."""

    if any(
        existing.category == issue.category
        and existing.evidence.source == issue.evidence.source
        and existing.evidence.details == issue.evidence.details
        for existing in output.issues
    ):
        return output
    output.issues.append(issue)
    return output


def build_user_prompt(submission: dict[str, object]) -> str:
    """Format a single submission package as user input."""

    sections = [
        "Evaluate this patient package for pre-op scheduling readiness using the provided policy.",
        "Return your response as JSON.",
        "Submission JSON:",
        json.dumps(submission, sort_keys=True),
    ]
    return "\n".join(sections)

def date_from_string(date_string: str) -> date:
    """Convert a date string to a date object."""

    return datetime.strptime(date_string, "%Y-%m-%d")

def datetime_from_string(datetime_string: str) -> datetime:
    """Convert a datetime string to a datetime object."""

    return datetime.fromisoformat(datetime_string)

def lab_looks_like_cbc(lab: LabResult) -> bool:
    """Returns True if the lab result looks like a CBC."""
    code = (lab.code or "").strip().upper()
    if code == "CBC" or code.startswith("LAB-CBC"):
        return True
    display = (lab.display or "").upper()
    return "COMPLETE BLOOD COUNT" in display or (
        "CBC" in display and "HBA1C" not in code
    )

def lab_looks_like_cmp(lab: LabResult) -> bool:
    """Returns True if the lab result looks like a CMP."""
    code = (lab.code or "").strip().upper()
    if code == "CMP":
        return True
    display = (lab.display or "").upper()
    return "COMPREHENSIVE METABOLIC" in display or re.search(
        r"\bCMP\b", display
    ) is not None

def document_looks_like_surgical_consent(document: Document) -> bool:
    """Returns True if the document looks like a surgical consent."""
    t = (document.type or "").lower()
    if re.search(r"consent.*(surgical|surgery|procedure|elective)|"
                 r"(surgical|surgery|procedure).*consent", t):
        return True
    if "consent" in t and "counseling" in t or "consent discussion" in t:
        return True  # optional: tighten with text check
    return False

def document_consent_text_looks_signed(document: Document) -> bool:
    """Returns True if the document consent text looks signed."""
    if not document.text:
        return False
    s = document.text.lower()
    if re.search(r"\bunsigned\b|awaiting.*signature", s):
        return False
    return bool(re.search(
        r"\bsigned\b|signature on file|obtained and signed|electronic consent obtained",
        s,
    ))

def medication_looks_like_anticoagulant(medication: Medication) -> bool:
    """Returns True if the medication looks like an anticoagulant."""
    if not medication.name:
        return False
    normalized = medication.name.lower().strip()
    return any(fragment in normalized for fragment in ANTICOAGULANT_NAME_FRAGMENTS)