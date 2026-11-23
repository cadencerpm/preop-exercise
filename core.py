"""Public entry point for pre-op triage, plus the submission input schema.

`triage_submission` delegates to the `preop` package:
LLM reads each document -> facts are verified against source text ->
a deterministic rules engine applies the policy in `preop/policy.toml`.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel as _PydanticBaseModel
from pydantic import ConfigDict, Field

from preop.output import (  # re-exported: run_evals.py imports these from core
    CheckResult,
    Decision,
    Evidence,
    IssueCategory,
    TriageIssue,
    TriageOutput,
)


class BaseModel(_PydanticBaseModel):
    # Keep unknown fields: real payloads carry fields this schema doesn't list
    # (e.g. value_c on a vital), and dropping them silently would hide evidence.
    model_config = ConfigDict(extra="allow")


ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]

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



# -------------------------
# Entry point
# -------------------------


def _make_extractor(model: str, kind: str):
    from preop.extract.heuristic import HeuristicExtractor
    from preop.extract.llm import LLMExtractor

    if kind == "heuristic":
        return HeuristicExtractor()
    if kind in {"llm", "crosscheck"}:
        return LLMExtractor(model)
    raise ValueError(f"Unknown PREOP_EXTRACTOR={kind!r} (expected llm | heuristic | crosscheck)")


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
    extractor: Any | None = None,
) -> TriageOutput:
    """Triage one submission package.

    Extractor selection (env PREOP_EXTRACTOR):
      crosscheck (default) LLM reader decides; the free offline heuristic reader
                 runs as an independent second opinion, and any check where the
                 two disagree becomes a review flag for the nurse
      llm        LLM reader only
      heuristic  offline regex reader only, no API key needed
    """
    from preop.extract.heuristic import HeuristicExtractor
    from preop.pipeline import triage

    if isinstance(submission, PatientSubmission):
        payload = submission.model_dump()
    else:
        payload = PatientSubmission.model_validate(submission).model_dump()

    kind = os.environ.get("PREOP_EXTRACTOR", "crosscheck")
    reader = extractor or _make_extractor(model, kind)
    second = HeuristicExtractor() if kind == "crosscheck" and extractor is None else None
    return triage(payload, reader, second_reader=second)
