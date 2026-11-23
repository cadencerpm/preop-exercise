"""Per-document facts extracted from free text.

This schema is the contract between *reading* (LLM or heuristic extractor) and
*deciding* (the deterministic rules engine). It deliberately asks for observations,
never for policy conclusions: the extractor reports "the signature line says
pending", and the rules engine decides what that means for scheduling.

Every fact carries a verbatim `quote` so it can be verified against the source text
and shown to the reviewing nurse. Fields have no defaults so the JSON schema is valid
for OpenAI strict structured outputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryAndPhysicalFacts(_Strict):
    is_history_and_physical: bool = Field(
        description="True only if this document IS a History and Physical (H&P) note, "
        "including pre-op H&Ps and older/annual H&Ps. False for anesthesia evaluations, "
        "nursing notes, consults, or notes that merely reference an H&P."
    )
    date_of_service: str | None = Field(description="H&P date of service, YYYY-MM-DD, if stated.")
    explicitly_incomplete: bool = Field(
        description="True only if the text says the H&P is draft, incomplete, or unsigned."
    )
    quote: str | None = Field(description="Verbatim text showing this is an H&P (e.g. its header).")


class ConsentFacts(_Strict):
    is_surgical_consent_record: bool = Field(
        description="True if this document is a surgical/procedure consent form or records "
        "the status of one (e.g. portal e-signature request, consent discussion note)."
    )
    signature_status: Literal["SIGNED", "NOT_SIGNED", "UNCLEAR"] | None = Field(
        description="SIGNED only if the text shows the patient (or legal representative) signed. "
        "NOT_SIGNED if pending, blank, not completed, or printed for review. Null if not a consent record."
    )
    signed_by: Literal["PATIENT", "LEGAL_REPRESENTATIVE", "UNKNOWN"] | None
    procedure_named: str | None = Field(description="Procedure named on the consent, verbatim.")
    matches_scheduled_procedure: Literal["YES", "NO", "NOT_STATED"] | None = Field(
        description="Whether procedure_named is the same operation as the scheduled procedure "
        "(ignore words like 'elective'; abbreviations count as the same)."
    )
    signed_date: str | None = Field(description="Date the patient signed, YYYY-MM-DD.")
    quote: str | None = Field(description="Verbatim signature-status text (e.g. the signature line).")


class MedicationMention(_Strict):
    drug_as_written: str
    drug_class: Literal["ANTICOAGULANT", "ANTIPLATELET", "OTHER"]
    status: Literal["CURRENTLY_TAKING", "STOPPED", "UNCERTAIN", "PLANNED_OR_BRIDGING_ONLY"] = Field(
        description="CURRENTLY_TAKING: text says the patient is on/taking it now. STOPPED: "
        "discontinued, replaced, or historical only. UNCERTAIN: text says it is unknown whether "
        "the patient still takes it. PLANNED_OR_BRIDGING_ONLY: only mentioned as part of a "
        "future/bridging plan (e.g. enoxaparin bridge)."
    )
    quote: str


class AnticoagulationPlan(_Strict):
    drug_as_written: str = Field(description="The anticoagulant this plan is written for.")
    pre_procedure_instruction: str | None = Field(
        description="Verbatim instruction for BEFORE surgery (hold/stop/continue with timing). "
        "Null if absent or deferred."
    )
    post_procedure_instruction: str | None = Field(
        description="Verbatim instruction for AFTER surgery (resume/restart with timing). "
        "Null if absent, deferred, or 'will be addressed later'."
    )
    deferred_or_pending: bool = Field(
        description="True if the note defers the decision (e.g. 'to be determined by surgical "
        "team', 'will follow up with recommendations')."
    )
    quote: str


class VitalReading(_Strict):
    kind: Literal["BLOOD_PRESSURE", "TEMPERATURE"]
    systolic: int | None
    diastolic: int | None
    temperature: float | None
    temperature_unit: Literal["F", "C"] | None
    date: str | None = Field(description="YYYY-MM-DD if the text gives the reading's date.")
    time: str | None = Field(
        description="HH:MM (24h) of the reading if stated or derivable (e.g. 'rechecked 24 minutes "
        "after 10:12' -> 10:36)."
    )
    quote: str


class DocumentFacts(_Strict):
    history_and_physical: HistoryAndPhysicalFacts
    consent: ConsentFacts
    antithrombotic_mentions: list[MedicationMention]
    anticoagulation_plans: list[AnticoagulationPlan]
    vital_readings: list[VitalReading]
    procedure_date_stated: str | None = Field(
        description="Scheduled procedure date stated in the text, YYYY-MM-DD. Null if TBD/pending."
    )
    procedure_date_quote: str | None
    procedure_risk_stated: Literal["LOW", "MODERATE", "HIGH"] | None = Field(
        description="Only if the text explicitly states the procedure risk category."
    )
    procedure_risk_quote: str | None


def empty_facts() -> DocumentFacts:
    return DocumentFacts(
        history_and_physical=HistoryAndPhysicalFacts(
            is_history_and_physical=False, date_of_service=None, explicitly_incomplete=False, quote=None
        ),
        consent=ConsentFacts(
            is_surgical_consent_record=False,
            signature_status=None,
            signed_by=None,
            procedure_named=None,
            matches_scheduled_procedure=None,
            signed_date=None,
            quote=None,
        ),
        antithrombotic_mentions=[],
        anticoagulation_plans=[],
        vital_readings=[],
        procedure_date_stated=None,
        procedure_date_quote=None,
        procedure_risk_stated=None,
        procedure_risk_quote=None,
    )
