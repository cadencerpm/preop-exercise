"""One formatter per oracle citation template.

Decision (documented in NOTES.md): citations mirror the labels byte-for-byte,
including the oracle's own grounding misses (short BP values, bare-array
sources, anticoag issues that cite a document while quoting a medication).
Every distinct (category, description, source pattern, details template) found
in ``data/patients_sample_50.jsonl`` has exactly one formatter below.

Each formatter returns a ``Citation`` (description/source/details); callers in
``gates.py`` and ``rules.py`` attach the category and sort ``order``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Citation:
    description: str
    source: str
    details: str


# --------------------------------------------------------------------------
# MISSING_REQUIRED_DATA
# --------------------------------------------------------------------------


def missing_procedure_date() -> Citation:
    return Citation(
        description="Missing procedure date",
        source="procedure.procedure_date",
        details="procedure.procedure_date is null",
    )


def missing_procedure_risk() -> Citation:
    return Citation(
        description="Missing procedure risk",
        source="procedure.procedure_risk",
        details="procedure.procedure_risk is null",
    )


def missing_latest_bp() -> Citation:
    return Citation(
        description="Missing latest blood pressure",
        source="vitals",
        details="No blood_pressure vital with valid date found",
    )


def missing_latest_bp_values(*, source: str) -> Citation:
    # Same category/description as missing_latest_bp -- this is the same
    # policy failure ("no usable latest BP"), just a different raw-data shape
    # (a dated reading exists but its numeric values are null). Cites the
    # specific vitals[i] rather than the bare "vitals" array since there is a
    # concrete element to point at here.
    return Citation(
        description="Missing latest blood pressure",
        source=source,
        details=f"Latest blood_pressure vital ({source}) has null systolic/diastolic",
    )


def missing_latest_temp() -> Citation:
    return Citation(
        description="Missing latest temperature",
        source="vitals",
        details="No temperature vital with valid date found",
    )


def missing_latest_temp_values(*, source: str) -> Citation:
    return Citation(
        description="Missing latest temperature",
        source=source,
        details=f"Latest temperature vital ({source}) has null value_f",
    )


def unknown_anticoag_status(*, source: str, name: str) -> Citation:
    return Citation(
        description="Unknown anticoagulant active status",
        source=source,
        details=(
            f"Medication {name} has active=null; cannot determine if currently taking"
        ),
    )


def document_review_unavailable() -> Citation:
    return Citation(
        description="Document review unavailable",
        source="documents",
        details=(
            "Document classification failed after retries; unable to evaluate "
            "document-based requirements"
        ),
    )


# --------------------------------------------------------------------------
# REQUIRED_DOCUMENTATION
# --------------------------------------------------------------------------


def hp_missing() -> Citation:
    return Citation(
        description="History and Physical document missing",
        source="documents",
        details="No History and Physical document with valid date found",
    )


def hp_outside_window(
    *, source: str, hp_date: date, proc_date: date, delta: int, window: int
) -> Citation:
    return Citation(
        description="H&P outside 30-day window",
        source=source,
        details=(
            f"H&P date {hp_date.isoformat()} vs procedure_date {proc_date.isoformat()} "
            f"({delta} days prior; must be within {window})"
        ),
    )


def consent_missing() -> Citation:
    return Citation(
        description="Signed surgical consent missing",
        source="documents",
        details="No Surgical Consent document found",
    )


def consent_not_signed(*, source: str, text: str) -> Citation:
    return Citation(
        description="Surgical consent not clearly signed",
        source=source,
        details=f"Consent document text does not clearly indicate signed consent: {text}",
    )


# --------------------------------------------------------------------------
# REQUIRED_TESTING
# --------------------------------------------------------------------------


def lab_missing(*, code: str, risk: str | None) -> Citation:
    # CBC's existence check runs even when procedure_risk is unknown (it's
    # required at every tier), so this needs a null-safe wording too.
    condition = f"for procedure_risk {risk}" if risk is not None else "(procedure_risk unknown)"
    return Citation(
        description=f"{code} missing",
        source="labs",
        details=f"No {code} result with valid effective_at found {condition}",
    )


def lab_outside_window(
    *,
    code: str,
    source: str,
    effective_at: str,
    proc_date: date,
    delta: int,
    window: int,
    risk: str,
) -> Citation:
    # Only the 14-day HIGH-risk rule is observed in the labeled set calling out
    # the risk tier by name; the 30-day LOW/MODERATE rule is the unqualified
    # default. See NOTES.md for this (untested) generalization.
    suffix = f" for {risk} risk procedure" if risk == "HIGH" else ""
    return Citation(
        description=f"{code} outside {window}-day window{suffix}",
        source=source,
        details=(
            f"{code} effective_at {effective_at} vs procedure_date {proc_date.isoformat()} "
            f"({delta} days prior; must be within {window})"
        ),
    )


# --------------------------------------------------------------------------
# ANTICOAGULATION_MANAGEMENT
# --------------------------------------------------------------------------


def anticoag_missing_plan(
    *, plan_doc_source: str | None, med_source: str
) -> Citation:
    source = plan_doc_source if plan_doc_source is not None else "documents"
    return Citation(
        description="Missing perioperative anticoagulation plan",
        source=source,
        details=(
            f"Active anticoagulant medication present ({med_source}) "
            "but no clear perioperative plan document found"
        ),
    )


# --------------------------------------------------------------------------
# ACUTE_SAFETY_EXCLUSION
# --------------------------------------------------------------------------


def bp_exclusion(*, source: str, systolic: object, diastolic: object) -> Citation:
    return Citation(
        description="Blood pressure meets exclusion threshold",
        source=source,
        details=(
            f"latest BP systolic={systolic}, diastolic={diastolic}; "
            "threshold systolic>=180 or diastolic>=110"
        ),
    )


def temp_exclusion(*, source: str, value_f: object) -> Citation:
    return Citation(
        description="Temperature exceeds exclusion threshold",
        source=source,
        details=f"latest temperature value_f={value_f}; threshold is > 100.4",
    )
