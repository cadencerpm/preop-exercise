from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preop.extract.base import ProcedureContext  # noqa: E402
from preop.facts import DocumentFacts, empty_facts  # noqa: E402


class FakeExtractor:
    """Returns preset facts per doc_id, so rule tests don't depend on any reader."""

    name = "fake"
    model = None
    prompt_version = "fake"

    def __init__(self, facts_by_doc: dict[str, DocumentFacts]):
        self.facts_by_doc = facts_by_doc

    def extract(self, document: dict[str, Any], context: ProcedureContext) -> DocumentFacts:
        return self.facts_by_doc.get(str(document.get("doc_id")), empty_facts())


def base_submission() -> dict[str, Any]:
    """A clean LOW-risk case that is READY under the policy."""
    return {
        "patient": {"id": "p1"},
        "procedure": {
            "case_id": "t1",
            "procedure_type": "Elective inguinal hernia repair",
            "procedure_risk": "LOW",
            "procedure_date": "2026-03-31",
        },
        "vitals": [
            {"type": "blood_pressure", "systolic": 120, "diastolic": 78, "date": "2026-03-20T09:00:00Z"},
            {"type": "temperature", "value_f": 98.4, "date": "2026-03-20T09:05:00Z"},
        ],
        "labs": [{"id": "lab1", "code": "CBC", "display": "Complete Blood Count",
                  "effective_at": "2026-03-15T08:00:00Z", "status": "final"}],
        "medications": [],
        "conditions": [],
        "documents": [
            {"doc_id": "hp", "type": "History and Physical", "date": "2026-03-10", "author": "Dr A",
             "text": "HISTORY AND PHYSICAL\nPlanned procedure: inguinal hernia repair"},
            {"doc_id": "consent", "type": "Surgical Consent", "date": "2026-03-12", "author": "Dr B",
             "text": "INFORMED CONSENT\nProcedure: inguinal hernia repair\nSigned by patient: J Doe, 03/12/2026."},
        ],
    }


def hp_facts(date: str | None = None, quote: str = "HISTORY AND PHYSICAL") -> DocumentFacts:
    f = empty_facts()
    f.history_and_physical.is_history_and_physical = True
    f.history_and_physical.date_of_service = date
    f.history_and_physical.quote = quote
    return f


def consent_facts(status: str = "SIGNED", matches: str = "YES",
                  quote: str = "Signed by patient: J Doe, 03/12/2026.") -> DocumentFacts:
    f = empty_facts()
    f.consent.is_surgical_consent_record = True
    f.consent.signature_status = status  # type: ignore[assignment]
    f.consent.signed_by = "PATIENT"
    f.consent.procedure_named = "inguinal hernia repair"
    f.consent.matches_scheduled_procedure = matches  # type: ignore[assignment]
    f.consent.quote = quote
    return f


def base_facts() -> dict[str, DocumentFacts]:
    return {"hp": hp_facts(), "consent": consent_facts()}


@pytest.fixture
def run_case() -> Callable[..., Any]:
    from preop.pipeline import triage

    def _run(submission: dict[str, Any] | None = None, facts: dict[str, DocumentFacts] | None = None):
        sub = copy.deepcopy(submission or base_submission())
        return triage(sub, FakeExtractor(facts if facts is not None else base_facts()))

    return _run


def categories(output) -> list[str]:
    return sorted({i.category for i in output.issues})


def load_sample_cases() -> list[dict[str, Any]]:
    path = ROOT / "data" / "patients_sample_50.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
