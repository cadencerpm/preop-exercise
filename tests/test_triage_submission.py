from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from core import PatientSubmission, TriageOutput, triage_submission


@pytest.fixture
def clean_submission_payload() -> dict[str, object]:
    """A submission with no documents: triage_submission should never touch the
    OpenAI client, because the pipeline skips document classification entirely
    when there are no documents (the document rules still fire "missing"
    issues on their own, so this lands on NEEDS_FOLLOW_UP, not READY)."""

    return {
        "patient": {"id": "patient-1"},
        "procedure": {
            "case_id": "case-1",
            "procedure_risk": "LOW",
            "procedure_date": "2026-02-01",
        },
        "vitals": [
            {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-01-25"},
            {"type": "temperature", "value_f": 98.6, "date": "2026-01-25"},
        ],
        "labs": [
            {"code": "CBC", "display": "Complete blood count", "effective_at": "2026-01-20", "status": "final"}
        ],
        "medications": [],
        "conditions": [],
        "documents": [],
    }


def test_triage_submission_handles_missing_documents_without_any_openai_call(
    clean_submission_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # sys.modules["openai"] = None makes any `import openai` (or `from openai
    # import X`) raise ImportError immediately, regardless of whether the
    # real package happens to be installed in this environment. Deleting the
    # entry (the old approach) does not give that guarantee -- a real,
    # installed `openai` package would just get re-imported.
    monkeypatch.setitem(sys.modules, "openai", None)

    output = triage_submission(clean_submission_payload, model="test-model")

    assert isinstance(output, TriageOutput)
    assert output.decision == "NEEDS_FOLLOW_UP"
    categories = {issue.category for issue in output.issues}
    assert categories == {"REQUIRED_DOCUMENTATION"}


def test_triage_submission_uses_the_given_model_for_document_classification(
    clean_submission_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = dict(clean_submission_payload)
    payload["documents"] = [
        {"type": "History and Physical", "date": "2026-01-25", "text": "H&P completed."}
    ]

    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "documents": [
                    {
                        "index": 0,
                        "role": "HISTORY_AND_PHYSICAL",
                        "is_current": True,
                        "consent_signed": None,
                        "is_clear_plan": None,
                        "excerpt": "H&P completed.",
                        "reason": "current episode H&P",
                    }
                ]
            }
        )
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))

    triage_submission(payload, model="my-classifier-model")

    assert client.responses.create.call_count == 1
    assert client.responses.create.call_args.kwargs["model"] == "my-classifier-model"


def test_triage_submission_accepts_a_patient_submission_instance(
    clean_submission_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    submission = PatientSubmission.model_validate(clean_submission_payload)

    output = triage_submission(submission, model="test-model")

    assert output.decision == "NEEDS_FOLLOW_UP"


def test_triage_submission_rejects_a_pipeline_result_outside_the_schema(
    clean_submission_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)

    def _bad_pipeline_run(raw: dict[str, object], classifier: object, **kwargs: object) -> dict[str, object]:
        return {"decision": "MAYBE", "issues": [], "explanation": "not a valid decision"}

    monkeypatch.setattr("triage.pipeline.run", _bad_pipeline_run)

    with pytest.raises(ValidationError):
        triage_submission(clean_submission_payload, model="test-model")
