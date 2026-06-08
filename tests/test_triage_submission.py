from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from core import (
    BASELINE_SYSTEM_PROMPT,
    PatientSubmission,
    TriageOutput,
    triage_output_json_schema,
    triage_submission,
)


@pytest.fixture
def openai_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "decision": "READY",
                "issues": [],
                "explanation": "All required criteria are satisfied.",
            }
        ),
    )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    return client


@pytest.fixture
def submission_payload() -> dict[str, object]:
    return {
        "patient": {"id": "patient-1"},
        "procedure": {
            "case_id": "case-1",
            "procedure_risk": "LOW",
            "procedure_date": "2026-02-01",
        },
        "vitals": [
            {
                "type": "blood_pressure",
                "systolic": 120,
                "diastolic": 80,
                "date": "2026-01-25",
            }
        ],
        "labs": [
            {
                "code": "CBC",
                "display": "Complete blood count",
                "effective_at": "2026-01-20",
                "status": "final",
            }
        ],
        "medications": [],
        "conditions": [],
        "documents": [
            {
                "type": "history_and_physical",
                "date": "2026-01-20",
                "text": "History and physical completed.",
            },
            {
                "type": "surgical_consent",
                "date": "2026-01-22",
                "text": "Signed surgical consent.",
            },
        ],
    }


def test_triage_submission_returns_structured_output(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    output = triage_submission(submission_payload, model="test-model")

    assert isinstance(output, TriageOutput)
    assert output.decision == "READY"
    assert output.issues == []
    assert output.explanation == "All required criteria are satisfied."

    assert openai_client.responses.create.call_count == 1
    call = openai_client.responses.create.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["instructions"] == BASELINE_SYSTEM_PROMPT

    text_config = call["text"]
    assert isinstance(text_config, dict)
    response_format = text_config["format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "preop_triage_output"
    assert response_format["schema"] == triage_output_json_schema()
    assert response_format["strict"] is False


def test_triage_submission_builds_prompt_from_validated_submission(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    submission = PatientSubmission.model_validate(submission_payload)

    triage_submission(submission, model="test-model")

    assert openai_client.responses.create.call_count == 1
    call = openai_client.responses.create.call_args.kwargs
    message = call["input"][0]
    content = message["content"][0]
    prompt = content["text"]

    assert message["type"] == "message"
    assert message["role"] == "user"
    assert content["type"] == "input_text"
    assert json.loads(prompt.split("Submission JSON:\n", 1)[1]) == submission.model_dump()


def test_triage_submission_rejects_invalid_model_json(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    openai_client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "decision": "MAYBE",
                "issues": [],
                "explanation": "Not a valid triage decision.",
            }
        )
    )

    with pytest.raises(ValidationError):
        triage_submission(submission_payload, model="test-model")

    assert openai_client.responses.create.call_count == 1
