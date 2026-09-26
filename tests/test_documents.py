from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from triage.documents import (
    ConsentStatus,
    DocClaim,
    DocRole,
    OpenAIClassifier,
    build_documents_input,
    validate,
)


def _claims_response(items: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(output_text=json.dumps({"documents": items}))


def _full_claim(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "index": 0,
        "role": "HISTORY_AND_PHYSICAL",
        "is_current": True,
        "consent_signed": None,
        "is_clear_plan": None,
        "excerpt": "pre-op evaluation complete",
        "reason": "explicitly the current H&P",
    }
    base.update(overrides)
    return base


@pytest.fixture
def openai_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.responses.create.return_value = _claims_response([_full_claim()])
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    return client


DOCS = [
    {"doc_id": "d0", "type": "History and Physical", "date": "2026-03-01", "author": "Dr. A", "text": "pre-op evaluation complete"},
]


def test_classify_request_sends_only_index_type_date_text(openai_client: Mock) -> None:
    classifier = OpenAIClassifier(model="test-model")
    classifier.classify(DOCS)

    call = openai_client.responses.create.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["temperature"] == 0

    message = call["input"][0]
    content = message["content"][0]
    payload = json.loads(content["text"])
    assert payload == {"documents": build_documents_input(DOCS)}
    # Nothing but index/type/date/text ever crosses the boundary.
    for doc in payload["documents"]:
        assert set(doc) == {"index", "type", "date", "text"}


def test_classify_request_uses_strict_json_schema(openai_client: Mock) -> None:
    classifier = OpenAIClassifier(model="test-model")
    classifier.classify(DOCS)

    call = openai_client.responses.create.call_args.kwargs
    response_format = call["text"]["format"]
    assert response_format["type"] == "json_schema"
    assert response_format["strict"] is True
    schema = response_format["schema"]
    assert schema["additionalProperties"] is False
    doc_schema = schema["properties"]["documents"]["items"]
    assert doc_schema["additionalProperties"] is False
    assert set(doc_schema["required"]) == set(doc_schema["properties"])


def test_classify_parses_claims_and_drops_the_reason_field(openai_client: Mock) -> None:
    classifier = OpenAIClassifier(model="test-model")
    claims = classifier.classify(DOCS)

    assert claims == [
        DocClaim(
            index=0,
            role=DocRole.HISTORY_AND_PHYSICAL,
            is_current=True,
            consent_signed=None,
            is_clear_plan=None,
            excerpt="pre-op evaluation complete",
        )
    ]
    assert not hasattr(claims[0], "reason")


def test_classify_caches_identical_requests(openai_client: Mock) -> None:
    classifier = OpenAIClassifier(model="test-model")
    classifier.classify(DOCS)
    classifier.classify(list(DOCS))  # same content, different list object

    assert openai_client.responses.create.call_count == 1


def test_classify_cache_is_shared_across_instances(openai_client: Mock) -> None:
    # core.triage_submission builds a new classifier per call, so the cache
    # must outlive any single instance for determinism replays to hit it.
    OpenAIClassifier(model="test-model").classify(DOCS)
    OpenAIClassifier(model="test-model").classify(DOCS)

    assert openai_client.responses.create.call_count == 1


def test_classify_does_not_cache_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.responses.create.side_effect = [
        RuntimeError("transient"),
        _claims_response([_full_claim()]),
    ]
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))

    classifier = OpenAIClassifier(model="test-model")
    assert classifier.classify(DOCS) is None
    assert classifier.classify(DOCS) is not None
    assert client.responses.create.call_count == 2


def test_classify_cache_misses_on_different_docs(openai_client: Mock) -> None:
    classifier = OpenAIClassifier(model="test-model")
    classifier.classify(DOCS)
    other_docs = [{**DOCS[0], "text": "a different document"}]
    classifier.classify(other_docs)

    assert openai_client.responses.create.call_count == 2


def test_classify_returns_none_when_the_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.responses.create.side_effect = RuntimeError("boom")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))

    classifier = OpenAIClassifier(model="test-model")
    assert classifier.classify(DOCS) is None


def test_classify_logs_the_exception_type_and_message_on_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A classifier failure degrades gracefully (returns None -> "Document
    # review unavailable"), but it must not be silent: the exception type and
    # message are logged so a real outage is visible in the logs.
    client = Mock()
    client.responses.create.side_effect = RuntimeError("boom")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))

    classifier = OpenAIClassifier(model="test-model")
    with caplog.at_level(logging.WARNING):
        result = classifier.classify(DOCS)

    assert result is None
    assert "RuntimeError" in caplog.text
    assert "boom" in caplog.text


def test_classify_returns_none_on_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text="not json")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))

    classifier = OpenAIClassifier(model="test-model")
    assert classifier.classify(DOCS) is None


def test_validate_drops_claim_with_out_of_range_index() -> None:
    claims = [
        DocClaim(
            index=5,
            role=DocRole.HISTORY_AND_PHYSICAL,
            is_current=True,
            consent_signed=None,
            is_clear_plan=None,
            excerpt="pre-op evaluation complete",
        )
    ]
    assert validate(claims, DOCS) == []


def test_validate_drops_claim_whose_excerpt_is_not_verbatim_in_the_text() -> None:
    claims = [
        DocClaim(
            index=0,
            role=DocRole.HISTORY_AND_PHYSICAL,
            is_current=True,
            consent_signed=None,
            is_clear_plan=None,
            excerpt="this text was invented by the model",
        )
    ]
    assert validate(claims, DOCS) == []


def test_validate_drops_claim_with_an_empty_excerpt() -> None:
    # An empty string is trivially "in" any text, so the verbatim check alone
    # would let this through; excerpt non-emptiness must be checked too.
    claims = [
        DocClaim(
            index=0,
            role=DocRole.HISTORY_AND_PHYSICAL,
            is_current=True,
            consent_signed=None,
            is_clear_plan=None,
            excerpt="",
        )
    ]
    assert validate(claims, DOCS) == []


def test_validate_drops_claim_with_a_whitespace_only_excerpt() -> None:
    # "   " is a literal substring of this doc's text, so the verbatim check
    # alone would pass it; it must still be rejected as blank.
    docs = [{**DOCS[0], "text": "pre-op   evaluation complete"}]
    claims = [
        DocClaim(
            index=0,
            role=DocRole.HISTORY_AND_PHYSICAL,
            is_current=True,
            consent_signed=None,
            is_clear_plan=None,
            excerpt="   ",
        )
    ]
    assert validate(claims, docs) == []


def test_validate_keeps_a_claim_with_a_verbatim_excerpt_at_a_valid_index() -> None:
    claim = DocClaim(
        index=0,
        role=DocRole.SURGICAL_CONSENT,
        is_current=None,
        consent_signed=ConsentStatus.SIGNED,
        is_clear_plan=None,
        excerpt="pre-op evaluation",
    )
    assert validate([claim], DOCS) == [claim]
