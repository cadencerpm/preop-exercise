"""Entry point, LLM reader contract, caching and determinism."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import load_sample_cases

from core import PatientSubmission, TriageOutput, triage_submission
from preop.extract.base import ProcedureContext
from preop.extract.heuristic import HeuristicExtractor
from preop.extract.llm import PROMPT_VERSION, LLMExtractor
from preop.facts import DocumentFacts, empty_facts
from run_evals import ScoredOutput

DOC = {"doc_id": "d1", "type": "H&P", "date": "2026-02-21", "author": "A", "text": "HISTORY AND PHYSICAL\nok"}
CTX = ProcedureContext("Elective hernia repair", "2026-03-01")


def _facts() -> DocumentFacts:
    f = empty_facts()
    f.history_and_physical.is_history_and_physical = True
    f.history_and_physical.quote = "HISTORY AND PHYSICAL"
    return f


@pytest.fixture
def fake_client() -> Mock:
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(output_parsed=_facts())
    return client


def test_llm_reader_uses_strict_structured_output_at_temperature_zero(fake_client, tmp_path):
    reader = LLMExtractor("gpt-4.1-mini", client=fake_client, cache_dir=tmp_path)
    facts = reader.extract(DOC, CTX)
    assert facts.history_and_physical.is_history_and_physical
    kwargs = fake_client.responses.parse.call_args.kwargs
    assert kwargs["text_format"] is DocumentFacts
    assert kwargs["temperature"] == 0
    assert kwargs["model"] == "gpt-4.1-mini"
    user = kwargs["input"][0]["content"]
    assert "Elective hernia repair" in user and DOC["text"] in user
    # the reader never sees policy thresholds; it only reports observations
    assert "30 days" not in kwargs["instructions"] and "180" not in kwargs["instructions"]


def test_llm_reader_caches_by_content(fake_client, tmp_path):
    reader = LLMExtractor("gpt-4.1-mini", client=fake_client, cache_dir=tmp_path)
    reader.extract(DOC, CTX)
    reader.extract(DOC, CTX)
    assert fake_client.responses.parse.call_count == 1  # replayed from cache
    reader.extract({**DOC, "text": DOC["text"] + " edited"}, CTX)
    assert fake_client.responses.parse.call_count == 2  # new content -> new call
    LLMExtractor("gpt-4.1", client=fake_client, cache_dir=tmp_path).extract(DOC, CTX)
    assert fake_client.responses.parse.call_count == 3  # model is part of the key
    assert PROMPT_VERSION


def test_llm_reader_raises_on_refusal(fake_client, tmp_path):
    fake_client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
    with pytest.raises(RuntimeError):
        LLMExtractor("m", client=fake_client, cache_dir=tmp_path).extract(DOC, CTX)


def test_output_satisfies_scorer_schema_and_is_json_serializable():
    case = load_sample_cases()[0]
    out = triage_submission(case["submission"], model="unused", extractor=HeuristicExtractor())
    assert isinstance(out, TriageOutput)
    payload = json.loads(json.dumps(out.model_dump()))
    ScoredOutput.model_validate(payload)
    assert payload["provenance"]["policy_version"] == "2026-01-01"


def test_accepts_validated_submission_model():
    case = load_sample_cases()[3]
    sub = PatientSubmission.model_validate(case["submission"])
    out = triage_submission(sub, model="unused", extractor=HeuristicExtractor())
    assert out.decision == case["label"]["decision"]


def test_byte_identical_output_across_runs():
    case = load_sample_cases()[16]
    runs = {
        json.dumps(triage_submission(case["submission"], model="m", extractor=HeuristicExtractor()).model_dump(),
                   sort_keys=True)
        for _ in range(5)
    }
    assert len(runs) == 1


def test_crosscheck_flags_reader_disagreement(monkeypatch, tmp_path):
    """LLM reader misses a consent the heuristic reader sees -> review flag, decision unchanged."""
    case = load_sample_cases()[0]
    client = Mock()
    client.responses.parse.side_effect = lambda **kw: SimpleNamespace(
        output_parsed=HeuristicExtractor().extract(_doc_from_prompt(case, kw["input"][0]["content"]), CTX)
        if "CONSENT" not in kw["input"][0]["content"] else empty_facts()
    )
    monkeypatch.setenv("PREOP_EXTRACTOR", "crosscheck")
    monkeypatch.setenv("PREOP_CACHE_DIR", str(tmp_path))
    import core

    monkeypatch.setattr(core, "_make_extractor", lambda model, kind: LLMExtractor(model, client=client, cache_dir=tmp_path))
    out = triage_submission(case["submission"], model="m")
    assert out.decision == "NEEDS_FOLLOW_UP"  # primary reader drives the decision
    assert any(f.code == "READERS_DISAGREE" and f.ref == "signed_surgical_consent" for f in out.review_flags)


def _doc_from_prompt(case, prompt: str) -> dict:
    return next(d for d in case["submission"]["documents"] if d["text"] in prompt)


def test_reader_outage_degrades_to_offline_reader_with_flag(tmp_path):
    client = Mock()
    client.responses.parse.side_effect = TimeoutError("upstream timeout")
    case = load_sample_cases()[1]
    out = triage_submission(case["submission"], model="m",
                            extractor=LLMExtractor("m", client=client, cache_dir=tmp_path))
    assert out.decision == case["label"]["decision"]
    assert any(f.code == "PRIMARY_READER_FAILED" for f in out.review_flags)


def test_missing_api_key_fails_loudly_instead_of_silently_degrading(monkeypatch, tmp_path):
    from preop.extract.base import ReaderConfigError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    case = load_sample_cases()[1]
    with pytest.raises(ReaderConfigError):
        triage_submission(case["submission"], model="m", extractor=LLMExtractor("m", cache_dir=tmp_path))
