"""The LLM boundary: documents only, in; role/adequacy claims, out.

Per the architecture decision, the model never sees vitals, labs, medications,
procedure dates, or policy text -- only ``{index, type, date, text}`` per
document. It answers two questions only: which role each document plays (and,
for an H&P, whether it is the current episode), and whether a perioperative
anticoag plan document is clear. It never computes dates, emits a decision, or
invents an index; code re-reads the raw record at the index it points to.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class DocRole(StrEnum):
    HISTORY_AND_PHYSICAL = "HISTORY_AND_PHYSICAL"
    SURGICAL_CONSENT = "SURGICAL_CONSENT"
    PERIOP_ANTICOAG_PLAN = "PERIOP_ANTICOAG_PLAN"
    OTHER = "OTHER"


class ConsentStatus(StrEnum):
    SIGNED = "SIGNED"
    UNSIGNED = "UNSIGNED"
    UNCLEAR = "UNCLEAR"


@dataclass(frozen=True)
class DocClaim:
    """A validated model claim about one document. ``excerpt`` only exists to
    support validation against the raw text; it (and the model's ``reason``,
    which never even makes it into this type) never reach the final output."""

    index: int
    role: DocRole
    is_current: bool | None
    consent_signed: ConsentStatus | None
    is_clear_plan: bool | None
    excerpt: str


class Classifier(Protocol):
    def classify(self, docs: list[dict[str, object]]) -> list[DocClaim] | None:
        """Return claims, or None if classification failed after retries."""
        ...


PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """
You are reading a list of clinical documents from a pre-op submission package.
For EACH document, decide its role and answer only the fields in the schema.

Roles:
- HISTORY_AND_PHYSICAL: a history & physical / pre-op evaluation note for the
  CURRENT surgical episode. If the text explicitly says it is a retained prior
  document (e.g. kept for longitudinal chart context), it is still this role,
  but is_current must be false.
- SURGICAL_CONSENT: a surgical consent document. Decide consent_signed from the
  text: SIGNED, UNSIGNED, or UNCLEAR if the text does not clearly say either way.
- PERIOP_ANTICOAG_PLAN: a document whose purpose is to describe a perioperative
  anticoagulation plan (how a blood thinner is held and resumed around
  surgery). Decide is_clear_plan: true only if the text actually specifies
  hold/resume instructions; false if it only mentions the medication, defers
  to another provider, or says a plan is pending/not yet documented.
- OTHER: anything else (nursing intake, anesthesia pre-assessment, follow-up
  notes, etc). A note that a medication list was reviewed is NOT a plan.

Hard rules:
- Use only the text given. Do not use outside medical knowledge.
- Never compute or reason about dates; only report is_current based on whether
  the text itself frames the document as the current episode vs. a retained
  prior.
- Never emit a decision or a policy category.
- Never invent a document index; only use indexes given in the input.
- `excerpt` must be copied verbatim from that document's `text`.
- Fields not applicable to a document's role (e.g. consent_signed for an H&P)
  must be null.
""".strip()


def build_documents_input(raw_docs: list[dict[str, object]]) -> list[dict[str, object]]:
    """The exact, narrow payload that crosses to the model: index/type/date/text
    only. Nothing else from the submission is ever included."""

    return [
        {
            "index": index,
            "type": doc.get("type"),
            "date": doc.get("date"),
            "text": doc.get("text"),
        }
        for index, doc in enumerate(raw_docs)
    ]


def _document_claim_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "index": {"type": "integer"},
            "role": {"type": "string", "enum": [role.value for role in DocRole]},
            "is_current": {"type": ["boolean", "null"]},
            "consent_signed": {
                "type": ["string", "null"],
                "enum": [status.value for status in ConsentStatus] + [None],
            },
            "is_clear_plan": {"type": ["boolean", "null"]},
            "excerpt": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": [
            "index",
            "role",
            "is_current",
            "consent_signed",
            "is_clear_plan",
            "excerpt",
            "reason",
        ],
    }


def response_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "documents": {"type": "array", "items": _document_claim_schema()},
        },
        "required": ["documents"],
    }


def _cache_key(model: str, docs_payload: list[dict[str, object]]) -> str:
    canonical = json.dumps(docs_payload, sort_keys=True, separators=(",", ":"))
    digest_input = f"{model}|{PROMPT_VERSION}|{canonical}".encode()
    return hashlib.sha256(digest_input).hexdigest()


def _claim_from_payload(item: dict[str, object]) -> DocClaim | None:
    try:
        role = DocRole(item["role"])
        index = int(item["index"])
    except (KeyError, ValueError, TypeError):
        return None
    consent_raw = item.get("consent_signed")
    consent_signed = ConsentStatus(consent_raw) if consent_raw else None
    return DocClaim(
        index=index,
        role=role,
        is_current=item.get("is_current"),
        consent_signed=consent_signed,
        is_clear_plan=item.get("is_clear_plan"),
        excerpt=str(item.get("excerpt") or ""),
    )


def _parse_claims(payload: dict[str, object]) -> list[DocClaim]:
    items = payload.get("documents")
    if not isinstance(items, list):
        return []
    claims = (_claim_from_payload(item) for item in items if isinstance(item, dict))
    return [claim for claim in claims if claim is not None]


def validate(
    claims: list[DocClaim], raw_docs: list[dict[str, object]]
) -> list[DocClaim]:
    """Anti-hallucination gate: drop any claim pointing at an index that does not
    exist, whose excerpt is empty, or whose excerpt is not verbatim in that
    document's text. An empty excerpt is never proof of anything -- every
    string trivially "contains" "", so it must be rejected explicitly rather
    than relying on the verbatim check alone."""

    validated: list[DocClaim] = []
    for claim in claims:
        if claim.index < 0 or claim.index >= len(raw_docs):
            continue
        if not claim.excerpt.strip():
            continue
        doc = raw_docs[claim.index]
        text = doc.get("text") if isinstance(doc, dict) else None
        if not isinstance(text, str) or claim.excerpt not in text:
            continue
        validated.append(claim)
    return validated


# Process-wide so it survives across OpenAIClassifier instances:
# core.triage_submission builds a fresh classifier per call, and the
# determinism harness replays the same record in one process.
_RESPONSE_CACHE: dict[str, list[DocClaim]] = {}


class OpenAIClassifier:
    """Responses API classifier. Strict JSON schema, temperature 0, a
    process-wide response cache keyed on sha256(model, PROMPT_VERSION, canonical
    docs JSON), and the SDK's own retries. Returns None (rather than raising) on
    failure so the pipeline can degrade to MISSING_REQUIRED_DATA. Failures are
    not cached, so a transient error does not stick."""

    def __init__(self, model: str, *, client: object | None = None) -> None:
        self.model = model
        self._client = client

    def classify(self, docs: list[dict[str, object]]) -> list[DocClaim] | None:
        payload = build_documents_input(docs)
        key = _cache_key(self.model, payload)
        if key in _RESPONSE_CACHE:
            return _RESPONSE_CACHE[key]
        result = self._call(payload)
        if result is not None:
            _RESPONSE_CACHE[key] = result
        return result

    def _resolve_client(self) -> object:
        if self._client is not None:
            return self._client
        from openai import OpenAI  # imported lazily; see core.triage_submission

        return OpenAI()

    def _call(self, payload: list[dict[str, object]]) -> list[DocClaim] | None:
        try:
            client = self._resolve_client()
            response = client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                temperature=0,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps({"documents": payload}, sort_keys=True),
                            }
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "document_claims",
                        "schema": response_json_schema(),
                        "strict": True,
                    }
                },
            )
            data = json.loads(response.output_text)
        except Exception as exc:  # noqa: BLE001 - any failure here (network,
            # malformed response, unexpected client shape) must degrade to
            # "unavailable" rather than crash triage_submission; see
            # pipeline.py's handling of a None result. The failure is not
            # silent, though: it is logged (type + message) so a real outage
            # is visible in the logs rather than only showing up as
            # "Document review unavailable" in triage output.
            logger.warning(
                "Document classification call failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None
        return _parse_claims(data)


class FakeClassifier:
    """Test double: returns a fixed list of claims regardless of input, and
    counts how many times it was called so tests can assert the short-circuit
    skips the LLM entirely."""

    def __init__(self, claims: list[DocClaim]) -> None:
        self._claims = claims
        self.calls = 0

    def classify(self, docs: list[dict[str, object]]) -> list[DocClaim] | None:
        self.calls += 1
        return list(self._claims)


def load_fixture_claims(path: Path | str) -> dict[str, list[DocClaim]]:
    """Load the hand-labeled DocClaims fixture (tests/fixtures/doc_claims.json),
    keyed by case_id. Shared by tests and eval/local_score.py."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        case_id: [_claim_from_fixture_row(row) for row in rows]
        for case_id, rows in data.items()
    }


def _claim_from_fixture_row(row: dict[str, object]) -> DocClaim:
    consent_raw = row.get("consent_signed")
    return DocClaim(
        index=int(row["index"]),
        role=DocRole(row["role"]),
        is_current=row.get("is_current"),
        consent_signed=ConsentStatus(consent_raw) if consent_raw else None,
        is_clear_plan=row.get("is_clear_plan"),
        excerpt=str(row.get("excerpt", "")),
    )
