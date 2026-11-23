"""LLM-backed document reader: one structured-output call per document.

Design choices:
- One call per document (not per case): small focused context, independently
  cacheable, parallelizable, and a failure is isolated to one document.
- OpenAI strict structured outputs via `responses.parse(text_format=...)`, so the
  response is guaranteed to match `DocumentFacts`.
- temperature=0 plus a content-addressed disk cache: re-running a case replays the
  exact same extraction, which makes outputs reproducible and auditable.
- The model reports observations with verbatim quotes. It never sees the policy
  thresholds and never makes the scheduling decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from preop.extract.base import ProcedureContext, ReaderConfigError
from preop.facts import DocumentFacts

PROMPT_VERSION = "doc-facts-v3"

SYSTEM_PROMPT = """
You read ONE clinical document from a pre-operative surgical package and report
observations as structured data. You do not decide whether the patient can be
scheduled; a separate system applies the policy. Report what the text says.

General rules
- Every `quote` field must be copied VERBATIM from the document (exact characters, no
  paraphrase, no added words). Keep quotes short: the single line or sentence that shows the fact.
- If something is not in this document, use null / false / an empty list. Never guess.
- Dates as YYYY-MM-DD. Times as 24h HH:MM.

history_and_physical
- is_history_and_physical = true only if this document itself is a History and Physical
  (titles vary: "H&P", "History & Physical", "Pre-op H and P", misspellings). Old or
  annual-wellness H&Ps still count as H&Ps. Anesthesia evaluations, nursing notes and
  consults are NOT H&Ps, even if they say "H&P reviewed".

consent
- is_surgical_consent_record = true if this document is a surgical consent form or
  records the status of one (consent discussion note, portal e-signature status).
- signature_status: SIGNED only when the text shows the patient (or legal representative)
  actually signed. Pending, blank signature line, "printed for review", "not completed",
  or "to be obtained" = NOT_SIGNED.
- matches_scheduled_procedure: compare the procedure on the consent with the scheduled
  procedure given below.

antithrombotic_mentions
- List every anticoagulant or antiplatelet drug mentioned (generic or brand), each with
  the patient's status as stated in THIS document. Use PLANNED_OR_BRIDGING_ONLY for drugs
  that appear only as part of a plan (e.g. "bridge with enoxaparin").
- A note saying the patient "reports taking" a drug not on their list = CURRENTLY_TAKING.
- "Discontinued", "stopped", "replaced with", "completed course in 2020" = STOPPED.
- "Unable to confirm whether still taking" = UNCERTAIN.

anticoagulation_plans
- For each anticoagulant with perioperative management instructions, copy the BEFORE-surgery
  instruction and the AFTER-surgery instruction verbatim. If either part is missing, deferred
  ("to be determined", "will be addressed after surgery", "will follow up"), set it to null.

vital_readings
- Every blood pressure and temperature reading in the text, with its number(s) and unit.
  Include initial readings AND rechecks as separate entries. Derive the time of a recheck
  when the text gives it (e.g. "rechecked 24 minutes later" after a 10:12 reading -> 10:36).
- Do not report "see flowsheet", "deferred", or "not recorded" as readings.

procedure_date_stated / procedure_risk_stated
- Only if the document explicitly states the scheduled date or the risk category.
  "To be determined" / "not yet assigned" = null.
""".strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class LLMExtractor:
    name = "llm"

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        cache_dir: Path | None = None,
        use_cache: bool | None = None,
    ) -> None:
        self.model = model
        self.prompt_version = PROMPT_VERSION
        self._client = client
        if use_cache is None:
            use_cache = os.environ.get("PREOP_CACHE", "1") != "0"
        self.use_cache = use_cache
        self.cache_dir = cache_dir or Path(
            os.environ.get("PREOP_CACHE_DIR", _repo_root() / ".cache" / "extractions")
        )
        self._schema_hash = hashlib.sha256(
            json.dumps(DocumentFacts.model_json_schema(), sort_keys=True).encode()
        ).hexdigest()[:12]

    # -- plumbing -------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            try:
                self._client = OpenAI(max_retries=4, timeout=60.0)
            except Exception as exc:  # e.g. OPENAI_API_KEY not set
                raise ReaderConfigError(
                    f"LLM reader not configured ({exc}). Set OPENAI_API_KEY, or run with "
                    "PREOP_EXTRACTOR=heuristic for the offline reader."
                ) from exc
        return self._client

    def _cache_key(self, document: dict[str, object], context: ProcedureContext) -> str:
        payload = {
            "model": self.model,
            "prompt": PROMPT_VERSION,
            "schema": self._schema_hash,
            "doc": {k: document.get(k) for k in ("type", "date", "author", "text")},
            "ctx": [context.procedure_type, context.procedure_date],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _user_message(self, document: dict[str, object], context: ProcedureContext) -> str:
        return "\n".join(
            [
                f"Scheduled procedure: {context.procedure_type or 'unknown'}",
                f"Scheduled procedure date (from scheduling system): {context.procedure_date or 'not set'}",
                "",
                f"Document title: {document.get('type') or ''}",
                f"Document date: {document.get('date') or ''}",
                f"Author: {document.get('author') or ''}",
                "----- DOCUMENT TEXT -----",
                str(document.get("text") or ""),
                "----- END -----",
            ]
        )

    # -- public ---------------------------------------------------------------

    def extract(self, document: dict[str, object], context: ProcedureContext) -> DocumentFacts:
        key = self._cache_key(document, context)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if self.use_cache and path.exists():
            return DocumentFacts.model_validate_json(path.read_text(encoding="utf-8"))

        response = self.client.responses.parse(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": self._user_message(document, context)}],
            text_format=DocumentFacts,
            temperature=0,
        )
        facts = response.output_parsed
        if facts is None:  # refusal or empty parse
            raise RuntimeError(f"No structured output for document {document.get('doc_id')}")
        if not isinstance(facts, DocumentFacts):
            facts = DocumentFacts.model_validate(facts)

        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(facts.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(path)
        return facts
