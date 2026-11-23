"""Extractor interface and quote verification."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from preop.facts import DocumentFacts


class ReaderConfigError(RuntimeError):
    """Misconfiguration (e.g. missing API key). Fails loudly instead of degrading."""


@dataclass(frozen=True)
class ProcedureContext:
    """What an extractor needs to know about the case to read one document."""

    procedure_type: str | None
    procedure_date: str | None


class Extractor(Protocol):
    name: str
    model: str | None
    prompt_version: str | None

    def extract(self, document: dict[str, object], context: ProcedureContext) -> DocumentFacts: ...


_WS = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return _WS.sub(" ", text).strip().lower()


def quote_in_text(quote: str | None, text: str | None) -> bool:
    """True if `quote` appears in `text`, modulo whitespace/case/unicode.

    An ellipsis in the quote is treated as a gap: every fragment must appear, in order.
    This is the guardrail against an extractor inventing a fact: a fact whose quote
    cannot be found in the document is not trusted.
    """
    if not quote or not text:
        return False
    haystack = _normalize(text)
    position = 0
    for fragment in _ELLIPSIS.split(quote):
        needle = _normalize(fragment).strip(" .;,")
        if not needle:
            continue
        found = haystack.find(needle, position)
        if found < 0:
            return False
        position = found + len(needle)
    return True
