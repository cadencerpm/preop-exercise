"""Ground every extracted fact in the source text before the rules engine sees it.

A fact whose verbatim quote cannot be found in its document is dropped and reported
as a review flag. Dropping is fail-closed for this policy: a missing consent
signature, plan, or vital reading leads to NEEDS_FOLLOW_UP, never to READY.
"""

from __future__ import annotations

import re

from preop.extract.base import quote_in_text
from preop.facts import DocumentFacts
from preop.output import ReviewFlag


def _numbers_in(quote: str, *values: float | int | None) -> bool:
    found = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", quote)}
    return all(v is None or float(v) in found for v in values)


def verify_document_facts(
    facts: DocumentFacts, text: str, doc_ref: str
) -> tuple[DocumentFacts, list[ReviewFlag]]:
    facts = facts.model_copy(deep=True)
    flags: list[ReviewFlag] = []

    def flag(code: str, what: str) -> None:
        flags.append(ReviewFlag(code=code, message=what, ref=doc_ref))

    # H&P identity can come from the title alone, so its quote is optional, but a
    # provided quote must still be real.
    hp = facts.history_and_physical
    if hp.quote and not quote_in_text(hp.quote, text):
        flag("UNVERIFIED_QUOTE", f"H&P header quote not found in document: {hp.quote!r}")
        hp.quote = None

    c = facts.consent
    if c.is_surgical_consent_record and c.signature_status == "SIGNED" and not quote_in_text(c.quote, text):
        flag("UNVERIFIED_QUOTE", f"Consent marked signed but quote not found; treated as unclear: {c.quote!r}")
        c.signature_status = "UNCLEAR"
        c.quote = None

    kept_mentions = []
    for m in facts.antithrombotic_mentions:
        if quote_in_text(m.quote, text):
            kept_mentions.append(m)
        else:
            flag("UNVERIFIED_QUOTE", f"Medication mention dropped, quote not found: {m.quote!r}")
    facts.antithrombotic_mentions = kept_mentions

    kept_plans = []
    for p in facts.anticoagulation_plans:
        if not quote_in_text(p.quote, text):
            flag("UNVERIFIED_QUOTE", f"Anticoagulation plan dropped, quote not found: {p.quote!r}")
            continue
        if p.pre_procedure_instruction and not quote_in_text(p.pre_procedure_instruction, text):
            flag("UNVERIFIED_QUOTE", f"Pre-op instruction not found verbatim: {p.pre_procedure_instruction!r}")
            p.pre_procedure_instruction = None
        if p.post_procedure_instruction and not quote_in_text(p.post_procedure_instruction, text):
            flag("UNVERIFIED_QUOTE", f"Post-op instruction not found verbatim: {p.post_procedure_instruction!r}")
            p.post_procedure_instruction = None
        kept_plans.append(p)
    facts.anticoagulation_plans = kept_plans

    kept_vitals = []
    for v in facts.vital_readings:
        values = (v.systolic, v.diastolic) if v.kind == "BLOOD_PRESSURE" else (v.temperature,)
        if quote_in_text(v.quote, text) and _numbers_in(v.quote, *values):
            kept_vitals.append(v)
        else:
            flag("UNVERIFIED_QUOTE", f"Vital reading dropped, quote/values not found: {v.quote!r}")
    facts.vital_readings = kept_vitals

    if facts.procedure_date_stated and not quote_in_text(facts.procedure_date_quote, text):
        flag("UNVERIFIED_QUOTE", "Stated procedure date dropped, quote not found")
        facts.procedure_date_stated = None
    if facts.procedure_risk_stated and not quote_in_text(facts.procedure_risk_quote, text):
        flag("UNVERIFIED_QUOTE", "Stated procedure risk dropped, quote not found")
        facts.procedure_risk_stated = None

    return facts, flags
