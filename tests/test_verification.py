"""Facts must be grounded in the source text before the rules engine trusts them."""

from __future__ import annotations

from conftest import base_facts, base_submission, categories, consent_facts

from preop.extract.base import quote_in_text
from preop.facts import VitalReading, empty_facts


def test_quote_matching_tolerates_whitespace_case_and_ellipsis():
    text = "VITAL SIGNS:\n  08:06   NIBP 189/112,  T 37.1 °C\nADDENDUM 08:30: Re-measured 137/78 mmHg"
    assert quote_in_text("nibp 189/112, t 37.1 °C", text)
    assert quote_in_text("08:06 NIBP 189/112 ... Re-measured 137/78", text)
    assert not quote_in_text("Re-measured 120/70", text)
    assert not quote_in_text("Re-measured 137/78 ... 08:06", text)  # order matters


def test_hallucinated_signature_is_not_trusted(run_case):
    facts = base_facts() | {"consent": consent_facts("SIGNED", quote="Patient signed electronically on 03/12")}
    out = run_case(facts=facts)
    assert categories(out) == ["REQUIRED_DOCUMENTATION"]
    assert any(f.code == "UNVERIFIED_QUOTE" for f in out.review_flags)


def test_vital_with_numbers_not_in_quote_is_dropped(run_case):
    sub = base_submission()
    sub["documents"].append({"doc_id": "rn", "type": "Nursing", "date": "2026-03-28", "text": "BP 120/80"})
    f = empty_facts()
    f.vital_readings = [VitalReading(kind="BLOOD_PRESSURE", systolic=190, diastolic=80, temperature=None,
                                     temperature_unit=None, date=None, time=None, quote="BP 120/80")]
    out = run_case(sub, base_facts() | {"rn": f})
    assert out.decision == "READY"  # the invented 190 never reached the rules
    assert any("Vital reading dropped" in fl.message for fl in out.review_flags)
