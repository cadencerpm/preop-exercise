"""Metamorphic + boundary tests generated from the real sample cases.

The 50 labeled cases test the engine at 50 points. These tests mutate those cases
in ways whose effect on the decision is known in advance:

- invariances: shifting every date by the same amount, or writing a drug by its
  generic instead of brand name, must not change the result;
- boundaries: a reading or result placed exactly on, or one unit past, a policy
  threshold must flip the decision exactly there.

They run on the offline heuristic reader, so they exercise reader + rules together.
"""

from __future__ import annotations

import copy
import re
from datetime import date, datetime, timedelta

import pytest
from conftest import load_sample_cases

from preop.extract.heuristic import HeuristicExtractor
from preop.pipeline import triage

CASES = load_sample_cases()
READY = [c for c in CASES if c["label"]["decision"] == "READY"]
READER = HeuristicExtractor()


def run(sub):
    out = triage(sub, READER)
    return out.decision, sorted({i.category for i in out.issues})


def _shift_iso(value, days):
    if not isinstance(value, str):
        return value
    m = re.match(r"(\d{4}-\d{2}-\d{2})(.*)", value)
    if not m:
        return value
    return (date.fromisoformat(m.group(1)) + timedelta(days=days)).isoformat() + m.group(2)


def shift_all_dates(sub, days):
    sub = copy.deepcopy(sub)
    sub["procedure"]["procedure_date"] = _shift_iso(sub["procedure"]["procedure_date"], days)
    for v in sub["vitals"]:
        v["date"] = _shift_iso(v["date"], days)
    for lab in sub["labs"]:
        lab["effective_at"] = _shift_iso(lab["effective_at"], days)
    for d in sub["documents"]:
        d["date"] = _shift_iso(d["date"], days)
    return sub


def latest_moment(sub) -> datetime:
    """A time after every vital in the case (structured or in notes)."""
    days = [d["date"] for d in sub["documents"] if d.get("date")] + [v["date"][:10] for v in sub["vitals"]]
    return datetime.fromisoformat(max(days)[:10]) + timedelta(days=1, hours=12)


def with_vital(sub, **vital):
    sub = copy.deepcopy(sub)
    vital["date"] = latest_moment(sub).strftime("%Y-%m-%dT%H:%M:00Z")
    sub["vitals"].append(vital)
    return sub


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
@pytest.mark.parametrize("days", [-40, 365])
def test_invariant_under_date_translation(case, days):
    assert run(shift_all_dates(case["submission"], days)) == run(case["submission"])


@pytest.mark.parametrize("case", READY, ids=lambda c: c["case_id"])
@pytest.mark.parametrize(
    "vital, expected",
    [
        ({"type": "blood_pressure", "systolic": 179, "diastolic": 109}, "READY"),
        ({"type": "blood_pressure", "systolic": 180, "diastolic": 70}, "NOT_CLEARED"),
        ({"type": "blood_pressure", "systolic": 130, "diastolic": 110}, "NOT_CLEARED"),
        ({"type": "temperature", "value_f": 100.4}, "READY"),
        ({"type": "temperature", "value_f": 100.5}, "NOT_CLEARED"),
    ],
    ids=["179/109", "180/70", "130/110", "100.4F", "100.5F"],
)
def test_acute_thresholds_flip_exactly_at_boundary(case, vital, expected):
    decision, cats = run(with_vital(case["submission"], **vital))
    assert decision == expected
    assert cats == (["ACUTE_SAFETY_EXCLUSION"] if expected == "NOT_CLEARED" else [])


@pytest.mark.parametrize("case", READY, ids=lambda c: c["case_id"])
def test_high_reading_then_normal_recheck_is_cleared(case):
    sub = with_vital(case["submission"], type="blood_pressure", systolic=195, diastolic=115)
    recheck = dict(sub["vitals"][-1], systolic=128, diastolic=76)
    recheck["date"] = recheck["date"].replace(":00Z", ":30Z").replace("T12:00", "T12:30")
    sub["vitals"].append(recheck)
    assert run(sub) == ("READY", [])


@pytest.mark.parametrize("case", READY, ids=lambda c: c["case_id"])
def test_lab_window_boundary(case):
    sub = copy.deepcopy(case["submission"])
    proc = date.fromisoformat(sub["procedure"]["procedure_date"])
    limit = 14 if sub["procedure"]["procedure_risk"] == "HIGH" else 30
    for lab in sub["labs"]:
        if lab["code"] in {"CBC", "LAB-CBC"}:
            lab["status"] = "entered-in-error"
    sub["labs"].append({"id": "edge", "code": "CBC", "display": "CBC", "status": "final",
                        "effective_at": (proc - timedelta(days=limit)).isoformat() + "T08:00:00Z"})
    assert run(sub) == ("READY", [])
    sub["labs"][-1]["effective_at"] = (proc - timedelta(days=limit + 1)).isoformat() + "T08:00:00Z"
    assert run(sub) == ("NEEDS_FOLLOW_UP", ["REQUIRED_TESTING"])


@pytest.mark.parametrize("case", READY, ids=lambda c: c["case_id"])
def test_removing_consent_documents_is_a_documentation_gap(case):
    sub = copy.deepcopy(case["submission"])
    sub["documents"] = [d for d in sub["documents"] if "consent" not in (d["type"] or "").lower()]
    decision, cats = run(sub)
    assert decision == "NEEDS_FOLLOW_UP" and "REQUIRED_DOCUMENTATION" in cats


BRANDS = {"Xarelto": "rivaroxaban", "Eliquis": "apixaban", "Coumadin": "warfarin",
          "Pradaxa": "dabigatran", "Savaysa": "edoxaban"}
ANTICOAG_CASES = [c for c in CASES if any(b in str(c["submission"]) for b in BRANDS)]


@pytest.mark.parametrize("case", ANTICOAG_CASES, ids=lambda c: c["case_id"])
def test_brand_and_generic_names_are_interchangeable(case):
    sub = copy.deepcopy(case["submission"])
    blob = str(sub)
    for brand, generic in BRANDS.items():
        for d in sub["documents"]:
            d["text"] = re.sub(rf"\b{brand}\b", generic, d["text"])
        for m in sub["medications"]:
            m["name"] = re.sub(rf"\b{brand}\b", generic, m["name"])
    assert blob != str(sub)
    assert run(sub) == run(case["submission"])
