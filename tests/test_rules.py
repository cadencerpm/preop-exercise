"""Policy rules, tested against hand-built facts (no reader involved)."""

from __future__ import annotations

from conftest import base_facts, base_submission, categories, consent_facts, hp_facts

from preop.facts import AnticoagulationPlan, MedicationMention, VitalReading, empty_facts


def test_clean_case_is_ready_with_full_audit_trail(run_case):
    out = run_case()
    assert out.decision == "READY"
    assert out.issues == []
    statuses = {c.check: c.status for c in out.checks}
    assert statuses["history_and_physical_current"] == "PASS"
    assert statuses["signed_surgical_consent"] == "PASS"
    assert statuses["cbc_within_window"] == "PASS"
    assert statuses["anticoagulation_plan"] == "NOT_APPLICABLE"
    # every PASS carries evidence a nurse can click through
    assert all(c.evidence for c in out.checks if c.status == "PASS")


# ---- Rule 1: documentation ---------------------------------------------------


def test_hp_exactly_30_days_passes_31_fails(run_case):
    sub = base_submission()
    sub["documents"][0]["date"] = "2026-03-01"  # 30 days before 03/31
    assert run_case(sub).decision == "READY"
    sub["documents"][0]["date"] = "2026-02-28"  # 31 days
    out = run_case(sub)
    assert out.decision == "NEEDS_FOLLOW_UP"
    assert categories(out) == ["REQUIRED_DOCUMENTATION"]
    issue = out.issues[0]
    assert issue.facts["age_days"] == 31 and issue.facts["max_age_days"] == 30
    assert "2026-03-01" in issue.action  # tells the nurse the earliest acceptable date


def test_old_hp_ignored_when_current_one_exists(run_case):
    sub = base_submission()
    sub["documents"].append({"doc_id": "old", "type": "H&P", "date": "2025-06-01", "text": "HISTORY AND PHYSICAL"})
    facts = base_facts() | {"old": hp_facts()}
    assert run_case(sub, facts).decision == "READY"


def test_missing_hp(run_case):
    facts = base_facts()
    del facts["hp"]
    out = run_case(facts=facts)
    assert categories(out) == ["REQUIRED_DOCUMENTATION"]


def test_unsigned_consent_fails_and_superseding_signed_consent_passes(run_case):
    sub = base_submission()
    sub["documents"].append({"doc_id": "c0", "type": "Consent", "date": "2026-03-01",
                             "text": "Patient signature: pending"})
    facts = base_facts() | {"c0": consent_facts("NOT_SIGNED", quote="Patient signature: pending")}
    assert run_case(sub, facts).decision == "READY"  # newer signed consent supersedes
    del facts["consent"]
    out = run_case(sub, facts)
    assert categories(out) == ["REQUIRED_DOCUMENTATION"]
    assert "not signed" in out.issues[0].summary


def test_consent_for_a_different_procedure_fails(run_case):
    facts = base_facts() | {"consent": consent_facts(matches="NO")}
    out = run_case(facts=facts)
    assert categories(out) == ["REQUIRED_DOCUMENTATION"]
    assert "different procedure" in out.issues[0].summary


# ---- Rule 2: testing ----------------------------------------------------------


def test_entered_in_error_lab_is_skipped_for_older_valid_one(run_case):
    sub = base_submission()
    sub["labs"] = [
        {"id": "new", "code": "CBC", "effective_at": "2026-03-25", "status": "entered-in-error"},
        {"id": "old", "code": "CBC", "effective_at": "2026-02-01", "status": "final"},
    ]
    out = run_case(sub)
    assert categories(out) == ["REQUIRED_TESTING"]
    assert out.issues[0].facts["most_recent_result"] == "2026-02-01"
    assert out.issues[0].facts["excluded_results"][0]["status"] == "entered-in-error"


def test_high_risk_needs_cmp_and_bmp_does_not_count(run_case):
    sub = base_submission()
    sub["procedure"]["procedure_risk"] = "HIGH"
    sub["labs"] = [
        {"id": "a", "code": "LAB-CBC", "display": "Complete Blood Count w/ Differential",
         "effective_at": "2026-03-17", "status": "final"},  # 14 days: ok
        {"id": "b", "code": "BMP", "display": "Basic Metabolic Panel", "effective_at": "2026-03-20", "status": "final"},
    ]
    out = run_case(sub)
    assert categories(out) == ["REQUIRED_TESTING"]
    assert out.issues[0].check == "cmp_within_window"


def test_high_risk_cbc_15_days_fails(run_case):
    sub = base_submission()
    sub["procedure"]["procedure_risk"] = "HIGH"
    sub["labs"] = [
        {"id": "a", "code": "CBC", "effective_at": "2026-03-16", "status": "final"},
        {"id": "b", "code": "CMP", "effective_at": "2026-03-20", "status": "final"},
    ]
    out = run_case(sub)
    assert [i.check for i in out.issues] == ["cbc_within_window"]


def test_unknown_risk_reports_missing_data_only_when_labs_ok_for_every_risk(run_case):
    sub = base_submission()
    sub["procedure"]["procedure_risk"] = None
    sub["labs"] = [
        {"id": "a", "code": "CBC", "effective_at": "2026-03-25", "status": "final"},
        {"id": "b", "code": "CMP", "effective_at": "2026-03-25", "status": "final"},
    ]
    assert categories(run_case(sub)) == ["MISSING_REQUIRED_DATA"]


def test_unknown_risk_with_no_cbc_is_also_a_testing_failure(run_case):
    sub = base_submission()
    sub["procedure"]["procedure_risk"] = None
    sub["labs"] = []
    assert categories(run_case(sub)) == ["MISSING_REQUIRED_DATA", "REQUIRED_TESTING"]


def test_unknown_procedure_date_blocks_windows_without_extra_issues(run_case):
    sub = base_submission()
    sub["procedure"]["procedure_date"] = None
    out = run_case(sub)
    assert categories(out) == ["MISSING_REQUIRED_DATA"]
    assert len(out.issues) == 1  # root cause reported once
    blocked = {c.check for c in out.checks if c.status == "BLOCKED"}
    assert {"history_and_physical_current", "cbc_within_window"} <= blocked


def test_procedure_date_recovered_from_documents(run_case):
    sub = base_submission()
    sub["procedure"]["procedure_date"] = None
    sub["documents"][0]["text"] += "\nProcedure date target: 2026-03-31."
    facts = base_facts()
    facts["hp"].procedure_date_stated = "2026-03-31"
    facts["hp"].procedure_date_quote = "Procedure date target: 2026-03-31"
    out = run_case(sub, facts)
    assert out.decision == "READY"
    assert any(f.code == "PROCEDURE_DATE_FROM_DOCUMENT" for f in out.review_flags)


# ---- Rule 3: anticoagulation ----------------------------------------------------

PLAN_TEXT = (
    "1. Hold Xarelto for the 2 days before surgery (last dose 03/28/2026). "
    "Resume Xarelto 24-48 hours after surgery."
)


def _anticoag_case(med_active, note_status=None, plan=True, post=True):
    sub = base_submission()
    if med_active != "absent":
        sub["medications"] = [{"name": "Xarelto 20 mg PO daily", "active": med_active}]
    facts = base_facts()
    if note_status:
        line = "- Xarelto 20 mg - patient note"
        sub["documents"].append({"doc_id": "rn", "type": "Nursing", "date": "2026-03-25", "text": line})
        f = empty_facts()
        f.antithrombotic_mentions = [MedicationMention(drug_as_written="Xarelto", drug_class="ANTICOAGULANT",
                                                       status=note_status, quote=line)]
        facts["rn"] = f
    if plan:
        sub["documents"].append({"doc_id": "plan", "type": "Cardiology", "date": "2026-03-20", "text": PLAN_TEXT})
        f = empty_facts()
        f.anticoagulation_plans = [AnticoagulationPlan(
            drug_as_written="rivaroxaban",  # generic name: matched to brand via lexicon
            pre_procedure_instruction="Hold Xarelto for the 2 days before surgery",
            post_procedure_instruction="Resume Xarelto 24-48 hours after surgery" if post else None,
            deferred_or_pending=False, quote=PLAN_TEXT)]
        facts["plan"] = f
    return sub, facts


def test_brand_name_plan_covers_active_anticoagulant(run_case):
    sub, facts = _anticoag_case(True)
    out = run_case(sub, facts)
    assert out.decision == "READY"
    check = next(c for c in out.checks if c.check == "anticoagulation_plan_rivaroxaban")
    assert check.status == "PASS" and check.facts["post_procedure_instruction"]


def test_plan_missing_post_op_restart_is_incomplete(run_case):
    sub, facts = _anticoag_case(True, post=False)
    out = run_case(sub, facts)
    assert categories(out) == ["ANTICOAGULATION_MANAGEMENT"]
    assert "post-op resume" in out.issues[0].summary


def test_patient_reported_anticoagulant_not_on_med_list_requires_plan(run_case):
    sub, facts = _anticoag_case("absent", note_status="CURRENTLY_TAKING", plan=False)
    assert categories(run_case(sub, facts)) == ["ANTICOAGULATION_MANAGEMENT"]


def test_inactive_and_confirmed_stopped_needs_no_plan(run_case):
    sub, facts = _anticoag_case(False, note_status="STOPPED", plan=False)
    assert run_case(sub, facts).decision == "READY"


def test_unknown_active_status_is_missing_data_not_anticoag_failure(run_case):
    sub, facts = _anticoag_case(None, note_status="UNCERTAIN", plan=False)
    assert categories(run_case(sub, facts)) == ["MISSING_REQUIRED_DATA"]


def test_antiplatelet_alone_does_not_trigger_rule_3(run_case):
    sub = base_submission()
    sub["medications"] = [{"name": "ASA 81 mg PO daily", "active": True},
                          {"name": "clopidogrel 75 mg PO daily", "active": True}]
    out = run_case(sub)
    assert out.decision == "READY"
    rule3 = next(c for c in out.checks if c.rule == "RULE_3_ANTICOAGULATION")
    assert rule3.status == "NOT_APPLICABLE" and "antiplatelet" in rule3.summary


# ---- Rule 4: acute safety ---------------------------------------------------------


def _with_note_vitals(readings: list[VitalReading], text: str, date: str = "2026-03-28"):
    sub = base_submission()
    sub["documents"].append({"doc_id": "rn", "type": "Nursing", "date": date, "text": text})
    f = empty_facts()
    f.vital_readings = readings
    return sub, base_facts() | {"rn": f}


def _bp(s, d, time, quote):
    return VitalReading(kind="BLOOD_PRESSURE", systolic=s, diastolic=d, temperature=None,
                        temperature_unit=None, date=None, time=time, quote=quote)


def _temp(v, unit, quote):
    return VitalReading(kind="TEMPERATURE", systolic=None, diastolic=None, temperature=v,
                        temperature_unit=unit, date=None, time=None, quote=quote)


def test_note_vitals_newer_than_structured_are_used(run_case):
    sub, facts = _with_note_vitals([_bp(157, 117, "09:05", "BP 157/117")], "09:05 BP 157/117")
    out = run_case(sub, facts)
    assert out.decision == "NOT_CLEARED"
    assert categories(out) == ["ACUTE_SAFETY_EXCLUSION"]


def test_recheck_within_range_clears_exclusion(run_case):
    text = "08:06 NIBP 189/112. ADDENDUM 08:30: Re-measured 137/78 mmHg"
    sub, facts = _with_note_vitals([_bp(189, 112, "08:06", "NIBP 189/112"), _bp(137, 78, "08:30", "Re-measured 137/78")], text)
    assert run_case(sub, facts).decision == "READY"


def test_recheck_still_high_is_not_cleared(run_case):
    text = "09:06: BP 196/110 mmHg. rechecked at 09:38: BP 181/98 mmHg"
    sub, facts = _with_note_vitals([_bp(196, 110, "09:06", "BP 196/110"), _bp(181, 98, "09:38", "BP 181/98")], text)
    assert run_case(sub, facts).decision == "NOT_CLEARED"


def test_diastolic_exactly_110_excludes_and_109_does_not(run_case):
    sub, facts = _with_note_vitals([_bp(148, 110, "09:00", "148/110")], "BP 148/110")
    assert run_case(sub, facts).decision == "NOT_CLEARED"
    sub, facts = _with_note_vitals([_bp(179, 109, "09:00", "179/109")], "BP 179/109")
    assert run_case(sub, facts).decision == "READY"


def test_celsius_converted_before_threshold(run_case):
    sub, facts = _with_note_vitals([_temp(37.9, "C", "T 37.9 C")], "T 37.9 C")  # 100.22 F
    assert run_case(sub, facts).decision == "READY"
    sub, facts = _with_note_vitals([_temp(38.1, "C", "T 38.1 C")], "T 38.1 C")  # 100.58 F
    assert run_case(sub, facts).decision == "NOT_CLEARED"
    sub, facts = _with_note_vitals([_temp(100.4, "F", "100.4 F")], "100.4 F")  # not > 100.4
    assert run_case(sub, facts).decision == "READY"


def test_missing_temperature_is_missing_data(run_case):
    sub = base_submission()
    sub["vitals"] = [v for v in sub["vitals"] if v["type"] != "temperature"]
    out = run_case(sub)
    assert out.decision == "NEEDS_FOLLOW_UP"
    assert categories(out) == ["MISSING_REQUIRED_DATA"]


def test_same_day_untimed_reading_is_judged_worst_case(run_case):
    sub = base_submission()
    sub["vitals"].append({"type": "blood_pressure", "systolic": 122, "diastolic": 70, "date": "2026-03-28T10:00:00Z"})
    sub, facts = sub, base_facts()
    sub["documents"].append({"doc_id": "rn", "type": "Nursing", "date": "2026-03-28", "text": "BP 185/90"})
    f = empty_facts()
    f.vital_readings = [_bp(185, 90, None, "BP 185/90")]
    facts["rn"] = f
    out = run_case(sub, facts)
    assert out.decision == "NOT_CLEARED"
    assert any(fl.code == "VITAL_ORDER_AMBIGUOUS" for fl in out.review_flags)


# ---- Decision precedence -----------------------------------------------------------


def test_not_cleared_still_lists_other_issues(run_case):
    sub, facts = _with_note_vitals([_temp(101.5, "F", "101.5 F")], "Temp 101.5 F")
    facts["consent"] = consent_facts("NOT_SIGNED", quote="Signed by patient: J Doe, 03/12/2026.")
    out = run_case(sub, facts)
    assert out.decision == "NOT_CLEARED"
    assert categories(out) == ["ACUTE_SAFETY_EXCLUSION", "REQUIRED_DOCUMENTATION"]
