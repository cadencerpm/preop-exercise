from __future__ import annotations

from datetime import date

from triage.facts import Field, canonical_lab_code, normalize, parse_item_date
from triage.rules import rule_cbc_window


def _submission(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "procedure": {"procedure_date": "2026-03-11", "procedure_risk": "HIGH"},
        "vitals": [],
        "labs": [],
        "medications": [],
        "documents": [],
    }
    base.update(overrides)
    return base


def test_canonical_lab_code_aliases_known_codes() -> None:
    assert canonical_lab_code("LAB-CBC") == "CBC"
    assert canonical_lab_code("LAB-CMP") == "CMP"
    assert canonical_lab_code("CBC") == "CBC"
    assert canonical_lab_code("HBA1C") == "HBA1C"  # not aliased; simply untracked
    assert canonical_lab_code(None) is None


def test_parse_item_date_handles_date_and_datetime_strings() -> None:
    assert parse_item_date("2026-02-19") == date(2026, 2, 19)
    assert parse_item_date("2026-02-19T09:10:00Z") == date(2026, 2, 19)
    assert parse_item_date(None) is None
    assert parse_item_date("not-a-date") is None
    assert parse_item_date("") is None


def test_hba1c_is_a_distractor_and_is_never_tracked() -> None:
    facts = normalize(
        _submission(
            labs=[
                {"code": "HBA1C", "effective_at": "2026-03-01", "display": "A1c"},
            ]
        )
    )
    assert facts.latest_lab == {}


def test_lab_most_recent_reduction_has_no_fallback_to_older_in_window_result() -> None:
    # The most-recent CBC (labs[0]) is dated *after* the procedure, so it is
    # out of window; an older CBC (labs[1]) would be in window. Policy says
    # only the most recent counts -- no fallback to the older, in-window one.
    # A fallback implementation would silently accept labs[1] and raise no
    # issue here; this asserts the opposite: a REQUIRED_TESTING issue that
    # cites the newest (labs[0]), proving there is no fallback.
    facts = normalize(
        _submission(
            labs=[
                {"code": "CBC", "effective_at": "2026-03-15"},  # newest by date, but after the procedure -> out of window
                {"code": "CBC", "effective_at": "2026-03-05"},  # older, 6 days prior -> would be in window
            ]
        )
    )
    assert facts.latest_lab["CBC"].obj["effective_at"] == "2026-03-15"
    assert facts.latest_lab["CBC"].path == "labs[0]"

    issues = rule_cbc_window(facts, [])
    assert [issue.description for issue in issues] == ["CBC outside 14-day window for HIGH risk procedure"]
    assert issues[0].source == "labs[0]"


def test_lab_most_recent_reduction_handles_mixed_date_only_and_datetime_formats() -> None:
    # A date-only string parses naive (midnight); a "...Z" string parses
    # timezone-aware. Comparing the two directly in max() used to raise
    # TypeError; both must normalize to UTC so the newest can be picked
    # without crashing.
    facts = normalize(
        _submission(
            labs=[
                {"code": "CBC", "effective_at": "2026-03-05"},
                {"code": "CBC", "effective_at": "2026-03-04T00:00:00Z"},
            ]
        )
    )
    assert facts.latest_lab["CBC"].obj["effective_at"] == "2026-03-05"


def test_lab_code_alias_reduces_together_with_canonical_form() -> None:
    facts = normalize(
        _submission(
            labs=[
                {"code": "LAB-CBC", "effective_at": "2026-02-01T00:00:00Z"},
                {"code": "CBC", "effective_at": "2026-03-01T00:00:00Z"},
            ]
        )
    )
    assert facts.latest_lab["CBC"].obj["effective_at"] == "2026-03-01T00:00:00Z"


def test_vitals_most_recent_reduction_applies_to_bp_and_temp_independently() -> None:
    facts = normalize(
        _submission(
            vitals=[
                {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-01T00:00:00Z"},
                {"type": "blood_pressure", "systolic": 184, "diastolic": 111, "date": "2026-03-09T00:00:00Z"},
                {"type": "temperature", "value_f": 98.6, "date": "2026-03-01T00:00:00Z"},
                {"type": "temperature", "value_f": 101.0, "date": "2026-03-09T00:00:00Z"},
            ]
        )
    )
    assert facts.latest_bp.obj["systolic"] == 184
    assert facts.latest_bp.path == "vitals[1]"
    assert facts.latest_temp.obj["value_f"] == 101.0
    assert facts.latest_temp.path == "vitals[3]"


def test_vitals_most_recent_reduction_handles_mixed_date_only_and_datetime_formats() -> None:
    # Same mixed-format crash as labs (see the CBC test above), exercised for
    # both vital types.
    facts = normalize(
        _submission(
            vitals=[
                {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-05"},
                {"type": "blood_pressure", "systolic": 184, "diastolic": 111, "date": "2026-03-04T00:00:00Z"},
                {"type": "temperature", "value_f": 98.6, "date": "2026-03-05"},
                {"type": "temperature", "value_f": 101.0, "date": "2026-03-04T00:00:00Z"},
            ]
        )
    )
    assert facts.latest_bp.obj["date"] == "2026-03-05"
    assert facts.latest_temp.obj["date"] == "2026-03-05"


def test_undated_vital_counts_as_absent_not_as_a_fallback_candidate() -> None:
    facts = normalize(
        _submission(
            vitals=[
                {"type": "blood_pressure", "systolic": 120, "diastolic": 80},  # no date
            ]
        )
    )
    assert facts.latest_bp is None


def test_medication_tri_state_active_true_false_null() -> None:
    facts = normalize(
        _submission(
            medications=[
                {"name": "apixaban", "active": True},
                {"name": "warfarin", "active": False},
                {"name": "warfarin", "active": None},
            ]
        )
    )
    statuses = {ref.path: status for ref, status in facts.anticoags}
    assert statuses == {
        "medications[0]": "active",
        "medications[1]": "inactive",
        "medications[2]": "unknown",
    }


def test_lisinopril_and_metformin_are_distractors_not_anticoagulants() -> None:
    facts = normalize(
        _submission(
            medications=[
                {"name": "lisinopril", "active": True},
                {"name": "metformin", "active": True},
            ]
        )
    )
    assert facts.anticoags == []


def test_missing_procedure_date_and_risk_parse_to_none() -> None:
    facts = normalize(_submission(procedure={"procedure_date": None, "procedure_risk": None}))
    assert facts.proc_date is None
    assert facts.risk is None


def test_documents_are_kept_in_order_with_provenance_paths() -> None:
    facts = normalize(
        _submission(
            documents=[
                {"type": "History and Physical", "date": "2026-03-01", "text": "a"},
                {"type": "Surgical Consent", "date": "2026-03-02", "text": "b"},
            ]
        )
    )
    assert [ref.path for ref in facts.documents] == ["documents[0]", "documents[1]"]
    assert facts.documents[0].obj["text"] == "a"


def test_field_enum_members_match_the_plan() -> None:
    assert {f.value for f in Field} == {"proc_date", "proc_risk", "bp", "temp", "doc_roles"}
