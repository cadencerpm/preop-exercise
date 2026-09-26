from __future__ import annotations

from triage.documents import ConsentStatus, DocClaim, DocRole, FakeClassifier
from triage.facts import Facts, normalize
from triage.pipeline import run as run_pipeline
from triage.rules import (
    Issue,
    rule_anticoag,
    rule_cbc_exists,
    rule_cbc_window,
    rule_cmp_exists,
    rule_cmp_window,
    rule_consent,
    rule_hp_exists,
    rule_hp_window,
)


def _submission(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "procedure": {"procedure_date": "2026-03-11", "procedure_risk": "HIGH"},
        "vitals": [
            {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-01"},
            {"type": "temperature", "value_f": 98.6, "date": "2026-03-01"},
        ],
        "labs": [],
        "medications": [],
        "documents": [],
    }
    base.update(overrides)
    return base


def _hp_claim(index: int = 0, *, is_current: bool | None = True) -> DocClaim:
    # "H&P" is a non-empty, verbatim substring of every H&P document's text
    # used in this module's fixtures ("H&P", "very old H&P", "stale H&P",
    # ...), so this claim survives documents.validate() for the tests that
    # exercise run_pipeline (and is simply unused by tests that call a rule
    # function directly).
    return DocClaim(
        index=index,
        role=DocRole.HISTORY_AND_PHYSICAL,
        is_current=is_current,
        consent_signed=None,
        is_clear_plan=None,
        excerpt="H&P",
    )


def _hp_issues(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    """rule_hp was split into an existence check and a window check (fix 5);
    calling both together, in registry order, reproduces the old combined
    rule_hp's behavior for tests that don't care about the split itself."""

    return rule_hp_exists(facts, claims) + rule_hp_window(facts, claims)


def _testing_issues(facts: Facts) -> list[Issue]:
    """rule_testing was split into CBC/CMP existence + window checks (fix 5);
    calling all four together, in registry order, reproduces the old combined
    rule_testing's behavior for tests that don't care about the split itself."""

    return (
        rule_cbc_exists(facts, [])
        + rule_cbc_window(facts, [])
        + rule_cmp_exists(facts, [])
        + rule_cmp_window(facts, [])
    )


# --------------------------------------------------------------------------
# Rule 2: required testing by procedure risk.
# --------------------------------------------------------------------------


def test_high_risk_requires_both_cbc_and_cmp_within_14_days() -> None:
    facts = normalize(
        _submission(
            procedure={"procedure_date": "2026-03-11", "procedure_risk": "HIGH"},
            labs=[{"code": "CBC", "effective_at": "2026-03-05"}],  # 6 days prior, in window
        )
    )
    issues = _testing_issues(facts)
    assert [issue.description for issue in issues] == ["CMP missing"]


def test_low_risk_only_requires_cbc_not_cmp() -> None:
    facts = normalize(
        _submission(
            procedure={"procedure_date": "2026-03-11", "procedure_risk": "LOW"},
            labs=[],
        )
    )
    issues = _testing_issues(facts)
    assert [issue.description for issue in issues] == ["CBC missing"]


def test_cbc_window_boundary_is_inclusive_of_14_days() -> None:
    facts = normalize(
        _submission(
            procedure={"procedure_date": "2026-03-15", "procedure_risk": "HIGH"},
            labs=[
                {"code": "CBC", "effective_at": "2026-03-01"},  # exactly 14 days prior
                {"code": "CMP", "effective_at": "2026-03-01"},
            ],
        )
    )
    assert _testing_issues(facts) == []


def test_cbc_window_boundary_fails_at_15_days() -> None:
    facts = normalize(
        _submission(
            procedure={"procedure_date": "2026-03-16", "procedure_risk": "HIGH"},
            labs=[
                {"code": "CBC", "effective_at": "2026-03-01"},  # 15 days prior
                {"code": "CMP", "effective_at": "2026-03-10"},
            ],
        )
    )
    issues = _testing_issues(facts)
    assert [issue.description for issue in issues] == ["CBC outside 14-day window for HIGH risk procedure"]


# --------------------------------------------------------------------------
# Rule 1: H&P window (content-aware selection).
# --------------------------------------------------------------------------


def test_hp_window_boundary_is_inclusive_of_30_days() -> None:
    facts = normalize(
        _submission(
            procedure={"procedure_date": "2026-03-31", "procedure_risk": "LOW"},
            documents=[{"type": "History and Physical", "date": "2026-03-01", "text": "H&P"}],
        )
    )
    assert _hp_issues(facts, [_hp_claim(0)]) == []


def test_hp_window_boundary_fails_at_31_days() -> None:
    facts = normalize(
        _submission(
            procedure={"procedure_date": "2026-04-01", "procedure_risk": "LOW"},
            documents=[{"type": "History and Physical", "date": "2026-03-01", "text": "H&P"}],
        )
    )
    issues = _hp_issues(facts, [_hp_claim(0)])
    assert [issue.description for issue in issues] == ["H&P outside 30-day window"]


def test_hp_missing_when_no_current_hp_claim_exists() -> None:
    facts = normalize(
        _submission(documents=[{"type": "Anesthesia Note", "date": "2026-03-01", "text": "n/a"}])
    )
    issues = _hp_issues(facts, [])
    assert [issue.description for issue in issues] == ["History and Physical document missing"]
    assert issues[0].source == "documents"


def test_hp_with_is_current_none_is_treated_as_not_current() -> None:
    # User decision: an unclear is_current (None) is treated the same as
    # False -- it never counts as the current episode's H&P -- rather than
    # being treated as an ambiguous "maybe current" that would still pass.
    facts = normalize(
        _submission(
            documents=[{"type": "History and Physical", "date": "2026-03-05", "text": "H&P, currency unclear"}]
        )
    )
    issues = _hp_issues(facts, [_hp_claim(0, is_current=None)])
    assert [issue.description for issue in issues] == ["History and Physical document missing"]


def test_hp_picks_most_recent_current_document_ignoring_retained_priors() -> None:
    facts = normalize(
        _submission(
            procedure={"procedure_date": "2026-03-11", "procedure_risk": "LOW"},
            documents=[
                {"type": "H&P (current)", "date": "2026-03-05", "text": "current"},
                {"type": "H&P (prior)", "date": "2026-01-01", "text": "retained prior"},
            ],
        )
    )
    claims = [_hp_claim(0, is_current=True), _hp_claim(1, is_current=False)]
    assert _hp_issues(facts, claims) == []


# --------------------------------------------------------------------------
# Rule 1: signed surgical consent.
# --------------------------------------------------------------------------


def test_consent_missing_when_no_consent_claim_exists() -> None:
    facts = normalize(_submission())
    issues = rule_consent(facts, [])
    assert [issue.description for issue in issues] == ["Signed surgical consent missing"]
    assert issues[0].source == "documents"


def test_consent_unsigned_cites_the_document_and_quotes_its_text() -> None:
    facts = normalize(
        _submission(documents=[{"type": "Surgical Consent", "date": "2026-03-01", "text": "Unsigned; awaiting signature."}])
    )
    claim = DocClaim(
        index=0,
        role=DocRole.SURGICAL_CONSENT,
        is_current=None,
        consent_signed=ConsentStatus.UNSIGNED,
        is_clear_plan=None,
        excerpt="",
    )
    issues = rule_consent(facts, [claim])
    assert issues[0].description == "Surgical consent not clearly signed"
    assert issues[0].source == "documents[0]"
    assert issues[0].details == "Consent document text does not clearly indicate signed consent: Unsigned; awaiting signature."


def test_consent_signed_raises_no_issue() -> None:
    facts = normalize(
        _submission(documents=[{"type": "Surgical Consent", "date": "2026-03-01", "text": "Signed."}])
    )
    claim = DocClaim(
        index=0,
        role=DocRole.SURGICAL_CONSENT,
        is_current=None,
        consent_signed=ConsentStatus.SIGNED,
        is_clear_plan=None,
        excerpt="",
    )
    assert rule_consent(facts, [claim]) == []


# --------------------------------------------------------------------------
# Rule 3: anticoagulation management.
# --------------------------------------------------------------------------


def test_lisinopril_and_metformin_distractors_never_trigger_rule_3() -> None:
    facts = normalize(
        _submission(medications=[{"name": "lisinopril", "active": True}, {"name": "metformin", "active": True}])
    )
    assert rule_anticoag(facts, []) == []


def test_active_anticoagulant_with_no_plan_document_cites_bare_documents() -> None:
    facts = normalize(_submission(medications=[{"name": "apixaban", "active": True}]))
    issues = rule_anticoag(facts, [])
    assert issues[0].description == "Missing perioperative anticoagulation plan"
    assert issues[0].source == "documents"
    assert issues[0].details == (
        "Active anticoagulant medication present (medications[0]) but no clear perioperative plan document found"
    )


def test_active_anticoagulant_with_inadequate_plan_document_cites_it() -> None:
    facts = normalize(
        _submission(
            medications=[{"name": "apixaban", "active": True}],
            documents=[{"type": "Anticoag Plan", "date": "2026-03-01", "text": "plan pending"}],
        )
    )
    claim = DocClaim(
        index=0,
        role=DocRole.PERIOP_ANTICOAG_PLAN,
        is_current=None,
        consent_signed=None,
        is_clear_plan=False,
        excerpt="",
    )
    issues = rule_anticoag(facts, [claim])
    assert issues[0].source == "documents[0]"


def test_active_anticoagulant_with_a_clear_plan_raises_no_issue() -> None:
    facts = normalize(
        _submission(
            medications=[{"name": "apixaban", "active": True}],
            documents=[{"type": "Anticoag Plan", "date": "2026-03-01", "text": "hold 2 days, resume post-op day 1"}],
        )
    )
    claim = DocClaim(
        index=0,
        role=DocRole.PERIOP_ANTICOAG_PLAN,
        is_current=None,
        consent_signed=None,
        is_clear_plan=True,
        excerpt="",
    )
    assert rule_anticoag(facts, [claim]) == []


def test_unknown_anticoag_status_never_triggers_rule_3() -> None:
    # active=None is a gates.py MISSING issue, not a rule_anticoag issue (it is
    # not "active", so the join's left-hand side is false).
    facts = normalize(_submission(medications=[{"name": "warfarin", "active": None}]))
    assert rule_anticoag(facts, []) == []


# --------------------------------------------------------------------------
# Fix 3: a present-but-valueless latest vital must never silently read as
# READY -- gates.py reports it as missing (see test_gates.py for the gate
# itself), which keeps rule_safety_bp/rule_safety_temp from running at all
# (they ``require`` Field.BP/Field.TEMP) instead of them no-oping quietly.
# --------------------------------------------------------------------------


def test_valueless_latest_bp_never_yields_ready() -> None:
    raw = _submission(
        procedure={"procedure_date": "2026-03-01", "procedure_risk": "LOW"},
        vitals=[
            {"type": "blood_pressure", "systolic": None, "diastolic": None, "date": "2026-02-20"},
            {"type": "temperature", "value_f": 98.6, "date": "2026-02-20"},
        ],
        labs=[{"code": "CBC", "effective_at": "2026-02-25"}],
    )
    output = run_pipeline(raw, FakeClassifier([]))
    assert output["decision"] == "NEEDS_FOLLOW_UP"
    categories = {issue["category"] for issue in output["issues"]}
    assert "MISSING_REQUIRED_DATA" in categories
    assert "ACUTE_SAFETY_EXCLUSION" not in categories


def test_valueless_latest_temp_never_yields_ready() -> None:
    raw = _submission(
        procedure={"procedure_date": "2026-03-01", "procedure_risk": "LOW"},
        vitals=[
            {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-02-20"},
            {"type": "temperature", "value_f": None, "date": "2026-02-20"},
        ],
        labs=[{"code": "CBC", "effective_at": "2026-02-25"}],
    )
    output = run_pipeline(raw, FakeClassifier([]))
    assert output["decision"] == "NEEDS_FOLLOW_UP"
    categories = {issue["category"] for issue in output["issues"]}
    assert "MISSING_REQUIRED_DATA" in categories
    assert "ACUTE_SAFETY_EXCLUSION" not in categories


# --------------------------------------------------------------------------
# Fix 5: existence checks (CBC missing / H&P missing) must not be suppressed
# just because a *different* field (procedure_risk / procedure_date) is also
# null -- missing data is valuable feedback on its own. Window checks still
# require both procedure_date and procedure_risk, so they're skipped instead.
# These are pipeline-level (not direct rule calls) because the point is the
# registry's ``requires``-based skip, which only applies through run_pipeline.
# --------------------------------------------------------------------------


def test_null_risk_with_no_cbc_reports_missing_risk_and_missing_cbc() -> None:
    raw = _submission(
        procedure={"procedure_date": "2026-03-11", "procedure_risk": None},
        labs=[],
        documents=[{"type": "History and Physical", "date": "2026-03-01", "text": "H&P"}],
    )
    output = run_pipeline(raw, FakeClassifier([_hp_claim(0)]))
    descriptions = [issue["description"] for issue in output["issues"]]
    assert "Missing procedure risk" in descriptions
    assert "CBC missing" in descriptions


def test_null_risk_never_reports_a_cmp_issue() -> None:
    raw = _submission(
        procedure={"procedure_date": "2026-03-11", "procedure_risk": None},
        labs=[{"code": "CBC", "effective_at": "2026-03-05"}],  # CBC present; no CMP at all
        documents=[{"type": "History and Physical", "date": "2026-03-01", "text": "H&P"}],
    )
    output = run_pipeline(raw, FakeClassifier([_hp_claim(0)]))
    descriptions = [issue["description"] for issue in output["issues"]]
    assert not any("CMP" in description for description in descriptions)


def test_null_date_with_no_hp_doc_reports_missing_date_and_missing_hp() -> None:
    raw = _submission(
        procedure={"procedure_date": None, "procedure_risk": "LOW"},
        labs=[{"code": "CBC", "effective_at": "2026-03-05"}],
        documents=[],
    )
    output = run_pipeline(raw, FakeClassifier([]))
    descriptions = [issue["description"] for issue in output["issues"]]
    assert "Missing procedure date" in descriptions
    assert "History and Physical document missing" in descriptions


def test_null_date_with_stale_hp_reports_only_missing_date_no_window_issue() -> None:
    raw = _submission(
        procedure={"procedure_date": None, "procedure_risk": "LOW"},
        labs=[{"code": "CBC", "effective_at": "2026-03-05"}],
        documents=[
            {"type": "History and Physical", "date": "2020-01-01", "text": "very old H&P"},
            {"type": "Surgical Consent", "date": "2020-01-01", "text": "Signed."},
        ],
    )
    claims = [
        _hp_claim(0, is_current=True),
        DocClaim(
            index=1,
            role=DocRole.SURGICAL_CONSENT,
            is_current=None,
            consent_signed=ConsentStatus.SIGNED,
            is_clear_plan=None,
            excerpt="Signed",
        ),
    ]
    output = run_pipeline(raw, FakeClassifier(claims))
    descriptions = [issue["description"] for issue in output["issues"]]
    # Only the gate fires -- no H&P window issue, because the window check is
    # skipped outright when procedure_date is unavailable (it never even
    # computes a delta against a null date).
    assert descriptions == ["Missing procedure date"]


# --------------------------------------------------------------------------
# Accumulation + short-circuit (pipeline-level; see pipeline.py docstring).
# --------------------------------------------------------------------------


def test_missing_data_gate_and_safety_issue_accumulate_together() -> None:
    raw = _submission(
        procedure={"procedure_date": None, "procedure_risk": "HIGH"},
        vitals=[
            {"type": "blood_pressure", "systolic": 184, "diastolic": 111, "date": "2026-03-01"},
            {"type": "temperature", "value_f": 98.6, "date": "2026-03-01"},
        ],
        # CBC/CMP present so this test isolates proc_date + safety accumulation
        # without also tripping the (now date-independent) CBC/CMP existence
        # checks; the window checks are skipped anyway since proc_date is null.
        labs=[
            {"code": "CBC", "effective_at": "2026-03-01"},
            {"code": "CMP", "effective_at": "2026-03-01"},
        ],
    )
    output = run_pipeline(raw, FakeClassifier([]))
    categories = [issue["category"] for issue in output["issues"]]
    assert categories == ["MISSING_REQUIRED_DATA", "ACUTE_SAFETY_EXCLUSION"]
    assert output["decision"] == "NOT_CLEARED"


def test_short_circuit_skips_the_classifier_but_keeps_testing_and_missing_issues() -> None:
    raw = _submission(
        procedure={"procedure_date": "2026-03-11", "procedure_risk": "LOW"},
        vitals=[{"type": "blood_pressure", "systolic": 184, "diastolic": 111, "date": "2026-03-01"}],
        labs=[],
        documents=[{"type": "History and Physical", "date": "2026-03-01", "text": "H&P"}],
    )
    classifier = FakeClassifier([_hp_claim(0)])
    output = run_pipeline(raw, classifier)

    assert classifier.calls == 0
    categories = [issue["category"] for issue in output["issues"]]
    assert "ACUTE_SAFETY_EXCLUSION" in categories
    assert "REQUIRED_TESTING" in categories  # CBC missing -- a non-document rule
    assert output["decision"] == "NOT_CLEARED"


def test_safety_still_fires_when_document_classification_is_unavailable() -> None:
    # review_documents_on_not_cleared=True forces the classifier to actually be
    # called despite the safety exclusion, so this exercises the "LLM failed"
    # degrade path specifically (as opposed to the short-circuit skip).
    raw = _submission(
        vitals=[
            {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-01"},
            {"type": "temperature", "value_f": 101.0, "date": "2026-03-01"},
        ],
        documents=[{"type": "History and Physical", "date": "2026-03-01", "text": "H&P"}],
    )

    class FailingClassifier:
        def classify(self, docs: list[dict[str, object]]) -> None:
            return None

    output = run_pipeline(raw, FailingClassifier(), review_documents_on_not_cleared=True)
    categories = [issue["category"] for issue in output["issues"]]
    assert "ACUTE_SAFETY_EXCLUSION" in categories
    assert "MISSING_REQUIRED_DATA" in categories  # "Document review unavailable"
    assert output["decision"] == "NOT_CLEARED"


def test_review_documents_on_not_cleared_accumulates_document_and_safety_issues() -> None:
    # Mirrors case_00002's structure (a real document issue alongside a safety
    # exclusion) without relying on the oracle's own H&P selection quirk. Every
    # other rule is deliberately satisfied so only the H&P and safety issues
    # remain, demonstrating "accumulate, don't short-circuit" for rules.
    raw = _submission(
        procedure={"procedure_date": "2026-04-01", "procedure_risk": "LOW"},
        vitals=[
            {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-20"},
            {"type": "temperature", "value_f": 101.0, "date": "2026-03-20"},
        ],
        labs=[{"code": "CBC", "effective_at": "2026-03-25"}],
        documents=[
            {"type": "History and Physical", "date": "2026-01-01", "text": "stale H&P"},
            {"type": "Surgical Consent", "date": "2026-03-20", "text": "Signed consent."},
        ],
    )
    claims = [
        _hp_claim(0, is_current=True),
        DocClaim(
            index=1,
            role=DocRole.SURGICAL_CONSENT,
            is_current=None,
            consent_signed=ConsentStatus.SIGNED,
            is_clear_plan=None,
            excerpt="Signed",
        ),
    ]
    classifier = FakeClassifier(claims)
    output = run_pipeline(raw, classifier, review_documents_on_not_cleared=True)

    assert classifier.calls == 1
    categories = [issue["category"] for issue in output["issues"]]
    assert categories == ["REQUIRED_DOCUMENTATION", "ACUTE_SAFETY_EXCLUSION"]
    assert output["decision"] == "NOT_CLEARED"
