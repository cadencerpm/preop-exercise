"""One pure function per policy rule; a small registry drives which ones run.

Each rule fn has the signature ``(Facts, list[DocClaim]) -> list[Issue]`` and
declares the ``Field``s it ``requires``. The runner (``non_document_specs`` /
``document_specs`` + ``run_rules``, driven from ``pipeline.py``) skips a rule
outright when any required field is unavailable, rather than each rule having
to null-check its own inputs.

``order`` on every Issue is what makes output ordering deterministic and lets
it match the oracle's issue ordering exactly (see NOTES.md for how the order
values were reverse-engineered from the labeled set).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from . import cite
from .documents import ConsentStatus, DocClaim, DocRole
from .facts import Facts, Field, parse_item_date

# --------------------------------------------------------------------------
# Issue + rule registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Issue:
    category: str
    description: str
    source: str
    details: str
    order: int


RuleFn = Callable[[Facts, list[DocClaim]], list[Issue]]


@dataclass(frozen=True)
class _RuleSpec:
    order: int
    requires: frozenset[Field]
    fn: RuleFn


_REGISTRY: list[_RuleSpec] = []


def rule(*, order: int, requires: frozenset[Field] = frozenset()) -> Callable[[RuleFn], RuleFn]:
    def decorator(fn: RuleFn) -> RuleFn:
        _REGISTRY.append(_RuleSpec(order=order, requires=requires, fn=fn))
        return fn

    return decorator


def non_document_specs() -> list[_RuleSpec]:
    return sorted((spec for spec in _REGISTRY if Field.DOC_ROLES not in spec.requires), key=lambda s: s.order)


def document_specs() -> list[_RuleSpec]:
    return sorted((spec for spec in _REGISTRY if Field.DOC_ROLES in spec.requires), key=lambda s: s.order)


def run_rules(
    specs: list[_RuleSpec],
    facts: Facts,
    unavailable: frozenset[Field],
    claims: list[DocClaim],
) -> list[Issue]:
    issues: list[Issue] = []
    for spec in specs:
        if spec.requires & unavailable:
            continue
        issues.extend(spec.fn(facts, claims))
    return issues


# --------------------------------------------------------------------------
# Order values (reverse-engineered from the oracle's issue ordering; see
# NOTES.md). Gate order values (10-100) live here too so every Issue's sort
# position is defined in one place.
# --------------------------------------------------------------------------

ORDER_MISSING_PROC_DATE = 10
ORDER_MISSING_PROC_RISK = 20
ORDER_HP = 30
ORDER_CONSENT = 40
ORDER_CBC = 50
ORDER_CMP = 60
ORDER_ANTICOAG = 70
ORDER_MISSING_BP = 80
ORDER_MISSING_TEMP = 90
ORDER_MISSING_ANTICOAG_UNKNOWN = 100
ORDER_SAFETY_BP = 110
ORDER_SAFETY_TEMP = 120
ORDER_DOC_REVIEW_UNAVAILABLE = 25

WINDOW_HP_DAYS = 30
WINDOW_LOW_MODERATE_LAB_DAYS = 30
WINDOW_HIGH_RISK_LAB_DAYS = 14

BP_SYSTOLIC_THRESHOLD = 180
BP_DIASTOLIC_THRESHOLD = 110
TEMP_THRESHOLD_F = 100.4

CATEGORY_REQUIRED_DOCUMENTATION = "REQUIRED_DOCUMENTATION"
CATEGORY_REQUIRED_TESTING = "REQUIRED_TESTING"
CATEGORY_ANTICOAGULATION_MANAGEMENT = "ANTICOAGULATION_MANAGEMENT"
CATEGORY_ACUTE_SAFETY_EXCLUSION = "ACUTE_SAFETY_EXCLUSION"
CATEGORY_MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"


def make_issue(category: str, order: int, citation: cite.Citation) -> Issue:
    return Issue(
        category=category,
        description=citation.description,
        source=citation.source,
        details=citation.details,
        order=order,
    )


# --------------------------------------------------------------------------
# Rule 1: History & Physical (content-aware: LLM role + is_current; among
# current H&Ps, code picks the most recent by date and does the window math).
#
# Split into an existence check and a window check (user decision: missing
# data is valuable feedback, so "no current H&P" must be reported even when
# procedure_date is null -- it should not be masked by a different missing-
# data issue). Both share ``_current_hp_candidates``, which itself has no
# dependency on ``facts.proc_date`` at all, so only the window rule needs
# Field.PROC_DATE in ``requires``; the existence rule needs only DOC_ROLES.
# The two are mutually exclusive in practice (window has nothing to check
# when there are no candidates, which is exactly when existence fires), so
# sharing ORDER_HP is safe.
# --------------------------------------------------------------------------


def _current_hp_candidates(facts: Facts, claims: list[DocClaim]) -> list[tuple[date, DocClaim]]:
    candidates: list[tuple[date, DocClaim]] = []
    for claim in claims:
        if claim.role is not DocRole.HISTORY_AND_PHYSICAL or not claim.is_current:
            continue
        doc = facts.documents[claim.index].obj
        doc_date = parse_item_date(doc.get("date"))
        if doc_date is None:
            continue
        candidates.append((doc_date, claim))
    return candidates


@rule(order=ORDER_HP, requires=frozenset({Field.DOC_ROLES}))
def rule_hp_exists(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    if _current_hp_candidates(facts, claims):
        return []
    return [make_issue(CATEGORY_REQUIRED_DOCUMENTATION, ORDER_HP, cite.hp_missing())]


@rule(order=ORDER_HP, requires=frozenset({Field.PROC_DATE, Field.DOC_ROLES}))
def rule_hp_window(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    candidates = _current_hp_candidates(facts, claims)
    if not candidates:
        return []  # no candidate to check the window against; rule_hp_exists already reported it

    hp_date, claim = max(candidates, key=lambda pair: pair[0])
    delta = (facts.proc_date - hp_date).days
    if 0 <= delta <= WINDOW_HP_DAYS:
        return []

    source = facts.documents[claim.index].path
    citation = cite.hp_outside_window(
        source=source, hp_date=hp_date, proc_date=facts.proc_date, delta=delta, window=WINDOW_HP_DAYS
    )
    return [make_issue(CATEGORY_REQUIRED_DOCUMENTATION, ORDER_HP, citation)]


# --------------------------------------------------------------------------
# Rule 1: Signed surgical consent.
# --------------------------------------------------------------------------


@rule(order=ORDER_CONSENT, requires=frozenset({Field.DOC_ROLES}))
def rule_consent(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    consent_claims = [claim for claim in claims if claim.role is DocRole.SURGICAL_CONSENT]
    if not consent_claims:
        return [make_issue(CATEGORY_REQUIRED_DOCUMENTATION, ORDER_CONSENT, cite.consent_missing())]

    claim = max(
        consent_claims,
        key=lambda c: parse_item_date(facts.documents[c.index].obj.get("date")) or date.min,
    )
    if claim.consent_signed is ConsentStatus.SIGNED:
        return []

    doc = facts.documents[claim.index]
    text = str(doc.obj.get("text") or "")
    citation = cite.consent_not_signed(source=doc.path, text=text)
    return [make_issue(CATEGORY_REQUIRED_DOCUMENTATION, ORDER_CONSENT, citation)]


# --------------------------------------------------------------------------
# Rule 2: Required testing by procedure risk.
#
# Split into an existence check and a window check per lab (user decision:
# missing data is valuable feedback, not something to withhold just because
# another field is also missing):
#   - CBC is required at every risk tier, so its existence check runs even
#     with procedure_risk and/or procedure_date null (``requires=frozenset()``).
#   - CMP is only required for HIGH risk, so its existence check needs
#     Field.PROC_RISK to even ask the question -- an unknown risk skips CMP
#     entirely, same as before.
#   - Both windows need to know *which* window (30 vs 14, by risk) and how
#     many days have elapsed, so both require PROC_DATE and PROC_RISK.
# Existence and window are mutually exclusive per lab (window is a no-op
# when there is no result to check), so sharing ORDER_CBC/ORDER_CMP is safe.
# --------------------------------------------------------------------------


def _lab_missing_issue(facts: Facts, *, code: str, order: int) -> list[Issue]:
    if facts.latest_lab.get(code) is not None:
        return []
    citation = cite.lab_missing(code=code, risk=facts.risk)
    return [make_issue(CATEGORY_REQUIRED_TESTING, order, citation)]


def _lab_window_issue(facts: Facts, *, code: str, window: int, order: int) -> list[Issue]:
    ref = facts.latest_lab.get(code)
    if ref is None:
        return []  # no result to check the window against; the existence rule already reported it

    lab_date = parse_item_date(ref.obj.get("effective_at"))
    delta = (facts.proc_date - lab_date).days
    if 0 <= delta <= window:
        return []

    citation = cite.lab_outside_window(
        code=code,
        source=ref.path,
        effective_at=str(ref.obj.get("effective_at")),
        proc_date=facts.proc_date,
        delta=delta,
        window=window,
        risk=facts.risk,
    )
    return [make_issue(CATEGORY_REQUIRED_TESTING, order, citation)]


@rule(order=ORDER_CBC, requires=frozenset())
def rule_cbc_exists(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    return _lab_missing_issue(facts, code="CBC", order=ORDER_CBC)


@rule(order=ORDER_CBC, requires=frozenset({Field.PROC_DATE, Field.PROC_RISK}))
def rule_cbc_window(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    window = WINDOW_HIGH_RISK_LAB_DAYS if facts.risk == "HIGH" else WINDOW_LOW_MODERATE_LAB_DAYS
    return _lab_window_issue(facts, code="CBC", window=window, order=ORDER_CBC)


@rule(order=ORDER_CMP, requires=frozenset({Field.PROC_RISK}))
def rule_cmp_exists(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    if facts.risk != "HIGH":
        return []
    return _lab_missing_issue(facts, code="CMP", order=ORDER_CMP)


@rule(order=ORDER_CMP, requires=frozenset({Field.PROC_DATE, Field.PROC_RISK}))
def rule_cmp_window(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    if facts.risk != "HIGH":
        return []
    return _lab_window_issue(facts, code="CMP", window=WINDOW_HIGH_RISK_LAB_DAYS, order=ORDER_CMP)


# --------------------------------------------------------------------------
# Rule 3: Anticoagulation management. A join: fires only when an anticoagulant
# is active AND no validated document is a clear plan.
# --------------------------------------------------------------------------


def _clear_plan_exists(claims: list[DocClaim]) -> bool:
    return any(
        claim.role is DocRole.PERIOP_ANTICOAG_PLAN and claim.is_clear_plan is True
        for claim in claims
    )


def _plan_doc_source(facts: Facts, claims: list[DocClaim]) -> str | None:
    candidates = [
        (parse_item_date(facts.documents[claim.index].obj.get("date")) or date.min, claim)
        for claim in claims
        if claim.role is DocRole.PERIOP_ANTICOAG_PLAN
    ]
    if not candidates:
        return None
    _, claim = max(candidates, key=lambda pair: pair[0])
    return facts.documents[claim.index].path


@rule(order=ORDER_ANTICOAG, requires=frozenset({Field.DOC_ROLES}))
def rule_anticoag(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    active = [(ref, status) for ref, status in facts.anticoags if status == "active"]
    if not active:
        return []
    if _clear_plan_exists(claims):
        return []

    med_ref, _status = active[0]
    citation = cite.anticoag_missing_plan(
        plan_doc_source=_plan_doc_source(facts, claims), med_source=med_ref.path
    )
    return [make_issue(CATEGORY_ANTICOAGULATION_MANAGEMENT, ORDER_ANTICOAG, citation)]


# --------------------------------------------------------------------------
# Rule 4: Acute safety exclusions. Each check ``requires`` its own vital
# field (Field.BP / Field.TEMP): gates.py marks that field unavailable
# whenever the latest reading is either absent or present-but-valueless (no
# numeric systolic/diastolic, or no numeric value_f), so a rule here never
# needs to null-check its own inputs -- missing vitals can never produce
# NOT_CLEARED; that's MISSING_REQUIRED_DATA, from gates.py.
# --------------------------------------------------------------------------


@rule(order=ORDER_SAFETY_BP, requires=frozenset({Field.BP}))
def rule_safety_bp(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    systolic = facts.latest_bp.obj["systolic"]
    diastolic = facts.latest_bp.obj["diastolic"]
    if systolic < BP_SYSTOLIC_THRESHOLD and diastolic < BP_DIASTOLIC_THRESHOLD:
        return []
    citation = cite.bp_exclusion(source=facts.latest_bp.path, systolic=systolic, diastolic=diastolic)
    return [make_issue(CATEGORY_ACUTE_SAFETY_EXCLUSION, ORDER_SAFETY_BP, citation)]


@rule(order=ORDER_SAFETY_TEMP, requires=frozenset({Field.TEMP}))
def rule_safety_temp(facts: Facts, claims: list[DocClaim]) -> list[Issue]:
    value_f = facts.latest_temp.obj["value_f"]
    if value_f <= TEMP_THRESHOLD_F:
        return []
    citation = cite.temp_exclusion(source=facts.latest_temp.path, value_f=value_f)
    return [make_issue(CATEGORY_ACUTE_SAFETY_EXCLUSION, ORDER_SAFETY_TEMP, citation)]


# --------------------------------------------------------------------------
# Decision fold + explanation.
# --------------------------------------------------------------------------


READY_EXPLANATION = (
    "All required documentation, testing, anticoagulation planning, and safety "
    "checks are satisfied."
)


def decide(issues: list[Issue]) -> str:
    if any(issue.category == CATEGORY_ACUTE_SAFETY_EXCLUSION for issue in issues):
        return "NOT_CLEARED"
    if issues:
        return "NEEDS_FOLLOW_UP"
    return "READY"


def build_explanation(issues: list[Issue], *, review_skipped_note: str | None) -> str:
    if not issues:
        return READY_EXPLANATION
    parts = [f"{issue.category}: {issue.description}" for issue in issues]
    explanation = " | ".join(parts)
    if review_skipped_note:
        explanation = f"{explanation} | {review_skipped_note}"
    return explanation


def issue_to_dict(issue: Issue) -> dict[str, object]:
    return {
        "category": issue.category,
        "description": issue.description,
        "evidence": {"source": issue.source, "details": issue.details},
    }
