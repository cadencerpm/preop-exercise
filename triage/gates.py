"""Null/missing-field gates: run unconditionally, before any rule.

Each gate emits a MISSING_REQUIRED_DATA issue for one missing fact and marks
that fact unavailable so window/threshold rules that need it are skipped
outright (rather than each rule re-deriving "missing" on its own). Anticoag
status is a tri-state per medication: an ``active=null`` anticoagulant is
reported here as MISSING and deliberately does NOT feed Rule 3 (see rules.py).

The latest BP/temp gates fire in two shapes: no dated vital of that type at
all (``facts.latest_bp``/``latest_temp`` is ``None``), or one exists -- and
most-recent reduction already picked it, with no fallback to an older
reading -- but its required numeric field(s) are null. Both shapes mark the
same ``Field`` unavailable, so ``rule_safety_bp``/``rule_safety_temp`` (which
``requires`` that field) are skipped either way instead of silently no-oping
into a false READY.
"""

from __future__ import annotations

from . import cite
from .facts import Facts, Field
from .rules import (
    CATEGORY_MISSING_REQUIRED_DATA,
    ORDER_MISSING_ANTICOAG_UNKNOWN,
    ORDER_MISSING_BP,
    ORDER_MISSING_PROC_DATE,
    ORDER_MISSING_PROC_RISK,
    ORDER_MISSING_TEMP,
    Issue,
    make_issue,
)


def _bp_has_values(obj: dict[str, object]) -> bool:
    systolic = obj.get("systolic")
    diastolic = obj.get("diastolic")
    return isinstance(systolic, (int, float)) and isinstance(diastolic, (int, float))


def _temp_has_value(obj: dict[str, object]) -> bool:
    return isinstance(obj.get("value_f"), (int, float))


def gates(facts: Facts) -> tuple[list[Issue], frozenset[Field]]:
    issues: list[Issue] = []
    unavailable: set[Field] = set()

    if facts.proc_date is None:
        issues.append(make_issue(CATEGORY_MISSING_REQUIRED_DATA, ORDER_MISSING_PROC_DATE, cite.missing_procedure_date()))
        unavailable.add(Field.PROC_DATE)

    if facts.risk is None:
        issues.append(make_issue(CATEGORY_MISSING_REQUIRED_DATA, ORDER_MISSING_PROC_RISK, cite.missing_procedure_risk()))
        unavailable.add(Field.PROC_RISK)

    if facts.latest_bp is None:
        issues.append(make_issue(CATEGORY_MISSING_REQUIRED_DATA, ORDER_MISSING_BP, cite.missing_latest_bp()))
        unavailable.add(Field.BP)
    elif not _bp_has_values(facts.latest_bp.obj):
        # A dated blood_pressure vital exists and reduction already picked it
        # as the most recent -- per policy there is no fallback to an older
        # reading -- but it is missing the numeric values a safety check
        # needs. Treat it the same as "no latest BP" rather than silently
        # letting rule_safety_bp no-op into READY.
        citation = cite.missing_latest_bp_values(source=facts.latest_bp.path)
        issues.append(make_issue(CATEGORY_MISSING_REQUIRED_DATA, ORDER_MISSING_BP, citation))
        unavailable.add(Field.BP)

    if facts.latest_temp is None:
        issues.append(make_issue(CATEGORY_MISSING_REQUIRED_DATA, ORDER_MISSING_TEMP, cite.missing_latest_temp()))
        unavailable.add(Field.TEMP)
    elif not _temp_has_value(facts.latest_temp.obj):
        citation = cite.missing_latest_temp_values(source=facts.latest_temp.path)
        issues.append(make_issue(CATEGORY_MISSING_REQUIRED_DATA, ORDER_MISSING_TEMP, citation))
        unavailable.add(Field.TEMP)

    for ref, status in facts.anticoags:
        if status != "unknown":
            continue
        name = str(ref.obj.get("name") or "")
        citation = cite.unknown_anticoag_status(source=ref.path, name=name)
        issues.append(make_issue(CATEGORY_MISSING_REQUIRED_DATA, ORDER_MISSING_ANTICOAG_UNKNOWN, citation))

    return issues, frozenset(unavailable)
