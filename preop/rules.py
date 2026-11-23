"""Deterministic policy engine.

Input: the structured submission plus verified per-document facts.
Output: one CheckResult per policy check, each with status, evidence, computed
values, and a resolution action.

Status semantics:
  PASS            requirement met
  FAIL            requirement not met (category = the rule's category)
  UNKNOWN         a field needed to evaluate the rule is missing -> MISSING_REQUIRED_DATA
  BLOCKED         cannot be evaluated because of an UNKNOWN reported by another check;
                  not an issue by itself, so the root cause is reported once
  NOT_APPLICABLE  rule does not apply (e.g. no anticoagulant)

No LLM is involved here; the same facts always produce the same result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from preop.facts import AnticoagulationPlan, DocumentFacts, VitalReading
from preop.lexicon import ANTICOAGULANT, VALID_LAB_STATUSES, canonical_lab_test, classify_drug
from preop.output import CheckResult, Evidence, ReviewFlag
from preop.policy import RISK_LEVELS, Policy
from preop.timeutil import Moment, days_between, moment_from, parse_date

R_DATA = "PROCEDURE_DATA"
R1 = "RULE_1_DOCUMENTATION"
R2 = "RULE_2_TESTING"
R3 = "RULE_3_ANTICOAGULATION"
R4 = "RULE_4_ACUTE_SAFETY"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class DocView:
    index: int
    doc: dict[str, Any]
    facts: DocumentFacts

    @property
    def ref(self) -> str:
        return str(self.doc.get("doc_id") or f"documents[{self.index}]")

    @property
    def day(self) -> date | None:
        return parse_date(self.doc.get("date")) or parse_date(
            self.facts.history_and_physical.date_of_service
        )

    @property
    def label(self) -> str:
        author = self.doc.get("author")
        parts = [str(self.doc.get("type") or "Document"), str(self.doc.get("date") or "undated")]
        return " ".join(parts) + (f" ({author})" if author else "")

    def evidence(self, quote: str | None = None) -> Evidence:
        return Evidence(
            source="document",
            ref=self.ref,
            label=self.label,
            date=self.day.isoformat() if self.day else None,
            quote=quote,
            quote_verified=True if quote else None,
        )


@dataclass
class CaseInputs:
    submission: dict[str, Any]
    docs: list[DocView]
    flags: list[ReviewFlag] = field(default_factory=list)


@dataclass
class Resolved:
    """Procedure facts resolved across structured data and documents."""

    procedure_date: date | None
    risk: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proc(sub: dict[str, Any]) -> dict[str, Any]:
    return sub.get("procedure") or {}


def _proc_evidence(field_name: str, value: Any) -> Evidence:
    return Evidence(source="procedure", ref=f"procedure.{field_name}", label=f"procedure.{field_name}", quote=str(value))


def _window_start(proc_date: date, max_age: int) -> str:
    return (proc_date - timedelta(days=max_age)).isoformat()


# ---------------------------------------------------------------------------
# Procedure date / risk
# ---------------------------------------------------------------------------


def check_procedure_date(case: CaseInputs) -> tuple[CheckResult, date | None]:
    raw = _proc(case.submission).get("procedure_date")
    parsed = parse_date(raw)
    if parsed:
        return (
            CheckResult(
                rule=R_DATA, check="procedure_date_known", status="PASS",
                summary=f"Procedure date {parsed.isoformat()} (scheduling system).",
                evidence=[_proc_evidence("procedure_date", raw)],
                facts={"procedure_date": parsed.isoformat(), "source": "structured"},
            ),
            parsed,
        )
    stated = [(d, parse_date(d.facts.procedure_date_stated)) for d in case.docs if d.facts.procedure_date_stated]
    distinct = {s for _, s in stated if s}
    if len(distinct) == 1:
        value = distinct.pop()
        case.flags.append(
            ReviewFlag(
                code="PROCEDURE_DATE_FROM_DOCUMENT",
                message=f"procedure_date is blank; using {value.isoformat()} stated consistently in documents.",
            )
        )
        return (
            CheckResult(
                rule=R_DATA, check="procedure_date_known", status="PASS",
                summary=f"Procedure date blank in scheduling system; {value.isoformat()} stated in documents.",
                evidence=[d.evidence(d.facts.procedure_date_quote) for d, s in stated if s == value],
                facts={"procedure_date": value.isoformat(), "source": "document"},
            ),
            value,
        )
    summary = "Procedure date is missing, so the H&P and lab time windows cannot be evaluated."
    if len(distinct) > 1:
        summary = "Procedure date is blank and documents state conflicting dates: " + ", ".join(
            sorted(x.isoformat() for x in distinct)
        )
    return (
        CheckResult(
            rule=R_DATA, check="procedure_date_known", status="UNKNOWN", category="MISSING_REQUIRED_DATA",
            summary=summary,
            evidence=[_proc_evidence("procedure_date", raw)] + [d.evidence(d.facts.procedure_date_quote) for d, _ in stated],
            facts={"procedure_date": None},
            action="Confirm the scheduled procedure date, then re-run triage.",
        ),
        None,
    )


def check_procedure_risk(case: CaseInputs) -> tuple[CheckResult, str | None]:
    raw = _proc(case.submission).get("procedure_risk")
    if raw in RISK_LEVELS:
        return (
            CheckResult(
                rule=R_DATA, check="procedure_risk_known", status="PASS",
                summary=f"Procedure risk {raw} (scheduling system).",
                evidence=[_proc_evidence("procedure_risk", raw)], facts={"procedure_risk": raw, "source": "structured"},
            ),
            raw,
        )
    stated = {d.facts.procedure_risk_stated for d in case.docs if d.facts.procedure_risk_stated}
    if len(stated) == 1:
        value = stated.pop()
        case.flags.append(
            ReviewFlag(code="PROCEDURE_RISK_FROM_DOCUMENT", message=f"procedure_risk blank; using {value} stated in documents.")
        )
        return (
            CheckResult(
                rule=R_DATA, check="procedure_risk_known", status="PASS",
                summary=f"Procedure risk blank in scheduling system; {value} stated in documents.",
                evidence=[d.evidence(d.facts.procedure_risk_quote) for d in case.docs if d.facts.procedure_risk_stated == value],
                facts={"procedure_risk": value, "source": "document"},
            ),
            value,
        )
    return (
        CheckResult(
            rule=R_DATA, check="procedure_risk_known", status="UNKNOWN", category="MISSING_REQUIRED_DATA",
            summary="Procedure risk category is not assigned, so the required lab panel cannot be determined.",
            evidence=[_proc_evidence("procedure_risk", raw)], facts={"procedure_risk": None},
            action="Have the surgical team assign a LOW / MODERATE / HIGH risk category.",
        ),
        None,
    )


# ---------------------------------------------------------------------------
# Rule 1: documentation
# ---------------------------------------------------------------------------


def check_history_and_physical(case: CaseInputs, policy: Policy, proc_date: date | None) -> CheckResult:
    limit = policy.hp_max_age_days
    hps = [d for d in case.docs if d.facts.history_and_physical.is_history_and_physical]
    complete = [d for d in hps if not d.facts.history_and_physical.explicitly_incomplete]
    base = {"rule": R1, "check": "history_and_physical_current"}
    if not complete:
        why = "No History and Physical found in the package."
        if hps:
            why = "Only H&P(s) explicitly marked incomplete/draft were found."
        return CheckResult(
            **base, status="FAIL", category="REQUIRED_DOCUMENTATION", summary=why,
            evidence=[d.evidence(d.facts.history_and_physical.quote) for d in hps],
            facts={"h_and_p_found": len(hps), "max_age_days": limit},
            action="Obtain a completed H&P"
            + (f" dated on or after {_window_start(proc_date, limit)}." if proc_date else "."),
        )
    dated = sorted([d for d in complete if d.day], key=lambda d: d.day)  # type: ignore[arg-type,return-value]
    if not dated:
        return CheckResult(
            **base, status="UNKNOWN", category="MISSING_REQUIRED_DATA",
            summary="H&P found but it has no date, so the 30-day window cannot be checked.",
            evidence=[d.evidence() for d in complete], action="Confirm the H&P date of service.",
        )
    if proc_date is None:
        latest = dated[-1]
        return CheckResult(
            **base, status="BLOCKED",
            summary=f"H&P on file ({latest.day.isoformat()}); window check waits on the procedure date.",  # type: ignore[union-attr]
            evidence=[latest.evidence(latest.facts.history_and_physical.quote)],
        )
    eligible = [d for d in dated if d.day <= proc_date]  # type: ignore[operator]
    latest = eligible[-1] if eligible else dated[-1]
    age = days_between(latest.day, proc_date)  # type: ignore[arg-type]
    facts = {"h_and_p_date": latest.day.isoformat(), "procedure_date": proc_date.isoformat(),  # type: ignore[union-attr]
             "age_days": age, "max_age_days": limit,
             "older_h_and_ps_ignored": [d.day.isoformat() for d in dated if d is not latest]}  # type: ignore[union-attr]
    ev = [latest.evidence(latest.facts.history_and_physical.quote)]
    if 0 <= age <= limit:
        return CheckResult(**base, status="PASS", summary=f"H&P dated {latest.day.isoformat()} is {age} days before the procedure (limit {limit}).",  # type: ignore[union-attr]
                           evidence=ev, facts=facts)
    reason = f"{age} days before the procedure (limit {limit})" if age >= 0 else "dated after the procedure date"
    return CheckResult(
        **base, status="FAIL", category="REQUIRED_DOCUMENTATION",
        summary=f"Most recent H&P ({latest.day.isoformat()}) is {reason}.",  # type: ignore[union-attr]
        evidence=ev, facts=facts,
        action=f"Obtain an updated H&P dated on or after {_window_start(proc_date, limit)}.",
    )


def check_consent(case: CaseInputs) -> CheckResult:
    base = {"rule": R1, "check": "signed_surgical_consent"}
    records = [d for d in case.docs if d.facts.consent.is_surgical_consent_record]
    valid = [
        d for d in records
        if d.facts.consent.signature_status == "SIGNED" and d.facts.consent.matches_scheduled_procedure != "NO"
    ]
    if valid:
        best = sorted(valid, key=lambda d: d.day or date.min)[-1]
        c = best.facts.consent
        return CheckResult(
            **base, status="PASS",
            summary=f"Signed consent for '{c.procedure_named or 'procedure'}' ({best.label}).",
            evidence=[best.evidence(c.quote)],
            facts={"signed_date": c.signed_date, "procedure_named": c.procedure_named,
                   "superseded_or_invalid_records": [d.label for d in records if d is not best]},
        )
    if not records:
        return CheckResult(
            **base, status="FAIL", category="REQUIRED_DOCUMENTATION",
            summary="No surgical consent found in the package.",
            action="Obtain a patient-signed surgical consent for the scheduled procedure.",
        )
    problems = []
    for d in records:
        c = d.facts.consent
        if c.signature_status == "SIGNED" and c.matches_scheduled_procedure == "NO":
            problems.append(f"{d.label}: signed, but for a different procedure ('{c.procedure_named}')")
        elif c.signature_status == "NOT_SIGNED":
            problems.append(f"{d.label}: not signed by patient")
        else:
            problems.append(f"{d.label}: signature status unclear")
    return CheckResult(
        **base, status="FAIL", category="REQUIRED_DOCUMENTATION",
        summary="No valid signed consent for the scheduled procedure. " + "; ".join(problems) + ".",
        evidence=[d.evidence(d.facts.consent.quote) for d in records],
        facts={"scheduled_procedure": _proc(case.submission).get("procedure_type")},
        action="Obtain a patient-signed surgical consent for the scheduled procedure.",
    )


# ---------------------------------------------------------------------------
# Rule 2: testing
# ---------------------------------------------------------------------------


@dataclass
class LabView:
    index: int
    raw: dict[str, Any]
    test: str
    day: date

    def evidence(self) -> Evidence:
        r = self.raw
        return Evidence(
            source="lab", ref=str(r.get("id") or f"labs[{self.index}]"),
            label=f"{r.get('display') or r.get('code')} [{r.get('status')}] {self.day.isoformat()}",
            date=self.day.isoformat(),
        )


def _labs(case: CaseInputs) -> tuple[dict[str, list[LabView]], list[dict[str, Any]]]:
    usable: dict[str, list[LabView]] = {}
    excluded: list[dict[str, Any]] = []
    for i, lab in enumerate(case.submission.get("labs") or []):
        test = canonical_lab_test(lab.get("code"), lab.get("display"))
        day = parse_date(lab.get("effective_at"))
        if test is None:
            continue
        status = (lab.get("status") or "").lower()
        if status and status not in VALID_LAB_STATUSES:
            excluded.append({"test": test, "date": lab.get("effective_at"), "status": status, "id": lab.get("id")})
            continue
        if day is None:
            excluded.append({"test": test, "date": None, "status": status or None, "id": lab.get("id")})
            continue
        usable.setdefault(test, []).append(LabView(i, lab, test, day))
    for views in usable.values():
        views.sort(key=lambda v: v.day)
    return usable, excluded


def _test_outcome(latest: LabView | None, max_age: int, proc_date: date) -> tuple[str, int | None]:
    if latest is None:
        return "FAIL", None
    age = days_between(latest.day, proc_date)
    return ("PASS" if 0 <= age <= max_age else "FAIL"), age


def check_testing(case: CaseInputs, policy: Policy, proc_date: date | None, risk: str | None) -> list[CheckResult]:
    usable, excluded = _labs(case)
    risks = [risk] if risk else list(RISK_LEVELS)
    tests = sorted({t for r in risks for t in policy.required_tests(r)})
    results: list[CheckResult] = []
    for test in tests:
        views = [v for v in usable.get(test, []) if proc_date is None or v.day <= proc_date]
        latest = views[-1] if views else None
        ignored_excluded = [e for e in excluded if e["test"] == test]
        base = {"rule": R2, "check": f"{test.lower()}_within_window"}
        facts: dict[str, Any] = {"test": test, "most_recent_result": latest.day.isoformat() if latest else None,
                                 "excluded_results": ignored_excluded}
        ev = [latest.evidence()] if latest else []
        required_by = [r for r in risks if test in policy.required_tests(r)]
        required_always = len(required_by) == len(risks)

        if latest is None and required_always:
            results.append(CheckResult(
                **base, status="FAIL", category="REQUIRED_TESTING",
                summary=f"No valid {test} result on file" + (" (entered-in-error results excluded)." if ignored_excluded else "."),
                evidence=ev, facts=facts,
                action=f"Order a {test}" + (f" resulted on or after {_window_start(proc_date, policy.required_tests(risks[0])[test])}." if proc_date and risk else "."),
            ))
            continue
        if proc_date is None:
            results.append(CheckResult(**base, status="BLOCKED", evidence=ev, facts=facts,
                                       summary=f"{test} window check waits on the procedure date."))
            continue

        outcomes = {r: _test_outcome(latest, policy.required_tests(r)[test], proc_date) for r in required_by}
        age = next(iter(outcomes.values()))[1]
        facts.update({"age_days": age, "procedure_date": proc_date.isoformat(),
                      "max_age_days_by_risk": {r: policy.required_tests(r)[test] for r in required_by}})
        statuses = {o for o, _ in outcomes.values()}
        if risk:
            limit = policy.required_tests(risk)[test]
            if statuses == {"PASS"}:
                results.append(CheckResult(**base, status="PASS", evidence=ev, facts=facts,
                                           summary=f"{test} from {latest.day.isoformat()} is {age} days before the procedure (limit {limit} for {risk} risk)."))  # type: ignore[union-attr]
            else:
                what = f"Most recent {test} ({latest.day.isoformat()}) is {age} days before the procedure" if latest else f"No valid {test} on file"
                results.append(CheckResult(
                    **base, status="FAIL", category="REQUIRED_TESTING", evidence=ev, facts=facts,
                    summary=f"{what} (limit {limit} for {risk} risk).",
                    action=f"Repeat {test}, resulted on or after {_window_start(proc_date, limit)}.",
                ))
        else:
            # Risk unknown: evaluate under every possible risk level. Only report a
            # testing failure if it fails no matter which risk is assigned.
            if statuses == {"PASS"} and required_always:
                results.append(CheckResult(**base, status="PASS", evidence=ev, facts=facts,
                                           summary=f"{test} ({age} days old) satisfies every risk level's window."))
            elif statuses == {"FAIL"} and required_always:
                results.append(CheckResult(
                    **base, status="FAIL", category="REQUIRED_TESTING", evidence=ev, facts=facts,
                    summary=f"{test} fails the window under every possible risk level.",
                    action=f"Repeat {test}.",
                ))
            else:
                results.append(CheckResult(
                    **base, status="BLOCKED", evidence=ev, facts={**facts, "outcome_by_risk": {r: o for r, (o, _) in outcomes.items()}},
                    summary=f"Whether {test} is required/current depends on the unassigned risk category.",
                ))
    return results


# ---------------------------------------------------------------------------
# Rule 3: anticoagulation
# ---------------------------------------------------------------------------

_STATUS_PRIORITY = {"UNCERTAIN": 3, "CURRENTLY_TAKING": 2, "STOPPED": 1}


@dataclass
class DrugState:
    generic: str
    status: str  # TAKING | NOT_TAKING | UNKNOWN
    reason: str
    evidence: list[Evidence]


def _anticoagulant_states(case: CaseInputs) -> tuple[dict[str, DrugState], list[str]]:
    med_list: dict[str, tuple[bool | None, Evidence]] = {}
    antiplatelets: list[str] = []
    for i, med in enumerate(case.submission.get("medications") or []):
        generic, cls = classify_drug(med.get("name"))
        if cls == ANTICOAGULANT and generic:
            ev = Evidence(source="medication", ref=f"medications[{i}]",
                          label=f"Medication list: {med.get('name')} (active={med.get('active')})")
            med_list[generic] = (med.get("active"), ev)
        elif cls and generic:
            antiplatelets.append(f"{med.get('name')} ({generic}, antiplatelet)")

    notes: dict[str, list[tuple[date, str, Evidence]]] = {}
    for d in case.docs:
        for m in d.facts.antithrombotic_mentions:
            generic, cls = classify_drug(m.drug_as_written)
            if generic is None and m.drug_class == "ANTICOAGULANT":
                generic, cls = m.drug_as_written.lower(), ANTICOAGULANT
                case.flags.append(ReviewFlag(code="UNRECOGNIZED_ANTICOAGULANT", ref=d.ref,
                                             message=f"'{m.drug_as_written}' treated as anticoagulant per extractor; not in lexicon."))
            if cls != ANTICOAGULANT or m.status == "PLANNED_OR_BRIDGING_ONLY":
                continue
            notes.setdefault(generic, []).append((d.day or date.min, m.status, d.evidence(m.quote)))  # type: ignore[arg-type]

    states: dict[str, DrugState] = {}
    for generic in sorted(set(med_list) | set(notes)):
        active, med_ev = med_list.get(generic, (None, None))
        in_list = generic in med_list
        mentions = sorted(notes.get(generic, []), key=lambda t: t[0])
        evidence = ([med_ev] if med_ev else []) + [e for _, _, e in mentions]
        if mentions:
            latest_day = mentions[-1][0]
            latest = max((s for dd, s, _ in mentions if dd == latest_day), key=lambda s: _STATUS_PRIORITY[s])
            if latest == "UNCERTAIN":
                states[generic] = DrugState(generic, "UNKNOWN", "most recent note says current use is unconfirmed", evidence)
            elif latest == "CURRENTLY_TAKING":
                states[generic] = DrugState(generic, "TAKING", "most recent note documents current use", evidence)
            elif in_list and active is True:
                states[generic] = DrugState(generic, "UNKNOWN", "medication list says active but most recent note says stopped", evidence)
            else:
                states[generic] = DrugState(generic, "NOT_TAKING", "most recent note documents it was stopped", evidence)
        elif active is True:
            states[generic] = DrugState(generic, "TAKING", "active on medication list", evidence)
        elif active is False:
            states[generic] = DrugState(generic, "NOT_TAKING", "inactive on medication list", evidence)
        else:
            states[generic] = DrugState(generic, "UNKNOWN", "medication list active status is blank", evidence)
    return states, antiplatelets


def _plans_for(case: CaseInputs, generic: str) -> list[tuple[DocView, AnticoagulationPlan]]:
    out = []
    for d in case.docs:
        for p in d.facts.anticoagulation_plans:
            g, _ = classify_drug(p.drug_as_written)
            if (g or p.drug_as_written.lower()) == generic:
                out.append((d, p))
    return sorted(out, key=lambda dp: (dp[0].day or date.min, dp[1].pre_procedure_instruction is not None and dp[1].post_procedure_instruction is not None))


def _other_plans(case: CaseInputs, generic: str) -> list[str]:
    seen = []
    for d in case.docs:
        for p in d.facts.anticoagulation_plans:
            g, _ = classify_drug(p.drug_as_written)
            if g and g != generic:
                seen.append(f"{d.label} covers {g}")
    return seen


def check_anticoagulation(case: CaseInputs, policy: Policy) -> list[CheckResult]:
    states, antiplatelets = _anticoagulant_states(case)
    relevant = {g: s for g, s in states.items() if s.status in {"TAKING", "UNKNOWN"}}
    if not relevant:
        summary = "No current anticoagulant; Rule 3 does not apply."
        if antiplatelets:
            summary += " Antiplatelet(s) noted, not anticoagulants: " + ", ".join(antiplatelets) + "."
        return [CheckResult(
            rule=R3, check="anticoagulation_plan", status="NOT_APPLICABLE", summary=summary,
            evidence=[e for s in states.values() for e in s.evidence],
            facts={"not_currently_taking": {g: s.reason for g, s in states.items()}},
        )]

    results = []
    for generic, state in sorted(relevant.items()):
        base = {"rule": R3, "check": f"anticoagulation_plan_{generic}"}
        plans = _plans_for(case, generic)
        latest = plans[-1] if plans else None
        pre = latest[1].pre_procedure_instruction if latest else None
        post = latest[1].post_procedure_instruction if latest else None
        complete = (pre is not None or not policy.require_pre_procedure_instruction) and (
            post is not None or not policy.require_post_procedure_instruction
        )
        ev = list(state.evidence)
        if latest:
            ev.append(latest[0].evidence(latest[1].quote))
        facts = {"drug": generic, "use_status": state.status, "use_status_reason": state.reason,
                 "plan_source": latest[0].label if latest else None,
                 "pre_procedure_instruction": pre, "post_procedure_instruction": post}
        if latest and complete:
            results.append(CheckResult(**base, status="PASS", evidence=ev, facts=facts,
                                       summary=f"{generic}: plan documents both pre-op ('{pre}') and post-op ('{post}') management."))
            continue
        if state.status == "UNKNOWN":
            results.append(CheckResult(
                **base, status="UNKNOWN", category="MISSING_REQUIRED_DATA", evidence=ev, facts=facts,
                summary=f"Cannot determine whether the patient currently takes {generic}: {state.reason}",
                action=f"Confirm current {generic} use (medication reconciliation / pharmacy fill history); "
                       "if taking, obtain a perioperative plan.",
            ))
            continue
        if latest is None:
            others = _other_plans(case, generic)
            note = f" Plans on file are for other drugs ({'; '.join(others)})." if others else ""
            results.append(CheckResult(
                **base, status="FAIL", category="ANTICOAGULATION_MANAGEMENT", evidence=ev, facts=facts,
                summary=f"Patient is taking {generic} ({state.reason}) but no perioperative management plan is documented.{note}",
                action=f"Obtain a perioperative plan for {generic} stating when to stop before and when to resume after surgery.",
            ))
            continue
        missing = [part for part, val in (("pre-op hold instruction", pre), ("post-op resume instruction", post)) if val is None]
        results.append(CheckResult(
            **base, status="FAIL", category="ANTICOAGULATION_MANAGEMENT", evidence=ev, facts=facts,
            summary=f"{generic} plan ({latest[0].label}) is incomplete: missing {' and '.join(missing)}.",
            action=f"Ask the prescriber to document the {' and '.join(missing)} for {generic}.",
        ))
    return results


# ---------------------------------------------------------------------------
# Rule 4: acute safety
# ---------------------------------------------------------------------------


@dataclass
class Reading:
    moment: Moment
    value: tuple[float, ...]
    evidence: Evidence


def _readings(case: CaseInputs) -> tuple[list[Reading], list[Reading]]:
    bps: list[Reading] = []
    temps: list[Reading] = []
    for i, v in enumerate(case.submission.get("vitals") or []):
        moment = moment_from(v.get("date"))
        if moment is None:
            continue
        label = f"Structured vital ({v.get('source') or 'unknown source'}) {moment.as_text()}"
        if v.get("systolic") is not None and v.get("diastolic") is not None:
            ev = Evidence(source="vital", ref=f"vitals[{i}]", label=label, date=moment.day.isoformat(),
                          quote=f"{v['systolic']}/{v['diastolic']} mmHg")
            bps.append(Reading(moment, (float(v["systolic"]), float(v["diastolic"])), ev))
        temp_f = v.get("value_f")
        if temp_f is None and v.get("value_c") is not None:
            temp_f = float(v["value_c"]) * 9 / 5 + 32
        if temp_f is not None:
            ev = Evidence(source="vital", ref=f"vitals[{i}]", label=label, date=moment.day.isoformat(),
                          quote=f"{float(temp_f):.1f} F")
            temps.append(Reading(moment, (round(float(temp_f), 2),), ev))
    for d in case.docs:
        for r in d.facts.vital_readings:
            moment = moment_from(r.date or (d.day.isoformat() if d.day else None), r.time)
            if moment is None:
                continue
            ev = d.evidence(r.quote)
            ev.label = f"{d.label} @ {moment.as_text()}"
            if r.kind == "BLOOD_PRESSURE" and r.systolic is not None and r.diastolic is not None:
                bps.append(Reading(moment, (float(r.systolic), float(r.diastolic)), ev))
            elif r.kind == "TEMPERATURE" and r.temperature is not None:
                f = r.temperature * 9 / 5 + 32 if r.temperature_unit == "C" else r.temperature
                temps.append(Reading(moment, (round(f, 2),), ev))
    return bps, temps


def _latest(readings: list[Reading]) -> tuple[list[Reading], bool]:
    """Readings that could be 'the most recent'. Returns (candidates, ambiguous)."""
    last_day = max(r.moment.day for r in readings)
    same_day = [r for r in readings if r.moment.day == last_day]
    timed = [r for r in same_day if r.moment.clock is not None]
    untimed = [r for r in same_day if r.moment.clock is None]
    if timed:
        top = max(r.moment.clock for r in timed)  # type: ignore[type-var]
        candidates = [r for r in timed if r.moment.clock == top] + untimed
    else:
        candidates = untimed
    distinct = {r.value for r in candidates}
    return candidates, len(distinct) > 1


def check_acute_safety(case: CaseInputs, policy: Policy) -> list[CheckResult]:
    bps, temps = _readings(case)
    results: list[CheckResult] = []

    def bp_bad(v: tuple[float, ...]) -> bool:
        return v[0] >= policy.systolic_exclusion_at_or_above or v[1] >= policy.diastolic_exclusion_at_or_above

    def t_bad(v: tuple[float, ...]) -> bool:
        return v[0] > policy.temperature_f_exclusion_above

    for name, readings, bad, fmt, limit in (
        ("blood_pressure", bps, bp_bad, lambda v: f"{v[0]:.0f}/{v[1]:.0f} mmHg",
         f"systolic >= {policy.systolic_exclusion_at_or_above:.0f} or diastolic >= {policy.diastolic_exclusion_at_or_above:.0f}"),
        ("temperature", temps, t_bad, lambda v: f"{v[0]:.1f} F", f"> {policy.temperature_f_exclusion_above} F"),
    ):
        base = {"rule": R4, "check": f"latest_{name}"}
        pretty = name.replace("_", " ")
        if not readings:
            results.append(CheckResult(
                **base, status="UNKNOWN", category="MISSING_REQUIRED_DATA",
                summary=f"No {pretty} recorded anywhere in the package (structured vitals or notes).",
                facts={"exclusion_threshold": limit},
                action=f"Record a current {pretty} before scheduling.",
            ))
            continue
        candidates, ambiguous = _latest(readings)
        # Fail closed: if the most recent reading can't be singled out, judge the worst.
        worst = max(candidates, key=lambda r: (bad(r.value), r.value))
        history = sorted(readings, key=lambda r: r.moment.sort_key())
        facts = {"most_recent": fmt(worst.value), "taken_at": worst.moment.as_text(), "exclusion_threshold": limit,
                 "superseded_readings": [f"{fmt(r.value)} @ {r.moment.as_text()}" for r in history if r not in candidates][-5:]}
        if ambiguous:
            case.flags.append(ReviewFlag(code="VITAL_ORDER_AMBIGUOUS",
                                         message=f"Several {pretty} readings share the latest time; evaluated the worst ({fmt(worst.value)})."))
        if bad(worst.value):
            results.append(CheckResult(
                **base, status="FAIL", category="ACUTE_SAFETY_EXCLUSION", evidence=[worst.evidence], facts=facts,
                summary=f"Most recent {pretty} {fmt(worst.value)} ({worst.moment.as_text()}) meets the exclusion threshold ({limit}).",
                action=f"Do not schedule. Re-evaluate {pretty}; a rechecked value within range clears this exclusion.",
            ))
        else:
            results.append(CheckResult(
                **base, status="PASS", evidence=[worst.evidence], facts=facts,
                summary=f"Most recent {pretty} {fmt(worst.value)} ({worst.moment.as_text()}) is within range.",
            ))
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate(case: CaseInputs, policy: Policy) -> list[CheckResult]:
    date_check, proc_date = check_procedure_date(case)
    risk_check, risk = check_procedure_risk(case)
    checks = [date_check, risk_check]
    checks.append(check_history_and_physical(case, policy, proc_date))
    checks.append(check_consent(case))
    checks.extend(check_testing(case, policy, proc_date, risk))
    checks.extend(check_anticoagulation(case, policy))
    checks.extend(check_acute_safety(case, policy))
    return checks
