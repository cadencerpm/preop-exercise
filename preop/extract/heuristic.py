"""Deterministic, offline document reader (regex + lexicon).

Purpose:
1. Runs with no API key, so the rules engine, tests, and CI work offline.
2. Acts as an independent second reader: when both extractors run, disagreements
   become review flags for the nurse (see `pipeline.cross_check`).

It is intentionally conservative and is NOT the primary reader. Free-text phrasing
it has not seen will make it return "not found", which the rules engine treats as
missing (fail closed) rather than as a pass.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from preop.extract.base import ProcedureContext
from preop.facts import (
    AnticoagulationPlan,
    ConsentFacts,
    DocumentFacts,
    HistoryAndPhysicalFacts,
    MedicationMention,
    VitalReading,
    empty_facts,
)
from preop.lexicon import ANTICOAGULANT, ANTIPLATELETS, find_anticoagulants
from preop.timeutil import parse_date

_HP_TITLE = re.compile(r"\bh\s*(?:&|and)\s*p\b|history\s*(?:&|and)\s*p[a-z]*", re.I)
_HP_HEADER = re.compile(r"^\W*(?:pre-?operative\s+)?history\s*(?:&|and)\s*physical", re.I)
_NOT_HP_TITLE = re.compile(r"anesthesia|nursing|consent", re.I)
_DATE_OF_SERVICE = re.compile(r"date of service:\s*(\d{1,2}/\d{1,2}/\d{4})", re.I)
_INCOMPLETE = re.compile(r"\b(draft|incomplete|preliminary|unsigned)\b", re.I)

_CONSENT_TITLE = re.compile(r"consent", re.I)
_CONSENT_TEXT = re.compile(r"informed consent|e-?signature request|consent for (?:surgery|surgical)", re.I)
_SIG_LINE = re.compile(r"^.*(?:patient signature|signed by patient|signature request).*$", re.I | re.M)
_SIG_NOT = re.compile(
    r"pending|not (?:yet )?signed|not completed|_{4,}|printed for (?:patient )?review|to be obtained",
    re.I,
)
_SIG_YES = re.compile(
    r"signed by patient|signature on file|/s/|electronically signed \d|signed in clinic|e-?signed", re.I
)
_PROCEDURE_LINE = re.compile(r"^\s*(?:procedure|planned procedure)\s*:\s*(.+)$", re.I | re.M)
_SIGNED_DATE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")

_STOPPED = re.compile(
    r"discontinued|was stopped|stopped (?:last|in)|transition off|completed \d+ months|not currently anticoagulated",
    re.I,
)
_UNCERTAIN = re.compile(r"unable to confirm|unsure whether|not sure (?:if|whether)|unclear whether", re.I)
_PLAN_LINE = re.compile(
    r"\b(hold|stop|resume|restart|bridge|interruption|periprocedural|perioperative management)\b", re.I
)
_PLAN_TIMING = re.compile(r"before|prior|after|post-?op|pre-?op|surgery|procedure|determined", re.I)
_DEFERRED = re.compile(
    r"to be determined|will follow up|discuss interruption|happy to assist|will be addressed|defer", re.I
)
_PRE = re.compile(r"\b(hold|stop|discontinue|interrupt|continue|last dose|bridge)\b", re.I)
_POST = re.compile(r"\b(resume|restart|reinitiate|re-?start)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.;])\s+(?=[A-Z])")

_BP = re.compile(r"(?<![\d/])(\d{2,3})\s*/\s*(\d{2,3})(?![\d/])")
_BP_CONTEXT = re.compile(r"\b(?:BP|NIBP|mmHg|blood pressure|vitals|re-?measured)\b", re.I)
_TEMP = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d+)?)\s*°?\s*([FC])\b")
_CLOCK = re.compile(r"(?<![\d/:])([01]?\d|2[0-3]):([0-5]\d)(?!\d|:\d)")
_MINUTES_LATER = re.compile(r"(\d+)\s*min(?:ute)?s?\b", re.I)

_SCHEDULED_FOR = re.compile(r"scheduled for (\d{1,2}/\d{1,2}/\d{4})|procedure date(?: target)?:\s*(\S+)", re.I)
_RISK_STATED = re.compile(r"risk (?:category|level)\s*[:=]\s*(LOW|MODERATE|HIGH)\b", re.I)

_ANTIPLATELET_WORDS = re.compile(
    r"\b(" + "|".join(a for names in ANTIPLATELETS.values() for a in names if a != "asa") + r"|asa\s+\d+)\b",
    re.I,
)


def _iso(us_date: str | None) -> str | None:
    d = parse_date(us_date)
    return d.isoformat() if d else None


def _line_at(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start : end if end >= 0 else len(text)].strip()


def _norm_proc(name: str) -> set[str]:
    words = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    return {w for w in words if w not in {"elective", "procedure", "surgery", "the", "of", "with"}}


class HeuristicExtractor:
    name = "heuristic"
    model = None
    prompt_version = "heuristic-v1"

    def extract(self, document: dict[str, object], context: ProcedureContext) -> DocumentFacts:
        text = str(document.get("text") or "")
        title = str(document.get("type") or "")
        facts = empty_facts()
        facts.history_and_physical = self._hp(title, text)
        facts.consent = self._consent(title, text, context)
        facts.antithrombotic_mentions = self._mentions(text)
        facts.anticoagulation_plans = self._plans(text)
        facts.vital_readings = self._vitals(text)
        m = _SCHEDULED_FOR.search(text)
        if m:
            raw = m.group(1) or m.group(2)
            if parse_date(raw):
                facts.procedure_date_stated = parse_date(raw).isoformat()  # type: ignore[union-attr]
                facts.procedure_date_quote = m.group(0)
        r = _RISK_STATED.search(text)
        if r:
            facts.procedure_risk_stated = r.group(1).upper()  # type: ignore[assignment]
            facts.procedure_risk_quote = r.group(0)
        return facts

    # -- H&P ------------------------------------------------------------------

    def _hp(self, title: str, text: str) -> HistoryAndPhysicalFacts:
        first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
        by_title = bool(_HP_TITLE.search(title)) and not _NOT_HP_TITLE.search(title)
        by_header = bool(_HP_HEADER.search(first_line))
        if not (by_title or by_header):
            return HistoryAndPhysicalFacts(
                is_history_and_physical=False, date_of_service=None, explicitly_incomplete=False, quote=None
            )
        dos = _DATE_OF_SERVICE.search(text)
        return HistoryAndPhysicalFacts(
            is_history_and_physical=True,
            date_of_service=_iso(dos.group(1)) if dos else None,
            explicitly_incomplete=bool(_INCOMPLETE.search(text[:400])),
            quote=first_line.strip() if by_header else None,
        )

    # -- consent ----------------------------------------------------------------

    def _consent(self, title: str, text: str, context: ProcedureContext) -> ConsentFacts:
        is_consent = bool(_CONSENT_TITLE.search(title)) or bool(_CONSENT_TEXT.search(text[:300]))
        if not is_consent:
            return empty_facts().consent
        lines = _SIG_LINE.findall(text)
        status, quote, signed_date = "UNCLEAR", None, None
        for line in lines:
            if _SIG_NOT.search(line):
                status, quote = "NOT_SIGNED", line.strip()
                break
            if _SIG_YES.search(line):
                status, quote = "SIGNED", line.strip()
                d = _SIGNED_DATE.search(line)
                signed_date = _iso(d.group(1)) if d else None
        if status != "NOT_SIGNED":
            blank = re.search(r"^.*(?:not yet signed|printed for patient review).*$", text, re.I | re.M)
            if blank:
                status, quote, signed_date = "NOT_SIGNED", blank.group(0).strip(), None
        proc = _PROCEDURE_LINE.search(text)
        procedure_named = proc.group(1).strip() if proc else None
        if procedure_named and context.procedure_type:
            a, b = _norm_proc(procedure_named), _norm_proc(context.procedure_type)
            match = "YES" if a and b and (a <= b or b <= a) else "NO"
        else:
            match = "NOT_STATED"
        return ConsentFacts(
            is_surgical_consent_record=True,
            signature_status=status,  # type: ignore[arg-type]
            signed_by="PATIENT" if status == "SIGNED" else "UNKNOWN",
            procedure_named=procedure_named,
            matches_scheduled_procedure=match,  # type: ignore[arg-type]
            signed_date=signed_date,
            quote=quote,
        )

    # -- anticoagulants -----------------------------------------------------------

    def _mentions(self, text: str) -> list[MedicationMention]:
        out: list[MedicationMention] = []
        seen: set[tuple[str, str, str]] = set()
        for generic, match in find_anticoagulants(text):
            line = _line_at(text, match.start())
            status = self._status_for(generic, match.group(0), line)
            key = (generic, status, line)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                MedicationMention(
                    drug_as_written=match.group(0),
                    drug_class=ANTICOAGULANT,  # type: ignore[arg-type]
                    status=status,  # type: ignore[arg-type]
                    quote=line,
                )
            )
        for m in _ANTIPLATELET_WORDS.finditer(text):
            line = _line_at(text, m.start())
            out.append(
                MedicationMention(
                    drug_as_written=m.group(0), drug_class="ANTIPLATELET", status="CURRENTLY_TAKING", quote=line
                )
            )
        return out

    def _status_for(self, generic: str, written: str, line: str) -> str:
        replaced = re.search(r"replaced with\s+(\w+)", line, re.I)
        if replaced:
            return "CURRENTLY_TAKING" if replaced.group(1).lower() == written.lower() else "STOPPED"
        if _UNCERTAIN.search(line):
            return "UNCERTAIN"
        if _STOPPED.search(line):
            return "STOPPED"
        if generic in {"enoxaparin", "dalteparin", "heparin"} and re.search(r"bridg", line, re.I):
            return "PLANNED_OR_BRIDGING_ONLY"
        if _PLAN_LINE.search(line) and _PLAN_TIMING.search(line) and not re.search(
            r"maintained on|taking|reports", line, re.I
        ):
            return "PLANNED_OR_BRIDGING_ONLY"
        return "CURRENTLY_TAKING"

    def _plans(self, text: str) -> list[AnticoagulationPlan]:
        plans: list[AnticoagulationPlan] = []
        for line in text.splitlines():
            line = line.strip()
            if not (_PLAN_LINE.search(line) and _PLAN_TIMING.search(line)):
                continue
            drugs = find_anticoagulants(line)
            if not drugs or re.search(r"maintained on|reports taking|taking as prescribed", line, re.I):
                continue
            if _STOPPED.search(line) and not _PRE.search(line):
                continue
            drug = drugs[0][1].group(0)
            pre: list[str] = []
            post: list[str] = []
            for sentence in _SENTENCE.split(re.sub(r"^\d+\.\s*", "", line)):
                if _DEFERRED.search(sentence):
                    continue
                if _POST.search(sentence):
                    post.append(sentence.strip())
                elif _PRE.search(sentence):
                    pre.append(sentence.strip())
            plans.append(
                AnticoagulationPlan(
                    drug_as_written=drug,
                    pre_procedure_instruction=" ".join(pre) or None,
                    post_procedure_instruction=" ".join(post) or None,
                    deferred_or_pending=bool(_DEFERRED.search(line)),
                    quote=line,
                )
            )
        return plans

    # -- vitals ---------------------------------------------------------------------

    def _vitals(self, text: str) -> list[VitalReading]:
        readings: list[VitalReading] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or re.search(r"\bINR\b|goal|mg/kg", line):
                continue
            events: list[tuple[int, str, re.Match[str]]] = []
            events += [(m.start(), "clock", m) for m in _CLOCK.finditer(line)]
            if _BP_CONTEXT.search(line):
                events += [(m.start(), "bp", m) for m in _BP.finditer(line)]
            events += [(m.start(), "temp", m) for m in _TEMP.finditer(line)]
            events.sort(key=lambda e: e[0])
            current: datetime | None = None
            last_pos = 0
            for pos, kind, m in events:
                if kind == "clock":
                    current = datetime(2000, 1, 1, int(m.group(1)), int(m.group(2)))
                    last_pos = m.end()
                    continue
                # "rechecked 24 minutes later" with no explicit new clock time
                segment = line[last_pos:pos]
                later = _MINUTES_LATER.search(segment)
                if current is not None and later and re.search(r"later|after|following|recheck", segment, re.I):
                    current = current + timedelta(minutes=int(later.group(1)))
                    last_pos = pos
                clock = current.strftime("%H:%M") if current else None
                if kind == "bp":
                    sys_v, dia_v = int(m.group(1)), int(m.group(2))
                    if not (60 <= sys_v <= 260 and 30 <= dia_v <= 160):
                        continue
                    readings.append(
                        VitalReading(
                            kind="BLOOD_PRESSURE", systolic=sys_v, diastolic=dia_v, temperature=None,
                            temperature_unit=None, date=None, time=clock, quote=m.group(0),
                        )
                    )
                else:
                    value, unit = float(m.group(1)), m.group(2).upper()
                    if not ((unit == "C" and 30 <= value <= 45) or (unit == "F" and 90 <= value <= 110)):
                        continue
                    readings.append(
                        VitalReading(
                            kind="TEMPERATURE", systolic=None, diastolic=None, temperature=value,
                            temperature_unit=unit, date=None, time=clock,  # type: ignore[arg-type]
                            quote=m.group(0),
                        )
                    )
        return readings
