"""Raw submission JSON -> typed facts with provenance.

Assumption (per the architecture review): the structured sections of a submission
(``procedure``, ``vitals``, ``labs``, ``medications``) are standardized -- ISO
dates/datetimes, a fixed ``procedure_risk`` vocabulary, numeric vitals, a fixed
``vitals[].type`` vocabulary. So normalization here is strict: it does no fuzzy
repair. The one exception is the small lab-code alias table, because both
``LAB-CBC``/``CBC`` and ``LAB-CMP``/``CMP`` forms appear in the data.

Only ``documents[].type`` is free text (100+ variants observed) and is
deliberately never string-matched by this module -- document meaning is read by
the LLM in ``documents.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from enum import StrEnum
from typing import Literal

AnticoagStatus = Literal["active", "inactive", "unknown"]

# Both forms of these codes appear in the sample data for the same underlying test.
LAB_ALIASES: dict[str, str] = {
    "LAB-CBC": "CBC",
    "LAB-CMP": "CMP",
}

# The only labs the policy cares about. HBA1C appears throughout the data as a
# distractor and is intentionally never tracked.
TRACKED_LAB_CODES: frozenset[str] = frozenset({"CBC", "CMP"})

# Small, adversarial vocabulary (per the handoff): lisinopril (BP) and metformin
# (diabetes) are distractors that must never be treated as anticoagulants.
ANTICOAGULANTS: frozenset[str] = frozenset({"apixaban", "warfarin"})


class Field(StrEnum):
    """Facts a rule may depend on. Gates and the pipeline mark these unavailable
    so a rule can declare ``requires`` and be skipped cleanly instead of each
    rule null-checking its own inputs."""

    PROC_DATE = "proc_date"
    PROC_RISK = "proc_risk"
    BP = "bp"
    TEMP = "temp"
    DOC_ROLES = "doc_roles"


@dataclass(frozen=True)
class Ref:
    """Provenance pointer: which raw JSON object (and array path) a fact came
    from. Citations are assembled from these, never reconstructed after the fact."""

    path: str
    obj: dict[str, object]


@dataclass(frozen=True)
class Facts:
    proc_date: date | None
    risk: str | None
    latest_bp: Ref | None
    latest_temp: Ref | None
    latest_lab: dict[str, Ref]
    anticoags: list[tuple[Ref, AnticoagStatus]]
    documents: list[Ref]


def canonical_lab_code(code: object) -> str | None:
    if not isinstance(code, str) or not code:
        return None
    return LAB_ALIASES.get(code, code)


def parse_item_date(value: object) -> date | None:
    """Parse a raw date/datetime string to a calendar date for window math.

    Per the decided date semantics: ``delta = proc_date - date(item_date[:10])``.
    Undated or unparsable values count as absent -- there is no fuzzy repair.
    """

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _sort_key(value: object) -> datetime | None:
    """Full timestamp (falling back to midnight for date-only strings), used only
    to pick the single most-recent item within a group -- never for window math.

    Normalized to timezone-aware UTC: a date-only string parses naive (midnight)
    and an offset-bearing string (e.g. trailing ``Z``) parses aware, and ``max()``
    over a mix of the two raises ``TypeError``. Treating a naive result as UTC
    midnight makes every value comparable without changing same-format
    comparisons (all-naive or all-aware groups sort exactly as before).
    """

    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        if "T" in text:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            parsed = datetime.combine(date.fromisoformat(text[:10]), time.min)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _most_recent(refs: list[Ref], date_field: str) -> Ref | None:
    """Most-recent reduction with no fallback: an item with no parsable date
    cannot be "the most recent" of anything, so it is dropped, not skipped-over."""

    dated = [
        (ref, key)
        for ref in refs
        if (key := _sort_key(ref.obj.get(date_field))) is not None
    ]
    if not dated:
        return None
    return max(dated, key=lambda pair: pair[1])[0]


def normalize(raw: dict[str, object]) -> Facts:
    """Raw submission dict -> Facts. Strict parsing; see module docstring."""

    procedure = raw.get("procedure") or {}
    proc_date = parse_item_date(procedure.get("procedure_date"))
    risk = procedure.get("procedure_risk")

    vitals = raw.get("vitals") or []
    bp_refs = [
        Ref(path=f"vitals[{i}]", obj=vital)
        for i, vital in enumerate(vitals)
        if isinstance(vital, dict) and vital.get("type") == "blood_pressure"
    ]
    temp_refs = [
        Ref(path=f"vitals[{i}]", obj=vital)
        for i, vital in enumerate(vitals)
        if isinstance(vital, dict) and vital.get("type") == "temperature"
    ]
    latest_bp = _most_recent(bp_refs, "date")
    latest_temp = _most_recent(temp_refs, "date")

    labs_by_code: dict[str, list[Ref]] = {}
    for i, lab in enumerate(raw.get("labs") or []):
        if not isinstance(lab, dict):
            continue
        code = canonical_lab_code(lab.get("code"))
        if code not in TRACKED_LAB_CODES:
            continue
        labs_by_code.setdefault(code, []).append(Ref(path=f"labs[{i}]", obj=lab))
    latest_lab = {
        code: ref
        for code, refs in labs_by_code.items()
        if (ref := _most_recent(refs, "effective_at")) is not None
    }

    anticoags: list[tuple[Ref, AnticoagStatus]] = []
    for i, med in enumerate(raw.get("medications") or []):
        if not isinstance(med, dict):
            continue
        name = str(med.get("name") or "").strip().lower()
        if name not in ANTICOAGULANTS:
            continue
        active = med.get("active")
        status: AnticoagStatus
        if active is None:
            status = "unknown"
        elif active:
            status = "active"
        else:
            status = "inactive"
        anticoags.append((Ref(path=f"medications[{i}]", obj=med), status))

    documents = [
        Ref(path=f"documents[{i}]", obj=doc)
        for i, doc in enumerate(raw.get("documents") or [])
        if isinstance(doc, dict)
    ]

    return Facts(
        proc_date=proc_date,
        risk=risk if isinstance(risk, str) else None,
        latest_bp=latest_bp,
        latest_temp=latest_temp,
        latest_lab=latest_lab,
        anticoags=anticoags,
        documents=documents,
    )
