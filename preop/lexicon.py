"""Reference vocabularies used to *read* the record (not to add policy criteria).

The policy allows general clinical knowledge for interpretation: drug brand names,
lab naming variants, unit conversions. Keeping that knowledge in one reviewed table
means drug classification is deterministic and auditable instead of being left to
an LLM's judgment on every call.
"""

from __future__ import annotations

import re

# generic name -> names it may appear under (generic + common US brands)
ANTICOAGULANTS: dict[str, tuple[str, ...]] = {
    "warfarin": ("warfarin", "coumadin", "jantoven"),
    "apixaban": ("apixaban", "eliquis"),
    "rivaroxaban": ("rivaroxaban", "xarelto"),
    "dabigatran": ("dabigatran", "pradaxa"),
    "edoxaban": ("edoxaban", "savaysa", "lixiana"),
    "betrixaban": ("betrixaban", "bevyxxa"),
    "enoxaparin": ("enoxaparin", "lovenox"),
    "dalteparin": ("dalteparin", "fragmin"),
    "tinzaparin": ("tinzaparin", "innohep"),
    "fondaparinux": ("fondaparinux", "arixtra"),
    "heparin": ("heparin",),
    "argatroban": ("argatroban",),
    "bivalirudin": ("bivalirudin", "angiomax"),
}

# Antiplatelets are NOT anticoagulants under Rule 3; listed so they are classified
# explicitly rather than silently ignored.
ANTIPLATELETS: dict[str, tuple[str, ...]] = {
    "aspirin": ("aspirin", "asa", "ecotrin", "bayer"),
    "clopidogrel": ("clopidogrel", "plavix"),
    "prasugrel": ("prasugrel", "effient"),
    "ticagrelor": ("ticagrelor", "brilinta"),
    "cilostazol": ("cilostazol", "pletal"),
    "dipyridamole": ("dipyridamole", "aggrenox", "persantine"),
    "vorapaxar": ("vorapaxar", "zontivity"),
}

ANTICOAGULANT = "ANTICOAGULANT"
ANTIPLATELET = "ANTIPLATELET"


def _alias_pattern(table: dict[str, tuple[str, ...]]) -> list[tuple[str, re.Pattern[str]]]:
    return [
        (generic, re.compile(r"\b(" + "|".join(map(re.escape, aliases)) + r")\b", re.I))
        for generic, aliases in table.items()
    ]


_ANTICOAG_PATTERNS = _alias_pattern(ANTICOAGULANTS)
_ANTIPLATELET_PATTERNS = _alias_pattern(ANTIPLATELETS)


def classify_drug(text: str | None) -> tuple[str | None, str | None]:
    """Return (generic_name, drug_class) for a medication string, or (None, None)."""
    if not text:
        return None, None
    for generic, pattern in _ANTICOAG_PATTERNS:
        if pattern.search(text):
            return generic, ANTICOAGULANT
    for generic, pattern in _ANTIPLATELET_PATTERNS:
        if pattern.search(text):
            return generic, ANTIPLATELET
    return None, None


def find_anticoagulants(text: str) -> list[tuple[str, re.Match[str]]]:
    """All anticoagulant name occurrences in free text, as (generic, match)."""
    hits: list[tuple[str, re.Match[str]]] = []
    for generic, pattern in _ANTICOAG_PATTERNS:
        hits.extend((generic, m) for m in pattern.finditer(text))
    return sorted(hits, key=lambda h: h[1].start())


# Required pre-op tests. Codes are matched first (after stripping a "LAB-" prefix),
# display text second. BMP is deliberately absent: a basic metabolic panel does not
# satisfy a CMP requirement.
LAB_TESTS: dict[str, dict[str, object]] = {
    "CBC": {
        "codes": {"CBC", "CBCD", "CBC-DIFF", "CBCWD"},
        "display": re.compile(r"\bcbc\b|complete blood count|hemogram", re.I),
    },
    "CMP": {
        "codes": {"CMP", "CMP14"},
        "display": re.compile(r"\bcmp\b|comprehensive metabolic", re.I),
    },
}

# Lab statuses that represent a usable, resulted value (FHIR Observation.status).
VALID_LAB_STATUSES = {"final", "amended", "corrected"}


def canonical_lab_test(code: str | None, display: str | None) -> str | None:
    norm_code = (code or "").strip().upper()
    if norm_code.startswith("LAB-"):
        norm_code = norm_code[4:]
    for test, spec in LAB_TESTS.items():
        if norm_code in spec["codes"]:  # type: ignore[operator]
            return test
    if norm_code in {"BMP", "BMP8"}:
        return None
    for test, spec in LAB_TESTS.items():
        if display and spec["display"].search(display):  # type: ignore[union-attr]
            return test
    return None
