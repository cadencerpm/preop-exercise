"""Load the scheduling policy from `policy.toml` into a typed, immutable object."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

POLICY_PATH = Path(__file__).with_name("policy.toml")

RISK_LEVELS = ("LOW", "MODERATE", "HIGH")


@dataclass(frozen=True)
class Policy:
    name: str
    version: str
    effective_date: str
    hp_max_age_days: int
    require_signed_consent: bool
    testing: dict[str, dict[str, int]] = field(default_factory=dict)
    require_pre_procedure_instruction: bool = True
    require_post_procedure_instruction: bool = True
    systolic_exclusion_at_or_above: float = 180
    diastolic_exclusion_at_or_above: float = 110
    temperature_f_exclusion_above: float = 100.4

    def required_tests(self, risk: str) -> dict[str, int]:
        return dict(self.testing.get(risk, {}))


def load_policy(path: Path = POLICY_PATH) -> Policy:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    testing = {risk: {k: int(v) for k, v in raw["testing"][risk].items()} for risk in RISK_LEVELS}
    return Policy(
        name=raw["policy"]["name"],
        version=raw["policy"]["version"],
        effective_date=raw["policy"]["effective_date"],
        hp_max_age_days=int(raw["documentation"]["history_and_physical_max_age_days"]),
        require_signed_consent=bool(raw["documentation"]["require_signed_consent"]),
        testing=testing,
        require_pre_procedure_instruction=bool(
            raw["anticoagulation"]["require_pre_procedure_instruction"]
        ),
        require_post_procedure_instruction=bool(
            raw["anticoagulation"]["require_post_procedure_instruction"]
        ),
        systolic_exclusion_at_or_above=float(
            raw["acute_safety"]["systolic_bp_exclusion_at_or_above"]
        ),
        diastolic_exclusion_at_or_above=float(
            raw["acute_safety"]["diastolic_bp_exclusion_at_or_above"]
        ),
        temperature_f_exclusion_above=float(raw["acute_safety"]["temperature_f_exclusion_above"]),
    )


@lru_cache(maxsize=1)
def default_policy() -> Policy:
    return load_policy()
