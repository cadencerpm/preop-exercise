"""Triage output contract, designed as an evidence packet for the reviewing nurse.

Scored fields: `decision` and `issues[].category`. Everything else exists so a human
can verify the result in seconds:

- each issue names the policy rule and check, the computed values (`facts`), the
  exact source records (`evidence`, with verbatim quotes checked against the source),
  and a concrete `action` that would resolve it;
- `checks` is the full audit trail, including every PASS, so sampled READY cases can
  be verified just as quickly as flagged ones;
- `review_flags` surface things the engine could not settle cleanly (unverifiable
  quotes, conflicting sources) without changing the policy decision.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]
CheckStatus = Literal["PASS", "FAIL", "UNKNOWN", "BLOCKED", "NOT_APPLICABLE"]
EvidenceSource = Literal["document", "lab", "vital", "medication", "procedure"]


class Evidence(BaseModel):
    source: EvidenceSource
    ref: str | None = Field(default=None, description="doc_id, lab id, or index into the submission")
    label: str = Field(description="Human-readable pointer, e.g. 'H&P [PDF] 2026-02-21 (A. Patel)'")
    date: str | None = None
    quote: str | None = Field(default=None, description="Verbatim source text")
    quote_verified: bool | None = Field(
        default=None, description="True if the quote was found verbatim in the source document"
    )


class CheckResult(BaseModel):
    rule: str
    check: str
    status: CheckStatus
    category: IssueCategory | None = None
    summary: str
    evidence: list[Evidence] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    action: str | None = None


class TriageIssue(BaseModel):
    category: IssueCategory
    rule: str
    check: str
    summary: str
    evidence: list[Evidence] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    action: str | None = None


class ReviewFlag(BaseModel):
    code: str
    message: str
    ref: str | None = None


class Provenance(BaseModel):
    engine_version: str
    policy_version: str
    extractor: str
    model: str | None = None
    prompt_version: str | None = None


class TriageOutput(BaseModel):
    decision: Decision
    issues: list[TriageIssue] = Field(validation_alias=AliasChoices("issues"))
    explanation: str
    checks: list[CheckResult] = Field(default_factory=list)
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    provenance: Provenance | None = None
