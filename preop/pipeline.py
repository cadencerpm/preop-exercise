"""End-to-end triage: read documents -> verify facts -> apply policy -> evidence packet."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from preop.extract.base import Extractor, ProcedureContext, ReaderConfigError
from preop.facts import DocumentFacts
from preop.output import CheckResult, Provenance, ReviewFlag, TriageIssue, TriageOutput
from preop.policy import Policy, default_policy
from preop.rules import CaseInputs, DocView, evaluate
from preop.verify import verify_document_facts

ENGINE_VERSION = "1.0.0"
MAX_WORKERS = int(os.environ.get("PREOP_MAX_WORKERS", "8"))


def _extract_all(
    extractor: Extractor, docs: list[dict[str, Any]], ctx: ProcedureContext, flags: list[ReviewFlag]
) -> list[DocumentFacts]:
    """Read every document in parallel. If the primary reader fails on a document
    (API outage after retries, refusal), fall back to the offline reader for that
    document and flag it, so one bad call degrades one document, not the whole case."""
    if not docs:
        return []

    def read(doc: dict[str, Any]) -> DocumentFacts:
        try:
            return extractor.extract(doc, ctx)
        except ReaderConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - any reader failure is handled the same way
            if extractor.name == "heuristic":
                raise
            from preop.extract.heuristic import HeuristicExtractor

            flags.append(ReviewFlag(
                code="PRIMARY_READER_FAILED", ref=str(doc.get("doc_id")),
                message=f"{extractor.name} reader failed ({type(exc).__name__}); used offline reader for this document.",
            ))
            return HeuristicExtractor().extract(doc, ctx)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(docs))) as pool:
        return list(pool.map(read, docs))


def build_case(submission: dict[str, Any], extractor: Extractor) -> CaseInputs:
    proc = submission.get("procedure") or {}
    ctx = ProcedureContext(procedure_type=proc.get("procedure_type"), procedure_date=proc.get("procedure_date"))
    raw_docs = [d for d in (submission.get("documents") or []) if isinstance(d, dict)]
    flags: list[ReviewFlag] = []
    extracted = _extract_all(extractor, raw_docs, ctx, flags)
    views: list[DocView] = []
    for i, (doc, facts) in enumerate(zip(raw_docs, extracted)):
        ref = str(doc.get("doc_id") or f"documents[{i}]")
        clean, doc_flags = verify_document_facts(facts, str(doc.get("text") or ""), ref)
        flags.extend(doc_flags)
        views.append(DocView(i, doc, clean))
    return CaseInputs(submission=submission, docs=views, flags=flags)


def decide(checks: list[CheckResult]) -> str:
    if any(c.status == "FAIL" and c.category == "ACUTE_SAFETY_EXCLUSION" for c in checks):
        return "NOT_CLEARED"
    if any(c.status in {"FAIL", "UNKNOWN"} for c in checks):
        return "NEEDS_FOLLOW_UP"
    return "READY"


def _explain(decision: str, issues: list[TriageIssue], checks: list[CheckResult]) -> str:
    if not issues:
        evaluated = [c for c in checks if c.status == "PASS"]
        return f"READY: all {len(evaluated)} applicable checks passed. " + " ".join(c.summary for c in evaluated)
    lead = {
        "NOT_CLEARED": "NOT_CLEARED: an acute safety exclusion is present.",
        "NEEDS_FOLLOW_UP": "NEEDS_FOLLOW_UP: required items are missing or out of window.",
    }[decision]
    return lead + " " + " ".join(f"[{i.category}] {i.summary}" for i in issues)


def triage(
    submission: dict[str, Any],
    extractor: Extractor,
    *,
    policy: Policy | None = None,
    second_reader: Extractor | None = None,
) -> TriageOutput:
    policy = policy or default_policy()
    case = build_case(submission, extractor)
    checks = evaluate(case, policy)

    if second_reader is not None:
        case.flags.extend(cross_check(submission, checks, second_reader, policy))

    issues = [
        TriageIssue(
            category=c.category,  # type: ignore[arg-type]
            rule=c.rule, check=c.check, summary=c.summary,
            evidence=c.evidence, facts=c.facts, action=c.action,
        )
        for c in checks
        if c.status in {"FAIL", "UNKNOWN"} and c.category
    ]
    # Safety blockers first, so the reviewer reads the reason for NOT_CLEARED first.
    issues.sort(key=lambda i: i.category != "ACUTE_SAFETY_EXCLUSION")
    decision = decide(checks)
    return TriageOutput(
        decision=decision,  # type: ignore[arg-type]
        issues=issues,
        explanation=_explain(decision, issues, checks),
        checks=checks,
        review_flags=_dedupe(case.flags),
        provenance=Provenance(
            engine_version=ENGINE_VERSION, policy_version=policy.version,
            extractor=extractor.name, model=extractor.model, prompt_version=extractor.prompt_version,
        ),
    )


def cross_check(
    submission: dict[str, Any], primary: list[CheckResult], reader: Extractor, policy: Policy
) -> list[ReviewFlag]:
    """Re-run the rules on an independent reader's facts; disagreements -> review flags.

    Never changes the decision: it tells the nurse which checks deserve a second look.
    """
    other = {c.check: c for c in evaluate(build_case(submission, reader), policy)}
    flags = []
    for c in primary:
        o = other.get(c.check)
        if o is None or o.status != c.status:
            flags.append(ReviewFlag(
                code="READERS_DISAGREE", ref=c.check,
                message=f"{c.check}: primary reader -> {c.status}; {reader.name} reader -> {o.status if o else 'not evaluated'}.",
            ))
    return flags


def _dedupe(flags: list[ReviewFlag]) -> list[ReviewFlag]:
    seen, out = set(), []
    for f in flags:
        key = (f.code, f.message, f.ref)
        if key not in seen:
            seen.add(key)
            out.append(f)
    # stable order: reader threads may append in any order
    return sorted(out, key=lambda f: (f.code, f.ref or "", f.message))
