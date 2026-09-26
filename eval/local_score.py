#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=2.0.0",
#   "pydantic>=2.8.0",
# ]
# ///

"""Offline scorer for the deterministic triage pipeline.

Runs ``triage.pipeline.run`` over every labeled case in the jsonl and reuses
``run_evals._local_metrics_for_row`` for the same four metrics ``make evals``
reports (json schema valid, decision match, issue-category-set match, issue
value grounding), plus a per-case issue-category diff against the oracle.

With the default (fake) classifier this makes no network calls at all -- it
replays the hand-labeled ``tests/fixtures/doc_claims.json`` fixture, so it is
safe to run repeatedly while iterating on the rule engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_evals import _local_metrics_for_row, load_cases
from triage import pipeline
from triage.documents import (
    Classifier,
    FakeClassifier,
    OpenAIClassifier,
    load_fixture_claims,
)

DEFAULT_INPUT = ROOT / "data" / "patients_sample_50.jsonl"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "doc_claims.json"

METRIC_NAMES = [
    "json_schema_valid",
    "decision_match_oracle",
    "issue_categories_match_oracle",
    "issues_value_grounding",
]

# Known, documented divergence from the oracle issue *category set* (see
# NOTES.md): case_00042 also diverges, but only on citation source, not on
# category set, so it never shows up in this script's divergence check
# below (which only compares category sets) and has no entry here.
#   - case_00002: content-aware H&P selection picks the in-window current H&P
#     (documents[0]) over the oracle's out-of-window retained prior
#     (documents[1]); the ACUTE_SAFETY_EXCLUSION issue still matches.
DEFAULT_ALLOW_LIST: dict[str, str] = {
    "case_00002": (
        "content-aware H&P selection: the current H&P (documents[0]) is in-window, "
        "so no REQUIRED_DOCUMENTATION issue is raised; the oracle instead flags the "
        "retained prior (documents[1]). Decision still matches (temperature dominates)."
    ),
}


def build_classifier(name: str, *, model: str, claims: list[object]) -> Classifier:
    if name == "fake":
        return FakeClassifier(claims)
    if name == "openai":
        return OpenAIClassifier(model=model)
    raise ValueError(f"Unknown classifier {name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument(
        "--classifier",
        choices=["fake", "openai"],
        default="fake",
        help="fake replays tests/fixtures/doc_claims.json (no network calls); "
        "openai makes real API calls and is not used by the automated tests",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--review-documents-on-not-cleared",
        action="store_true",
        help="disable the ACUTE_SAFETY_EXCLUSION short-circuit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.input))
    claims_by_case = load_fixture_claims(Path(args.fixture)) if args.classifier == "fake" else {}

    rows: list[dict[str, object]] = []
    divergences: list[tuple[str, list[str], list[str]]] = []

    for case in cases:
        submission = case.submission.model_dump()
        classifier = build_classifier(
            args.classifier, model=args.model, claims=claims_by_case.get(case.case_id, [])
        )
        output_payload = pipeline.run(
            submission,
            classifier,
            review_documents_on_not_cleared=args.review_documents_on_not_cleared,
        )
        local = _local_metrics_for_row(submission, case.expected_output, output_payload)
        rows.append(local)

        if local["expected_categories"] != local["actual_categories"]:
            divergences.append((case.case_id, local["expected_categories"], local["actual_categories"]))

    _print_report(rows, divergences)


def _print_report(
    rows: list[dict[str, object]], divergences: list[tuple[str, list[str], list[str]]]
) -> None:
    total = len(rows)
    print(f"Scored {total} cases (offline; use --classifier openai for a live run)\n")

    for name in METRIC_NAMES:
        rate = 100.0 * sum(1 for row in rows if row["metrics"][name]) / total
        print(f"  {name}: {rate:.2f}%")

    aggregate = sum(row["aggregate_local_score"] for row in rows) / total
    print(f"  aggregate_local_score_pct: {aggregate:.2f}%\n")

    print(f"Issue-category divergences from the oracle: {len(divergences)}")
    unexpected = 0
    for case_id, expected, actual in divergences:
        reason = DEFAULT_ALLOW_LIST.get(case_id)
        if reason is not None:
            print(f"  [allow-listed] {case_id}: expected={expected} actual={actual}\n      reason: {reason}")
        else:
            unexpected += 1
            print(f"  [UNEXPECTED]   {case_id}: expected={expected} actual={actual}")

    if unexpected:
        print(f"\n{unexpected} unexpected divergence(s) not on the allow-list.")
    else:
        print("\nNo unexpected divergences.")


if __name__ == "__main__":
    main()
