#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=2.0.0", "pydantic>=2.8.0"]
# ///
"""Score triage against the labels locally, with per-category error analysis.

Uses the same scoring function as run_evals.py but needs no OpenAI Evals run, and
prints exactly which category was missed or invented for each failing case.

    uv run tools/offline_eval.py                       # heuristic reader (offline)
    PREOP_EXTRACTOR=llm uv run tools/offline_eval.py   # LLM reader
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import triage_submission  # noqa: E402
from run_evals import EvalCase, _local_metrics_for_row, load_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT / "data" / "patients_sample_50.jsonl"))
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--out", default=str(ROOT / "data" / "offline_outputs.jsonl"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    os.environ.setdefault("PREOP_EXTRACTOR", "heuristic")

    cases = [EvalCase.model_validate(r) for r in load_jsonl(Path(args.input))]
    totals: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    extra: Counter[str] = Counter()
    decision_confusion: Counter[tuple[str, str]] = Counter()
    flag_counts: Counter[str] = Counter()
    rows = []
    for idx, case in enumerate(cases):
        out = triage_submission(case.submission, model=args.model).model_dump()
        rows.append({"record_index": idx, "case_id": case.case_id, "output": out})
        flag_counts.update(f["code"] for f in out["review_flags"])
        m = _local_metrics_for_row(case.label, out)
        for k, v in m["metrics"].items():
            totals[k] += int(bool(v))
        exp, act = set(m["expected_categories"]), set(m["actual_categories"])
        missed.update(exp - act)
        extra.update(act - exp)
        decision_confusion[(case.label.decision, out["decision"])] += 1
        if not all(m["metrics"].values()):
            print(f"\n✗ {case.case_id}: expected {case.label.decision} {sorted(exp)} | got {out['decision']} {sorted(act)}")
            for i in out["issues"]:
                print(f"    - [{i['category']}] {i['summary']}")
            if args.verbose:
                print(f"    label: {case.label.rationale}")

    n = len(cases)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print("\n=== Summary ===")
    for k in ("json_schema_valid", "decision_match_oracle", "issue_categories_match_oracle"):
        print(f"{k:32s} {totals[k]}/{n}  ({100 * totals[k] / n:.1f}%)")
    print(f"{'aggregate score':32s} {100 * sum(totals.values()) / (3 * n):.1f}%")
    print("missed categories :", dict(missed))
    print("extra categories  :", dict(extra))
    unsafe = sum(v for (e, a), v in decision_confusion.items() if a == "READY" and e != "READY")
    print(f"false READY (unsafe) : {unsafe}")
    print("review flags:", dict(flag_counts))
    print("decision confusion (label -> ours):", {f"{e}->{a}": v for (e, a), v in sorted(decision_confusion.items())})


if __name__ == "__main__":
    main()
