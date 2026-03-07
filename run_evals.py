#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=2.0.0",
#   "pydantic>=2.8.0",
# ]
# ///

"""Run local OpenAI Evals scoring and determinism checks for baseline triage outputs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import time
from typing import Any

from openai import OpenAI

from core import (
    PreparedPatientCase,
    TriageIssue,
    TriageOutput,
    triage_submission,
)

ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = "gpt-4.1-mini"
TERMINAL_RUN_STATUSES = {"completed", "failed", "canceled"}
PRIMARY_SCORE_NAME = "aggregate_local_score_pct"

# issues_value_grounding is worth half a point — it rewards structured evidence
# but is harder to satisfy than the other binary metrics.
METRIC_WEIGHTS: dict[str, float] = {
    "json_schema_valid": 1.0,
    "decision_match_oracle": 1.0,
    "issue_categories_match_oracle": 1.0,
    "issues_value_grounding": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "patients_sample_50.jsonl"),
        help="Input patient JSONL file",
    )
    parser.add_argument(
        "--outputs",
        default=str(ROOT / "data" / "baseline_outputs.jsonl"),
        help="Baseline outputs JSONL file",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "data" / "eval_report.json"),
        help="Report output path",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id used for determinism mode",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
        help="Polling interval for eval run status checks",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout while waiting for eval run completion",
    )
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="Run 10x deterministic replay check instead of eval scoring",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of repeated calls for determinism mode",
    )
    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="Record index to use for determinism mode",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_cases(path: Path) -> list[PreparedPatientCase]:
    cases: list[PreparedPatientCase] = []
    for idx, row in enumerate(load_jsonl(path)):
        if not (
            isinstance(row, dict)
            and "submission" in row
            and "case_id" in row
            and "expected_output" in row
        ):
            raise ValueError(
                f"Input row {idx} is missing required keys "
                "(case_id, submission, expected_output). Run make prepare first."
            )
        case = PreparedPatientCase.model_validate(row)
        cases.append(case)
    return cases


def load_baseline_outputs(path: Path) -> dict[int, dict[str, object]]:
    outputs_by_index: dict[int, dict[str, object]] = {}
    for row in load_jsonl(path):
        idx = int(row.get("record_index", len(outputs_by_index)))
        outputs_by_index[idx] = row
    return outputs_by_index


_SOURCE_PATH_RE = re.compile(
    r"^(?P<base>[a-z_]+)(?:\[(?P<index>\d+)\])?(?:\.(?P<field>[a-z_][\w]*))?$"
)
_LIST_BASES = {"documents", "labs", "vitals", "medications", "conditions"}
_ALLOWED_BASES = _LIST_BASES | {"procedure", "patient", "metadata"}
def _resolve_source(
    submission: dict[str, object], source: str
) -> tuple[object | None, dict[str, object] | None]:
    if not source:
        return None, None
    match = _SOURCE_PATH_RE.match(source.strip().lower())
    if not match:
        return None, None

    base = match.group("base")
    if base not in _ALLOWED_BASES:
        return None, None

    index = match.group("index")
    field = match.group("field")
    if base in _LIST_BASES:
        if field and index is None:
            return None, None
        items = submission.get(base)
        if not isinstance(items, list):
            return None, None
        if index is None:
            return items, {"base": base, "index": None, "field": field}
        idx = int(index)
        if idx < 0 or idx >= len(items):
            return None, None
        item = items[idx]
        if field:
            if isinstance(item, dict):
                return item.get(field), {"base": base, "index": idx, "field": field}
            return None, None
        return item, {"base": base, "index": idx, "field": None}

    if index is not None:
        return None, None
    obj = submission.get(base)
    if field:
        if isinstance(obj, dict):
            return obj.get(field), {"base": base, "index": None, "field": field}
        return None, None
    return obj, {"base": base, "index": None, "field": None}


def _is_missing_issue(issue: TriageIssue) -> bool:
    return str(issue.category or "").upper() == "MISSING_REQUIRED_DATA"



def _collect_candidate_values(ref: object) -> list[str]:
    values: list[str] = []
    if isinstance(ref, dict):
        for value in ref.values():
            if isinstance(value, str) and len(value.strip()) >= 4:
                values.append(value)
            elif isinstance(value, (int, float)):
                values.append(str(value))
    elif isinstance(ref, str):
        values.append(ref)
    elif isinstance(ref, (int, float)):
        values.append(str(ref))
    return values


def _details_mentions_value(details: str, ref: object) -> bool:
    if not details:
        return False
    details_lower = details.lower()
    candidates = _collect_candidate_values(ref)

    for value in candidates:
        value_lower = value.lower()
        if len(value_lower) >= 4 and value_lower in details_lower:
            return True

    dates_in_details = set(re.findall(r"\d{4}-\d{2}-\d{2}", details_lower))
    if dates_in_details:
        for value in candidates:
            value_lower = value.lower()
            if any(date in value_lower for date in dates_in_details):
                return True

    return False


def _is_quote_from_doc(details: str, ref: object) -> bool:
    if not details:
        return False
    if isinstance(ref, dict):
        text = ref.get("text")
    elif isinstance(ref, str):
        text = ref
    else:
        text = None
    if not isinstance(text, str):
        return False
    snippet = details.strip().strip("'\"")
    if len(snippet) < 8:
        return False
    return snippet.lower() in text.lower()


def _fuzzy_grounded(
    submission: dict[str, object],
    issue: TriageIssue,
    details: str,
) -> bool:
    """Lenient fallback: passes if details mentions any concrete value from a
    submission object whose field appears in the source string, or from the
    category-relevant collection."""
    source_lower = str(issue.evidence.source or "").strip().lower()

    # Strategy (a): find submission objects whose field value appears in source.
    # E.g. vital.source="Primary care visit" contained in candidate source string.
    for collection in ("vitals", "labs", "documents", "medications", "conditions"):
        for item in submission.get(collection, []) or []:
            if not isinstance(item, dict):
                continue
            for v in item.values():
                if isinstance(v, str) and len(v) >= 4 and v.lower() in source_lower:
                    if _details_mentions_value(details, item) or _is_quote_from_doc(
                        details, item
                    ):
                        return True
    for top_key in ("procedure", "patient", "metadata"):
        obj = submission.get(top_key)
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str) and len(v) >= 4 and v.lower() in source_lower:
                    if _details_mentions_value(details, obj):
                        return True

    # Strategy (b): scan the category-relevant section for any mentioned value.
    category = str(issue.category or "").upper()
    section_refs: list[object] = []
    if category == "REQUIRED_TESTING":
        section_refs = list(submission.get("labs", []) or [])
    elif category == "ACUTE_SAFETY_EXCLUSION":
        section_refs = list(submission.get("vitals", []) or [])
    elif category == "REQUIRED_DOCUMENTATION":
        section_refs = list(submission.get("documents", []) or [])
    elif category == "ANTICOAGULATION_MANAGEMENT":
        section_refs = list(submission.get("documents", []) or []) + list(
            submission.get("medications", []) or []
        )
    elif category == "MISSING_REQUIRED_DATA":
        for key in ("procedure", "vitals", "labs", "medications", "documents"):
            item = submission.get(key)
            if isinstance(item, list):
                section_refs.extend(item)
            elif isinstance(item, dict):
                section_refs.append(item)

    return any(
        _details_mentions_value(details, ref) or _is_quote_from_doc(details, ref)
        for ref in section_refs
    )


def _check_issues_value_grounding(
    submission: dict[str, object],
    output: TriageOutput,
) -> bool:
    for issue in output.issues:
        source = issue.evidence.source
        details = issue.evidence.details
        if not source or not details:
            return False
        # Missing issues don't need value grounding — there's nothing to ground against.
        if _is_missing_issue(issue):
            continue
        # Non-missing issues must cite a concrete value from the submission.
        ref, _meta = _resolve_source(submission, source)
        if ref is not None and (
            _details_mentions_value(details, ref) or _is_quote_from_doc(details, ref)
        ):
            continue
        if _fuzzy_grounded(submission, issue, details):
            continue
        return False
    return True


def _extract_output_payload(
    row: dict[str, object],
) -> tuple[dict[str, object] | None, str | None]:
    output_payload = row.get("output")
    error = row.get("error")

    if isinstance(output_payload, dict):
        return output_payload, None if error is None else str(error)

    if (
        isinstance(row, dict)
        and "decision" in row
        and "explanation" in row
        and "issues" in row
    ):
        return row, None

    return None, None if error is None else str(error)



def _local_metrics_for_row(
    submission: dict[str, object],
    oracle_output: TriageOutput,
    model_output_payload: dict[str, object] | None,
) -> dict[str, object]:
    json_schema_valid = False
    parsed_output: TriageOutput | None = None
    parse_error: str | None = None

    if model_output_payload is not None:
        try:
            parsed_output = TriageOutput.model_validate(model_output_payload)
            json_schema_valid = True
        except Exception as exc:  # pragma: no cover - malformed output path
            parse_error = str(exc)

    actual_decision = parsed_output.decision if parsed_output else "INVALID"
    decision_match_oracle = (
        parsed_output is not None and actual_decision == oracle_output.decision
    )

    expected_categories = sorted({issue.category for issue in oracle_output.issues})
    actual_categories = (
        sorted({issue.category for issue in parsed_output.issues})
        if parsed_output
        else []
    )
    issue_categories_match_oracle = expected_categories == actual_categories

    issues_value_grounding = (
        _check_issues_value_grounding(submission, parsed_output)
        if parsed_output
        else False
    )

    metric_bools = {
        "json_schema_valid": json_schema_valid,
        "decision_match_oracle": decision_match_oracle,
        "issue_categories_match_oracle": issue_categories_match_oracle,
        "issues_value_grounding": issues_value_grounding,
    }

    aggregate_local_score = 100.0 * (
        sum(METRIC_WEIGHTS[k] for k, v in metric_bools.items() if v)
        / sum(METRIC_WEIGHTS.values())
    )

    return {
        "oracle": oracle_output.model_dump(),
        "parsed_output": parsed_output.model_dump() if parsed_output else None,
        "parse_error": parse_error,
        "metrics": metric_bools,
        "aggregate_local_score": aggregate_local_score,
        "expected_categories": expected_categories,
        "actual_categories": actual_categories,
        "actual_decision": actual_decision,
    }


def _build_eval_items(
    cases: list[PreparedPatientCase],
    outputs_by_index: dict[int, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    content_rows: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []

    for idx, case in enumerate(cases):
        submission = case.submission.model_dump()
        oracle_output = case.expected_output

        baseline_row = outputs_by_index.get(idx, {})
        output_payload, output_error = _extract_output_payload(baseline_row)

        local = _local_metrics_for_row(
            submission=submission,
            oracle_output=oracle_output,
            model_output_payload=output_payload,
        )

        metrics = local["metrics"]

        item = {
            "record_index": idx,
            "case_id": case.case_id,
            "oracle_decision": oracle_output.decision,
            "expected_issue_categories": "|".join(local["expected_categories"]),
            "expected_json_schema_valid": "true",
            "expected_issues_value_grounding": "true",
            "actual_decision": local["actual_decision"],
            "actual_issue_categories": "|".join(local["actual_categories"]),
            "json_schema_valid": "true" if metrics["json_schema_valid"] else "false",
            "issues_value_grounding": (
                "true" if metrics["issues_value_grounding"] else "false"
            ),
        }
        content_rows.append({"item": item})

        local_rows.append(
            {
                "record_index": idx,
                "case_id": case.case_id,
                "submission": submission,
                "baseline_error": output_error,
                "oracle": local["oracle"],
                "parsed_output": local["parsed_output"],
                "parse_error": local["parse_error"],
                "metrics": metrics,
                "aggregate_local_score": local["aggregate_local_score"],
            }
        )

    return content_rows, local_rows


def _create_eval(client: OpenAI) -> Any:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return client.evals.create(
        name=f"preop-triage-eval-{timestamp}",
        data_source_config={
            "type": "custom",
            "item_schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "record_index": {"type": "integer"},
                    "case_id": {"type": "string"},
                    "oracle_decision": {"type": "string"},
                    "expected_issue_categories": {"type": "string"},
                    "expected_json_schema_valid": {"type": "string"},
                    "expected_issues_value_grounding": {"type": "string"},
                    "actual_decision": {"type": "string"},
                    "actual_issue_categories": {"type": "string"},
                    "json_schema_valid": {"type": "string"},
                    "issues_value_grounding": {"type": "string"},
                },
                "required": [
                    "record_index",
                    "case_id",
                    "oracle_decision",
                    "expected_issue_categories",
                    "expected_json_schema_valid",
                    "expected_issues_value_grounding",
                    "actual_decision",
                    "actual_issue_categories",
                    "json_schema_valid",
                    "issues_value_grounding",
                ],
            },
        },
        testing_criteria=[
            {
                "type": "string_check",
                "name": "decision_match_oracle",
                "input": "{{item.actual_decision}}",
                "reference": "{{item.oracle_decision}}",
                "operation": "eq",
            },
            {
                "type": "string_check",
                "name": "json_schema_valid",
                "input": "{{item.json_schema_valid}}",
                "reference": "{{item.expected_json_schema_valid}}",
                "operation": "eq",
            },
            {
                "type": "string_check",
                "name": "issues_value_grounding",
                "input": "{{item.issues_value_grounding}}",
                "reference": "{{item.expected_issues_value_grounding}}",
                "operation": "eq",
            },
            {
                "type": "string_check",
                "name": "issue_categories_match_oracle",
                "input": "{{item.actual_issue_categories}}",
                "reference": "{{item.expected_issue_categories}}",
                "operation": "eq",
            },
        ],
        metadata={
            "project": "preop-triage-takehome",
            "source": "local-python-runner",
        },
    )


def _create_eval_run(
    client: OpenAI, eval_id: str, content_rows: list[dict[str, object]]
) -> Any:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return client.evals.runs.create(
        eval_id=eval_id,
        name=f"preop-triage-run-{timestamp}",
        data_source={
            "type": "jsonl",
            "source": {
                "type": "file_content",
                "content": content_rows,
            },
        },
        metadata={
            "source": "local-python-runner",
        },
    )


def _wait_for_run_completion(
    client: OpenAI,
    *,
    eval_id: str,
    run_id: str,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> Any:
    deadline = time.time() + timeout_seconds
    while True:
        run = client.evals.runs.retrieve(run_id=run_id, eval_id=eval_id)
        if run.status in TERMINAL_RUN_STATUSES:
            return run
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for eval run completion (eval_id={eval_id}, run_id={run_id})"
            )
        time.sleep(poll_interval_seconds)


def _summarize_output_items(output_items: list[Any]) -> dict[str, object]:
    by_criterion: dict[str, dict[str, int]] = {}

    for output_item in output_items:
        for result in output_item.results:
            bucket = by_criterion.setdefault(result.name, {"passed": 0, "failed": 0})
            if result.passed:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1

    summary: dict[str, object] = {}
    for criterion_name, counts in by_criterion.items():
        total = counts["passed"] + counts["failed"]
        pass_rate = (counts["passed"] / total) * 100 if total > 0 else 0.0
        summary[criterion_name] = {
            "passed": counts["passed"],
            "failed": counts["failed"],
            "total": total,
            "pass_rate_pct": round(pass_rate, 2),
        }

    return summary


def _summarize_local_rows(local_rows: list[dict[str, object]]) -> dict[str, object]:
    if not local_rows:
        return {
            "records": 0,
            "aggregate_local_score_pct": 0.0,
        }

    metric_names = [
        "json_schema_valid",
        "decision_match_oracle",
        "issue_categories_match_oracle",
        "issues_value_grounding",
    ]

    summary: dict[str, object] = {"records": len(local_rows)}

    for metric_name in metric_names:
        metric_values = [bool(row["metrics"][metric_name]) for row in local_rows]
        rate = (sum(1 for value in metric_values if value) / len(metric_values)) * 100
        summary[f"{metric_name}_rate_pct"] = round(rate, 2)

    aggregate_values = [float(row["aggregate_local_score"]) for row in local_rows]
    summary["aggregate_local_score_pct"] = round(
        sum(aggregate_values) / len(aggregate_values), 2
    )
    summary["primary_score_pct"] = summary["aggregate_local_score_pct"]

    return summary


def run_eval_mode(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input)
    outputs_path = Path(args.outputs)

    cases = load_cases(input_path)
    outputs_by_index = load_baseline_outputs(outputs_path)

    content_rows, local_rows = _build_eval_items(cases, outputs_by_index)

    client = OpenAI()
    eval_obj = _create_eval(client)
    run_obj = _create_eval_run(client, eval_obj.id, content_rows)

    final_run = _wait_for_run_completion(
        client,
        eval_id=eval_obj.id,
        run_id=run_obj.id,
        poll_interval_seconds=args.poll_interval_seconds,
        timeout_seconds=args.timeout_seconds,
    )

    output_items_page = client.evals.runs.output_items.list(
        run_id=run_obj.id,
        eval_id=eval_obj.id,
        limit=max(100, len(content_rows)),
    )
    output_items = list(output_items_page.data)

    local_metrics_summary = _summarize_local_rows(local_rows)
    primary_score_pct = float(local_metrics_summary.get(PRIMARY_SCORE_NAME, 0.0))

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "mode": "eval",
        "primary_score": {
            "name": PRIMARY_SCORE_NAME,
            "value_pct": primary_score_pct,
            "goal": "maximize",
        },
        "eval_id": eval_obj.id,
        "run_id": run_obj.id,
        "run_status": final_run.status,
        "run_report_url": final_run.report_url,
        "run_result_counts": final_run.result_counts.model_dump(),
        "run_per_testing_criteria_results": [
            result.model_dump() for result in final_run.per_testing_criteria_results
        ],
        "criteria_summary_from_output_items": _summarize_output_items(output_items),
        "local_metrics_summary": local_metrics_summary,
        "records": local_rows,
        "output_items": [item.model_dump() for item in output_items],
    }

    return report


def run_determinism_mode(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input)
    cases = load_cases(input_path)

    if not cases:
        raise ValueError(f"No patient records found in {input_path}")
    if args.record_index < 0 or args.record_index >= len(cases):
        raise IndexError(
            f"record-index {args.record_index} out of range for {len(cases)} records"
        )

    case = cases[args.record_index]
    submission = case.submission.model_dump()

    run_rows: list[dict[str, object]] = []
    decisions: list[str] = []
    canonical_outputs: list[str] = []

    for run_index in range(args.runs):
        try:
            output = triage_submission(
                submission=submission,
                model=args.model,
            )
            decision = output.decision
            canonical = json.dumps(
                output.model_dump(), sort_keys=True, separators=(",", ":")
            )
            valid_json = True
            error = None
        except Exception as exc:  # pragma: no cover - network/runtime failure path
            decision = "INVALID"
            canonical = "INVALID"
            valid_json = False
            error = str(exc)

        run_rows.append(
            {
                "run_index": run_index,
                "decision": decision,
                "json_schema_valid": valid_json,
                "error": error,
            }
        )
        decisions.append(decision)
        canonical_outputs.append(canonical)

    decision_counts = Counter(decisions)
    canonical_counts = Counter(canonical_outputs)
    json_valid_count = sum(1 for row in run_rows if row["json_schema_valid"])

    decision_stability_pct = (
        100.0 * max(decision_counts.values()) / args.runs if args.runs else 0.0
    )
    json_format_stability_pct = (
        100.0 * json_valid_count / args.runs if args.runs else 0.0
    )
    exact_output_match_pct = (
        100.0 * max(canonical_counts.values()) / args.runs if args.runs else 0.0
    )

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "mode": "determinism",
        "record_index": args.record_index,
        "case_id": case.case_id,
        "model": args.model,
        "runs": args.runs,
        "decision_stability_pct": round(decision_stability_pct, 2),
        "json_format_stability_pct": round(json_format_stability_pct, 2),
        "exact_output_match_pct": round(exact_output_match_pct, 2),
        "decision_counts": dict(decision_counts),
        "run_rows": run_rows,
    }
    return report


def main() -> None:
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be > 0")

    if args.determinism:
        report = run_determinism_mode(args)
    else:
        report = run_eval_mode(args)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"Wrote report -> {report_path}")
    if report.get("mode") == "eval":
        primary_score = report.get("primary_score", {})
        name = primary_score.get("name", PRIMARY_SCORE_NAME)
        value = primary_score.get(
            "value_pct",
            report.get("local_metrics_summary", {}).get(PRIMARY_SCORE_NAME, 0.0),
        )
        print(f"Primary score ({name}): {value}%")


if __name__ == "__main__":
    main()
