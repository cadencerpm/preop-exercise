# Pre-Op Triage Take-Home

## Objective

`triage_submission(...)` in `core.py` triages a single pre-op submission package against the Cadence Surgical Center policy. It started as a naive single LLM call. It is now a thin wrapper around a deterministic pipeline in `triage/`, where all policy logic lives in tested Python code. The LLM is used only to read free-text documents: which one is the current H&P, whether consent is signed, and whether an anticoagulation plan is clear. See `NOTES.md` for the design, assumptions and known differences from the labels.

Your output must match this schema:

- `decision`: `READY | NEEDS_FOLLOW_UP | NOT_CLEARED`
- `issues[]`: category + evidence (`source`, `details`)
- `explanation`

## How the pipeline works

```
submission → normalize → gates → code-only rules (testing, safety)
           ├─ safety exclusion fired? → NOT_CLEARED (LLM skipped)
           └─ otherwise → LLM classifies documents → validate → document rules (H&P, consent, anticoag) → decision
```

| Module | Responsibility |
|---|---|
| `triage/facts.py` | Parse the submission into typed facts; keep only the most recent vital or lab; map lab-code aliases; classify anticoagulants |
| `triage/gates.py` | Missing-data checks (procedure date/risk, latest BP/temp, anticoagulant with unknown active status) |
| `triage/rules.py` | One function per policy rule, the decision fold, and the explanation |
| `triage/cite.py` | Builds evidence `source`/`details` strings |
| `triage/documents.py` | The only LLM call (documents only), anti-hallucination validation, response cache |
| `triage/pipeline.py` | Orchestration and the NOT_CLEARED short-circuit |
| `eval/local_score.py` | Offline scorer that uses the same metrics as `run_evals.py` |

If the LLM is unavailable (no key, network error), the pipeline still runs. It logs a warning and reports `Document review unavailable`. Safety exclusions are still detected, but a case can never be `READY` without document review.

## What Is Provided

- `data/patients_sample_50.jsonl` includes:
  - `case_id`
  - `submission`
  - `expected_output`
- `run_baseline.py` runs your `triage_submission` implementation and writes outputs.
- `run_evals.py` scores outputs against provided `expected_output` and can run determinism checks.

## Completion

Note: this exercise is evaluated on engineering judgment. You may not reach a 100% score, and that is OK! We are looking to understand how you approached the problem and designed a working solution.

## Setup

1. Confirm `uv` is installed. All commands run through `uv`, which provides Python ≥3.11 and the dependencies. The system Python can be older.

```bash
uv --version
```

2. Set your OpenAI API key. It is only needed for live runs; tests and offline scoring work without one.

```bash
export OPENAI_API_KEY="<your_api_key>"
```

## Running without an API key

Run the unit tests (no network):

```bash
make test
```

Score the pipeline offline. This replays hand-labelled document classifications from `tests/fixtures/doc_claims.json` instead of calling the LLM:

```bash
uv run --with 'openai>=2.0.0' --with 'pydantic>=2.8.0' python3 eval/local_score.py
```

It prints the same four metrics as `make evals`, plus a per-case diff against the labels. Add `--review-documents-on-not-cleared` to disable the short-circuit.

## Running live (requires `OPENAI_API_KEY`)

1. Smoke-test one case. `case_00012` should return `READY`:

```bash
uv run --with 'openai>=2.0.0' --with 'pydantic>=2.8.0' python3 -c '
import json; from core import triage_submission
rows = [json.loads(l) for l in open("data/patients_sample_50.jsonl")]
print(triage_submission(rows[12]["submission"], model="gpt-4.1-mini").model_dump_json(indent=2))'
```

If it returns `NEEDS_FOLLOW_UP` with `Document review unavailable`, check stderr for the `Document classification call failed` warning (usually a missing or invalid key).

2. Generate outputs for all cases. About 43 of the 50 cases call the LLM; the rest are short-circuited:

```bash
make baseline 2>&1 | tee data/baseline.log
grep -c "Document classification call failed" data/baseline.log   # expect 0
```

3. Run eval scoring. This uploads results to the OpenAI Evals API, so it also needs the key:

```bash
make evals
```

Alternatively, score a live run locally without the Evals upload:

```bash
uv run --with 'openai>=2.0.0' --with 'pydantic>=2.8.0' python3 eval/local_score.py --classifier openai
```

4. Run the determinism check:

```bash
make determinism
```

LLM responses are cached per process (keyed on model, prompt version and document content), so repeated calls in one process return identical output. See `NOTES.md` for what this number does and doesn't measure.

5. Print the score:

```bash
make score
```

6. View the interactive report (TUI):

```bash
make report
```

This opens a terminal UI (`view_report.py`) that shows per-case results side-by-side with oracle expectations. You can browse records, see metric pass/fail status, and inspect submission data. Press `f` on a metric row to filter the case list to failures. Press `q` to quit.

## Outputs

- Baseline outputs: `data/baseline_outputs.jsonl`
- Eval report: `data/eval_report.json`
- Determinism report: `data/determinism_report.json`

## Configurable Variables

- `MODEL` (default `gpt-4.1-mini`)
- `INPUT` (default `data/patients_sample_50.jsonl`)
- `OUTPUT` (default `data/baseline_outputs.jsonl`)
- `REPORT` (default `data/eval_report.json`)
- `DETERMINISM_REPORT` (default `data/determinism_report.json`)
