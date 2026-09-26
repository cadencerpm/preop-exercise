# Design & Implementation Notes

## 1. Starting point

The baseline sent the whole policy and submission to one LLM call and trusted its answer. The
handoff (`HANDOFF_preop_triage.md`) set the direction:

- **Policy lives in code; the LLM only reads text.** Deterministic, tested rules make every
  decision. The model does the one thing code can't: interpret free-text documents.
- **The data is labeled.** Build the scorer first and use the 50 cases as a feedback loop, but
  don't overfit to them.
- **Accumulate issues, then fold.** Collect every issue and derive the decision by precedence:
  `NOT_CLEARED` > `NEEDS_FOLLOW_UP` > `READY`.

## 2. What we learned before designing

| Finding | Impact |
|---|---|
| The grader (`run_evals.py`) scores schema (1), decision (1), issue-category **set** (1), and evidence grounding (0.5). `description` is not graded. | Category sets matter more than exact issue lists. |
| Null `procedure_risk` and warfarin with `active: null` both occur; the handoff missed them. | Missing-data handling has to be generic. The anticoagulant check has three states. |
| Consent fails two ways: missing, or present but unsigned. | Two separate issue types. |
| The labels' own citations fail the grounding check in 22 cases (3-digit BP values, bare-array sources, anticoag issues citing one array while quoting another). | We chose to mirror the labels anyway (see §5). |
| Document `type` strings are free text: 100+ variants, typos, retained-prior decoys. | Document meaning must come from `text`, via the LLM. |

## 3. Design

```
submission → normalize → gates → code-only rules (testing, safety)
           ├─ safety exclusion? → NOT_CLEARED, LLM skipped
           └─ else → LLM classifies documents → validate → document rules → decide
```

**Code owns** (structured fields are assumed standardized):
- most-recent reduction per lab and vital, *before* any window or threshold check, with no
  fallback to older results
- lab-code aliases (`LAB-CBC` → `CBC`)
- anticoagulant lookup (apixaban, warfarin; lisinopril and metformin are ignored)
- date windows (calendar days, inclusive) and vital thresholds
- the decision fold, citations, and explanation

**The LLM owns** only document interpretation. It sees `{index, type, date, text}` per document
and nothing else: no vitals, meds, dates, or policy. It returns:
- a role for each document: H&P, consent, anticoag plan, or other
- whether an H&P is current
- whether a consent is signed (`SIGNED` / `UNSIGNED` / `UNCLEAR`)
- whether an anticoag plan is clear

**The model picks pointers; code reads the values.** The model says "`documents[0]` is the
current H&P", and code reads that document's `date` and does the window math.

**Anti-hallucination checks:**
- A strict JSON schema at temperature 0.
- `validate()` drops any claim whose index doesn't exist or whose excerpt is empty or not
  verbatim in the document.
- The model's `reason`/`excerpt` text never reaches the output.

**Missing data:**
- The gates emit `MISSING_REQUIRED_DATA` and mark the field unavailable.
- Each rule declares the fields it `requires`, and is skipped if any of them is unavailable.
- Existence checks are separate from window checks, so "CBC missing" still fires when the risk
  is unknown.

## 4. Implementation

| Module | Role |
|---|---|
| `triage/facts.py` | Raw dict → typed `Facts`, each with a `Ref(path, obj)` for citations |
| `triage/gates.py` | Missing procedure date/risk, latest BP/temp (absent or valueless), anticoagulant with unknown status |
| `triage/rules.py` | `@rule(order, requires)` registry: H&P exists/window, consent, CBC/CMP exists/window, anticoag plan, BP/temp safety; `decide()` |
| `triage/cite.py` | One formatter per citation template in the labels |
| `triage/documents.py` | `OpenAIClassifier` (Responses API, strict schema, process-wide cache), `FakeClassifier`, `validate()` |
| `triage/pipeline.py` | Orchestration and the short-circuit |
| `core.py` | `triage_submission()`: same signature, now a thin wrapper around the pipeline |
| `eval/local_score.py` | Offline scorer that reuses the grader's metric code |

**Issue order:** each rule and gate has a fixed `order` value, chosen so that issues come out in
the same order as in the labels (e.g. documentation before safety; missing data before
anticoag). Output is stable-sorted by that value.

**Tests:** 218 tests, covering every edge case above, plus exact-match checks on all 50
labeled cases. They run with a fixture classifier whose labels were hand-written from the
document text, not from the expected outputs. The tests make no network calls.

## 5. Key decisions & trade-offs

- **Short-circuit on NOT_CLEARED.** A safety exclusion decides the case, so the LLM is skipped
  (7 of 50 cases). Cost: document issues aren't listed on those cases. On the labeled set this
  costs nothing. `review_documents_on_not_cleared=True` restores full review.
- **Content-aware H&P choice (case_00002).** The label flags a "retained prior" H&P that is out
  of window. We pick the real current H&P, which is in window. The decision still matches;
  we lose the category point on this one case. Clinically correct beats label-matching.
- **Mirror the labels' citations exactly.** We reproduce the labels' `description`, `source`, and
  `details` strings, which means inheriting their grounding misses (≈56% grounding). Each
  template is a single function in `cite.py`, so switching to grounding-friendly strings is a
  local change.
- **Response cache.** The cache is keyed on (model, prompt version, documents) and makes repeat
  calls identical. `make determinism` therefore reads 100% by construction: it measures
  repeatability, not how much the model's answers vary between calls.
- **Fail safe without the LLM.** On an API failure the pipeline logs a warning and emits
  "Document review unavailable". Safety rules still run, and a case can never be `READY`
  without document review.

## 6. Results (offline, fixture classifier)

| Metric | Score |
|---|---|
| Schema valid | 100% |
| Decision match | 100% |
| Category-set match | 98% (case_00002 by design) |
| Grounding | 56% (mirrored labels) |
| **Aggregate** | **93.14%** |

With no LLM at all (code only), the decision match is 94%. All 7 NOT_CLEARED cases are still
caught, and the 3 READY cases become NEEDS_FOLLOW_UP.

Live results depend on the model agreeing with the fixture's document labels. The live run
hasn't been done yet; see README.

## 7. Assumptions & known limits

- Windows are inclusive (`0 ≤ days ≤ N`). A result dated after the procedure is out of window.
- Undated items count as absent. Timestamps are normalized to UTC before comparison.
- An H&P with `is_current = null` is treated as not current.
- Not covered by the labeled data:
  - With multiple active anticoagulants, the first one is used.
  - With multiple plan documents, the most recent one is used.
  - For LOW/MODERATE window violations, the "for HIGH risk procedure" suffix is omitted.
- case_00042: we cite the actual (inadequate) plan document, while the label cites bare
  `documents`. The category still matches.
