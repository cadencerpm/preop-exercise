# Handoff: Pre-Op Scheduling Triage — implementation brief

**To:** Claude Code
**From:** the design discussion (human + Claude)
**Status:** design agreed at a high level. **Do not implement yet.** Read this, then
produce an architecture sketch (module layout, key types, interfaces, open questions)
for our review. We'll green-light before you write real code.

---

## 0. What this is

We're building a clinical triage agent for a take-home exercise (Cadence Surgical
Center). Input is one patient submission package as JSON. Output is a JSON decision:
one of `READY` / `NEEDS_FOLLOW_UP` / `NOT_CLEARED`, plus an exhaustive list of
`issues`, each citing the exact field/date/document it's based on. The policy in the
exercise appendix is the **only** source of truth — no external medical knowledge.

The exercise explicitly values *approach and engineering judgment over a perfect score*.
So: clean architecture, honest handling of ambiguity, documented assumptions, tests.

Language: **Python**, to match the provided starter harness and report viewer. Keep it
runnable with the existing harness.

---

## 1. The most important thing we learned: the dataset is labeled

`patients_sample_50.jsonl` is not 50 raw patients. Each line is:

```
{ "case_id": ..., "submission": {...}, "expected_output": {...} }
```

It's a **labeled eval set**. That changes the build order:

- **Build the scorer FIRST**, before any rule logic. It's our feedback loop.
- Distribution: 40 `NEEDS_FOLLOW_UP`, 7 `NOT_CLEARED`, 3 `READY`.
- The five exact `category` strings the grader uses (hardcode these as an enum, do not
  invent others):
  `MISSING_REQUIRED_DATA`, `REQUIRED_DOCUMENTATION`, `REQUIRED_TESTING`,
  `ANTICOAGULATION_MANAGEMENT`, `ACUTE_SAFETY_EXCLUSION`.

**Scoring caution:** we don't yet know if the harness scores decision-only, exact-issue-set,
or partial credit on issues. The scorer should report *all three* views (decision accuracy,
per-issue precision/recall, exact-match rate) so we can see where we stand regardless of how
the official harness weights it. **Do not overfit to these 50 cases** (see §8).

---

## 2. Core principle: the LLM feeds the rule engine; it is NOT the rule engine

The naive baseline hands the whole policy to one LLM call. We're doing the opposite:
push all reliability-critical logic into deterministic code, and confine the model to the
one thing code can't do — reading unstructured document text. Intelligence lives at the
edges; policy lives in tested code.

### The five-stage pipeline

```
submission
   │
   ▼
[1] NORMALIZE      (deterministic)  raw JSON → typed facts + provenance
   │                                 most-recent reduction, code canonicalization,
   │                                 med classification, null/missing detection
   ▼
[2] ENRICH         (LLM, narrow)    documents only → role + adequacy claims
   │                                 (index, role, excerpt, reason). No math, no verdicts.
   ▼
[3] VALIDATE       (deterministic)  check every model claim against the raw record
   │                                 (index exists, excerpt present). Anti-hallucination gate.
   ▼
[4] RULES          (deterministic)  one pure fn per rule → Issue[]; ACCUMULATE, no short-circuit
   │
   ▼
[5] DECIDE         (deterministic)  fold Issue[] → decision by precedence; build output JSON
   │
   ▼
{ decision, issues[], explanation }
```

Design consequence to preserve: **if you stub the LLM entirely, the rest of the system
should still run and produce well-formed, correctly-cited output** — just with worse
document classification. That's the test that the model is properly quarantined.

---

## 3. The LLM boundary (exact I/O contract)

**Input to the model: documents only.** For each document: `index`, `type`, `date`, `text`.
Nothing else crosses — no vitals, labs, thresholds, procedure_date, patient demographics,
med list, or policy text. (Feeding it numbers invites it to do arithmetic; feeding it the
policy invites it to emit a decision. Both are forbidden.)

**The model answers two questions, and only these:**

1. **Role assignment + currency.** Map each document to a fixed enum
   `{ HISTORY_AND_PHYSICAL, SURGICAL_CONSENT, PERIOP_ANTICOAG_PLAN, OTHER }`, and for an
   H&P decide `is_current` (current episode vs. retained prior). For consent, read
   `signed: true/false`.
2. **Anticoag plan adequacy.** For the document (if any) that is a perioperative anticoag
   plan: `is_clear_plan: true/false` — does the text actually describe how the medication
   is held before and resumed after surgery, or does it just gesture at one?

**Output shape (illustrative):**

```json
{
  "documents": [
    { "index": 0, "role": "HISTORY_AND_PHYSICAL", "is_current": true,
      "excerpt": "pre-op evaluation complete for planned procedure", "reason": "..." },
    { "index": 1, "role": "OTHER", "is_current": false,
      "excerpt": "Prior pre-op H&P retained for longitudinal chart context",
      "reason": "explicitly a retained prior" }
  ],
  "anticoag_plan": {
    "document_index": 4, "is_clear_plan": false,
    "excerpt": "Follow up with cardiology for peri-op recommendations",
    "reason": "defers to cardiology; no hold/resume instructions"
  }
}
```

**Hard rules for the model:** returns labels/booleans + `index` + verbatim `excerpt` +
short `reason`. **Never** computes dates or date differences. **Never** emits a decision or
a category. **Never** invents an index. Enforce with structured outputs / JSON-schema /
tool-call. Temperature 0.

**The model picks pointers; code reads values at pointers.** The model says "the H&P is
`documents[0]`"; code reads `documents[0].date` from the raw record and does the window math.
This is the single biggest hallucination-reducer.

### Open design decision to flag in the sketch: the anticoagulant context field

Earlier we considered passing the resolved anticoagulant name (e.g. `"apixaban"`) into the
model as context to scope the plan-adequacy judgment. **We've since leaned against it.**
Every real plan document names its own drug in the text, so the field is usually redundant,
it can drift out of sync with what code resolved, and naming the drug can nudge the model
toward volunteering outside guideline knowledge (forbidden). **Recommended:** don't pass it.
Code owns med detection; the model just says "which document, if any, is an adequate plan";
the two meet only at Rule 3 (see §5). Propose it this way in the sketch, but call it out as a
reversible choice.

---

## 4. What determinism owns (everything except reading text)

- **Most-recent reduction.** For each required test (labs) and each vital type, keep only the
  single most-recent result. Reduce **before** the window/threshold check. **No fallback** to
  an older in-window result — policy says only the most recent counts.
- **Lab code canonicalization.** Codes vary (`LAB-CBC` vs `CBC`). Needs an alias/normalization
  step or required tests won't be found.
- **Date-window arithmetic.** H&P within 30 days; CBC within 30 (LOW/MODERATE) or 14 (HIGH);
  CMP within 14 (HIGH). Compute against `procedure_date`.
- **Threshold comparisons.** Rule 4 vitals: systolic ≥180, diastolic ≥110, temp >100.4°F.
- **Medication classification.** Lookup table against a known anticoagulant set. Do NOT use the
  model for this.
- **Null / missing-field gating** (see §6).
- **The decision fold** (§5) and **citation string formatting** (§7).

---

## 5. Decision model: accumulate, then fold by precedence

**Accumulate — do not short-circuit.** Collect issues from every applicable rule. Proven by
`case_00002`: it's `NOT_CLEARED` but its `issues[]` contains *both* a `REQUIRED_DOCUMENTATION`
issue *and* the `ACUTE_SAFETY_EXCLUSION`. So safety does not stop rule evaluation.

**Fold by precedence:**

```
if any issue.category == ACUTE_SAFETY_EXCLUSION:  decision = NOT_CLEARED
elif issues is non-empty:                          decision = NEEDS_FOLLOW_UP
else:                                              decision = READY
```

Rationale: the three verdicts mean different things. `NOT_CLEARED` = positive evidence of a
specific dangerous condition (narrow, terminal, must win). `READY` = everything affirmatively
checks out. `NEEDS_FOLLOW_UP` = the honest fallback for anything missing, stale, ambiguous, or
un-evaluable.

**Safety must be evaluated even when other data is missing** — missing paperwork must never
suppress a real `NOT_CLEARED`. But missing *vitals* can't trigger `NOT_CLEARED` (you can't
assert danger you didn't measure); absent vitals become `NEEDS_FOLLOW_UP` instead (§6).

---

## 6. Edge cases & traps (all observed in the data — turn each into a test)

1. **`procedure_date == null` → gate, don't fail.** Emit one `MISSING_REQUIRED_DATA`
   ("Missing procedure date", source `procedure.procedure_date`). Then **suppress** every
   date-window-dependent rule (Rule 1's H&P-within-30 half; all of Rule 2) — mark them
   *un-evaluable*, do NOT emit window-violation issues for them. Keep date-*independent* checks
   running (consent presence, Rule 3, Rule 4). Confirmed on `case_00000` and 3 others: their
   labels contain only the missing-date issue (+ anticoag where applicable), never a window issue.
   Model "un-evaluable" explicitly rather than null-checking inside each rule.

2. **"Only most recent" applies to vitals too, not just labs.** Reduce BP to latest-BP, temp to
   latest-temp, then threshold-check those. (`case_00003`: "latest BP systolic=184".)

3. **Missing vitals are their own issues.** If no BP reading → a `MISSING_REQUIRED_DATA`
   "Missing latest blood pressure"; same for temperature. Source is the bare array `vitals`
   (there's no element to index). Two separate issues if both absent (`case_00004`). Implication:
   `READY` requires BP and temperature both present and safe.

4. **Anticoagulant vocabulary is small and adversarial.** Recognize `apixaban`, `warfarin`.
   `lisinopril` (BP med) and `metformin` (diabetes) appear alongside as **distractors** and must
   NOT trigger Rule 3. Classification is a lookup, not a substring match on "does a med exist".

5. **Document type strings are unreliable — this is why Rule 1/3 doc selection is the LLM's job.**
   Observed: misspellings (`"History & Phsyical"`), retained-prior decoys
   (`"...retained for longitudinal chart context"`), and near-miss distractors
   (nursing "home medications reviewed" is NOT an anticoag plan). A `contains("H&P")` filter
   fails all three. The model reads meaning; code never string-matches document types.

6. **Anticoag plan adequacy generalizes beyond one phrase.** `documents[4]` "follow up with
   cardiology" = defers = not a clear plan. But the *rule* is "does the text specify how the drug
   is held/resumed," not "does it contain the word cardiology." Prompt for the general judgment;
   do not hardcode the phrase (that's overfitting — see §8).

7. **Rule 3 is a join.** It fires only when (code: an active anticoagulant exists) AND
   (model: no document is a clear plan). Either half alone does nothing.

---

## 7. Citation conventions (mirror the labels exactly — grader may check `source`/`details`)

- Specific element you're pointing at → indexed: `documents[1]`, `vitals[2]`.
- Something **absent** (no element to index) → bare array name: `vitals`.
- A structured scalar field → dotted path: `procedure.procedure_date`.
- `details` is a human-readable string carrying **the observed value AND the rule it violated**,
  e.g. `"H&P date 2026-01-30 vs procedure_date 2026-03-03 (32 days prior; must be within 30)"`,
  or `"latest temperature value_f=101.0; threshold is > 100.4"`.
- Because every fact carries provenance from Stage 1, citations are assembled from data in hand —
  never reconstructed after the fact.

---

## 8. Known judgment forks — document, don't overfit

- **H&P selection can diverge from the labels.** In `case_00002` the label flags `documents[1]`
  (a retained prior, out of window) as the H&P — because the reference appears to select the H&P by
  the *first type-string match in array order*, which skips the misspelled `documents[0]` (the real,
  in-window current H&P). A content-aware classifier correctly picks `documents[0]` → no doc issue →
  **one fewer issue than the label** (though the decision still matches: temp dominates).
  **Decision:** do the clinically-correct thing (content-aware), and *write this divergence into the
  submission notes* as a deliberate, defensible choice. Optionally: when documents genuinely conflict
  (a current-looking H&P and a stale one both plausible, or a malformed type), lower confidence and
  emit a follow-up flag rather than silently choosing — that matches the triage spirit ("get human
  eyes on it").
- **Don't pattern-match the labels.** Treat the 50 cases as the spec for the *rules*, and write the
  fuzzy checks to generalize (what makes an H&P current, what makes a plan a plan) rather than to
  reproduce these exact strings.

---

## 9. Suggested module layout (propose your own if better)

```
triage/
  models.py        # typed submission model + Fact/Issue/Decision + category enum + LLM output schema
  normalize.py     # raw JSON → NormalizedFacts (most-recent reduction, code aliasing, med classify, null detection)
  llm/
    client.py      # single call, structured-output enforced, temperature 0
    schema.py      # the documents-only input + role/adequacy output schema
    prompt.py      # system prompt (the "hard rules for the model" contract)
  validate.py      # check LLM claims against raw record; drop/flag invalid ones
  rules.py         # one pure fn per rule; each returns Issue[]; "un-evaluable" support
  decide.py        # accumulate + precedence fold + explanation string + output assembly
  pipeline.py      # orchestrates 1→5; LLM-stub mode for deterministic-only runs
  cli.py           # run one submission / run over jsonl
eval/
  scorer.py        # decision accuracy + per-issue P/R + exact-match, over the labeled jsonl
tests/
  test_rules.py    # each edge case in §6 as a fixture-backed unit test
  test_normalize.py
  test_decide.py
```

---

## 10. Your task: SKETCH FIRST, do not implement

Produce for review, before writing implementation code:

1. **Module/interface sketch** — file layout + the signatures of the key functions
   (`normalize()`, `classify_documents()`, `validate()`, each rule fn, `decide()`, `score()`).
2. **Key type definitions** — the `NormalizedFacts` model, the `Issue` object (with its
   provenance/citation fields), and the LLM output schema.
3. **The rule-engine interface** — how a rule declares itself evaluable/un-evaluable and how
   issues carry pre-attached citations.
4. **The "un-evaluable" mechanism** for the null-procedure-date gate — show how you'd model it.
5. **Open questions / assumptions** you'd want us to confirm (e.g. the anticoag-context field in
   §3, the H&P divergence stance in §8, anything ambiguous in the policy).

Keep the sketch tight. We'll review, adjust, then you build — scorer first, then normalize +
rules against the labeled set, then wire in the LLM step last.
