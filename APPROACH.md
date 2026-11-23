# Approach: Pre-Op Scheduling Triage

## The problem

A surgery can't be scheduled until four policy rules pass. The deciding facts are often buried in free-text notes: a recheck BP in a nursing addendum, a consent whose signature line says "pending", an anticoagulant the patient mentions but that isn't on the med list. A wrong READY can harm a patient. A wrong NEEDS_FOLLOW_UP costs a nurse a few minutes. The design is built around that asymmetry.

## Design in one line

**The LLM reads; code decides.**

```
submission ─┬─ structured fields (labs, vitals, meds, procedure) ───────────────┐
            └─ each document ─► reader (LLM, strict schema) ─► verify quotes ──┼─► rules engine ─► decision + evidence packet
                               └─► offline reader (cross-check) ─► review flags  │      (pure Python, policy.toml)
```

| Layer | What it does | Why |
|---|---|---|
| **Reader** (`preop/extract/llm.py`) | One OpenAI call per document, strict structured output (`responses.parse`), temperature 0, content-addressed cache. Reports *observations* with verbatim quotes. It never sees thresholds and never decides. | Reading free text is the only fuzzy part of the problem. Per-document calls keep context small, so they're cacheable, parallel, and failures stay isolated. |
| **Verification** (`preop/verify.py`) | Every extracted quote must appear in its source document, and vital values must appear in their quote. Unverified facts are dropped and flagged. | A guardrail against hallucination. Dropping a fact fails closed: a missing signature or plan gives NEEDS_FOLLOW_UP, never READY. |
| **Rules engine** (`preop/rules.py`) | Pure functions, one check per requirement. Each returns PASS / FAIL / UNKNOWN / BLOCKED / NOT_APPLICABLE with evidence, computed values, and a fix action. | Deterministic, unit-testable, and auditable. The same facts always produce the same decision. |
| **Policy as data** (`preop/policy.toml`) | Every window and threshold, plus the policy version stamped into each output. | A policy revision is a reviewed config diff, not a code change. |
| **Second reader** (`preop/extract/heuristic.py`) | Offline regex + drug lexicon. Runs free alongside the LLM. Any check where the two readers disagree becomes a `READERS_DISAGREE` review flag. It also serves as the per-document fallback if the API fails. | Two independent readers show the nurse *where* to look. It also lets tests and CI run without an API key. |

## The evidence packet

Each issue carries `rule`, `check`, `summary`, `facts` (computed values such as `age_days: 31, max_age_days: 30`), `evidence` (doc ID / lab ID plus a verified verbatim quote), and an `action` ("Obtain an updated H&P dated on or after 2026-02-13"). The output also includes:

- `checks`: the full audit trail, **including every PASS**, so sampled READY cases can be verified as quickly as flagged ones.
- `review_flags`: anything the engine couldn't settle cleanly (unverified quote, reader disagreement, ambiguous same-day vital order, procedure date taken from a note). These never change the decision. They direct human attention.
- `provenance`: engine, policy, prompt version, and model.

## Policy interpretation log

Places where the policy needed a reading, the choice made, and the sample cases that confirm it.

| Question | Choice | Confirmed by |
|---|---|---|
| Is a window inclusive? | Day 30 passes, day 31 fails (calendar days). | case_00012, 00014, 00009 |
| "Most recent result" vs. entered-in-error | Most recent **valid** result (final/amended/corrected). | 00004, 00006, 00041 |
| Does a BMP satisfy a CMP? | No. | 00017 |
| Aspirin / clopidogrel | Antiplatelets, not anticoagulants. Rule 3 doesn't apply. | 00002, 00025 |
| Anticoagulant in notes but not on the med list | Counts as current use. | 00000, 00027, 00038 |
| Inactive on list and a note confirms it was stopped | No plan needed. | 00019, 00043 |
| Active status blank / patient unsure | MISSING_REQUIRED_DATA, not ANTICOAGULATION_MANAGEMENT. | 00033 |
| Plan written for a different drug (switched warfarin → apixaban) | Doesn't count. Plans match by generic name. | 00018 |
| Plan with only a hold, or a deferred restart | Incomplete: needs both pre-op and post-op instructions. | 00020, 00023, 00036, 00048 |
| Vitals in notes | Count. Most recent across all sources wins. A normal recheck clears an exclusion. | 00005, 00016, 00032, 00046, 00047 |
| No temperature anywhere | MISSING_REQUIRED_DATA (Rule 4 can't be evaluated). | 00009, 00011, 00013, 00021, 00022 |
| Consent pending, or for a different procedure | Not a valid consent. A newer signed consent supersedes. | 00007, 00010, 00024, 00049 |
| Procedure date / risk blank | One MISSING_REQUIRED_DATA issue. Dependent window checks are BLOCKED, not double-reported. | 00011, 00029, 00039, 00044 |
| Risk blank: which labs? | Evaluate under every possible risk. Flag REQUIRED_TESTING only if labs fail under all of them. | 00029, 00039 |
| Acute exclusion plus other gaps | NOT_CLEARED, and still list every other issue. | 00002, 00028, 00031, 00032 |
| Which H&P counts? | Any H&P within 30 days. The policy doesn't require a "pre-op" H&P, so I don't add that criterion. | 00001, 00045 (typo "Pyhsical") |
| Procedure date blank but a note states one | Use it if documents agree, and flag it for review. (Judgment call, not covered by the sample labels; the PDF example has this shape.) | none |
| Note times have no timezone | Wall-clock comparison. If the most recent reading can't be singled out, evaluate the worst one and flag it. | none |

## Testing

`make test` runs 287 tests offline in about 3 seconds.

- **Rule tests** (`test_rules.py`): each rule and boundary, built from hand-written facts with no reader involved.
- **Metamorphic tests** (`test_metamorphic.py`): generated from the 50 real cases. Shifting every date by −40 or +365 days must not change any result. Writing drugs by generic instead of brand name must not change any result. A BP of 179/109 vs. 180/70 vs. 130/110, a temperature of 100.4 vs. 100.5 °F, and a CBC exactly at vs. one day past its window must each flip the decision exactly at the threshold.
- **Verification tests**: hallucinated quotes and invented vital values never reach the rules.
- **Reader contract tests**: strict schema, temperature 0, cache keyed on model + prompt + content, refusal handling, outage fallback, byte-identical output across runs.

## Results

| Reader | Decision | Categories | Aggregate | False READY |
|---|---|---|---|---|
| Heuristic (offline) | 50/50 | 50/50 | 100% | 0 |
| LLM (`gpt-4.1-mini`) | run `make baseline && make evals` | | | |

**Read the 100% honestly.** The heuristic reader was written against these 50 cases, so it shows the *rules engine* is correct. It doesn't show that the regexes generalize. The LLM reader is the generalization path, and the cross-check flags show exactly where the two readers diverge on new data.

## Running

```bash
export OPENAI_API_KEY=...
make baseline && make evals && make determinism && make score   # default: LLM reader + offline cross-check
make offline                         # offline score + per-category error analysis (no API key)
make offline EXTRACTOR=crosscheck    # same report using the LLM reader
make test
```
Set `PREOP_EXTRACTOR=llm|heuristic|crosscheck` to pick the reader. Set `PREOP_CACHE=0` to bypass the extraction cache.

## What I'd do next

1. Run the LLM reader on the sample, review every `READERS_DISAGREE` flag, and tighten the prompt.
2. Add a nurse-facing view that highlights each evidence quote in its source document. The data for it is already in the packet.
3. Track reader agreement rate and the rate of false READYs from nurse spot-checks as production metrics. Alert on drift.
