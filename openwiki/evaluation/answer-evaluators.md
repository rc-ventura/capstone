---
type: evaluator subsystem
title: Answer Evaluators
description: The generation-side evaluators in evaluators.py — deterministic format_validator checks, abstention detection via normalized marker families, abstention_summary's recall/over-refusal reporting with Wilson CIs, and the token-F1 summary evaluator — plus the LangSmith evaluator contract they follow.
tags: [evaluators, abstention, format-validation, wilson-ci, langsmith, generation-metrics]
verified:
  - by: openwiki/0.6.0
    at: 2026-10-11T10:08:11.054Z
sources:
  - id: openwiki-source-3416dd8228bd00246c52dbba
    resource: repo://docs/adrs/0007-abstention-detection-and-reporting.md
  - id: openwiki-source-128a7649af2cd72bb2ba4f06
    resource: repo://docs/adrs/0008-answer-correctness-signals-and-calibration.md
  - id: openwiki-source-5e8cfa6ef22542c9ea8aaf4a
    resource: repo://docs/decisions.md
  - id: openwiki-source-a2b903fa68b9b04e7d5df273
    resource: repo://evaluators.py
generated: { by: "claude-code", at: "2026-10-11T10:08:11.054Z" }
---

# Answer Evaluators

`evaluators.py`'s generation side (Surface 2 in the plan) holds four evaluators: the deterministic `format_validator`, the abstention detector `abstained`, the `abstention_summary` aggregate, and `f1_summary_evaluator`. The module deliberately imports nothing from `config` or `utils`, so all of it runs without API keys.

## format_validator — literal instruction-following (FP5)

`format_validator(question, answer)` is pure code — zero LLM cost. It dispatches on the format the *question* demands and returns `{"score": 0|1, "reason": ..., "key": "format_valid"}`:

| Question contains | Check |
|---|---|
| `markdown table` | Table present, separator row, required columns in the header, uniform column counts, expected data-row count ("three papers" → 3 rows) |
| `JSON array` / `JSON object` | `json.loads` succeeds; exact key set match against backtick-quoted keys; array length constraints ("two objects") |
| `as CSV` | Header row equals the backtick-quoted spec; requested arXiv ids appear in order in column 1 |
| `as YAML` | All backtick-quoted keys present as `key:` lines |
| `bullet point` | Exact bullet count ("exactly N bullet points"); per-bullet word cap ("at most N words") |
| `sentences` + `citation` | Exact sentence count; answer ends with the required `[arxiv_id]` citation |

An unrecognized instruction scores 0 with reason `"no known format instruction detected"` — the checker is deliberately literal, which is its contract with the `format` slice: those questions encode the checkable constraints in their phrasing.

## abstained — did the answer refuse?

`abstained(answer)` normalizes the text (lowercase, typographic apostrophes → straight, apostrophes stripped, whitespace collapsed) and matches against `_ABSTENTION_MARKERS`: two families — **first-person** ("i dont know", "i cannot find") where the speaker owns the ignorance, and **source-negative** ("not in the context", "the corpus doesnt") where the context is said to lack the answer.

This replaced a two-phrase `startswith` check that missed every paraphrase (ADR-007 reproduces the bug: `"I'm sorry, the context doesn't say."` scored as an answer). Deliberate boundaries:

- `"dont know"` alone is **not** a marker — an answer reporting a *paper's* uncertainty ("the authors dont know whether…") must not read as a refusal.
- **English only**, by design: the marker list isn't extended by translation (an ever-growing list is the original trap); the `abstention_quality` LLM judge is the multilingual ground truth.
- Known, accepted false positive: a first-person hedge inside a real answer. EXP-0 reports heuristic × judge divergence with a ≥90% agreement target.

## abstention_summary — recall and over-refusal, never accuracy

`abstention_summary(outputs, reference_outputs, examples)` is a LangSmith **summary evaluator** (it sees the whole run). For each example it pairs `(should_abstain, did_abstain)` — `should` from the golden label, `did` from `abstained()` — and emits, total and per `metadata.slice`:

- **`abstention_recall`** — of should-refuse examples, the fraction refused (the release-blocking gate: ≥90% on `unanswerable`);
- **`over_refusal_rate`** — of should-answer examples, the fraction refused (the guardrail: a recall gain bought by refusing everything shows up here — RefusalBench shows the two errors trade off at r ≈ −0.78);
- **`abstention_f1`** — diagnostic only; a single number hides the trade-off.

Every score's comment carries raw counts plus a **Wilson 95% CI** from `wilson_ci(k, n)` — honest intervals at n = 10–25 where one example swings the estimate by ~7 points and Wald intervals can fall outside [0, 1]. Accuracy was removed entirely: a refuse-everything system would pass a single-number gate. `decisions.md` records the regression rule: *a gain in abstention_recall cannot come from an increase in over_refusal_rate*.

## f1_summary_evaluator — token-F1 vs gold

`f1_summary_evaluator(outputs, reference_outputs)` aggregates `_token_f1(answer, gold)` over examples that have a gold answer — a cheap substantive-coverage proxy. ADR-008 documents its known defects (correct paraphrases score low, wrong answers reusing reference words score high, citations count as tokens, refusal slices pollute the mean) and fixes the escalation order: free signals first, embedding cosine next, an LLM judge only if local calibration demands it. The `abstention_accuracy` key it used to emit was removed when `abstention_summary` took over.

## LangSmith contract

Per LangSmith's summary-evaluator protocol (`extra="forbid"` on `EvaluationResult`), these functions return exactly `{"results": [{"key", "score", "comment"?}, ...]}` — any extra top-level key fails the whole evaluator and silently drops every metric (the exception is only logged). `abstention_summary` additionally uses the positional `examples` argument to read `metadata.slice` for per-slice breakdowns; examples without a `should_abstain` label are skipped.

## Representative tests

`tests/test_evaluators.py` pins: each format checker's accept/reject paths; abstention paraphrase coverage, the hedge false-positive, English scoping, and empty answers; Wilson anchors (10/10 → [0.72, 1.00]) and invalid-count rejection; the two "dumb systems" (never-refuses vs always-refuses) proving the two error rates are reported separately; per-slice suffixes; and the exact `{"results": [...]}` return contract.
