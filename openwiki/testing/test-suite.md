---
type: test suite
title: Offline Test Suite
description: The pytest suite in tests/ — evaluator semantics, the ir-measures oracle cross-check, frozen-corpus manifest verification, and chunking/RunConfig invariants — all offline, with API keys required only because config reads them at import.
tags: [testing, pytest, offline, oracle, evaluators, invariants]
sources:
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-fca896e236afd96cf1228c0f
    resource: repo://tests/test_chunking_strategy.py
  - id: openwiki-source-f5beecb1cfd369c30ed5f9f6
    resource: repo://tests/test_corpus_manifest.py
  - id: openwiki-source-243e2ba3461465d15f5c241d
    resource: repo://tests/test_evaluators_oracle.py
  - id: openwiki-source-864eac58c65e882a533713e7
    resource: repo://tests/test_evaluators.py
  - id: openwiki-source-0720c5a3812dee82f153074a
    resource: repo://tests/test_ui.py
generated: { by: "claude-code", at: "2026-10-12T14:31:19.877Z" }
verified:
  - by: openwiki/0.6.0
    at: 2026-10-12T14:31:19.877Z
---

# Offline Test Suite

Five test files under `tests/`, all **offline**: no network calls, no LLM calls, no LangSmith access. `pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`, so `uv run pytest` imports the flat top-level modules directly. The README notes the one environmental caveat: `config.py` reads `OPENAI_API_KEY`/`LANGSMITH_API_KEY` at import, so the keys must be *set* (any value) for `utils`/`config` imports — which is also why `evaluators.py` is kept free of those imports.

## tests/test_evaluators.py — semantics, not just math

The largest file (~680 lines) pins the *contracts* of `evaluators.py` with controlled example dicts:

- **Metric semantics**: recall/hit/MRR values and edge cases; empty gold → `None` (skip, not zero).
- **Level handling**: paper-level dedup keeps first-occurrence rank; papers derived from chunk ids when `retrieved_arxiv_ids` isn't passed; explicit `gold_arxiv_ids` preferred over derived; chunk level ignores the arxiv list; unknown level raises.
- **k semantics**: `k` cuts the raw retrieved list before dedup.
- **Precision gating**: `None` without completeness evidence, `None` for single-paper gold, scored under `{"review": {"state": "adjudicated"}}`.
- **format_validator**: accept/reject per check (wrong JSON key, missing table separator, over-word bullets, wrong CSV order, sentence+citation, unknown instruction).
- **Abstention**: paraphrase coverage, the first-person hedge documented as a *known false positive*, English-only scoping, empty answers.
- **wilson_ci**: sanity anchors (10/10 → [0.72, 1.00]) and invalid-count rejection.
- **abstention_summary**: the two "dumb systems" tests (never-refuses vs always-refuses) that prove recall and over-refusal are reported separately; per-slice suffixes; unlabeled examples skipped.
- **normalize_text** (ADR-010): lowercase / punctuation-as-space / article removal spelled out case by case, including hyphen-splitting and em-dash-vanishing, and the deliberate absence of a stopword list (`not` survives).
- **arXiv id removal**: `ARXIV_ID` stripped in every spelling — bracketed, bare, versionless, uppercase `V`, several ids per bracket, parenthesized, `arXiv:` prefix — while lookalikes (`12345 trials`, `v3.14`) survive.
- **token_f1 / word_recall**: basics plus the semantics that distinguish them — a verbose answer covering the reference scores Recall 1.0 but F1 < 1, and multiset overlap caps repeated words at the reference count.
- **ADR-010 evidence batteries**: T0–T4 reformatting invariance (hyphen→space, em dash→comma, uppercase+loose punctuation on real multi-doc golds all score 1.0), U1–U4 id-spelling invariance, an ids-only answer scoring 0.0, and the function-word floor test pinning the ~0.22 Recall level as a *known trade-off, not quality*.
- **token_f1_summary**: slice scoping (`unanswerable` excluded even with a gold answer, empty golds skipped), `__<slice>` per-slice keys for both metrics, "never guesses the slice" (no `examples` or out-of-scope slices → `{"results": []}`), the rename guard (`f1_summary_evaluator`/`_token_f1` gone, `f1_summary` key gone), and the LangSmith contract test run through the real `_normalize_summary_evaluator` + `EvaluationResult` parsing.
- **LangSmith contract**: tests assert the exact `{"results": [...]}` shape.
- **Cross-module pin**: `test_arxiv_id_derivation_matches_utils_chunk_id` guards the string-split paper derivation against `utils.chunk_id`.

## tests/test_evaluators_oracle.py — the trec_eval cross-check

Verifies the retrieval formulas against `ir-measures` (the community's `trec_eval` wrapper, a dev-only dependency): 300 randomized cases each for paper level (with dedup and gold derived from chunk ids), chunk level, k cutoffs, and the `hit_at_ks` curve. The file documents the semantic gap it bridges — `ir-measures` ranks by score and P@k divides by the cutoff, while `precision_at_k` divides by distinct retrieved papers, so test runs are pre-deduplicated. The oracle itself never runs in production; it's cheap insurance that the reimplemented formulas agree with the reference implementation.

## tests/test_corpus_manifest.py — frozen-corpus invariants

Pins the manifest contract with stub fetchers: `abstract_digest` ignores whitespace but not content; write/load round-trip; rebuild preserves manifest order even when the fetcher returns a different order; and both fail-loud paths — missing arXiv id and revised abstract — raise `RuntimeError`.

## tests/test_chunking_strategy.py — retrieval-unit invariants

Pins ADR-005/RunConfig behavior: `chunk_size=0` returns one untouched record per paper; 500/0 splitting preserves metadata and the ≤500-char bound; `BASELINE` is `chunk_strategy == "abstract"` vs `CHUNKED_500_0` == `"500/0"`; overlap-without-chunking is rejected; per-strategy cache filenames (`arxiv_text-embedding-3-small_abstract.parquet` vs `_500-0.parquet`); `chunk_id`'s paper prefix; and lossless `papers_from_chunks` round-trip including re-derived chunk ids.

## tests/test_ui.py — the UI helpers

<!-- openwiki: broken internal link [url] file "url" does not exist. Fix the href or restore the target, then delete this comment. -->
Pure-function tests for `ui.py` (docs/ui/ui-plan.md §2 contract): `format_sources` dedups by `arxiv_id` while preserving retrieval rank, renders the exact `**[id]** title · date — [abs](url)` line format, and returns the neutral "Nenhuma fonte recuperada." for an empty list; `filter_history` keeps only `role`/`content`, drops the `metadata.title` sources blocks so they can't leak into the next turn's prompt, accepts both dicts and `gr.ChatMessage` objects, and ignores non-string content. `respond` — the only function that touches the pipeline — is verified manually, not in this suite.

## What's deliberately not tested

Anything requiring network or API keys: arXiv fetch, embedding builds, LLM generation, LangSmith sync. Those paths are exercised manually (`docs/manual-usage.md`) and will be covered by the experiment runner's own verification step (`results/` reports). The suite's role is to pin *evaluation semantics and corpus invariants* — the things a wrong number could silently corrupt.
