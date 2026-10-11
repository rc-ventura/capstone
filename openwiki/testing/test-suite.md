---
type: test suite
title: Offline Test Suite
description: The pytest suite in tests/ — evaluator semantics, the ir-measures oracle cross-check, frozen-corpus manifest verification, and chunking/RunConfig invariants — all offline, with API keys required only because config reads them at import.
tags: [testing, pytest, offline, oracle, evaluators, invariants]
verified:
  - by: openwiki/0.6.0
    at: 2026-10-11T10:08:11.054Z
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
generated: { by: "claude-code", at: "2026-10-11T10:08:11.054Z" }
---

# Offline Test Suite

Four test files under `tests/`, all **offline**: no network calls, no LLM calls, no LangSmith access. `pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`, so `uv run pytest` imports the flat top-level modules directly. The README notes the one environmental caveat: `config.py` reads `OPENAI_API_KEY`/`LANGSMITH_API_KEY` at import, so the keys must be *set* (any value) for `utils`/`config` imports — which is also why `evaluators.py` is kept free of those imports.

## tests/test_evaluators.py — semantics, not just math

The largest file (~530 lines) pins the *contracts* of `evaluators.py` with controlled example dicts:

- **Metric semantics**: recall/hit/MRR values and edge cases; empty gold → `None` (skip, not zero).
- **Level handling**: paper-level dedup keeps first-occurrence rank; papers derived from chunk ids when `retrieved_arxiv_ids` isn't passed; explicit `gold_arxiv_ids` preferred over derived; chunk level ignores the arxiv list; unknown level raises.
- **k semantics**: `k` cuts the raw retrieved list before dedup.
- **Precision gating**: `None` without completeness evidence, `None` for single-paper gold, scored under `{"review": {"state": "adjudicated"}}`.
- **format_validator**: accept/reject per check (wrong JSON key, missing table separator, over-word bullets, wrong CSV order, sentence+citation, unknown instruction).
- **Abstention**: paraphrase coverage, the first-person hedge documented as a *known false positive*, English-only scoping, empty answers.
- **wilson_ci**: sanity anchors (10/10 → [0.72, 1.00]) and invalid-count rejection.
- **abstention_summary**: the two "dumb systems" tests (never-refuses vs always-refuses) that prove recall and over-refusal are reported separately; per-slice suffixes; unlabeled examples skipped.
- **LangSmith contract**: tests assert the exact `{"results": [...]}` shape.
- **Cross-module pin**: `test_arxiv_id_derivation_matches_utils_chunk_id` guards the string-split paper derivation against `utils.chunk_id`.

## tests/test_evaluators_oracle.py — the trec_eval cross-check

Verifies the retrieval formulas against `ir-measures` (the community's `trec_eval` wrapper, a dev-only dependency): 300 randomized cases each for paper level (with dedup and gold derived from chunk ids), chunk level, k cutoffs, and the `hit_at_ks` curve. The file documents the semantic gap it bridges — `ir-measures` ranks by score and P@k divides by the cutoff, while `precision_at_k` divides by distinct retrieved papers, so test runs are pre-deduplicated. The oracle itself never runs in production; it's cheap insurance that the reimplemented formulas agree with the reference implementation.

## tests/test_corpus_manifest.py — frozen-corpus invariants

Pins the manifest contract with stub fetchers: `abstract_digest` ignores whitespace but not content; write/load round-trip; rebuild preserves manifest order even when the fetcher returns a different order; and both fail-loud paths — missing arXiv id and revised abstract — raise `RuntimeError`.

## tests/test_chunking_strategy.py — retrieval-unit invariants

Pins ADR-005/RunConfig behavior: `chunk_size=0` returns one untouched record per paper; 500/0 splitting preserves metadata and the ≤500-char bound; `BASELINE` is `chunk_strategy == "abstract"` vs `CHUNKED_500_0` == `"500/0"`; overlap-without-chunking is rejected; per-strategy cache filenames (`arxiv_text-embedding-3-small_abstract.parquet` vs `_500-0.parquet`); `chunk_id`'s paper prefix; and lossless `papers_from_chunks` round-trip including re-derived chunk ids.

## What's deliberately not tested

Anything requiring network or API keys: arXiv fetch, embedding builds, LLM generation, LangSmith sync. Those paths are exercised manually (`docs/manual-usage.md`) and will be covered by the experiment runner's own verification step (`results/` reports). The suite's role is to pin *evaluation semantics and corpus invariants* — the things a wrong number could silently corrupt.
