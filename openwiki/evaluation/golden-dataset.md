---
type: evaluation dataset
title: Golden Dataset
description: The arxiv-copilot-golden LangSmith dataset — the uniform example schema, two-axis provenance model, six slices and their construction methods, synthetic-by-construction generation, and the idempotent sync contract in build_golden_dataset.py.
tags: [golden-dataset, langsmith, evaluation, provenance, slices, synthetic-data]
sources:
  - id: openwiki-source-0bfdf81f1534614019fe4429
    resource: repo://build_golden_dataset.py
  - id: openwiki-source-2b7d80a0e7d874afa5d401f9
    resource: repo://docs/adrs/0001-golden-set-provenance-data-model.md
  - id: openwiki-source-8f04a5996c3b2c8616f1e345
    resource: repo://docs/adrs/0009-incomplete-gold-and-pool-bias-mini-pooling.md
  - id: openwiki-source-a2b903fa68b9b04e7d5df273
    resource: repo://evaluators.py
generated: { by: "claude-code", at: "2026-10-12T14:24:36.766Z" }
verified:
  - by: openwiki/0.6.0
    at: 2026-10-12T14:24:36.766Z
---

# Golden Dataset

The golden set is the LangSmith dataset `arxiv-copilot-golden` (~100 examples in the `core` split; an `extended` split is reserved for CKPT-7's final A/B). `build_golden_dataset.py` builds it and keeps it in sync. It exists to make the seven RAG failure points measurable — each slice targets specific FPs.

## Uniform example schema

Every example, every slice, always carries the same shape (so evaluators never branch on slice):

```python
{
    "inputs":  {"question": ...},
    "outputs": {"answer": ..., "gold_chunk_ids": [], "gold_arxiv_ids": [], "should_abstain": bool},
    "metadata": {"slice": ..., "authoring": {"origin": "llm", "method": ...},
                 "review": {"state": "draft"}, ...},
}
```

Absent gold labels are empty lists, not missing keys. `should_abstain` is the label the abstention evaluators consume.

## Two-axis provenance (ADR-001)

Metadata separates **who wrote it** from **how deeply it was verified**, replacing a single `provenance` label that admitted impossible states ("curated but never reviewed" — the original "hand-written" examples were actually LLM drafts approved at batch level):

- `authoring.origin` ∈ `llm | human | llm+human`
- `authoring.method` ∈ `from-chunk | topic-authored | empirical`
- `review.state` ∈ `draft | spot_checked | human_reviewed | adjudicated` (+ `review.reviewer` required once reviewed)

Everything the builders emit is `origin=llm`, `review.state=draft` — human anchoring is recorded in review state at review gates, never in authorship. The `adjudicated` state is also load-bearing downstream: `evaluators.is_gold_complete` treats it as evidence the retrieval gold is the complete relevant set.

## The slices

Target counts live in `config.GOLDEN_SLICE_TARGETS`; builders live in `SLICE_BUILDERS`.

| Slice | n | FP | Construction | Retrieval gold |
|---|---|---|---|---|
| `answerable` | 40 | baseline | **Synthetic-by-construction**: an LLM writes a question *from* a sampled chunk (plus a 3-sentence cited gold answer), so that chunk is trivially the gold | `gold_chunk_ids=[source chunk]` |
| `persona` | 10 | FP6 | Same mechanism, but one base question per chunk gets **two** gold answers — `lay` (zero jargon + analogy) vs `phd` (mechanism + caveats); the stark-contrast pair is the FP6 calibration instrument (ADR-004) | source chunk |
| `unanswerable` | 15 | FP1 | Topic-authored: domains unambiguously outside cs.AI (LIGO, batteries, oncology…) | none; gold answer is abstention |
| `stale` | 10 | FP1′/R3 | Topic-authored: questions about papers published *after* the snapshot date on covered topics, tempting retrieval with old material | none pre-refresh; `corpus_snapshot` metadata records the pin |
| `format` | 10 | FP5 | Topic-authored: the format instruction IS the test (markdown tables, JSON, CSV, YAML, bullets, sentence+citation); gold answers are correctly-formatted exemplars | none needed |
| `multi-doc` | 15 | FP3/FP7 | Topic-authored hybrid (ADR-003): **10 known-item** questions naming the required paper ids (mechanical `gold_arxiv_ids`) + **5 open-topic** questions naming none | known-item: `gold_arxiv_ids`; open-topic: `pending-adjudication` |
| `deep-hit` | 15 | FP2/R5 | **No builder** — to be found empirically during EXP-0 (questions whose answer doc ranks low in baseline) | filled at EXP-0 |

Open-topic multi-doc examples carry `retrieval_gold: "pending-adjudication"` — `is_retrieval_evaluable` skips them until a human judges the retrieved pool. Per ADR-009, that adjudication is a **pilot mini-pooling** at CKPT-0.8 (~80 judgments: the 5 open-topic questions + 10 stratified random single-gold examples + up to 4 where the gold wasn't rank 1), not a full review — papers judged relevant go into `gold_arxiv_ids` and the example's `review.state` becomes `adjudicated`, while judged-irrelevant storage is still an open data-model question. Their gold answer doubles as the completeness reference.

## Generation mechanics

- `_sample_chunks` draws **stratified, seed-fixed** samples from the `CHUNKED_500_0` cache (seeds 42/43): shuffle paper ids, take ≤2 chunks per paper — preventing topic collapse while keeping every source-chunk id resolvable. The builder pins `CHUNKED_500_0` even though the baseline is whole-abstract, because the questions were generated from those chunks.
- `_llm_json` calls `GENERATION_MODEL` with `response_format: json_object`, one retry on parse failure.
- `_QA_GEN_PROMPT` bans definitional questions ("What is the main focus…") and requires the question to target a concrete detail — a guard against questions answerable from any paper in the area.
- `regenerate_vague_answerable` is the remediation path: it finds core-split `answerable` examples whose question matches `_GENERIC_Q_PAT`, regenerates at temperature 0.8 (up to 3 attempts, then raises) *from the same source chunk* so `gold_chunk_ids` stay valid, and updates examples in place, marking them `regenerated: v2-specific-question`.

## Idempotent sync contract

`build_golden_dataset()` creates or reads the dataset, counts existing examples per slice in the `core` split, then per slice:

- **at target** → skip (a second run is a no-op);
- **empty** → build and `create_examples` in bulk (asserting builder count == target);
- **partially populated** → **raise** — partial state implies manual UI edits a human must reconcile;
- **no builder** (`deep-hit`) → log and continue.

`main.py` runs this on every invocation; `python build_golden_dataset.py` runs it standalone.

## Relationship to evaluators

The schema is the contract the evaluators read: `gold_chunk_ids`/`gold_arxiv_ids` feed [retrieval metrics](./retrieval-metrics.md) (paper-level gold derived from the id prefix), `should_abstain` feeds [abstention_summary](./answer-evaluators.md), `metadata.slice` drives per-slice reporting, and `review.state`/`open_topic`/`retrieval_gold` metadata gate which examples are retrieval-evaluable and precision-eligible.
