---
type: evaluator subsystem
title: Retrieval Metrics
description: The retrieval metrics in evaluators.py — paper-level vs chunk-level comparison, hit/recall/MRR/precision semantics, the gold-completeness gate on precision, the per-example reporting policy, and the anti-pooling-bias design verified against the ir-measures oracle.
tags: [retrieval, metrics, evaluation, recall, mrr, precision, ir-measures]
sources:
  - id: openwiki-source-9a2dbdf1e918930fa0a85df7
    resource: repo://docs/adrs/0006-retrieval-gold-granularity-and-metrics.md
  - id: openwiki-source-8f04a5996c3b2c8616f1e345
    resource: repo://docs/adrs/0009-incomplete-gold-and-pool-bias-mini-pooling.md
  - id: openwiki-source-a2b903fa68b9b04e7d5df273
    resource: repo://evaluators.py
  - id: openwiki-source-243e2ba3461465d15f5c241d
    resource: repo://tests/test_evaluators_oracle.py
  - id: openwiki-source-864eac58c65e882a533713e7
    resource: repo://tests/test_evaluators.py
generated: { by: "claude-code", at: "2026-10-12T14:24:36.766Z" }
verified:
  - by: openwiki/0.6.0
    at: 2026-10-12T14:24:36.766Z
---

# Retrieval Metrics

`evaluators.py`'s first surface measures Surface 1 — did retrieval find the right papers? All metrics share one core, `_ranked_and_gold`, which ranks retrieved items at **one** comparison level and pairs them with gold at the same level. The design exists to fix a real bug class: mixing chunk ids and arxiv ids in one list shifted MRR ranks and zeroed precision (metrics roadmap P1–P3, ADR-006).

## Levels and gold derivation

- **`level="paper"` (default, ADR-006 D-1).** Gold = `gold_arxiv_ids`, or the papers of `gold_chunk_ids` (the arxiv id is the prefix of `chunk_id` — `_arxiv_id_of_chunk` does a string split; a test pins it against `utils.chunk_id`). Retrieved rank = **first occurrence** of each distinct paper (`dict.fromkeys` — duplicate chunks collapse, order kept). The dataset is *not* migrated; paper gold is derived at evaluation time.
- **`level="chunk"` (diagnostic).** Gold = `gold_chunk_ids` only; retrieved = raw chunk ids. It answers "did we retrieve the exact source chunk?" — meaningful only within one chunking config, since `chunk_id` hashes the text (chunk gold can't be compared across chunking arms).
- **No gold at the requested level → `None`** — the metric is undefined for that example and skipped, never scored as failure.

`k` cuts the **raw** retrieved list (the retriever's top-k) *before* deduplication — so `k=5` means "the retriever's top-5 slots," not "5 distinct papers."

## The four metrics

| Function | Meaning | Notes |
|---|---|---|
| `hit_rate` | 1 if ≥1 gold in top-k, else 0 ("success") | primary signal for 1-paper gold |
| `recall_at_k` | fraction of gold in top-k ("coverage") | identical to hit for 1-gold — see reporting policy |
| `mrr` | 1 / rank of first gold, else 0 | averaged over examples → MRR |
| `precision_at_k` | fraction of top-k that is gold | **gated** — see below |

`hit_at_ks` returns the hit curve over several cutoffs (default 1/3/5) — it feeds CKPT-1's k sweep by showing how much each extra position buys.

## precision_at_k is gated on gold completeness

`precision_at_k` reads "not in gold" as an error, which is only legitimate when the gold is the **complete** relevant set. `is_gold_complete` derives that from existing metadata (no stored flag — ADR-001 forbids two sources of truth):

- `review.state == "adjudicated"` — a human judged the retrieved candidate pool (mini-pooling), so gold is complete w.r.t. that pool;
- `slice == "multi-doc"` and not `open_topic` — known-item gold is complete *by definition* ("relevant = the papers named in the question").

Everything else → precision returns `None`. The motivating case: a 1-paper gold at k=5 caps precision at 0.2 even for perfect retrieval, because the other 4 papers are *unjudged, not wrong* — reporting it would manufacture errors. At paper level the denominator is distinct papers retrieved. Precision is a diagnostic of noise reaching the generator (a *cause* of FP4), explicitly not an FP3 measure.

## Which examples are scored

`is_retrieval_evaluable` skips examples with no judged retrieval gold: zero-gold slices by design (`unanswerable`, `stale`, `format`) and open-topic `multi-doc` still `pending-adjudication`.

### Completing the gold — the mini-pooling pilot (ADR-009)

`precision_at_k`'s gating exists because **unjudged is never irrelevant**: papers outside the gold may answer the question, and treating them as errors punishes a better retriever (the classic incomplete-judgments problem). The fix is a **pilot mini-pooling at CKPT-0.8** — ~80 human judgments, not all ~225: the 5 open-topic questions (all top-5 candidates), 10 stratified random single-gold examples, plus up to 4 where the gold wasn't rank 1. Papers judged relevant go into `gold_arxiv_ids` and the example becomes `adjudicated` — which is exactly what flips `is_gold_complete` on and starts reporting precision.

A pre-registered stop rule decides what happens next: judged-relevant share ≤10% → keep the single gold; ≥25% → judge the remaining 40 examples; in between → review and decide. `Hole@k` (share of a retriever's top-k nobody judged) is to be built alongside — required before the first retriever comparison at CKPT-1 — but it needs judged-*irrelevant* storage that doesn't exist yet. Pool bias is bounded, not removed: candidates come only from the baseline retriever until a second retriever type (e.g. BM25, open item D-3) joins the pool.

`primary_retrieval_metrics(reference_outputs)` is the **reporting policy** (D-2) — which metrics an example should report:

| Gold papers | Reported |
|---|---|
| 1 | `["hit", "mrr"]` — recall would just repeat hit |
| 2+ | `["recall", "hit", "mrr"]` — recall is the coverage signal |
| 0 | `[]` — nothing to score |

## Oracle verification

`tests/test_evaluators_oracle.py` cross-checks all four formulas against `ir-measures` (the community-reference wrapper of `trec_eval`) over 300 randomized cases each: paper level with dedup, chunk level, k cutoffs, paper gold derived from chunk ids, and the hit curve. One semantics note encoded there: `ir-measures` P@k divides by the cutoff, while `precision_at_k` divides by distinct retrieved papers — the tests de-duplicate before comparing so both agree. The oracle is dev-only; in LangSmith the evaluators run per-example where building qrels/run objects would be pure overhead.

## Module boundary

The module never imports `utils` or `config` — paper ids come from a string split, and reference outputs arrive as plain dicts — which keeps the entire evaluator surface runnable offline without API keys.
