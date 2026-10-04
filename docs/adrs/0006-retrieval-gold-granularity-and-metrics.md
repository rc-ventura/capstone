# ADR-006: Retrieval gold granularity and metric reporting policy

**Status**: Accepted
**Date**: 2026-10-03
**Related**: [Lesson: Retrieval Unit and Gold Granularity](../learning-lessons/retrieval_unit_and_gold_granularity.md) · [Roadmap decisions D-1, D-2, L-4](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) · [ADR-005](./0005-retrieval-unit-whole-abstract.md)
**Code**: `evaluators.py` (`_ranked_and_gold`, `_gold_papers`, `hit_rate`, `recall_at_k`, `mrr`, `precision_at_k`, `hit_at_ks`, `primary_retrieval_metrics`); tests in `tests/test_evaluators.py` and `tests/test_evaluators_oracle.py`

---

## Context

The golden set has two kinds of retrieval gold, a consequence of how each slice was built (ADR-001, ADR-003):

| Slice | Gold in the dataset |
|---|---|
| `answerable` (40), `persona` (10) | `gold_chunk_ids`: **one chunk**, the one a question was generated from |
| `multi-doc` known-item (10) | `gold_arxiv_ids`: 2–3 **papers** |
| `unanswerable`, `stale`, `format`, `multi-doc` open-topic | none (open-topic pending adjudication) |

Problems found (CKPT-0.6.1b):

1. **Levels were mixed.** The metrics mixed chunk ids and arxiv ids in one list: with paper-level gold, MRR ranks were shifted by k
   (1/6 instead of 1.0) and `precision_at_k` returned 0.0.
2. **A one-chunk gold is an incomplete relevant set.** Each source paper has 2.44 sibling chunks on average (1 to 3); in 38 of 50
   examples the top-5 contains a sibling of the gold chunk. Chunk-level metrics are a **floor** (only the exact chunk counts);
   paper-level metrics are a **ceiling**.
3. **The two levels barely diverge today.** Baseline (k=5, 50 single-gold examples): hit@5 = 100% at both levels; hit@1 48/50 vs
   50/50; MRR 0.980 vs 1.000; only 2 of 50 examples differ.
4. **Chunk ids are not comparable across chunking configurations**: `chunk_id` hashes the chunk text (see ADR-005).
5. **With a single gold document, recall@k and hit@k are the same number** (`recall = 1[gold in top-k] = hit`): reporting both
   duplicates one signal (confirmed against `ir-measures`: `Success@5 = R@5`).

## Decision

1. **The gold of retrieval is the paper.** The paper-level gold is `gold_arxiv_ids`, or, when absent, the papers of `gold_chunk_ids`
   (the arxiv id is the prefix of the chunk id). The dataset is **not** migrated: it is derived at evaluation time.
2. **Evaluators score at paper level by default** (`level="paper"`). `level="chunk"` is a **diagnostic** ("did we retrieve the
   exact source chunk?"), meaningful only within one chunking configuration, and returns `None` when no chunk gold exists.
3. **Ranking rule:** a paper's rank is the first occurrence of that paper in the retrieved list (duplicate chunks collapse, order kept).
   **`k` cuts the raw retrieved list (the retriever's top-k) before de-duplication.**
4. **Reporting policy per example** (`primary_retrieval_metrics`):

   | Gold papers | Metrics reported |
   |---|---|
   | 1 | hit@k (success: is at least one gold document in the top-k?) and MRR (1 ÷ position of the first gold document), plus the hit@1/3/5 curve (`hit_at_ks`) |
   | 2 or more | recall@k (coverage: fraction of the gold papers found) as the main metric, plus hit@k and MRR |
   | 0 | none (nothing to score) |

5. **Out of scope here:** `precision_at_k` (fraction of the top-k that is gold). With a single gold document it is capped at 1/k and
   counts unjudged siblings as errors; its semantics are an open decision (roadmap P6).
6. **Verification:** the four formulas are cross-checked against `ir-measures` (a wrapper of the community-reference `trec_eval`)
   with randomized cases, including the `k` cutoff and paper gold derived from chunk ids (tests/test_evaluators_oracle.py).

## Alternatives considered

### Alternative A: Chunk-level gold only (status quo)

**Why not chosen**:
- Cannot be compared across chunking configurations (`chunk_id` changes), which breaks the planned EXP-3.
- Counts a correct sibling chunk of the right paper as a miss.

**Advantages** (not leveraged):
- Measures whether the exact source passage was retrieved.

### Alternative B: Paper level only, no chunk diagnostic

**Why not chosen**:
- Hides the "right paper, wrong passage" case in the 500/0 configuration, where the generator may lack the supporting text.

**Advantages** (not leveraged):
- Simplest reporting (one level).

### Alternative C: Report both levels combined or averaged

**Why not chosen**:
- Mixes a floor and a ceiling into one number that means neither.

### Alternative D: Complete the chunk gold by human adjudication of sibling chunks

**Why not chosen** (for now):
- Cost (an agent's unverified estimate: ~200–250 judgments) for a small gain, since floor and ceiling nearly coincide at baseline.
- Becomes moot if the baseline index is one record per abstract (ADR-005).

**Advantages** (not leveraged):
- Would raise the chunk-level floor and make chunk-level precision valid.

### Alternative E: Gold as a text span matched by content overlap

**Why not chosen**:
- More complex, and the content-based diagnostics it resembles (RAGChecker) were not validated against humans in the paper.

**Advantages** (not leveraged):
- Survives any re-chunking without relying on ids.

## Consequences

### Accepted

- Retrieval metrics are comparable across chunking configurations and indexing units (ADR-005).
- hit and recall are no longer reported as two columns carrying the same number.
- The metric code is verified against an independent reference (46 tests; 500 randomized cases with `k` and chunk-derived gold, 0 divergences).
- With one record per abstract (ADR-005) the chunk level collapses into the paper level and the diagnostic becomes unnecessary.

### Trade-offs

- The paper-level gold only knows the source paper: **other papers that also answer the question are unjudged**. Retrieval quality is
  a floor with respect to them; only the EXP-0 mini-pooling (human adjudication) can reveal them.
- At baseline `hit@k` is saturated (100%) on `answerable` and `persona`: those slices do not discriminate retrieval changes. The curve
  informs on `multi-doc` (hit@1/3/5 = 0.70/0.70/0.90) and later on `deep-hit`.
- Questions generated from the same chunk that is the gold may be easier than real queries and may favor the index they were generated
  from (lexical overlap hypothesis; no source studies synthetic gold, so this is an extrapolation).
- `precision_at_k` stays under-specified until P6 is closed.

### Conditions that invalidate this decision

This decision should be **revisited** if:

1. The EXP-0 mini-pooling shows many relevant, unjudged papers: then extend `gold_arxiv_ids` with the adjudicated ones.
2. The corpus moves from abstracts to long documents: passage-level gold becomes necessary again.
3. A metric for which the paper level is too coarse is needed (for example, answer-support by passage in the citation evaluator).

### Migration path when needed

1. Add adjudicated papers to `gold_arxiv_ids` (no code change: the evaluators already read it first).
2. If passage-level gold is needed, store the source text span next to the id and match by content overlap (Alternative E).
3. To make the chunk level comparable across configurations, regenerate `gold_chunk_ids` for each configuration from the stored source text.

## References

- Lesson: [Retrieval Unit and Gold Granularity for RAG over Abstracts](../learning-lessons/retrieval_unit_and_gold_granularity.md)
- Roadmap: [`ckpt-0.6.1b-metrics-roadmap.md`](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) (D-1, D-2, L-4, P5, P6)
- ADR-001 (golden set provenance), ADR-003 (hybrid multi-doc slice), ADR-005 (retrieval unit)
- Reading notes: `docs/research/fichamentos/ir-pooling-incomplete-judgments.md` (unjudged = floor; judge the holes), `thakur-2021-beir.md`, `ru-2024-ragchecker.md`, `bassani-2022-ranx.md`
