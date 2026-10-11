# ADR-009: Incomplete gold and pool bias — judge the holes (mini-pooling), never treat "unjudged" as "irrelevant"

**Status**: Accepted (decisions D-8 to D-11 are in force; D-3 and the adjudication data model are still open, see "Open items")
**Date**: 2026-10-11
**Related**: [Roadmap §6 P6, decisions D-3, D-8, D-9, D-10, D-11](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) · [Author note 2026-10-04](../notes/2026-10-04-gold-incompleto-e-vies-do-pool.md) · Reading notes: [IR pooling and incomplete judgments](../research/fichamentos/ir-pooling-incomplete-judgments.md) · [BEIR (Thakur 2021)](../research/fichamentos/thakur-2021-beir.md) · [ADR-001](./0001-golden-set-provenance-data-model.md) (review states) · [ADR-003](./0003-hybrid-multidoc-slice.md) (known-item vs. open-topic) · [ADR-005](./0005-retrieval-unit-whole-abstract.md) · [ADR-006](./0006-retrieval-gold-granularity-and-metrics.md) (paper-level gold; this record is its follow-up)
**Code**: `evaluators.py` (`is_gold_complete`, `precision_at_k`), committed in `5faf69e` + `a2b7e30`. **Not implemented yet**: the mini-pooling itself (runs at CKPT-0.8), `Hole@k`, and the storage of papers judged irrelevant.

---

> **This record is written for any reader, not only engineers.** The short version: our "answer key" for the search step lists one correct paper per question, but other papers may also answer it and nobody checked. Some of our scores punish the system for finding those other papers. This record explains how we stop that, how a small human review (the "mini-pooling") will measure how bad the problem is, and a limit of that review that we cannot remove.

## Context

### How the search step is scored, in plain words

Each golden question has an answer key for retrieval, called the **gold**: the arXiv paper(s) that should come back. For `answerable` and `persona` questions the gold is **the single paper the question was generated from** (ADR-006). The retriever returns its top 5 papers, and metrics compare them with the gold: **hit@k** (was the gold in the top k?), **MRR** (how high was it ranked?), **recall@k** (what share of the gold papers came back?) and **precision@k** (what share of the top k is in the gold?).

### Problem 1 — "not in the gold" is read as "wrong", but it only means "nobody looked"

Another paper in the corpus may answer the same question just as well. The gold does not list it, because it was never judged. This is the **incomplete judgments** problem studied in information retrieval (Buckley & Voorhees 2004; Buckley et al. 2007; Büttcher et al. 2007).

`precision_at_k` is the metric that breaks. With a 1-paper gold and k = 5 it cannot exceed 1/5 = 0.2, even for a perfect retriever. If a fraction *x* of the other four papers does answer the question, the true ceiling is (1 + 4*x*) / 5 (own arithmetic):

| Share of the other 4 papers that also answer (*x*) | True precision@5 ceiling |
|---|---|
| 0% | 0.20 |
| 10% | 0.28 |
| 25% | 0.40 |
| 50% | 0.60 |

hit@k, MRR and recall@k never penalize extra papers, so they were already safe. Treating unjudged as irrelevant gives a **lower bound** (conservative); the opposite shortcut, **ignoring** unjudged papers (bpref, condensed lists), is **not neutral**: it assumes the unjudged papers are relevant in the same proportion as the judged ones, which Büttcher et al. call "simply wrong", and bpref also needs papers *judged* irrelevant, which our gold does not store (it keeps positives only).

### Problem 2 — the human review itself inherits the baseline's blind spot (pool bias)

The planned fix is a human review of the baseline's top-5 for each question ("pooling": judge the candidates that systems return). But the candidates come **only from the baseline retriever**. A paper that only a different retriever would find is never shown to the reviewer, and never enters the gold, whereas every paper the baseline did return gets judged. After the review the gold is still a judgment made *on top of the baseline*, so it favors the baseline.

Evidence from the literature (BEIR, Table 4, after judging the unjudged results): nDCG@10 of a dense retriever (ANCE) rose from 0.654 to 0.735, while BM25 rose only from 0.656 to 0.668. A dense retriever looked tied with BM25 until its unjudged results were read. This is a different dataset and setup; we use it as a warning, not as a prediction for our corpus.

### Three facts that bound the problem (2026-10-04)

- **A — the stakes are limited today.** Baseline hit@5 is 100% on `answerable` (40/40) and `persona` (10/10). An incomplete gold cannot hurt hit@5 there; it can only distort the position metrics hit@1 (0.90) and MRR (0.934), for example when another valid paper ranks above the gold. Why hit@5 is 100% is an untested hypothesis (questions generated from the text share its vocabulary, ADR-006).
- **B — a structural limit.** Only the source of the candidates can reduce pool bias, not the amount of review. Candidates from a second retriever type (for example BM25, the lexical one) would do it.
- **C — a sequencing constraint.** A judgment says "this candidate answers *this* question". If `answerable`/`persona` are regenerated (the open contingency in the roadmap, D-7), every judgment is void. So judge a small sample before judging everything.

## Decision

1. **Unjudged is never irrelevant (D-8, implemented).** `precision_at_k` is reported only where the gold is **complete**, and returns `None` otherwise. The gold is complete when the example is a multi-doc **known-item** (relevant = the papers named in the question, by definition) **or** `review.state == "adjudicated"` (ADR-001). The flag is **derived** from existing metadata, not stored as a new field.

   ```python
   # evaluators.py
   is_gold_complete(reference_outputs, metadata)   # known-item multi-doc, or review.state == "adjudicated"
   precision_at_k(..., metadata=None)              # None unless the gold is complete; no metadata -> None
   ```

2. **Judge the holes with a pilot mini-pooling (D-9).** At CKPT-0.8, a human judges the candidates (baseline top-5 minus the gold; all 5 for open-topic questions): the 5 open-topic questions, plus 10 random `answerable`/`persona` (stratified 8 + 2, fixed seed), plus up to 4 `answerable` cases where the gold was not rank 1. About **80 judgments**, not all 50 single-gold examples (about 225). Known-item multi-doc is not pooled. Papers judged relevant go into `gold_arxiv_ids`, and the example becomes `adjudicated`.

3. **Stop rule fixed before seeing the data (D-11).** *x* = judged-relevant ÷ judged candidates in the 10-random sample:

   | *x* | Action |
   |---|---|
   | ≤ 10% | keep the single gold |
   | ≥ 25% | judge the remaining 40 examples (pilot judgments are reused) |
   | 10–25% | review the cases and decide |

   These numbers are engineering judgment; **no source gives them**. *x* = 25% means one extra answering paper per question on average.

4. **Measure the size of the blind spot with `Hole@k` (D-10).** `Hole@k` is the share of a retriever's top-k that nobody judged (defined in BEIR). It does not reveal the bias, which only appears when those papers are judged, but it tells how large the uncertainty in a score is, and it selects which papers to judge next. It is built **together with the mini-pooling** and must exist before the first retriever comparison (CKPT-1). For this the dataset must also store papers judged **irrelevant** (today it keeps positives only).

5. **Defer `distinct_papers@k` to EXP-3 (D-10).** It is always 5.00 on the whole-abstract baseline (ADR-005) and no paper defines it; it is our own diagnostic.

## Alternatives considered

### Alternative A: Keep `precision_at_k` as is (unjudged = irrelevant)

**Why not chosen**:
- With a 1-paper gold it is capped at 0.2 and would read a better retriever as worse whenever it finds other valid papers.

**Advantages** (not leveraged):
- No extra human work; the number is a conservative lower bound.

### Alternative B: Ignore unjudged papers (bpref-style tolerance, condensed lists)

**Why not chosen**:
- Not neutral: it overestimates systems outside the pool (Büttcher et al. 2007), and bpref needs papers judged irrelevant, which the gold does not store.

### Alternative C: Judge all 50 single-gold examples now (about 225 judgments)

**Why not chosen**:
- Fact C: regenerating the questions would void the judgments. The literature suggests a few hundred judgments are feasible, so the argument is the **order of work**, not feasibility. If the pilot rule says "judge the rest", the pilot judgments are reused.

### Alternative D: A stored `gold_complete` field

**Why not chosen**:
- It would create a second source of truth next to `review.state` (ADR-001), and the dataset sync only creates missing examples, so a new field would not reach the 100 existing ones without a remote update.

## Consequences

### Accepted

- `precision_at_k` can no longer punish a retriever for finding valid papers the gold does not list (62 offline tests green, including: `None` for a 1-paper gold, `None` without metadata, defined for known-item and adjudicated).
- The size of the incompleteness is measured on a small, pre-registered sample before any larger spend of human time.
- The decision rule is fixed before the data, so the pilot result cannot be read to fit a preferred outcome.

### Trade-offs

- **`precision_at_k` is empty where it matters most.** On the 50 single-gold examples it is `None` until they are adjudicated.
- **The pilot interval is wide.** With 40 judgments, 4 relevant of 40 (10%) has a Wilson 95% interval of about 4%–23% (own calculation), and the candidates of one question are not independent. The 10–25% band exists because of this.
- **The pool bias remains** after the pilot unless the candidate source changes (open item D-3, or judging each new retriever's holes at CKPT-1).
- **`Hole@k` costs reading time per retriever.** BEIR measured 6.4% (BM25) to 31.8% (TAS-B) of the top-10 without judgment; applied to our top-5 that is 0.3–1.6 papers per question, about 15–80 judgments per 50 questions per retriever (own arithmetic; different dataset, may not transfer).
- **One reviewer.** The protocol has the author reading each candidate; there is no second rater, so no agreement figure.

### Open items

- **D-3 (not decided):** run BM25 in parallel with every new retriever, and/or add BM25's top-5 to the pool, to detect lexical bias of the synthetic gold. To be decided **after the pilot**: if almost no baseline candidate answers, that suggests (inference, not measured) these questions have few relevant papers beyond the gold, which lowers the weight of the pool bias. No second retriever exists until CKPT-1.
- **Adjudication data model:** how judged-relevant and judged-irrelevant ids are stored in the dataset is not yet defined. The roadmap planned to record it with the pilot; this ADR should be amended then.

### Conditions that invalidate this decision

This decision should be **revisited** if:

1. The pilot gives *x* ≥ 25%: the single-paper gold is too incomplete, and the remaining 40 examples are judged.
2. `answerable`/`persona` are regenerated from the whole abstract (D-7 contingency): all judgments are void and the pilot is repeated on the new questions.
3. A second retriever type is added (CKPT-1): its holes must be judged, and D-3 is decided.
4. The corpus moves from abstracts to long documents (as in ADR-006).

### Migration path when needed

1. Add adjudicated papers to `gold_arxiv_ids` (no code change: the evaluators already read it first).
2. Store judged-irrelevant ids and compute `Hole@k` from them.
3. If D-3 is adopted, add BM25's top-5 to the candidate pool and re-adjudicate the affected examples.

## References

- Buckley, C. & Voorhees, E. M. (2004). *Retrieval Evaluation with Incomplete Information.* SIGIR '04.
- Buckley, C., Dimmick, D., Soboroff, I. & Voorhees, E. (2007). *Bias and the Limits of Pooling for Large Collections.* NIST report.
- Büttcher, S., Clarke, C. L. A., Yeung, P. C. K. & Soboroff, I. (2007). *Reliable Information Retrieval Evaluation with Incomplete and Biased Judgements.* SIGIR '07.
- Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) (Hole@k; Table 4).
- Cited as read through the project's reading notes (`ir-pooling-incomplete-judgments.md`, `thakur-2021-beir.md`), not re-read for this record.
