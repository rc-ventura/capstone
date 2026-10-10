# Experiment Decision Log

> One entry per experiment gate. Format: date, experiment, config change, metric before → after (target slice), regressions checked, verdict (PROMOTE / REJECT), rationale.

## Observations (non-gates, findings outside an experiment)

- **2026-10-06 — abstention metrics redesign (P7/P8, D-5/D-6; implemented 2026-10-06).** Two rules enter the
  gate's fine print:
  1. **Regression rule:** *a gain in `abstention_recall` cannot come from an increase in `over_refusal_rate`.*
  RefusalBench shows the two errors trade off (r = −0.78; GPT-4o refused 14.6× more than needed), so a system can
  pass the ≥ 90% gate by refusing indiscriminately. The `over_refusal_rate` guardrail is now reported alongside
  the gate (total and per slice, raw counts + Wilson 95% CI); its numeric tolerance is fixed at EXP-0 against the
  baseline's CI — no baseline exists yet to tolerate against.
  2. **Detection rule:** abstention is detected by `abstained()` (normalization + two marker families), a cheap
  floor — not the truth. The `abstention_quality` judge (0.6.3, T=0, answer-only, validated on ~50 real
  stratified answers) is the ground truth, and EXP-0 reports the heuristic × judge divergence (target ≥ 90%).
  `abstention_accuracy` no longer exists (it mixed under- and over-refusal into one number a refuse-everything
  system could pass); `f1_summary_evaluator` now reports token-F1 only.
- **2026-10-04 — rebuilding the frozen corpus by date scan no longer works; replaced by a manifest.**
  Building the whole-abstract index (ADR-005) with `fetch_arxiv_corpus()` failed: `Only 0 of 250 papers fell
  on/before the snapshot date 2026-09-17`. The snapshot pin has to skip every newer paper (the budget was sized on
  2026-09-24: ~766 newer papers, ~128 more per day) and 10 days later all 2,250 scanned results are newer. Fix:
  `resources/corpus_manifest.json` (**versioned in git**, 250 ids + the sha1 of each whitespace-normalized abstract).
  `utils.load_frozen_corpus()` now rebuilds the corpus **by arXiv id** (250 papers in ~8 s) and fails loudly if an id is
  missing or an abstract was revised after the freeze. Verified: all 250 ids return and all 250 digests match. Fallbacks:
  an existing parquet cache (offline), and the date scan only on a cold start (it also writes the manifest). The git-ignored
  parquet caches are therefore no longer the only source of the corpus.
- **2026-10-04 — whitespace in abstracts.** 10 of 250 abstracts contain line breaks. The first whole-abstract index was derived
  from the 500/0 chunks and had turned those breaks into spaces; the index was rebuilt from the original texts (by id) and the
  retrieval metrics did not change (`answerable` hit@1 0.90, MRR 0.934; `multi-doc` recall@5 0.617).

- **2026-09-27 — chunk dedup in the top-k (manual finding).** Running the baseline retriever
  (k=5) on questions from the `stale` slice, the top-5 returned the same paper 3× (e.g., 3 chunks of
  2609.19425v1; 3 of 2609.18471v1). Diversity collapse — top-k dominated by a single paper.
  Verified manually during the golden set review (CKPT-0.5-review). Impact: FP3 (multi-doc
  suffers from a top-k without diversity). Future action: `retrieval_precision_at_k` + a diversity metric
  in EXP-0; possible dedup by `arxiv_id` before cutting to k.

| Date | EXP | Change | Target slice | Metric before → after | Regressions | Verdict |
|------|-----|--------|--------------|-----------------------|-------------|---------|
| | EXP-0 | baseline (k=5, emb-small, one record per whole abstract (ADR-005), prompt v1) | all | — | — | BASELINE |
| | EXP-1 | | | | | |
| | EXP-2 | | | | | |
| | EXP-3 | | | | | |
| | EXP-4 | | | | | |
| | EXP-5 | | | | | |
| | EXP-6 | | | | | |
| | EXP-7 | | | | | |
| | EXP-8 | | | | | |
