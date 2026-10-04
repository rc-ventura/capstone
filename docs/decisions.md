# Experiment Decision Log

> One entry per experiment gate. Format: date, experiment, config change, metric before → after (target slice), regressions checked, verdict (PROMOTE / REJECT), rationale.

## Observations (non-gates, findings outside an experiment)

- **2026-09-27 — chunk dedup in the top-k (manual finding).** Running the baseline retriever
  (k=5) on questions from the `stale` slice, the top-5 returned the same paper 3× (e.g., 3 chunks of
  2609.19425v1; 3 of 2609.18471v1). Diversity collapse — top-k dominated by a single paper.
  Verified manually during the golden set review (CKPT-0.5-review). Impact: FP3 (multi-doc
  suffers from a top-k without diversity). Future action: `retrieval_precision_at_k` + a diversity metric
  in EXP-0; possible dedup by `arxiv_id` before cutting to k.

| Date | EXP | Change | Target slice | Metric before → after | Regressions | Verdict |
|------|-----|--------|--------------|-----------------------|-------------|---------|
| | EXP-0 | baseline (k=5, emb-small, chunk 500/0, prompt v1) | all | — | — | BASELINE |
| | EXP-1 | | | | | |
| | EXP-2 | | | | | |
| | EXP-3 | | | | | |
| | EXP-4 | | | | | |
| | EXP-5 | | | | | |
| | EXP-6 | | | | | |
| | EXP-7 | | | | | |
| | EXP-8 | | | | | |
