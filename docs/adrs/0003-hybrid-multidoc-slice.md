# ADR-003 — Hybrid `multi-doc` slice (10 known-item + 5 open-topic)

- **Status:** Accepted
- **Date:** 2026-09-27 (CKPT-0.5.2c)
- **Deciders:** product owner + Devin

## Context

The `multi-doc` slice (docs/plan.md §2) requires joint synthesis over 2–3 papers.
Two query archetypes exist: **known-item** (user names papers — "compare A and B",
which is literally product Features B/D) and **open-topic** (user asks a topical
question with no IDs — the dominant real-production shape). A pure known-item slice
is unrepresentative; a pure open-topic slice is unmeasurable for retrieval without
annotation.

## Decision

`multi-doc` = **10 known-item** (questions name the required paper IDs; mechanical
retrieval gold = `gold_arxiv_ids`) + **5 open-topic** (no IDs in the question;
`gold_arxiv_ids` intentionally empty, flagged `retrieval_gold=pending-adjudication`).
The open-topic retrieval gold is filled by human adjudication during the EXP-0
mini-pooling (TREC-style). Both kinds keep a synthesis gold answer (reference for
`completeness_judge`).

## Consequences

- **Positive:** the slice represents both real query modes; open-topic examples give
  CKPT-5 (query decomposition) a realistic target; adjudication makes the retrieval
  labels human-verified instead of assumed.
- **Negative:** until EXP-0 adjudication, the 5 open-topic examples have no
  computable retrieval metric — the baseline report must mark them as pending.

Rationale grounded in: TREC pooling (incomplete judgments), MultiHop-RAG
(construction-derived evidence), and the L1/L2 provenance layers documented in
`docs/learning-lessons/golden_dataset_construction_and_human_calibration.md`.
