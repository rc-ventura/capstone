# ADR-001 — Golden set provenance: two-axis data model

- **Status:** Accepted
- **Date:** 2026-09-27
- **Deciders:** product owner + Devin (review session CKPT-0.5-review)

## Context

The golden set (docs/plan.md §2) initially carried a single `provenance` field
(`synthetic` / `hand-written` / `derived`). During CKPT-0.5.3 it became clear that
`hand-written` was false: the 50 "hand-written" examples were actually drafted by an
LLM agent and approved at batch level — never reviewed item-by-item. A single field
also mixed two independent questions (*who wrote it* vs *who verified it*), which
admitted impossible states such as "curated but never reviewed".

## Decision

Replace the single `provenance` field with **two orthogonal axes** on every example's
metadata:

- `authoring.origin` ∈ `llm | human | llm+human` — who wrote the text (`llm+human` =
  human edited the draft);
- `authoring.method` ∈ `from-chunk | topic-authored | empirical` — how it was built
  (synthetic from a chunk / authored around a topic / found during EXP-0);
- `review.state` ∈ `draft | spot_checked | human_reviewed | adjudicated` — depth of
  human verification;
- `review.reviewer` — who reviewed (required whenever state is reviewed/adjudicated).

Consistency rule (enforced by a validation script): `human_reviewed`/`adjudicated`
require `reviewer`; `origin=llm+human` requires at least one human-edited field.

## Consequences

- **Positive:** provenance no longer lies; human anchoring is tracked where it
  actually happens (review gates, mini-pooling adjudication), not in authorship.
- **Positive:** downstream analyses can filter/slice by true review depth instead of
  assuming quality from a label.
- **Negative:** one migration (100 examples) and a rename away from a label
  ("hand-written"/"curated") that was already referenced in prose. Old
  `provenance` values are removed, not aliased, to avoid two sources of truth.

See: `docs/roadmap/ckpt-0.5-review-plan.md` §3.1, `results/ckpt-0.5-review.md`.
