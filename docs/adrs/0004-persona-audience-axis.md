# ADR-004 — Persona audience axis: lay vs phd (stark contrast)

- **Status:** Accepted
- **Date:** 2026-09-27 (CKPT-0.5.3a)
- **Deciders:** product owner + Devin

## Context

The `persona` slice (docs/plan.md §2) originally used a **pm vs phd** audience pair.
During human review (R3) the pair produced too little calibration gap: both profiles
are technical-adjacent, so the two gold answers were nearly identical for most bases
(e.g. spatial-reasoning pair differed only by expanding the STEM acronym). A slice
whose twin golds don't differ cannot exercise FP6 (specificity calibration).

## Decision

Change the audience axis to a **stark contrast**:

- `lay` = non-technical stakeholder — zero jargon, plain language + an everyday
  analogy, focus on impact/why-it-matters, no math;
- `phd` = domain researcher — method/mechanism, technical terms, caveats,
  quantitative detail.

Additional generation constraints adopted in the same decision: gold answers must be
**English** (a language guard caught Portuguese leakage from the model) and every
sentence must be **directly supported by the source chunk** (a lay gold had
hallucinated a generalization claim).

## Consequences

- **Positive:** the calibration gap is now large and measurable; the slice genuinely
  tests whether the model collapses jargon to zero (lay) vs densifies mechanism
  (phd).
- **Negative:** `docs/plan.md` §1 (Feature C) and §2 (persona row) had to be updated
  from pm/phd to lay/phd; all 10 persona golds were regenerated.
