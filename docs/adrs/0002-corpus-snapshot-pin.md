# ADR-002 — Corpus snapshot pin (frozen-at-date) + additive skip budget

- **Status:** Accepted
- **Date:** 2026-09-24 (CKPT-0.5.1)
- **Deciders:** product owner + Devin

## Context

`fetch_arxiv_corpus` always fetched the 250 most-recent cs.AI papers. The `stale`
golden slice (docs/plan.md §2) asks questions about papers published *after* the
corpus; any later cache rebuild (e.g. CKPT-2 changes the embedding model) would
silently pull newer papers and kill the slice's meaning mid-experiment.

## Decision

Freeze the corpus at `CORPUS_SNAPSHOT_DATE = 2026-09-17` (the baseline cache's max
published date). `fetch_arxiv_corpus` scans `target + CORPUS_SNAPSHOT_SKIP_BUDGET`
results and drops anything published after the pin, hard-failing with a clear message
if the budget runs short. Skip budget is **additive** (2000), not multiplicative.

## Consequences

- **Positive:** the `stale` slice keeps meaning "papers after the corpus" across all
  experiment cache rebuilds; the CKPT-8 refresh drill becomes an executable date bump
  + re-index instead of waiting on real time.
- **Positive:** no rebuild needed — the existing baseline cache's max date already
  equals the pin.
- **Negative:** the corpus stops tracking the newest arXiv papers by design; the pin
  must be bumped consciously (never automatically) at CKPT-8. The skip budget decays
  (~128 papers/day accumulate after the pin) and must be raised if the hard-fail
  guard trips.

Evidence: on 2026-09-24 there were 766 cs.AI papers published after the pin; the
first eligible paper sat at position 766 in the newest-first listing.

## Update 2026-10-04

Rebuilding the frozen corpus by scanning arXiv by date stopped working: the additive skip budget was exhausted ~10 days after it was
sized (~128 newer papers per day). The corpus is now also pinned by `resources/corpus_manifest.json` (ids + abstract digests,
versioned in git) and rebuilt by id; see ADR-005 and `docs/decisions.md` (Observations, 2026-10-04). The date pin keeps its meaning
for the `stale` slice and for a cold start.
