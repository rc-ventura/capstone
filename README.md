# Capstone: arXiv Research Copilot — RAG Failure-Mode Battleground

A cited Q&A assistant over ~250 arXiv cs.AI papers, engineered so that every
documented RAG failure point can be **detected** (golden set + evaluators),
**measured** (monitoring), and **mitigated** (experiments).

- **User**: engineers/researchers who need fast, trustworthy answers about recent AI research.
- **Job-to-be-done**: *"Answer my questions about recent AI papers with real citations — and tell me when you don't know."*

## Docs

| File | Purpose |
|---|---|
| [`docs/plan.md`](./docs/plan.md) | Full product + experiment plan, organized as checkpoints (CKPT-0 → CKPT-8) with decision gates |
| [`docs/rag_failure_modes_review.md`](./docs/rag_failure_modes_review.md) | Bibliographic review — the research foundation the plan is built on |
| [`docs/decisions.md`](./docs/decisions.md) | Experiment decision log (one row per gate: metric before/after, verdict) |

## Setup

```bash
uv sync
cp example.env .env   # then fill in OPENAI_API_KEY and LANGSMITH_API_KEY
```

## Status

Building checkpoint by checkpoint per `docs/plan.md` §5. Progress tracked via commits
on this repo and rows in `docs/decisions.md`.
