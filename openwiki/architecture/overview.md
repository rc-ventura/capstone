---
type: system overview
title: System Overview
description: What the arXiv Research Copilot is — a cited RAG Q&A assistant over 250 frozen arXiv cs.AI abstracts built as a vehicle for an evaluation system around Barnett et al.'s seven RAG failure points — and how its modules fit together.
tags: [overview, architecture, rag, evaluation, failure-taxonomy, langsmith]
verified:
  - by: openwiki/0.6.0
    at: 2026-10-11T10:08:11.054Z
sources:
  - id: openwiki-source-17dded95897e01ee430228e3
    resource: repo://app.py
  - id: openwiki-source-0bfdf81f1534614019fe4429
    resource: repo://build_golden_dataset.py
  - id: openwiki-source-bbbb757758dc109f3a4477fc
    resource: repo://config.py
  - id: openwiki-source-5e8cfa6ef22542c9ea8aaf4a
    resource: repo://docs/decisions.md
  - id: openwiki-source-0efdc1e16f1045bb19dad3ef
    resource: repo://docs/plan.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-53688c65a9b4548093faff14
    resource: repo://utils.py
generated: { by: "claude-code", at: "2026-10-11T10:08:11.054Z" }
---

# System Overview

The arXiv Research Copilot is a cited Q&A assistant over **250 arXiv cs.AI abstracts**. The product is deliberately simple — the real deliverable is the **evaluation system around it**: a human-reviewed golden set, code-based and LLM-judge evaluators, and an experiment log that promotes or rejects each configuration change against a recorded baseline.

The organizing taxonomy is **Barnett et al. (2024), "Seven Failure Points When Engineering a Retrieval Augmented Generation System"** (arXiv:2401.05856). The design rule, stated in `docs/plan.md` §9: every failure point gets a measuring evaluator, a golden-set slice, a mitigation experiment with a logged decision, and a production metric. The README maps each of FP1–FP7 to its slice and evaluator; the paper is treated as a *taxonomy for coverage*, not evidence of failure frequency — a caveat recorded in the project's own reading note.

## Module map

| Module | Responsibility |
|---|---|
| `app.py` | The traced RAG pipeline: `rewrite_query` → `retrieve_documents` → `rerank` → `generate_response`, all under `arxiv_copilot`. |
| `config.py` | Environment variables, corpus/golden-set constants, and `RunConfig` — the frozen dataclass modeling one experiment arm. |
| `utils.py` | Corpus acquisition and the index: arXiv fetch, frozen-corpus manifest verification, chunking, parquet vector-store caches, retriever. |
| `evaluators.py` | All evaluator code: retrieval metrics, `format_validator`, abstention detection/reporting, summary evaluators. Intentionally free of `config`/`utils` imports so tests run without API keys. |
| `build_golden_dataset.py` | Builds and idempotently syncs the golden dataset's slices into LangSmith. |
| `main.py` | End-to-end entry point: build the index, run one traced question, sync the golden dataset. |
| `tests/` | Offline suite (no network): evaluator semantics, an `ir-measures` oracle cross-check, manifest verification, chunking invariants. |
| `docs/` | The governance layer: master plan, ADRs, decision log, checkpoint roadmaps, learning lessons, research notes. |

## End-to-end flow

```
question ──► rewrite_query ──► retrieve_documents ──► rerank ──► generate_response ──► cited answer
              (off; CKPT-5)     k=5, whole abstracts   (off; CKPT-4)   gpt-4o-mini, prompt v1
```

1. **Corpus** — `utils.load_frozen_corpus()` rebuilds the same 250 papers from the git-versioned `resources/corpus_manifest.json` (arXiv ids + per-abstract sha1 digests), pinned at snapshot date 2026-09-17. Verified rebuilds fail loudly if an abstract was revised upstream.
2. **Index** — `utils.get_vector_store(run_config)` chunks per the arm's `chunk_strategy` (baseline: one record per whole abstract), embeds with `text-embedding-3-small`, and caches to a per-config parquet file under `resources/`.
3. **Pipeline** — `app.arxiv_copilot` runs the four stages, each producing a LangSmith run; the two disabled stages are reserved extension seams (rerank CKPT-4, rewrite CKPT-5) that raise `NotImplementedError` if toggled.
4. **Evaluation** — the golden dataset `arxiv-copilot-golden` (~100 core examples across six slices) lives in LangSmith; `evaluators.py` scores retrieval (Surface 1: hit/recall/MRR/precision at paper level) and generation (Surface 2: format, abstention, token-F1, with LLM judges planned).
5. **Governance** — each experiment (EXP-0 → EXP-7) changes one config axis, runs against the golden set, and ends in a PROMOTE/REJECT gate logged in `docs/decisions.md`; all gates also require no regression on non-target slices.

## Two measured failure surfaces

Following the plan (§3), retrieval and generation are instrumented and measured **independently, in that order**: retrieval metrics attach to the `retrieve_documents` retriever run; generation metrics attach to `generate_response`. Every run carries the same metadata (`app_version`, `embedding_model`, `k`, `chunk_strategy`, `rerank`, `query_rewrite`, `prompt_version`), so any metric can be sliced by any configuration axis.

## Design invariants worth knowing

- **Frozen corpus**: the manifest (not the git-ignored parquet caches) is the source of truth; a fresh clone reproduces the identical corpus by id.
- **Paper-level gold**: retrieval gold is the paper (`gold_arxiv_ids`, or the paper prefix of a `gold_chunk_id`), per ADR-006 — chunk labels are diagnostic only.
- **Idempotent dataset sync**: `build_golden_dataset` creates only missing slices, skips complete ones, and *raises* on partially populated slices (a sign of manual edits needing human reconciliation).
- **Small-n honesty**: core-set decision gates require ≥15% deltas; abstention metrics report raw counts with Wilson 95% CIs rather than bare proportions.

## Status

Development proceeds checkpoint by checkpoint (`docs/plan.md` §5), each with a written plan before code. Done: foundation, golden set (100 human-reviewed examples), retrieval metrics + format/summary evaluators, and metrics hardening P1–P8. Next: LLM judges, the experiment runner (CKPT-0.7), and the EXP-0 baseline run. A Gradio UI is planned but not implemented.
