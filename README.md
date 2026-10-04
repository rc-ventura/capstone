# arXiv Research Copilot — RAG Failure-Mode Battleground

A cited Q&A assistant over **250 arXiv cs.AI abstracts**, engineered so that every documented RAG
failure point can be **detected** (golden set + evaluators), **measured** (tracing and monitoring) and
**mitigated** (one experiment per fix, each with a decision gate).

- **User:** engineers and researchers who need fast, trustworthy answers about recent AI research.
- **Job to be done:** *"Answer my questions about recent AI papers with real citations — and tell me when you don't know."*

The failure taxonomy comes from **Barnett et al. (2024), "Seven Failure Points When Engineering a Retrieval Augmented
Generation System"** (see [Research foundation](#research-foundation)).

The product is the vehicle; the real deliverable is the **evaluation system around it**: a human-reviewed golden
set, code-based and LLM-judge evaluators, and a log of experiments (`docs/decisions.md`) that promote or reject
each change against a recorded baseline.

## What it does

| Feature | Behavior |
|---|---|
| **Cited Q&A** | Answers from the retrieved abstracts and cites the paper (`[arxiv_id]`) for every claim |
| **Honest abstention** | If the answer is not in the retrieved context it says it does not know (a release-blocking metric) |
| **Multi-turn threads** | Follow-up questions keep the conversation; every run carries a `thread_id` in LangSmith |
| **Structured / multi-part answers** | Tables, JSON, CSV, comparisons of several papers (exercised by the golden-set slices) |

## How it works

```
question ──► rewrite_query ──► retrieve_documents ──► rerank ──► generate_response ──► cited answer
              (CKPT-5, off)      k=5, one record        (CKPT-4, off)   gpt-4o-mini
                                 per whole abstract                     prompt v1
```

Every stage is traced in LangSmith (`@traceable`, `wrap_openai`, `trace`) and every run carries the same metadata
(`app_version`, `embedding_model`, `k`, `chunk_strategy`, `rerank`, `query_rewrite`, `prompt_version`), so any metric
can be sliced by any configuration axis.

**Baseline configuration** (`config.BASELINE`, the reference every later experiment is compared against):

| Setting | Value |
|---|---|
| Corpus | 250 cs.AI abstracts, frozen at the snapshot date 2026-09-17 |
| Retrieval unit | **one index record per whole abstract** ([ADR-005](./docs/adrs/0005-retrieval-unit-whole-abstract.md)) |
| Embeddings / retriever | `text-embedding-3-small`, top-k = 5 |
| Generation | `gpt-4o-mini`, temperature 0, prompt `v1` |
| Rewrite / rerank | off |

The 500-character chunking (`config.CHUNKED_500_0`) is kept as an **experimental arm** of the chunking experiment (CKPT-3).

## Evaluation approach

| Piece | What it is |
|---|---|
| **Golden set** | 100 examples in the LangSmith dataset `arxiv-copilot-golden`, **all human-reviewed**: `answerable` 40 · `unanswerable` 15 · `multi-doc` 15 · `format` 10 · `persona` 10 · `stale` 10 (a `deep-hit` slice is found empirically in EXP-0) |
| **Retrieval evaluators** | hit@k, recall@k, MRR, precision@k, scored at **paper level** ([ADR-006](./docs/adrs/0006-retrieval-gold-granularity-and-metrics.md)); cross-checked against a `trec_eval`-based oracle (`ir-measures`) |
| **Generation evaluators** | `format_validator` (code) and a summary evaluator are done; the LLM judges (faithfulness, citation accuracy, completeness, abstention, specificity, relevance) are next |
| **Experiments** | EXP-0 (baseline) → EXP-7 (final A/B), one per fix, each ending in a PROMOTE / REJECT verdict |

## Status

Work proceeds checkpoint by checkpoint (`docs/plan.md` §5); each checkpoint has a written plan before code.

| Step | State |
|---|---|
| Foundation: corpus, traced pipeline, baseline configuration | ✅ done |
| CKPT-0.5 — golden set (100 examples, human review 100/100) | ✅ done |
| CKPT-0.6 — evaluators: retrieval metrics and format/summary evaluators | ✅ done |
| CKPT-0.6 — LLM judges | ⬜ next |
| Metrics hardening ([roadmap](./docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md)): level-aware retrieval metrics, paper-level gold, whole-abstract baseline | ✅ P1–P5 done · P6–P11 under discussion |
| CKPT-0.7 — experiment runner · CKPT-0.8 — EXP-0 baseline run | ⬜ planned |
| CKPT-1 → CKPT-8 — experiments (k, embeddings, chunking, reranker, query rewriting, prompts, A/B) and production monitoring | ⬜ planned |
| Gradio web UI ([plan](./docs/ui/ui-plan.md)) | ⬜ planned |

## Quick start

**Requirements:** Python 3.12–3.13, [`uv`](https://docs.astral.sh/uv/), an OpenAI API key and a LangSmith API key.

```bash
uv sync
cp example.env .env        # then fill in OPENAI_API_KEY and LANGSMITH_API_KEY
uv run pytest              # offline suite: no API calls (the keys only need to be set, config reads them at import)
```

Run the pipeline end to end:

```bash
uv run python main.py
```

On the first run this (1) rebuilds the frozen corpus **by arXiv id** from `resources/corpus_manifest.json`
(verifying every abstract against its recorded digest) and embeds it, a few cents of OpenAI embeddings;
(2) answers one built-in question with a traced run; (3) syncs the golden dataset in LangSmith (idempotent: it only creates
missing slices). It writes traces and the dataset to your LangSmith project, so use a project you do not mind filling.

Inspect the retriever by hand (no LangSmith writes, see [`docs/manual-usage.md`](./docs/manual-usage.md)):

```python
import utils, config
for doc in utils.get_retriever(config.BASELINE).invoke("tool hallucination in LLM agents"):
    print(doc.metadata["arxiv_id"], doc.metadata["published"], doc.metadata["title"][:70])
```

## Project layout

```
app.py                     traced RAG pipeline (rewrite → retrieve → rerank → generate)
config.py                  environment, paths, models and RunConfig (BASELINE, CHUNKED_500_0)
utils.py                   corpus loading (manifest by id), indexing, retriever
evaluators.py              retrieval metrics, format validator, summary evaluator
build_golden_dataset.py    builds and syncs the golden dataset (idempotent)
main.py                    end-to-end entry point
resources/                 corpus_manifest.json (versioned) + parquet caches (git-ignored)
experiments/               experiment runners (planned)
tests/                     offline tests: evaluators, oracle cross-check, chunking strategy, corpus manifest
results/                   per-checkpoint result reports
docs/                      plan, ADRs, roadmap, decision log, lessons, research notes
```

**Reproducibility.** The corpus is frozen by `resources/corpus_manifest.json` (250 arXiv ids plus a digest of each
abstract, versioned in git). A fresh clone rebuilds the same 250 papers by id and fails loudly if any abstract was revised.
The parquet caches in `resources/` are git-ignored and can be rebuilt.

## Research foundation

The project is built on the seven failure points catalogued in:

> Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., Abdelrazek, M. (2024). **Seven Failure Points When Engineering a
> Retrieval Augmented Generation System.** *3rd International Conference on AI Engineering — Software Engineering for AI
> (CAIN 2024).* [arXiv:2401.05856](https://arxiv.org/abs/2401.05856)

| FP | Failure point (Barnett et al.) | How the project covers it |
|---|---|---|
| FP1 | Missing content: the answer is not in the corpus | `unanswerable` and `stale` slices, abstention evaluation |
| FP2 | Missed the top-ranked documents | Retrieval metrics; experiments on k, embeddings and reranking |
| FP3 | Not in context: consolidation limits | Context-size and k experiments |
| FP4 | Not extracted: the answer is in the context but the LLM misses it | Generation evaluators on the `answerable` slice |
| FP5 | Wrong format | `format` slice + `format_validator` |
| FP6 | Incorrect specificity | `persona` slice (lay vs. phd) |
| FP7 | Incomplete answers | `multi-doc` slice + completeness judge |

The rule that drives the plan: **every failure point gets a measuring evaluator, a golden-set slice, a mitigation experiment
with a logged decision, and a production metric** (`docs/plan.md` §9, "definition of done").

[`docs/research/rag_failure_modes_review.md`](./docs/research/rag_failure_modes_review.md) is the bibliographic review the plan
is built on: it adds practitioner and recent-taxonomy sources (clawRxiv, TrustNLP, Respan, Redis, Suthar) and the extended
retrieval failure modes. The [15 reading notes](./docs/research/fichamentos/) cover the papers behind the evaluation design.

Two caveats from our own reading of the paper ([note](./docs/research/fichamentos/barnett-2024-seven-failure-points.md)):
it is an **experience report** (three case studies, no per-failure-point counts), so we use it as a **taxonomy for coverage**,
not as evidence of how often each failure happens; and a few of our metric-to-failure-point mappings differ from the paper's
definitions (for example FP3 and FP4). The correction is tracked in the
[metrics roadmap](./docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md) (C-3) and is not yet applied to `docs/plan.md`.

## Key decisions

| ADR | Decision |
|---|---|
| [ADR-001](./docs/adrs/0001-golden-set-provenance-data-model.md) | Golden-set provenance as two axes: who wrote it × how deeply it was reviewed |
| [ADR-002](./docs/adrs/0002-corpus-snapshot-pin.md) | Corpus frozen at a snapshot date, so the `stale` slice keeps its meaning |
| [ADR-003](./docs/adrs/0003-hybrid-multidoc-slice.md) | `multi-doc` slice: 10 known-item + 5 open-topic questions |
| [ADR-004](./docs/adrs/0004-persona-audience-axis.md) | Persona axis: lay vs. phd |
| [ADR-005](./docs/adrs/0005-retrieval-unit-whole-abstract.md) | Baseline retrieval unit: one record per whole abstract |
| [ADR-006](./docs/adrs/0006-retrieval-gold-granularity-and-metrics.md) | Retrieval gold is the paper; metric reporting policy |

## Documentation

Start at the [docs index](./docs/README.md). Highlights: the [master plan](./docs/plan.md) (product, golden set,
evaluator catalog, checkpoints), the [metrics roadmap](./docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md) (living document),
the [decision log](./docs/decisions.md), the [learning lessons](./docs/learning-lessons/), and the
[reading notes](./docs/research/fichamentos/) on the 15 papers behind the evaluation design.

## Conventions

- A checkpoint gets a written plan in `docs/roadmap/` **before** its first line of code, approved before each batch.
- Architectural decisions become an ADR in `docs/adrs/`; every experiment gate adds a row to `docs/decisions.md`.
- Relevant technical discoveries become a learning lesson (indexed in `CLAUDE.md`).
- Project documentation is written in English.
