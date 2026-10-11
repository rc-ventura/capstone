---
type: quickstart
title: Quickstart
description: Get the arXiv Research Copilot running — requirements, environment keys, uv commands, what main.py does on first run, and a task-routing map to the rest of the wiki.
tags: [quickstart, setup, uv, environment, onboarding]
verified:
  - by: openwiki/0.6.0
    at: 2026-10-11T10:08:11.054Z
sources:
  - id: openwiki-source-2f0bddf9baece30bbb83c764
    resource: repo://docs/manual-usage.md
  - id: openwiki-source-0756aa10ba9e55148108b47d
    resource: repo://example.env
  - id: openwiki-source-833e692518af9eeaf8564cc6
    resource: repo://main.py
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-53688c65a9b4548093faff14
    resource: repo://utils.py
generated: { by: "claude-code", at: "2026-10-11T10:08:11.054Z" }
---

# Quickstart

A cited Q&A assistant over 250 frozen arXiv cs.AI abstracts, wrapped in an evaluation system (golden set, evaluators, experiment gates). This page gets it running; the [System Overview](architecture/overview.md) explains *why* it is built this way.

## Requirements

- **Python 3.12–3.13** and [`uv`](https://docs.astral.sh/uv/) for dependency management
- An **OpenAI API key** (embeddings + `gpt-4o-mini` generation)
- A **LangSmith API key** (traces + golden dataset — use a project you don't mind filling)

## Setup

```bash
uv sync
cp example.env .env   # then fill in OPENAI_API_KEY and LANGSMITH_API_KEY
```

`.env` fields (`config.py` reads them at import; the two API keys are required):

| Variable | Notes |
|---|---|
| `OPENAI_API_KEY` | required |
| `LANGSMITH_API_KEY` | required |
| `LANGSMITH_TRACING` | `"true"` in example.env |
| `LANGSMITH_PROJECT` | defaults to `capstone-arxiv-copilot` |
| `LANGSMITH_ENDPOINT` | commented out — set to `https://eu.api.smith.langchain.com` on the EU instance |

## Verify

```bash
uv run pytest    # offline suite — no API calls; keys just need to be set (config reads them at import)
```

## First run

```bash
uv run python main.py
```

Three things happen:

1. **Corpus → index.** `utils.get_vector_store(config.BASELINE)` opens the baseline parquet cache, or — on a cold cache — rebuilds the frozen corpus **by arXiv id** from the git-versioned `resources/corpus_manifest.json`, verifies each abstract against its recorded sha1 digest (fails loudly on any revision), embeds with `text-embedding-3-small`, and persists the parquet. First build costs a few cents of embeddings.
2. **One traced answer.** `app.arxiv_copilot` runs a built-in question through the pipeline and writes the full trace tree (with run metadata) to your LangSmith project.
3. **Golden sync.** `build_golden_dataset()` syncs `arxiv-copilot-golden` — idempotent: creates only missing slices, skips complete ones, raises on partially populated ones.

## Inspect by hand (no LangSmith writes)

```python
import utils, config
for doc in utils.get_retriever(config.BASELINE).invoke("tool hallucination in LLM agents"):
    print(doc.metadata["arxiv_id"], doc.metadata["published"], doc.metadata["title"][:70])
```

```python
import app, config
print(app.arxiv_copilot("What does TRACE retrieve and why?"))
```

```bash
uv run python build_golden_dataset.py   # standalone dataset sync; a 2nd run is a no-op
```

More snippets (SDK dataset inspection, diagnostic shortcuts): `docs/manual-usage.md`.

## Where to go next

| I want to… | Page |
|---|---|
| Understand what this project is and how the pieces fit | [System Overview](architecture/overview.md) |
| Follow a question through the pipeline stages | [RAG Pipeline](architecture/rag-pipeline.md) |
| Change k, embeddings, chunking, or define a new experiment arm | [Experiment Configuration](architecture/experiment-configuration.md) |
| Understand the frozen corpus, the manifest, and rebuilds | [Frozen Corpus & Manifest](corpus/frozen-corpus.md) |
| Understand indexing, caches, `chunk_id`, and the retriever | [Indexing & Retrieval](corpus/indexing-and-retrieval.md) |
| Work on the golden set's slices or sync behavior | [Golden Dataset](evaluation/golden-dataset.md) |
| Read the retrieval metric semantics (paper level, precision gating) | [Retrieval Metrics](evaluation/retrieval-metrics.md) |
| Work on format/abstention/F1 evaluators | [Answer Evaluators](evaluation/answer-evaluators.md) |
| Understand checkpoints, EXP gates, and the decision log | [Experiments & Decision Gates](evaluation/decision-gates.md) |
| Read traces, metadata, and dataset surfaces in LangSmith | [LangSmith Tracing & Datasets](observability/langsmith-tracing.md) |
| Run or extend the offline tests | [Offline Test Suite](testing/test-suite.md) |
