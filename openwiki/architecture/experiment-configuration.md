---
type: configuration model
title: Experiment Configuration
description: How config.py models one experiment arm as a frozen RunConfig, defines the BASELINE and CHUNKED_500_0 arms, loads environment variables, and turns every configuration field into LangSmith run metadata for metric slicing.
tags: [configuration, runconfig, experiment-arms, environment, metadata, baseline]
verified:
  - by: openwiki/0.6.0
    at: 2026-10-12T14:34:48.400Z
sources:
  - id: openwiki-source-17dded95897e01ee430228e3
    resource: repo://app.py
  - id: openwiki-source-0bfdf81f1534614019fe4429
    resource: repo://build_golden_dataset.py
  - id: openwiki-source-bbbb757758dc109f3a4477fc
    resource: repo://config.py
  - id: openwiki-source-0756aa10ba9e55148108b47d
    resource: repo://example.env
  - id: openwiki-source-53688c65a9b4548093faff14
    resource: repo://utils.py
generated: { by: "claude-code", at: "2026-10-11T10:08:11.054Z" }
---

# Experiment Configuration

`config.py` is the single place where the system's knobs live. It does three jobs: load and validate environment variables at import time, define the corpus and golden-dataset constants, and provide `RunConfig` — a frozen dataclass where **one instance is one arm of one experiment**.

## Environment loading

`load_dotenv(override=True)` runs at import, then `_require_env` reads the two mandatory keys:

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `LANGSMITH_API_KEY` | yes | — |
| `LANGSMITH_PROJECT` | no | `capstone-arxiv-copilot` |
| `CORPUS_SNAPSHOT_DATE` | no | `2026-09-17` |
| `GENERATION_MODEL` | no | `gpt-4o-mini` |
| `RERANK_MODEL` | no | `gpt-4o-mini` |

A missing required key raises `RuntimeError` with an instruction to copy `example.env` — so importing `config` without keys fails immediately. This has a test consequence: `evaluators.py` deliberately avoids importing `config` or `utils` so the offline evaluator tests never need API keys.

`CORPUS_SNAPSHOT_DATE` is the only corpus constant that is environment-overridable; everything else about the corpus (category `cs.AI`, target 250 papers, `CORPUS_SNAPSHOT_SKIP_BUDGET = 2000`) is a module constant. The skip budget is additive: the date-scan fallback asks arXiv for `target + budget` raw results and keeps the ones published on or before the pin, raising a clear error if the budget runs short.

## RunConfig — one arm of one experiment

```python
@dataclass(frozen=True)
class RunConfig:
    app_version: str
    embedding_model: str = "text-embedding-3-small"
    k: int = 5
    chunk_size: int = 0          # 0 = one record per whole abstract
    chunk_overlap: int = 0
    prompt_version: str = "v1"
    query_rewrite: bool = False
    decompose: bool = False
    rerank: bool = False
    rerank_top_n: int = 20
```

Invariants and derived names:

- `__post_init__` rejects `chunk_overlap != 0` when `chunk_size == 0` — overlap is meaningless without splitting.
- `chunk_size == 0` is the sentinel for "no chunking": the retrieval unit is the whole abstract (ADR-005). The `chunk_strategy` property renders it as `"abstract"`, otherwise `"{chunk_size}/{chunk_overlap}"` (e.g. `"500/0"`).
- `rerank_top_n` is the candidate pool fetched before the planned CKPT-4 reranker cuts back to `k`.
- `decompose` has no consumer yet — it is reserved for the CKPT-5 query-decomposition work, like `query_rewrite`.

The string forms feed two different systems: `utils._persist_path` turns `embedding_model` + `chunk_strategy` into the parquet cache filename (`arxiv_text-embedding-3-small_abstract.parquet` vs `..._500-0.parquet`), and `app._run_metadata` copies every field onto every LangSmith run.

## The two defined arms

```python
BASELINE      = RunConfig(app_version="baseline-ckpt0")
CHUNKED_500_0 = RunConfig(app_version="arm-chunked-500-0", chunk_size=500, chunk_overlap=0)
```

- **`BASELINE`** is EXP-0's reference configuration: `text-embedding-3-small`, k=5, whole-abstract records, prompt v1, no rewrite, no rerank. Every later experiment gate compares against it. The retrieval unit was switched from 500-char chunks to whole abstracts *before* EXP-0 ran, so the baseline has always been whole-abstract.
- **`CHUNKED_500_0`** is kept as the experimental arm for the CKPT-3 chunking experiment — and, importantly, it is the unit the golden set's `answerable`/`persona` questions were generated from. `build_golden_dataset` pins this config when it loads cached chunks so its stored `gold_chunk_ids` keep resolving against the same cache.

## Config fields as run metadata

Every `RunConfig` field doubles as a LangSmith metadata axis. `app._run_metadata` emits `app_version`, `embedding_model`, `k`, `chunk_strategy`, `rerank`, `query_rewrite`, and `prompt_version`, and `arxiv_copilot` attaches that dict to every stage's run. That is what lets any reported metric be sliced by any configuration axis in LangSmith — the configuration model and the observability model are deliberately the same object.

## Where this configuration is consumed

- `utils.get_vector_store(run_config)` — selects/creates the cache and embedding model; `get_retriever` wires `k`.
- `app.arxiv_copilot(question, run_config=config.BASELINE)` — the pipeline default; each stage reads its own toggles.
- `build_golden_dataset` — reads `CHUNKED_500_0` for chunk sampling, plus the golden-set constants (`GOLDEN_DATASET_NAME`, `CORE_SPLIT`, `GOLDEN_SLICE_TARGETS`).
