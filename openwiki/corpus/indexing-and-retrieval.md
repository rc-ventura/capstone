---
type: indexing subsystem
title: Indexing & Retrieval
description: How the corpus becomes a queryable index in utils.py — the whole-abstract vs 500-char retrieval units, deterministic chunk_id identity, per-config parquet caches backed by SKLearnVectorStore, and the retriever consumed by the pipeline.
tags: [indexing, retrieval, vectorstore, embeddings, chunking, cache]
sources:
  - id: openwiki-source-0bfdf81f1534614019fe4429
    resource: repo://build_golden_dataset.py
  - id: openwiki-source-d4f75835e0b32a78c2a186e2
    resource: repo://docs/adrs/0005-retrieval-unit-whole-abstract.md
  - id: openwiki-source-fca896e236afd96cf1228c0f
    resource: repo://tests/test_chunking_strategy.py
  - id: openwiki-source-53688c65a9b4548093faff14
    resource: repo://utils.py
generated: { by: "claude-code", at: "2026-10-11T10:08:11.054Z" }
verified:
  - by: openwiki/0.6.0
    at: 2026-10-12T14:34:48.400Z
---

# Indexing & Retrieval

`utils.py` owns everything between "250 paper Documents" and "a retriever that returns the top-k". The pipeline never touches arXiv or the cache directly — `app.retrieve_documents` just calls `utils.get_retriever(run_config).invoke(query)`.

## Retrieval units

`chunk_documents(docs, chunk_size, chunk_overlap)` has two regimes, selected by the arm's `RunConfig`:

- **`chunk_size == 0` — whole abstract (baseline, ADR-005).** Returns the papers unchanged: one index record per paper, page_content = the whole abstract. Retrieval unit == citation unit == paper, and a top-5 always holds 5 distinct papers.
- **`chunk_size > 0` — character-length splitting (experimental arm).** `RecursiveCharacterTextSplitter` cuts abstracts; `CHUNKED_500_0` produced 843 chunks from 250 papers (mean 3.37/paper). Splitting is by *characters*, deliberately not tokens: abstracts run ~255 tokens mean, so token-based 500-length chunks would rarely split — the plan's "~800-1200 chunks" sizing assumes characters.

Metadata flows through untouched: every chunk carries the source paper's `arxiv_id`, `title`, `authors`, `published`, `primary_category`, and `source` (set once in `_result_to_document`).

## chunk_id — deterministic identity

```python
def chunk_id(doc):  # f"{arxiv_id}:{sha1(page_content)[:12]}"
```

The vector store assigns random UUIDs per build, so `chunk_id` is the **single source of truth for retrieval gold labels**: the golden-set builder and the evaluators compute identical ids from any cache without a rebuild. Two consequences:

- The id is *content-addressed* — any chunking change produces entirely different ids, which is why chunk-level gold can't be compared across chunking configs (a key argument in ADR-005 for paper-level gold).
- The `arxiv_id:` prefix means the paper is always recoverable from the id by a string split — `evaluators.py` does exactly this to derive paper-level gold.

## Cache layout and build behavior

`_persist_path(embedding_model, chunk_strategy)` produces `resources/arxiv_{model}_{strategy}.parquet` (with `/` in the strategy rendered as `-`), so each experiment arm gets its own cache file:

| Arm | Cache file | Contents |
|---|---|---|
| `BASELINE` | `arxiv_text-embedding-3-small_abstract.parquet` | 250 records, whole abstracts |
| `CHUNKED_500_0` | `arxiv_text-embedding-3-small_500-0.parquet` | 843 chunks |

`get_vector_store(run_config)` is **cache-or-build**: if the parquet exists it opens an `SKLearnVectorStore` (FAISS-free, in-memory scikit-learn store) with the arm's `OpenAIEmbeddings` model; otherwise it loads the frozen corpus, chunks per the arm, embeds via `from_documents`, and persists. A cold build therefore costs a few cents of OpenAI embeddings; cache hits cost nothing and no network.

`get_retriever` is one line on top: `.as_retriever(search_kwargs={"k": run_config.k})` — `k` is the only retrieval-time knob the pipeline exposes (pre-rerank `rerank_top_n` is configured but unused until CKPT-4).

`load_cached_chunks(run_config)` reads a parquet cache back into `Document`s (row order preserved) and **raises** if the cache doesn't exist — it deliberately does not trigger a corpus rebuild. Its consumer is the golden-set builder, which samples source chunks for question generation.

## The measured trade-off (ADR-005)

The whole-abstract baseline was chosen from a same-corpus comparison: chunking gave `answerable` hit@1 1.00 but only 3.80 distinct papers in the top-5 (diversity collapse), while whole abstracts gave 0.90/1.00 hit@1/hit@5 with 5.00 distinct papers — at ~2.2× generation cost and +8% median latency. The hit@1 gap is acknowledged as possibly an artifact of generating questions *from* the chunked index. Chunking remains a first-class arm so the CKPT-3 experiment can measure "does splitting the abstract help or hurt?" at paper level.

## Failure behavior

- Missing cache for `load_cached_chunks` → `RuntimeError` (explicit; no silent rebuild).
- Missing cache for `get_vector_store` → full build path, which inherits all of the frozen corpus's fail-loud behavior (see [Frozen Corpus & Manifest](./frozen-corpus.md)).
- `chunk_overlap` without `chunk_size` is rejected at the `RunConfig` layer before reaching this code.

## Representative tests

`tests/test_chunking_strategy.py` pins: whole-abstract = one untouched record per paper; 500/0 splitting preserves metadata and ≤500-char chunks; per-strategy cache filenames; `chunk_id` paper-prefix extraction; and lossless `papers_from_chunks` round-trip including re-derived chunk ids.
