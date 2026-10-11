---
type: runtime pipeline
title: RAG Pipeline
description: The traced question-answering pipeline in app.py — rewrite, retrieve, rerank, generate — its prompt v1 contract for citations and abstention, multi-turn conversation support, and which stages are live versus reserved extension seams.
tags: [pipeline, rag, retrieval, generation, prompt, langsmith]
verified:
  - by: openwiki/0.6.0
    at: 2026-10-11T10:08:11.054Z
sources:
  - id: openwiki-source-17dded95897e01ee430228e3
    resource: repo://app.py
  - id: openwiki-source-833e692518af9eeaf8564cc6
    resource: repo://main.py
generated: { by: "claude-code", at: "2026-10-11T10:08:11.054Z" }
---

# RAG Pipeline

`app.py` implements the product's runtime: a single question in, a cited answer out. The entrypoint is `arxiv_copilot(question, run_config=config.BASELINE, conversation=None)`, which runs four stages in order and returns the answer string.

## Stage chain

```
arxiv_copilot (chain, root run)
├── rewrite_query(question, run_config)      → str
├── retrieve_documents(query, run_config)    → list[Document]
├── rerank(documents, query, run_config)     → list[Document]
└── generate_response(question, docs, run_config, conversation) → str
```

Each stage is controlled by `run_config`, and every stage receives the same metadata dict (`langsmith_extra={"metadata": metadata}`) so all runs in one invocation share the configuration axes.

| Stage | State | Behavior |
|---|---|---|
| `rewrite_query` | off (seam) | Returns the question unchanged when `run_config.query_rewrite` is false; raising `NotImplementedError` otherwise — the real rewriter lands in CKPT-5. |
| `retrieve_documents` | live | Delegates to `utils.get_retriever(run_config).invoke(query)` — an `SKLearnVectorStore` retriever at `search_kwargs={"k": run_config.k}` over the per-config parquet index. |
| `rerank` | off (seam) | Returns the document list unchanged when `run_config.rerank` is false; `NotImplementedError` otherwise — the LLM reranker (scoring `rerank_top_n` candidates down to `k`) lands in CKPT-4. |
| `generate_response` | live | Formats context, renders `PROMPT_V1`, calls `gpt-4o-mini` at `temperature=0.0`, returns `choices[0].message.content`. |

Note the seam asymmetry: the pipeline passes the *rewritten* query to retrieval and rerank, but passes the *original* `question` to `generate_response` — so a future rewriter can optimize retrieval terms without changing what the user actually asked.

## PROMPT_V1 — the generation contract

```text
You are an assistant for question-answering tasks over arXiv cs.AI papers.
Use the following pieces of retrieved context to answer the latest question in the conversation.
If the answer is not in the context, say you don't know — do not guess.
Cite the arXiv ID (e.g. [2609.12345]) for every claim you make.
The pre-existing conversation may provide important context to the question.
Use three sentences maximum and keep the answer concise.
```

Three obligations are baked into the baseline prompt, and each maps to an evaluation surface:

- **Abstention** ("say you don't know") — the honest-abstention product feature (FP1); detected downstream by `evaluators.abstained` and the planned `abstention_quality` judge. The prompt fixes no exact refusal phrase, which is why detection uses marker families rather than a prefix check.
- **Citation** ("cite the arXiv ID for every claim") — enabled by the context format: `generate_response` joins documents as `[{arxiv_id}] {page_content}`, so the model sees the id inline next to each abstract.
- **Brevity** ("three sentences maximum") — the course-style baseline; the `format` slice and `format_validator` exercise stricter instruction-following.

`_prompt_template` dispatches on `run_config.prompt_version` and raises `NotImplementedError` for anything other than `"v1"` — prompt variants are a CKPT-6 extension seam.

## Context formatting and citation integrity

```python
formatted_docs = "\n\n".join(
    f"[{doc.metadata['arxiv_id']}] {doc.page_content}" for doc in documents
)
```

Because the baseline retrieval unit is the whole abstract (one index record per paper), the bracketed id prefixing each block is also the citation unit the model is told to emit — retrieval unit == citation unit. Under the `CHUNKED_500_0` arm the same formatting holds (chunks carry the paper's full metadata), but a cited id then points at a chunk's paper rather than a uniquely retrieved block.

## Multi-turn support

`arxiv_copilot` accepts an optional `conversation: list[dict[str, str]]` which is rendered into the prompt's `Conversation:` slot (an empty list when absent). Turn identity is a *metadata* concern, not a prompt one: `main.run_pipeline` attaches a fresh `thread_id` uuid to the run metadata, which is what groups a conversation's runs in LangSmith.

## Failure behavior

- Toggling `query_rewrite`, `rerank`, or an unknown `prompt_version` raises `NotImplementedError` — deliberate seams, not fallbacks.
- `generate_response` makes exactly one chat completion at `temperature=0.0` with no retry logic; API failures propagate.
- An empty retrieved list still generates (the prompt just sees an empty context), which is precisely the abstention path the `unanswerable` slice exercises.

## Tracing surface

`rewrite_query`, `retrieve_documents`, `rerank`, and `arxiv_copilot` are `@traceable`-decorated (`run_type` `"chain"` or `"retriever"`); `generate_response` wraps its work in an explicit `trace()` context and the module-level `openai_client` is a `wrap_openai(OpenAI())`, so the LLM call appears as a child run. Details: [LangSmith Tracing & Datasets](../observability/langsmith-tracing.md).
