---
type: observability subsystem
title: LangSmith Tracing & Datasets
description: How the system is instrumented for LangSmith — the three tracing mechanisms used in app.py, the uniform run-metadata model that makes metrics sliceable by any configuration axis, thread_id for multi-turn runs, and the LangSmith surfaces (project, dataset, evaluators) the system writes to.
tags: [langsmith, tracing, observability, metadata, datasets, monitoring]
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
  - id: openwiki-source-0efdc1e16f1045bb19dad3ef
    resource: repo://docs/plan.md
  - id: openwiki-source-a2b903fa68b9b04e7d5df273
    resource: repo://evaluators.py
  - id: openwiki-source-0756aa10ba9e55148108b47d
    resource: repo://example.env
  - id: openwiki-source-833e692518af9eeaf8564cc6
    resource: repo://main.py
generated: { by: "claude-code", at: "2026-10-11T10:08:11.054Z" }
---

# LangSmith Tracing & Datasets

LangSmith is the system's single observability backend: pipeline traces, the golden dataset, and (planned) experiment and monitoring surfaces all land there. Tracing is not incidental — the course this capstone builds on is about AgentOps, and the metadata model is designed so *any metric can be sliced by any configuration axis*.

## The three tracing mechanisms in one pipeline

`app.py` deliberately exercises all three LangSmith instrumentation styles:

| Mechanism | Where | Effect |
|---|---|---|
| `@traceable` | `arxiv_copilot`, `rewrite_query`, `retrieve_documents`, `rerank` | Each stage becomes a named run under the root `arxiv_copilot` run, with `run_type` `"chain"` or `"retriever"` |
| `wrap_openai` | module-level `openai_client = wrap_openai(OpenAI())` | The raw OpenAI call inside `generate_response` is auto-traced as an `llm` child run |
| `trace()` context | inside `generate_response` | Groups the prompt-formatting + LLM call as a `chain` run with explicit `inputs` (`question`, `formatted_docs`) and `outputs` (`answer`) |

The resulting tree in LangSmith:

```
arxiv_copilot (chain, root)
├── rewrite_query (chain)
├── retrieve_documents (retriever)
├── rerank (chain)
└── generate_response (chain)
    └── ChatOpenAI (llm, via wrap_openai)
```

`@traceable` stage calls receive `langsmith_extra={"metadata": metadata}` so the same dict lands on every run; `generate_response` sets metadata directly on its `trace()` context.

## Run metadata — the slicing contract

`_run_metadata(run_config)` emits the fields below on **every** run (plan §3: "enables slicing any metric by any configuration"):

`app_version`, `embedding_model`, `k`, `chunk_strategy`, `rerank`, `query_rewrite`, `prompt_version`

This is what makes the experiment model observable: an EXP-4 rerank arm and the baseline produce identically-shaped traces differing only in metadata values, so per-slice metrics can be filtered and compared by `RunConfig` fields without any extra bookkeeping. `main.run_pipeline` additionally attaches a `thread_id` uuid, which groups the runs of a multi-turn conversation in LangSmith's thread view — turn identity lives in metadata, not in the function signatures.

## Configuration surface

| Variable | Purpose |
|---|---|
| `LANGSMITH_API_KEY` | required at `config` import — hard failure without it |
| `LANGSMITH_PROJECT` | trace destination; default `capstone-arxiv-copilot` |
| `LANGSMITH_TRACING` | `example.env` sets `"true"` |
| `LANGSMITH_ENDPOINT` | commented in `example.env` — set for the EU instance |

Everything traces into one project; the README warns to use a project "you do not mind filling," since `main.py` writes both traces and the dataset on first run.

## Dataset surface

`build_golden_dataset` uses the `langsmith.Client` SDK directly: `has_dataset` / `read_dataset` / `create_dataset` for `arxiv-copilot-golden`, `list_examples(dataset_id, splits=["core"])` for idempotency counts, `create_examples` for bulk slice creation, and `update_example` for `regenerate_vague_answerable`'s in-place fixes. Dataset examples carry the evaluation labels (`gold_chunk_ids`, `gold_arxiv_ids`, `should_abstain`) and metadata (`slice`, `authoring`, `review`) that the evaluators consume — see [Golden Dataset](../evaluation/golden-dataset.md).

## Evaluator surface

The evaluators are written against LangSmith's calling convention:

- **Per-example evaluators** (retrieval metrics, `format_validator`) take plain inputs and reference dicts.
- **Summary evaluators** (`abstention_summary`, `f1_summary_evaluator`) receive parallel `outputs`/`reference_outputs` lists plus `examples`, and must return exactly `{"results": [{"key", "score", "comment"?}]}` — `EvaluationResult` is `extra="forbid"`, so extra keys fail the evaluator and silently drop its metrics (a bug that already happened once, roadmap P4).

Planned CKPT-8 work adds online evaluators as rules on production traffic and a feedback endpoint — none of that exists yet.

## Manual inspection entry points

`docs/manual-usage.md` documents hands-on inspection without writing to LangSmith: invoke `utils.get_retriever` in a REPL to see ranked `arxiv_id`/`published`/`title`, run `app.arxiv_copilot` on a free-form question, or page through the dataset with the SDK (`list_examples`, `metadata["review"]["state"]` counts). In the LangSmith UI, the project shows `arxiv_copilot` traces with their run types and the full metadata dict; the dataset shows examples by split, slice, and review state.
