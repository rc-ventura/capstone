---
type: user interface
title: Web UI (Gradio)
description: The minimalist Gradio chat UI in ui.py — a local demo and debug tool that chats with the fixed baseline pipeline, shows cited sources in an accordion, and produces real multi-turn LangSmith traces grouped by a per-session thread_id.
tags: [ui, gradio, chat, demo, sources, thread-id, langsmith]
verified:
  - by: openwiki/0.6.0
    at: 2026-10-12T14:31:19.877Z
sources:
  - id: openwiki-source-a3f0492a6d8aa87271162233
    resource: repo://docs/ui/ui-plan.md
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-0720c5a3812dee82f153074a
    resource: repo://tests/test_ui.py
  - id: openwiki-source-4b14f496b381980b39391eea
    resource: repo://ui.py
generated: { by: "claude-code", at: "2026-10-12T14:31:19.877Z" }
---

# Web UI (Gradio)

`ui.py` is a single-page chat against the copilot, launched with `uv run python ui.py` (localhost only, `http://127.0.0.1:7860`). Per `docs/ui/ui-plan.md` it is a **local demo and debug tool**: it exists to chat with the baseline, see the cited sources, and generate real multi-turn traces for manual inspection in LangSmith. It precedes the planned CKPT-8 `server.py` (FastAPI) and does not replace it.

## What it deliberately is not

The UI is hardwired to `config.BASELINE` — no config/arm controls, no `share=True`, no feedback buttons (`.like()` and feedback URLs are CKPT-8 scope). Gradio was chosen over Streamlit because `gr.ChatInterface` already delivers message-format history and `gr.ChatMessage(metadata={"title": ...})` renders a native accordion for the sources block.

## Request flow

`respond(message, history, thread_state)` is the `ChatInterface` callback:

1. `filter_history(history)` converts the Gradio history into the `conversation` list the pipeline expects.
2. It calls `app.arxiv_copilot_with_sources(message, config.BASELINE, conversation, langsmith_extra={"metadata": ...})` — the twin entrypoint that returns `(answer, ranked_documents)` while tracing identically to `arxiv_copilot` (see [RAG Pipeline](../architecture/rag-pipeline.md)).
3. It returns two `gr.ChatMessage`s: the answer, then the sources block with `metadata={"title": "Fontes (N)"}` — the `title` is what makes Gradio render it as a collapsible accordion **and** what `filter_history` uses to strip it back out on the next turn.

The metadata attached to the root run is `app._run_metadata(RUN_CONFIG)` merged with `thread_id` — the same pattern `main.py` uses, so UI traffic lands in LangSmith with the full configuration axes plus thread grouping.

## The two pure helpers

<!-- openwiki: broken internal link [url] file "url" does not exist. Fix the href or restore the target, then delete this comment. -->
- **`format_sources(documents)`** — dedups by `arxiv_id` preserving retrieval rank, then renders one line per paper: `**[id]** title · published — [abs](url)`. An empty list returns the neutral string `"Nenhuma fonte recuperada."`. (The repo's docs and UI strings are in Portuguese; the wiki describes them in English.)
- **`filter_history(history)`** — keeps only user/assistant messages with string content and **drops every message carrying `metadata.title`**. Without this, the sources accordion of turn N would be rendered into the prompt of turn N+1. It accepts both dicts and `gr.ChatMessage` objects.

## Session state and layout

- `thread_id` is one `uuid4` per browser session held in `gr.State(_new_thread)`; `chat.chatbot.clear` regenerates it, so clearing the chat starts a new LangSmith thread.
- Layout: a centered ~760px column (`#main-col`), a one-line header with corpus size + snapshot date, three example questions (answerable, multi-doc, and one outside the corpus to demonstrate abstention), and a footer showing `app_version · k · GENERATION_MODEL`.
- `__main__` warms the vector store (`utils.get_vector_store(RUN_CONFIG)`) before launch so the first question doesn't pay the parquet load. Gradio 6 API notes the code documents: `theme`/`css` are arguments to `launch()`, not `Blocks`, and `ChatInterface` history is always in messages format. The dependency is `gradio>=5` in `pyproject.toml`.

## Tests

`tests/test_ui.py` covers the two pure helpers offline: dedup by paper, rank-order preservation, exact line format, the empty-list message, and the history filter (dicts and `ChatMessage` objects, extra keys dropped, non-string content ignored). `respond` itself — the only part touching the pipeline — is exercised manually per the verification list in `docs/ui/ui-plan.md`.

## Verification checklist (from the plan)

`uv run pytest tests/test_ui.py tests/test_evaluators.py` green; `main.py` output unchanged (`arxiv_copilot` contract preserved); the example question answers with citations and a "Fontes (N)" accordion; a follow-up uses conversation context; an out-of-corpus question abstains; clearing the chat yields a new `thread_id`; and in the `capstone-arxiv-copilot` project the session's runs group under the Threads tab.
