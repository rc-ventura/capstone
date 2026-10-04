# Plan — Minimalist web interface for the arXiv Research Copilot

## Context

Today the RAG (`app.arxiv_copilot`) only runs via `main.py` (one fixed question) or via the REPL
(`docs/manual-usage.md`). We lack a fast and pleasant way to **chat with the copilot,
see the cited sources, and generate real traces with a thread** for manual inspection. This comes before
`server.py` (FastAPI, CKPT-8) and does not replace it: it is a local demo and debug tool.

Goal: a clean, single-page chat that:
- answers questions using the baseline pipeline (`config.BASELINE`) without changing its behavior;
- shows the retrieved sources (arXiv ID, title, date, link) in a collapsible block;
- keeps a multi-turn conversation (Feature F) with a per-session `thread_id` in LangSmith.

**Framework: Gradio** (not Streamlit). `gr.ChatInterface(type="messages")` already delivers the
history in the `[{"role", "content"}]` format that `generate_response` expects, and comes with UI streaming,
examples, and a clear button out of the box, and `gr.ChatMessage(metadata={"title": ...})`
renders a native accordion — ideal for the sources without cluttering the screen. Streamlit would require
managing `session_state` and script re-execution by hand for the same result.

## Changes

### 1. `app.py` — expose the sources without changing the existing contract
`arxiv_copilot` returns only the string; the UI needs the documents. To avoid duplicating
the orchestration or retrieving twice:

- Extract the body of `arxiv_copilot` into an **untraced** helper
  `_run_pipeline(question, run_config, conversation) -> tuple[str, list[Document]]`
  (rewrite → retrieve → rerank → generate, with the same `langsmith_extra` metadata).
- `arxiv_copilot` remains `@traceable(run_type="chain")`, same signature, calls the
  helper and returns only `answer` → trace tree, `main.py`, and future evaluators intact.
- New `arxiv_copilot_with_sources(...)` with
  `@traceable(run_type="chain", name="arxiv_copilot", process_outputs=lambda o: {"output": o[0]})`
  → in LangSmith the root run is identical (same name, same output), but the caller
  receives `(answer, documents)`. (Confirm `process_outputs` in the installed version of
  `langsmith` via context7 before implementing.)

### 2. `ui.py` (new, project root, ~80 lines)
- `respond(message, history, request)`:
  - `conversation` = filtered history: only `role`/`content`, **discarding** the sources
    messages (those with `metadata.title`) — otherwise the sources block leaks into the prompt.
  - calls `app.arxiv_copilot_with_sources(message, config.BASELINE, conversation,
    langsmith_extra={"metadata": app._run_metadata(config.BASELINE) | {"thread_id": tid}})`
    — same pattern as `main.py:run_pipeline`.
  - returns `[gr.ChatMessage(answer), gr.ChatMessage(sources_md, metadata={"title": "Fontes (N)"})]`.
- `thread_id`: one `uuid4` per session in `gr.State`; renewed when the user clears the chat
  (`chatbot.clear`).
- `format_sources(documents) -> str` (pure, testable function): dedup by `arxiv_id`
  preserving rank order; each line `**[2609.12345]** Title · 2026-09-10 — [abs](source)`.
- Minimalist layout:
  - `gr.Blocks(theme=gr.themes.Base(primary_hue="slate", font=gr.themes.GoogleFont("Inter")),
    fill_height=True)`, central column `max-width ~760px` via a short `css`.
  - 1-line header: "arXiv Copilot" + discreet subtitle "~250 cs.AI papers · snapshot
    `CORPUS_SNAPSHOT_DATE`".
  - `gr.ChatInterface` with 3 examples (one answerable, one multi-doc, one outside the corpus to
    show abstention).
  - Small gray footer: `app_version · k · GENERATION_MODEL` (read from `config`).
  - No sidebar, no config controls — the baseline is fixed in this version.
- `if __name__ == "__main__": demo.launch()` (localhost only, no `share=True`).
- Warm up the vector store at import (`utils.get_vector_store(config.BASELINE)`) so the 1st
  question does not pay for the parquet load.

### 3. `pyproject.toml`
Add `gradio>=5` to the dependencies (`uv add gradio`).

### 4. Tests — `tests/test_ui.py`
- `format_sources`: dedup by paper, rank order preserved, empty list → neutral message.
- history filter: messages with `metadata.title` are removed; extra keys disappear.
No calls to OpenAI/LangSmith (pure functions, in the style of `tests/test_evaluators.py`).

### 5. Docs
New section in `docs/manual-usage.md`: "Web interface" with `uv run python ui.py` and what
to look at in LangSmith (runs grouped by `thread_id`).

## Out of scope
- 👍/👎 and feedback URLs (CKPT-8, `feedback.py`) — `ChatInterface` accepts `.like()` later.
- Switching config/arm via the UI (k, rerank, etc.) — comes in when there is more than one functional arm.
- Deploy/`share=True`.

## Git note
The working tree has uncommitted CKPT-0.6 changes (`evaluators.py`, tests,
`build_golden_dataset.py`). Implement the UI in a separate commit (or its own branch) so as not
to mix it with that work.

## Verification
1. `uv run pytest tests/test_ui.py tests/test_evaluators.py` — all green.
2. `uv run python main.py` — `Q:`/`A:` remains the same (`arxiv_copilot` contract preserved).
3. `uv run python ui.py` → open `http://127.0.0.1:7860`:
   - the example question is answered with `[arxiv_id]` citations and a "Fontes (N)" accordion;
   - follow-up ("and what are the limitations?") uses the conversation context;
   - question outside the corpus → abstention answer;
   - clear chat → the next question generates a new `thread_id`.
4. LangSmith, project `capstone-arxiv-copilot`: root run `arxiv_copilot` with string output,
   retriever/chain children as before, and the session's messages grouped in the Threads tab.
5. Screenshot via Playwright at desktop and mobile width (~390px) to check the layout.
