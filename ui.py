"""Minimalist web interface for the arXiv Research Copilot (docs/ui/ui-plan.md).

Local demo and debug tool only: chat with the baseline pipeline, see the cited
sources as an accordion, and generate real multi-turn traces (one thread_id per
session) for manual inspection in LangSmith. Run: `uv run python ui.py`.
"""

from __future__ import annotations

import uuid
from typing import Any

import gradio as gr
from langchain_core.documents import Document

import app
import config
import utils

RUN_CONFIG = config.BASELINE


def _unique_documents(documents: list[Document]) -> list[Document]:
    """Dedup by arxiv_id, preserving retrieval rank order."""
    unique = []
    seen: set[str] = set()
    for doc in documents:
        if doc.metadata["arxiv_id"] not in seen:
            seen.add(doc.metadata["arxiv_id"])
            unique.append(doc)
    return unique


def format_sources(documents: list[Document]) -> str:
    """Markdown for the sources accordion: one line per paper, in rank order."""
    lines = [
        f"**[{doc.metadata['arxiv_id']}]** {doc.metadata['title']} · {doc.metadata['published']}"
        f" — [abs]({doc.metadata['source']})"
        for doc in _unique_documents(documents)
    ]
    return "\n\n".join(lines) if lines else "Nenhuma fonte recuperada."


def filter_history(history: list[Any]) -> list[dict[str, str]]:
    """Gradio history → prompt conversation: keep only role/content and drop the
    sources blocks (messages with metadata.title), which must not leak into the prompt."""
    conversation = []
    for message in history:
        if isinstance(message, dict):
            role, content, metadata = (
                message.get("role"),
                message.get("content"),
                message.get("metadata") or {},
            )
        else:
            role, content, metadata = (
                getattr(message, "role", None),
                getattr(message, "content", None),
                getattr(message, "metadata", None) or {},
            )
        if role in ("user", "assistant") and isinstance(content, str) and not metadata.get("title"):
            conversation.append({"role": role, "content": content})
    return conversation


def respond(message: str, history: list[Any], thread_state: dict) -> list[gr.ChatMessage]:
    metadata = app._run_metadata(RUN_CONFIG) | {"thread_id": thread_state["thread_id"]}
    answer, documents = app.arxiv_copilot_with_sources(
        message,
        RUN_CONFIG,
        conversation=filter_history(history),
        langsmith_extra={"metadata": metadata},
    )
    sources = format_sources(documents)
    return [
        gr.ChatMessage(role="assistant", content=answer),
        gr.ChatMessage(
            role="assistant",
            content=sources,
            metadata={"title": f"Fontes ({len(_unique_documents(documents))})"},
        ),
    ]


def _new_thread() -> dict:
    return {"thread_id": str(uuid.uuid4())}


CSS = """
#main-col { max-width: 760px; margin: 0 auto; }
#footer { color: #8a8f98; font-size: 0.8rem; text-align: center; }
"""


def build_demo() -> gr.Blocks:
    # Gradio 6: theme/css live on launch(), not on Blocks; ChatInterface has no
    # type= param anymore (history is always in "messages" format).
    with gr.Blocks(title="arXiv Copilot", fill_height=True) as demo:
        with gr.Column(elem_id="main-col"):
            gr.Markdown(
                "# arXiv Copilot\n"
                f"~{config.CORPUS_TARGET_SIZE} cs.AI papers · snapshot `{config.CORPUS_SNAPSHOT_DATE}`"
            )
            thread_state = gr.State(_new_thread)
            chat = gr.ChatInterface(
                fn=respond,
                additional_inputs=[thread_state],
                examples=[
                    ["What are common failure modes of RAG systems?"],
                    ["Compare how two recent papers evaluate LLM agents."],
                    ["What are the latest results on protein folding with diffusion models?"],
                ],
            )
            gr.Markdown(
                f"{RUN_CONFIG.app_version} · k={RUN_CONFIG.k} · {config.GENERATION_MODEL}",
                elem_id="footer",
            )
            chat.chatbot.clear(_new_thread, outputs=thread_state)
    return demo


if __name__ == "__main__":
    utils.get_vector_store(RUN_CONFIG)  # warm up so the 1st question skips the parquet load
    demo = build_demo()
    demo.launch(  # localhost only; Gradio 6 takes theme/css here, not on Blocks
        theme=gr.themes.Base(primary_hue="slate", font=gr.themes.GoogleFont("Inter")),
        css=CSS,
    )
