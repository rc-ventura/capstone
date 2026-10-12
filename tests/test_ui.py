"""Offline unit tests for the UI helpers (docs/ui/ui-plan.md).

No network, no LLM — controlled Document/history inputs. Contract under test:
docs/ui/ui-plan.md §2 (sources block dedup/format, prompt history filtering).
"""

from langchain_core.documents import Document

import ui


def doc(arxiv_id: str, title: str = "Some paper") -> Document:
    return Document(
        page_content="abstract",
        metadata={
            "arxiv_id": arxiv_id,
            "title": title,
            "published": "2026-09-10",
            "source": f"http://arxiv.org/abs/{arxiv_id}",
        },
    )


# --- format_sources ------------------------------------------------------


def test_format_sources_dedups_by_paper():
    md = ui.format_sources([doc("2509.00001"), doc("2509.00002"), doc("2509.00001")])
    assert md.count("**[") == 2


def test_format_sources_preserves_rank_order():
    md = ui.format_sources([doc("2509.00002"), doc("2509.00001")])
    assert md.index("2509.00002") < md.index("2509.00001")


def test_format_sources_line_format():
    md = ui.format_sources([doc("2509.00001", title="Deep RAG")])
    assert "**[2509.00001]** Deep RAG · 2026-09-10 — [abs](http://arxiv.org/abs/2509.00001)" == md


def test_format_sources_empty_is_neutral():
    assert ui.format_sources([]) == "Nenhuma fonte recuperada."


# --- filter_history -------------------------------------------------------


SOURCES_MSG = {
    "role": "assistant",
    "content": "**[2509.00001]** ...",
    "metadata": {"title": "Fontes (1)"},
}


def test_filter_history_keeps_role_and_content_only():
    history = [{"role": "user", "content": "hi", "id": 7}, SOURCES_MSG]
    assert ui.filter_history(history) == [{"role": "user", "content": "hi"}]


def test_filter_history_drops_extra_keys():
    conv = ui.filter_history([{"role": "assistant", "content": "ok", "metadata": {}}])
    assert conv == [{"role": "assistant", "content": "ok"}]


def test_filter_history_accepts_chatmessage_objects():
    import gradio as gr

    history = [
        gr.ChatMessage(role="user", content="hi"),
        gr.ChatMessage(role="assistant", content="src", metadata={"title": "Fontes (1)"}),
    ]
    assert ui.filter_history(history) == [{"role": "user", "content": "hi"}]


def test_filter_history_ignores_non_text_parts():
    history = [{"role": "user", "content": {"text": "ignored"}}]
    assert ui.filter_history(history) == []
