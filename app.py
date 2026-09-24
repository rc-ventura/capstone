
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langsmith import traceable, trace
from langsmith.wrappers import wrap_openai
from openai import OpenAI

import config
import utils

openai_client = wrap_openai(OpenAI())

# copilot-prompt-v1 (docs/plan.md CKPT-6): course-style 3-sentence baseline,
# extended with citation + conversation — Feature A (cited Q&A) and Feature F
# (multi-turn threads) are baseline product scope, not experimental variants.
PROMPT_V1 = """You are an assistant for question-answering tasks over arXiv cs.AI papers.
Use the following pieces of retrieved context to answer the latest question in the conversation.
If the answer is not in the context, say you don't know — do not guess.
Cite the arXiv ID (e.g. [2609.12345]) for every claim you make.
The pre-existing conversation may provide important context to the question.
Use three sentences maximum and keep the answer concise.

Conversation: {conversation}
Context: {context}
Question: {question}
Answer:"""


def _prompt_template(prompt_version: str) -> str:
    if prompt_version == "v1":
        return PROMPT_V1
    raise NotImplementedError(f"Prompt {prompt_version} lands in CKPT-6 (docs/plan.md)")


def _run_metadata(run_config: config.RunConfig) -> dict[str, Any]:
    """Every field here is attached to every run so any metric can be sliced
    by any configuration axis (docs/plan.md §3)."""
    return {
        "app_version": run_config.app_version,
        "embedding_model": run_config.embedding_model,
        "k": run_config.k,
        "chunk_strategy": run_config.chunk_strategy,
        "rerank": run_config.rerank,
        "query_rewrite": run_config.query_rewrite,
        "prompt_version": run_config.prompt_version,
    }


@traceable(run_type="chain")
def rewrite_query(question: str, run_config: config.RunConfig) -> str:
    if not run_config.query_rewrite:
        return question
    raise NotImplementedError("Query rewrite lands in CKPT-5 (docs/plan.md)")


@traceable(run_type="retriever")
def retrieve_documents(query: str, run_config: config.RunConfig) -> list[Document]:
    retriever = utils.get_retriever(run_config)
    return retriever.invoke(query)


@traceable(run_type="chain")
def rerank(documents: list[Document], query: str, run_config: config.RunConfig) -> list[Document]:
    if not run_config.rerank:
        return documents
    raise NotImplementedError(f"LLM rerank lands in CKPT-4 (docs/plan.md); got query={query!r}")


def generate_response(
    question: str,
    documents: list[Document],
    run_config: config.RunConfig,
    conversation: list[dict[str, str]] | None = None,
) -> str:
    formatted_docs = "\n\n".join(
        f"[{doc.metadata['arxiv_id']}] {doc.page_content}" for doc in documents
    )
    prompt = _prompt_template(run_config.prompt_version).format(
        conversation=conversation or [],
        context=formatted_docs,
        question=question,
    )
    with trace(
        name="generate_response",
        run_type="chain",
        inputs={"question": question, "formatted_docs": formatted_docs},
        metadata=_run_metadata(run_config),
    ) as ls_trace:
        response = openai_client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = response.choices[0].message.content
        ls_trace.end(outputs={"answer": answer})
    return answer


@traceable(run_type="chain")
def arxiv_copilot(
    question: str,
    run_config: config.RunConfig = config.BASELINE,
    conversation: list[dict[str, str]] | None = None,
) -> str:
    metadata = _run_metadata(run_config)
    rewritten = rewrite_query(question, run_config, langsmith_extra={"metadata": metadata})
    documents = retrieve_documents(rewritten, run_config, langsmith_extra={"metadata": metadata})
    ranked = rerank(documents, rewritten, run_config, langsmith_extra={"metadata": metadata})
    return generate_response(question, ranked, run_config, conversation)
