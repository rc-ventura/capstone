from __future__ import annotations

import logging
import uuid

import app
import config
import utils

logger = logging.getLogger(__name__)


def build_corpus() -> None:
    """Build/load the baseline vector store and confirm it's queryable."""
    retriever = utils.get_retriever(config.BASELINE)
    sample = retriever.invoke("retrieval augmented generation hallucination")
    print(f"Indexed corpus ready. Sample query returned {len(sample)} chunks:")
    for doc in sample:
        print(f"- [{doc.metadata['arxiv_id']}] {doc.metadata['title'][:70]}")


def run_pipeline() -> None:
    """Run the traced RAG pipeline once end-to-end, confirming it reaches LangSmith."""
    question = "What are common failure modes of RAG systems?"
    metadata = app._run_metadata(config.BASELINE) | {"thread_id": str(uuid.uuid4())}
    answer = app.arxiv_copilot(question, langsmith_extra={"metadata": metadata})
    print(f"Q: {question}\nA: {answer}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_corpus()
    run_pipeline()
    # build_golden_dataset() joins this flow in the next checkpoint (docs/plan.md CKPT-0)


if __name__ == "__main__":
    main()
