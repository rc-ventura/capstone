from __future__ import annotations

import logging

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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_corpus()


if __name__ == "__main__":
    main()
