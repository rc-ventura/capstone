from __future__ import annotations

import logging
from pathlib import Path

import arxiv
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)


def fetch_arxiv_corpus(
    category: str = config.ARXIV_CATEGORY,
    max_results: int = config.CORPUS_TARGET_SIZE,
) -> list[Document]:
    """Fetch the most recent arXiv abstracts for `category` as LangChain Documents.

    One Document per paper (page_content = abstract, not full text — docs/plan.md
    §1 sizes the corpus assuming ~250-300 abstracts yield ~800-1200 chunks).
    `published` is kept on every chunk's metadata; it's what the `stale` golden-set
    slice (docs/plan.md §2) filters on.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    docs = [
        Document(
            page_content=result.summary,
            metadata={
                "arxiv_id": result.get_short_id(),
                "title": result.title,
                "authors": ", ".join(a.name for a in result.authors),
                "published": result.published.date().isoformat(),
                "primary_category": result.primary_category,
                "source": result.entry_id,
            },
        )
        for result in client.results(search)
    ]
    logger.info("Fetched %d papers from arXiv (category=%s)", len(docs), category)
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split papers into chunks, carrying each paper's metadata onto every chunk.

    Character-length splitting, not `.from_tiktoken_encoder` (the course's pattern
    for long doc pages): docs/plan.md §1 sizes the corpus in characters
    ("~800-1200 chunks @ 500 chars"), and abstracts are short enough (~300-450
    tokens) that token-based 500-length chunks would rarely split them at all —
    that math only lands at ~250-300 chunks, not 800-1200.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def _persist_path(embedding_model: str, chunk_strategy: str) -> Path:
    safe_chunk = chunk_strategy.replace("/", "-")
    return config.RESOURCES_DIR / f"arxiv_{embedding_model}_{safe_chunk}.parquet"


def get_vector_store(run_config: config.RunConfig = config.BASELINE) -> SKLearnVectorStore:
    """Load the cached vector store for `run_config`, building it from arXiv if missing."""
    persist_path = _persist_path(run_config.embedding_model, run_config.chunk_strategy)
    embeddings = OpenAIEmbeddings(model=run_config.embedding_model)

    if persist_path.exists():
        logger.info("Loading cached vector store from %s", persist_path)
        return SKLearnVectorStore(
            embedding=embeddings,
            persist_path=str(persist_path),
            serializer="parquet",
        )

    logger.info("No cache at %s — building corpus from arXiv", persist_path)
    papers = fetch_arxiv_corpus()
    chunks = chunk_documents(papers, run_config.chunk_size, run_config.chunk_overlap)
    vectorstore = SKLearnVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_path=str(persist_path),
        serializer="parquet",
    )
    vectorstore.persist()
    logger.info("Cached %d chunks from %d papers to %s", len(chunks), len(papers), persist_path)
    return vectorstore


def get_retriever(run_config: config.RunConfig = config.BASELINE) -> VectorStoreRetriever:
    """Return a retriever configured with `run_config.k` (docs/plan.md CKPT-1)."""
    return get_vector_store(run_config).as_retriever(search_kwargs={"k": run_config.k})
