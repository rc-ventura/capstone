from __future__ import annotations

import hashlib
import itertools
import json
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

    The corpus is pinned at config.CORPUS_SNAPSHOT_DATE (docs/plan/ckpt-0.5-plan.md
    §4.2): arXiv returns newest-first, so we scan target+snapshot-skip-budget raw
    results and drop anything published after the pin, keeping the corpus frozen
    across embedding/chunk cache rebuilds.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results + config.CORPUS_SNAPSHOT_SKIP_BUDGET,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    snapshot_results = itertools.islice(
        (
            result
            for result in client.results(search)
            if result.published.date() <= config.CORPUS_SNAPSHOT_DATE
        ),
        max_results,
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
        for result in snapshot_results
    ]
    if len(docs) < max_results:
        raise RuntimeError(
            f"Only {len(docs)} of {max_results} papers fell on/before the snapshot "
            f"date {config.CORPUS_SNAPSHOT_DATE} — raise CORPUS_SNAPSHOT_SKIP_BUDGET"
        )
    logger.info(
        "Fetched %d papers from arXiv (category=%s, snapshot<=%s)",
        len(docs),
        category,
        config.CORPUS_SNAPSHOT_DATE,
    )
    return docs


def chunk_id(doc: Document) -> str:
    """Deterministic chunk identity — the single source of truth for retrieval gold
    labels (docs/plan/ckpt-0.5-plan.md §4.1).

    Derived from persisted fields, so the dataset builder (CKPT-0.5) and the
    retrieval evaluators (CKPT-0.6) compute identical labels from any cache without
    a rebuild. The vector store's own UUIDs are random per build and can't be used.
    """
    digest = hashlib.sha1(doc.page_content.encode()).hexdigest()[:12]
    return f"{doc.metadata['arxiv_id']}:{digest}"


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


def load_cached_chunks(run_config: config.RunConfig = config.BASELINE) -> list[Document]:
    """Reconstruct every chunk from the cached parquet store, as Documents.

    Used by the golden-dataset builder (CKPT-0.5) to sample source chunks for
    synthetic question generation without triggering a corpus rebuild.
    """
    import pandas as pd

    persist_path = _persist_path(run_config.embedding_model, run_config.chunk_strategy)
    if not persist_path.exists():
        raise RuntimeError(f"No cached store at {persist_path} — build it via get_vector_store first")
    df = pd.read_parquet(persist_path)
    docs = []
    for _, row in df.iterrows():
        meta = row["metadatas"]
        docs.append(
            Document(
                page_content=row["texts"],
                metadata=json.loads(meta) if isinstance(meta, str) else meta,
            )
        )
    return docs
