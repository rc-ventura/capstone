from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

import arxiv
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)


def _result_to_document(result: arxiv.Result) -> Document:
    """One Document per paper: page_content = abstract, metadata as every chunk will carry it."""
    return Document(
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
    docs = [_result_to_document(result) for result in snapshot_results]
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

    `chunk_size=0` means NO splitting (ADR-005): each paper stays one record holding the whole
    abstract, so retrieval unit == citation unit == paper.

    Character-length splitting, not `.from_tiktoken_encoder` (the course's pattern
    for long doc pages): docs/plan.md §1 sizes the corpus in characters
    ("~800-1200 chunks @ 500 chars"), and abstracts are short enough (~300-450
    tokens) that token-based 500-length chunks would rarely split them at all —
    that math only lands at ~250-300 chunks, not 800-1200.
    """
    if chunk_size == 0:
        return list(docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def _persist_path(embedding_model: str, chunk_strategy: str) -> Path:
    safe_chunk = chunk_strategy.replace("/", "-")
    return config.RESOURCES_DIR / f"arxiv_{embedding_model}_{safe_chunk}.parquet"


def abstract_digest(text: str) -> str:
    """sha1 of the abstract with whitespace collapsed: a line break vs a space does not change it."""
    return hashlib.sha1(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def build_corpus_manifest(papers: list[Document]) -> dict:
    return {
        "snapshot_date": config.CORPUS_SNAPSHOT_DATE.isoformat(),
        "category": config.ARXIV_CATEGORY,
        "n_papers": len(papers),
        "papers": [
            {
                "arxiv_id": p.metadata["arxiv_id"],
                "published": p.metadata["published"],
                "abstract_sha1": abstract_digest(p.page_content),
            }
            for p in papers
        ],
    }


def write_corpus_manifest(papers: list[Document], path: Path = config.CORPUS_MANIFEST_PATH) -> None:
    path.write_text(json.dumps(build_corpus_manifest(papers), indent=1) + "\n")


def load_corpus_manifest(path: Path = config.CORPUS_MANIFEST_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def fetch_arxiv_papers_by_id(ids: list[str], batch_size: int = 100) -> list[Document]:
    """Fetch exact papers by arXiv id (cheap and deterministic, unlike scanning by date)."""
    client = arxiv.Client()
    docs: list[Document] = []
    for i in range(0, len(ids), batch_size):
        docs.extend(_result_to_document(r) for r in client.results(arxiv.Search(id_list=ids[i : i + batch_size])))
    return docs


def papers_from_manifest(
    manifest: dict, fetch: Callable[[list[str]], list[Document]] = fetch_arxiv_papers_by_id
) -> list[Document]:
    """Rebuild the frozen corpus from the manifest and VERIFY it: every id must come back and every
    abstract must match its recorded digest (an abstract revised after the freeze fails loudly)."""
    entries = manifest["papers"]
    fetched = {d.metadata["arxiv_id"]: d for d in fetch([e["arxiv_id"] for e in entries])}
    missing = [e["arxiv_id"] for e in entries if e["arxiv_id"] not in fetched]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest papers not returned by arXiv, e.g. {missing[:3]}")
    changed = [e["arxiv_id"] for e in entries if abstract_digest(fetched[e["arxiv_id"]].page_content) != e["abstract_sha1"]]
    if changed:
        raise RuntimeError(f"{len(changed)} abstracts differ from the frozen corpus, e.g. {changed[:3]}")
    return [fetched[e["arxiv_id"]] for e in entries]


def _read_cache(persist_path: Path) -> list[Document]:
    """Read every record of a cached store back as Documents (row order preserved)."""
    import pandas as pd

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


def papers_from_chunks(chunks: list[Document]) -> list[Document]:
    """Reassemble one Document per paper (the whole abstract) from its ordered chunks.

    chunk_documents() drops the separator at each cut, so joining the chunks with a single space
    restores the abstract's words and sentences exactly. The only loss is whitespace: where the
    original abstract had a line break at a chunk boundary it becomes a space (10 of the 250 cached
    papers; re-chunking the result gives 840 chunks instead of 843). Gold labels are paper ids, so
    this does not affect evaluation.
    """
    by_paper: dict[str, list[Document]] = {}
    for chunk in chunks:
        by_paper.setdefault(chunk.metadata["arxiv_id"], []).append(chunk)
    return [
        Document(page_content=" ".join(c.page_content for c in group), metadata=dict(group[0].metadata))
        for group in by_paper.values()
    ]


def _papers_from_existing_cache() -> list[Document] | None:
    """Offline fallback: derive the corpus from a cache already built (see papers_from_chunks)."""
    for cache in sorted(config.RESOURCES_DIR.glob("arxiv_*_*.parquet")):
        logger.info("Deriving the corpus from the existing cache %s", cache.name)
        return papers_from_chunks(_read_cache(cache))
    return None


def load_frozen_corpus() -> list[Document]:
    """The frozen corpus (ADR-002), in order of preference:

    1. the git-versioned manifest, fetched by id and verified against the recorded digests;
    2. an existing parquet cache (offline; also bootstraps the manifest);
    3. a date scan of arXiv (cold start only: it gets more expensive every day, because the snapshot
       pin must skip every newer paper, ~128/day) which also writes the manifest.
    """
    manifest = load_corpus_manifest()
    if manifest is not None:
        logger.info("Rebuilding the frozen corpus from the manifest (%d papers)", manifest["n_papers"])
        return papers_from_manifest(manifest)
    papers = _papers_from_existing_cache()
    if papers is None:
        papers = fetch_arxiv_corpus()
    write_corpus_manifest(papers)
    logger.info("Wrote the corpus manifest to %s", config.CORPUS_MANIFEST_PATH)
    return papers


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

    logger.info("No cache at %s — building the index", persist_path)
    papers = load_frozen_corpus()
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
    persist_path = _persist_path(run_config.embedding_model, run_config.chunk_strategy)
    if not persist_path.exists():
        raise RuntimeError(f"No cached store at {persist_path} — build it via get_vector_store first")
    return _read_cache(persist_path)
