"""Offline tests for the retrieval-unit configuration (ADR-005): whole abstract vs 500/0 chunks."""

import pytest
from langchain_core.documents import Document

import config
import utils

ABSTRACT = " ".join(f"Sentence number {i} talks about retrieval and agents." for i in range(40))


def _paper(arxiv_id="2609.00001v1"):
    return Document(page_content=ABSTRACT, metadata={"arxiv_id": arxiv_id, "title": "T"})


def test_whole_abstract_is_one_record_per_paper():
    papers = [_paper("2609.00001v1"), _paper("2609.00002v1")]
    records = utils.chunk_documents(papers, 0, 0)
    assert len(records) == 2
    assert [r.page_content for r in records] == [ABSTRACT, ABSTRACT]  # untouched, not split
    assert [r.metadata["arxiv_id"] for r in records] == ["2609.00001v1", "2609.00002v1"]


def test_chunking_500_splits_a_long_abstract_and_keeps_metadata():
    chunks = utils.chunk_documents([_paper()], 500, 0)
    assert len(chunks) > 1
    assert all(c.metadata["arxiv_id"] == "2609.00001v1" for c in chunks)
    assert all(len(c.page_content) <= 500 for c in chunks)


def test_baseline_is_the_whole_abstract_and_the_arm_is_500_0():
    assert config.BASELINE.chunk_size == 0 and config.BASELINE.chunk_strategy == "abstract"
    assert config.CHUNKED_500_0.chunk_strategy == "500/0"
    assert config.BASELINE.app_version != config.CHUNKED_500_0.app_version


def test_overlap_without_chunking_is_rejected():
    with pytest.raises(ValueError):
        config.RunConfig(app_version="x", chunk_size=0, chunk_overlap=50)


def test_each_strategy_has_its_own_cache_file():
    abstract = utils._persist_path("text-embedding-3-small", config.BASELINE.chunk_strategy)
    chunked = utils._persist_path("text-embedding-3-small", config.CHUNKED_500_0.chunk_strategy)
    assert abstract.name == "arxiv_text-embedding-3-small_abstract.parquet"
    assert chunked.name == "arxiv_text-embedding-3-small_500-0.parquet"


def test_whole_abstract_chunk_id_is_the_paper_prefix_plus_hash():
    # chunk id == arxiv_id:sha1(text): with one record per paper the prefix IS the paper id (ADR-006).
    record = utils.chunk_documents([_paper()], 0, 0)[0]
    assert utils.chunk_id(record).split(":", 1)[0] == "2609.00001v1"


def test_papers_from_chunks_restores_the_abstract_and_keeps_order_and_metadata():
    chunks = utils.chunk_documents([_paper("2609.00001v1"), _paper("2609.00002v1")], 500, 0)
    papers = utils.papers_from_chunks(chunks)
    assert [p.metadata["arxiv_id"] for p in papers] == ["2609.00001v1", "2609.00002v1"]
    assert [p.page_content for p in papers] == [ABSTRACT, ABSTRACT]  # single-spaced text: lossless
    # re-chunking the reassembled papers reproduces the same chunk ids
    assert [utils.chunk_id(c) for c in utils.chunk_documents(papers, 500, 0)] == [
        utils.chunk_id(c) for c in chunks
    ]
