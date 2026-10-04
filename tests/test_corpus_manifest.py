"""Offline tests for the frozen-corpus manifest (ADR-002 / ADR-005): ids + abstract digests."""

import pytest
from langchain_core.documents import Document

import utils


def _paper(arxiv_id: str, text: str = "An abstract about agents.\nIt has two lines.") -> Document:
    return Document(page_content=text, metadata={"arxiv_id": arxiv_id, "title": "T", "published": "2026-09-16"})


def _fetcher(papers: list[Document]):
    by_id = {p.metadata["arxiv_id"]: p for p in papers}
    return lambda ids: [by_id[i] for i in reversed(ids) if i in by_id]  # arXiv order is not the request order


def test_digest_ignores_whitespace_differences_but_not_content():
    assert utils.abstract_digest("a b\nc") == utils.abstract_digest("a   b c ")
    assert utils.abstract_digest("a b c") != utils.abstract_digest("a b d")


def test_manifest_roundtrip(tmp_path):
    papers = [_paper("2609.00001v1"), _paper("2609.00002v1")]
    path = tmp_path / "manifest.json"
    utils.write_corpus_manifest(papers, path)
    manifest = utils.load_corpus_manifest(path)
    assert manifest["n_papers"] == 2
    assert [e["arxiv_id"] for e in manifest["papers"]] == ["2609.00001v1", "2609.00002v1"]
    assert utils.load_corpus_manifest(tmp_path / "missing.json") is None


def test_rebuild_from_manifest_restores_manifest_order():
    papers = [_paper("2609.00001v1"), _paper("2609.00002v1"), _paper("2609.00003v1")]
    manifest = utils.build_corpus_manifest(papers)
    rebuilt = utils.papers_from_manifest(manifest, fetch=_fetcher(papers))
    assert [p.metadata["arxiv_id"] for p in rebuilt] == ["2609.00001v1", "2609.00002v1", "2609.00003v1"]


def test_rebuild_fails_loudly_when_a_paper_is_missing():
    papers = [_paper("2609.00001v1"), _paper("2609.00002v1")]
    manifest = utils.build_corpus_manifest(papers)
    with pytest.raises(RuntimeError, match="not returned"):
        utils.papers_from_manifest(manifest, fetch=_fetcher(papers[:1]))


def test_rebuild_fails_loudly_when_an_abstract_was_revised():
    papers = [_paper("2609.00001v1")]
    manifest = utils.build_corpus_manifest(papers)
    revised = [_paper("2609.00001v1", text="A revised abstract with different content.")]
    with pytest.raises(RuntimeError, match="differ from the frozen corpus"):
        utils.papers_from_manifest(manifest, fetch=_fetcher(revised))
