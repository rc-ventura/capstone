"""Offline unit tests for the retrieval evaluators (CKPT-0.6.1).

No network, no LLM — controlled example dicts. Contract under test:
docs/roadmap/ckpt-0.6-plan.md §3 (anti-pooling-bias semantics).
"""

import math

import evaluators as ev


def ref(chunk_ids=(), arxiv_ids=()):
    return {"gold_chunk_ids": list(chunk_ids), "gold_arxiv_ids": list(arxiv_ids)}


# --- recall_at_k ---------------------------------------------------------

def test_recall_full():
    assert ev.recall_at_k(["a", "b"], [], ref(chunk_ids=["a", "b"])) == 1.0


def test_recall_partial():
    assert ev.recall_at_k(["a"], [], ref(chunk_ids=["a", "b"])) == 0.5


def test_recall_none():
    assert ev.recall_at_k(["x"], [], ref(chunk_ids=["a"])) == 0.0


def test_recall_empty_gold_skips():
    assert ev.recall_at_k(["a"], [], ref()) is None


def test_recall_uses_arxiv_ids_for_multidoc():
    assert ev.recall_at_k([], ["p1", "p2"], ref(arxiv_ids=["p1", "p2"])) == 1.0


# --- hit_rate ------------------------------------------------------------

def test_hit_rate():
    assert ev.hit_rate(["x", "a"], [], ref(chunk_ids=["a"])) == 1.0
    assert ev.hit_rate(["x"], [], ref(chunk_ids=["a"])) == 0.0
    assert ev.hit_rate(["x"], [], ref()) is None


# --- mrr -----------------------------------------------------------------

def test_mrr_first_position():
    assert ev.mrr(["a", "x"], [], ref(chunk_ids=["a"])) == 1.0


def test_mrr_third_position():
    assert math.isclose(ev.mrr(["x", "y", "a"], [], ref(chunk_ids=["a"])), 1 / 3)


def test_mrr_absent():
    assert ev.mrr(["x"], [], ref(chunk_ids=["a"])) == 0.0


# --- precision_at_k ------------------------------------------------------

def test_precision_at_k():
    # 2 of 3 retrieved are gold
    assert math.isclose(ev.precision_at_k(["a", "b", "x"], [], ref(chunk_ids=["a", "b"])), 2 / 3)


def test_precision_empty_gold_skips():
    # unanswerable/stale: no gold labels -> precision is undefined by design
    assert ev.precision_at_k(["a"], [], ref()) is None


# --- is_retrieval_evaluable (skip rules) --------------------------------

def test_open_topic_skipped():
    md = {"retrieval_gold": "pending-adjudication"}
    assert not ev.is_retrieval_evaluable(ref(chunk_ids=["shouldnt-matter"]), md)


def test_zero_gold_skipped():
    assert not ev.is_retrieval_evaluable(ref(), {"slice": "unanswerable"})


def test_known_item_evaluable():
    assert ev.is_retrieval_evaluable(ref(arxiv_ids=["p1"]), {"slice": "multi-doc"})
