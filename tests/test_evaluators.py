"""Offline unit tests for the retrieval evaluators (CKPT-0.6.1).

No network, no LLM — controlled example dicts. Contract under test:
docs/roadmap/ckpt-0.6-plan.md §3 (anti-pooling-bias semantics).
"""

import json
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


# --- format_validator (0.6.2) -------------------------------------------

import build_golden_dataset as gds


def test_format_golds_dogfood_all_pass():
    """The 10 curated format golds must pass the validator themselves."""
    for e in gds.format_examples():
        r = ev.format_validator(e["inputs"]["question"], e["outputs"]["answer"])
        assert r["score"] == 1, (e["inputs"]["question"][:60], r)


def test_format_json_wrong_key_fails():
    q = 'Return a JSON object with exactly the keys `paper_id`, `title`, `core_problem`, `approach` describing paper X. No other text.'
    r = ev.format_validator(q, json.dumps({"paper_id": 1, "title": "t"}))
    assert r["score"] == 0 and "keys" in r["reason"]


def test_format_table_missing_separator_fails():
    q = "In a markdown table with columns `Paper` and `Phenomenon`, compare A and B."
    r = ev.format_validator(q, "| Paper | Phenomenon |\n| a | b |")
    assert r["score"] == 0 and "separator" in r["reason"]


def test_format_bullets_over_words_fails():
    q = "In exactly 3 bullet points of at most 12 words each, summarize X."
    bad = "- one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    r = ev.format_validator(q, bad + "\n- ok\n- ok2")
    assert r["score"] == 0 and "words" in r["reason"]


def test_format_csv_order_fails():
    q = "Reply as CSV with header `arxiv_id,title,published` listing papers 2609.18471v1, 2609.18515v1 and 2609.18820v2, in that order. No other text."
    bad = "arxiv_id,title,published\n2609.18515v1,B,2026-09-16\n2609.18471v1,A,2026-09-16\n2609.18820v2,C,2026-09-16"
    r = ev.format_validator(q, bad)
    assert r["score"] == 0 and "order" in r["reason"]


def test_format_sentences_citation():
    q = "Answer in exactly 2 sentences, ending the second sentence with the citation [2609.18460v1]: how can a local deviation become collective loss of control?"
    good = "A deviation seeds contagion. Failure emerges when spread outpaces correction [2609.18460v1]."
    assert ev.format_validator(q, good)["score"] == 1
    assert ev.format_validator(q, "One sentence only [2609.18460v1].")["score"] == 0
    assert ev.format_validator(q, "Two sentences here. But no citation at end.")["score"] == 0


def test_format_unknown_instruction():
    r = ev.format_validator("Just answer plainly.", "ok")
    assert r["score"] == 0 and "no known" in r["reason"]


# --- f1_summary_evaluator (0.6.2) ----------------------------------------

def _scores(out):
    return {r["key"]: r["score"] for r in out["results"]}


def test_f1_summary_basic():
    outputs = [{"answer": "the sky is blue"}, {"answer": "I don't know."}]
    refs = [
        {"answer": "the sky is blue", "should_abstain": False},
        {"answer": "I don't know.", "should_abstain": True},
    ]
    scores = _scores(ev.f1_summary_evaluator(outputs, refs))
    assert scores["f1_summary"] == 1.0
    assert scores["abstention_accuracy"] == 1.0


def test_f1_summary_detects_wrong_abstention():
    outputs = [{"answer": "here is my confident answer"}]
    refs = [{"answer": "gold", "should_abstain": True}]
    assert _scores(ev.f1_summary_evaluator(outputs, refs))["abstention_accuracy"] == 0.0


def test_f1_summary_omits_abstention_without_labels():
    scores = _scores(ev.f1_summary_evaluator([{"answer": "a b"}], [{"answer": "a b"}]))
    assert set(scores) == {"f1_summary"}


def test_f1_summary_contract_with_langsmith():
    """Regression for P4: langsmith builds EvaluationResult(extra="forbid") per entry and
    only accepts summary evaluators whose args are runs/examples/inputs/outputs/
    reference_outputs. The old shape (extra top-level key, single `results` arg) failed both."""
    from types import SimpleNamespace

    from langsmith.evaluation.evaluator import EvaluationResult, _normalize_summary_evaluator

    wrapped = _normalize_summary_evaluator(ev.f1_summary_evaluator)  # raises on bad signature
    runs = [SimpleNamespace(outputs={"answer": "I don't know."})]
    examples = [SimpleNamespace(inputs={}, outputs={"answer": "I don't know.", "should_abstain": True})]
    out = wrapped(runs, examples)
    results = out["results"] if isinstance(out, dict) else out.results
    parsed = [EvaluationResult(**r) for r in results]  # raises on any extra key
    assert {p.key for p in parsed} == {"f1_summary", "abstention_accuracy"}


# --- paper-level gold (CKPT-0.6.1b P1-P3) ---------------------------------

CHUNKS = ["p1:aaa", "p2:bbb", "p3:ccc", "p4:ddd", "p5:eee"]
PAPERS = ["p1", "p2", "p3", "p4", "p5"]


def test_mrr_paper_gold_not_shifted_by_chunk_list():
    # Regression: ranks used to be offset by len(chunk_ids) -> 1/6 instead of 1.
    assert ev.mrr(CHUNKS, PAPERS, ref(arxiv_ids=["p1"])) == 1.0
    assert math.isclose(ev.mrr(CHUNKS, PAPERS, ref(arxiv_ids=["p3"])), 1 / 3)


def test_precision_paper_gold_with_chunk_ids_present():
    # Regression: chunk ids were compared against paper gold -> always 0.0.
    assert math.isclose(ev.precision_at_k(CHUNKS, PAPERS, ref(arxiv_ids=["p1"])), 1 / 5)
    assert math.isclose(ev.precision_at_k(CHUNKS, PAPERS, ref(arxiv_ids=["p1", "p2"])), 2 / 5)


def test_paper_level_rank_dedups_duplicate_chunks_keeping_order():
    # decisions.md 2026-09-27: top-5 can hold 3 chunks of one paper.
    dup_chunks = ["pA:1", "pA:2", "pA:3", "pB:4", "pC:5"]
    assert ev.mrr(dup_chunks, [], ref(arxiv_ids=["pB"])) == 1 / 2  # rank among distinct papers
    assert math.isclose(ev.precision_at_k(dup_chunks, [], ref(arxiv_ids=["pB"])), 1 / 3)


def test_paper_ids_derived_from_chunk_ids_when_not_provided():
    assert ev.recall_at_k(CHUNKS, [], ref(arxiv_ids=["p2", "p9"])) == 0.5
    assert ev.hit_rate(CHUNKS, [], ref(arxiv_ids=["p9"])) == 0.0


def test_paper_level_default_prefers_explicit_paper_gold_over_chunk_gold():
    # roadmap D-1: the paper is the primary level; the chunk gold is diagnostic only.
    r = {"gold_chunk_ids": ["p2:bbb"], "gold_arxiv_ids": ["p1"]}
    assert ev.mrr(CHUNKS, PAPERS, r) == 1.0
    assert ev.mrr(CHUNKS, PAPERS, r, level="chunk") == 1 / 2


def test_chunk_level_ignores_arxiv_list():
    assert ev.mrr(["x:1", "a:1"], ["a", "x"], ref(chunk_ids=["a:1"]), level="chunk") == 1 / 2


def test_empty_retrieval_with_gold():
    assert ev.mrr([], [], ref(arxiv_ids=["p1"])) == 0.0
    assert ev.precision_at_k([], [], ref(arxiv_ids=["p1"])) == 0.0
    assert ev.recall_at_k([], [], ref(arxiv_ids=["p1"])) == 0.0


def test_arxiv_id_derivation_matches_utils_chunk_id():
    from langchain_core.documents import Document

    import utils

    doc = Document(page_content="some chunk text", metadata={"arxiv_id": "2609.12345v1"})
    assert ev._arxiv_id_of_chunk(utils.chunk_id(doc)) == "2609.12345v1"


# --- D-1 / D-2 / P5: paper level by default, chunk level as diagnostic ----------

# Real case (golden set, "checkpoint handoff"): the gold chunk (2nd chunk of the paper)
# came 2nd; another chunk of the SAME paper came 1st.
HANDOFF_CHUNKS = [
    "2609.19636v1:7765a13e5f1b",  # same paper, sibling chunk
    "2609.19636v1:7e548892cc14",  # gold chunk (generated the question)
    "2609.18460v1:061c23fc26f2",
    "2609.19892v1:facb0f33496f",
    "2609.18864v1:1f1c23f8779a",
]
HANDOFF_REF = {"gold_chunk_ids": ["2609.19636v1:7e548892cc14"], "gold_arxiv_ids": []}


def test_chunk_gold_is_scored_at_paper_level_by_default():
    assert ev.mrr(HANDOFF_CHUNKS, [], HANDOFF_REF) == 1.0  # sibling in 1st = right paper
    assert ev.hit_rate(HANDOFF_CHUNKS, [], HANDOFF_REF, k=1) == 1.0


def test_exact_chunk_is_a_diagnostic_level():
    assert ev.mrr(HANDOFF_CHUNKS, [], HANDOFF_REF, level="chunk") == 1 / 2
    assert ev.hit_rate(HANDOFF_CHUNKS, [], HANDOFF_REF, level="chunk", k=1) == 0.0
    assert ev.hit_rate(HANDOFF_CHUNKS, [], HANDOFF_REF, level="chunk", k=2) == 1.0


def test_chunk_level_is_undefined_without_chunk_gold():
    assert ev.mrr(CHUNKS, PAPERS, ref(arxiv_ids=["p1"]), level="chunk") is None


def test_unknown_level_raises():
    import pytest

    with pytest.raises(ValueError):
        ev.hit_rate(CHUNKS, PAPERS, ref(arxiv_ids=["p1"]), level="sentence")


def test_k_cuts_the_raw_retrieved_list_before_dedup():
    dup = ["pA:1", "pA:2", "pB:3", "pC:4", "pD:5"]
    # top-2 raw = [pA, pA] -> 1 distinct paper; pB is not in the top-2
    assert ev.hit_rate(dup, [], ref(arxiv_ids=["pB"]), k=2) == 0.0
    assert ev.hit_rate(dup, [], ref(arxiv_ids=["pB"]), k=3) == 1.0
    assert ev.mrr(dup, [], ref(arxiv_ids=["pB"]), k=3) == 1 / 2  # rank among distinct papers


def test_hit_at_ks_curve():
    curve = ev.hit_at_ks(CHUNKS, PAPERS, ref(arxiv_ids=["p3"]), ks=(1, 3, 5))
    assert curve == {1: 0.0, 3: 1.0, 5: 1.0}
    assert ev.hit_at_ks(CHUNKS, PAPERS, ref(), ks=(1,)) == {1: None}


def test_single_gold_paper_makes_recall_equal_hit():
    # why D-2 reports hit+MRR (not recall) for 1 gold paper
    for retrieved in (PAPERS, ["x", "y"], ["p1", "x"]):
        r = ref(arxiv_ids=["p1"])
        assert ev.recall_at_k([], retrieved, r) == ev.hit_rate([], retrieved, r)


def test_multi_gold_makes_recall_differ_from_hit():
    r = ref(arxiv_ids=["p1", "p2"])
    assert ev.hit_rate([], ["p1", "x"], r) == 1.0
    assert ev.recall_at_k([], ["p1", "x"], r) == 0.5


def test_primary_retrieval_metrics_policy():
    assert ev.primary_retrieval_metrics(HANDOFF_REF) == ["hit", "mrr"]  # chunk gold -> 1 paper
    assert ev.primary_retrieval_metrics(ref(arxiv_ids=["p1"])) == ["hit", "mrr"]
    assert ev.primary_retrieval_metrics(ref(arxiv_ids=["p1", "p2"])) == ["recall", "hit", "mrr"]
    assert ev.primary_retrieval_metrics(ref()) == []  # unanswerable / stale / open-topic
