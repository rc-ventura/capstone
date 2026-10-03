"""Cross-check the retrieval evaluators against ranx (CKPT-0.6.1b P3).

ranx is a dev-only oracle: our evaluators run per example inside LangSmith, where
building ranx Qrels/Run objects per call would be pure overhead. Here, randomized
cases confirm our four formulas agree with the reference implementation.

ranx ranks by score, so each retrieved list gets strictly descending scores.
Lists are de-duplicated and k == len(list): ranx's precision@k divides by the
cutoff k, ours by the (deduplicated) list length.
"""

import random

import pytest
from ranx import Qrels, Run, evaluate

import evaluators as ev

N_CASES = 300
UNIVERSE = [f"p{i}" for i in range(12)]


def _ranx(ranked: list[str], gold: set[str]) -> dict[str, float]:
    k = len(ranked)
    qrels = Qrels.from_dict({"q": {g: 1 for g in gold}})
    run = Run.from_dict({"q": {rid: float(k - i) for i, rid in enumerate(ranked)}})
    metrics = [f"recall@{k}", f"hit_rate@{k}", f"mrr@{k}", f"precision@{k}"]
    scores = evaluate(qrels, run, metrics)
    return {
        "recall": scores[metrics[0]],
        "hit": scores[metrics[1]],
        "mrr": scores[metrics[2]],
        "precision": scores[metrics[3]],
    }


def _ours(chunks: list[str], papers: list[str], reference: dict) -> dict[str, float]:
    return {
        "recall": ev.recall_at_k(chunks, papers, reference),
        "hit": ev.hit_rate(chunks, papers, reference),
        "mrr": ev.mrr(chunks, papers, reference),
        "precision": ev.precision_at_k(chunks, papers, reference),
    }


def _assert_close(ours: dict[str, float], theirs: dict[str, float], case):
    for name, value in ours.items():
        assert value == pytest.approx(theirs[name]), (name, case)


def test_paper_level_matches_ranx():
    rng = random.Random(0)
    for _ in range(N_CASES):
        gold = set(rng.sample(UNIVERSE, rng.randint(1, 4)))
        # several chunks per paper, as in the 500/0 corpus: duplicates must collapse
        picked = [rng.choice(UNIVERSE) for _ in range(rng.randint(1, 8))]
        chunks = [f"{p}:{i:03d}" for i, p in enumerate(picked)]
        papers = list(picked)
        ranked = list(dict.fromkeys(picked))  # first-occurrence rank, distinct papers
        case = (picked, gold)
        _assert_close(
            _ours(chunks, papers, {"gold_arxiv_ids": sorted(gold)}), _ranx(ranked, gold), case
        )
        # same result when only chunk ids are provided (papers derived from them)
        _assert_close(
            _ours(chunks, [], {"gold_arxiv_ids": sorted(gold)}), _ranx(ranked, gold), case
        )


def test_chunk_level_matches_ranx():
    rng = random.Random(1)
    chunk_universe = [f"{p}:{h}" for p in UNIVERSE for h in ("a", "b")]
    for _ in range(N_CASES):
        gold = set(rng.sample(chunk_universe, rng.randint(1, 4)))
        ranked = rng.sample(chunk_universe, rng.randint(1, 8))
        papers = [c.split(":")[0] for c in ranked]
        case = (ranked, gold)
        _assert_close(
            _ours(ranked, papers, {"gold_chunk_ids": sorted(gold)}), _ranx(ranked, gold), case
        )
