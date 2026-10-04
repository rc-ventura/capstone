"""Cross-check the retrieval evaluators against ir-measures (CKPT-0.6.1b; roadmap L-3).

ir-measures wraps trec_eval, the community's reference implementation. It is a dev-only
oracle: our evaluators run once per example inside LangSmith, where building qrels/run
objects per call would be pure overhead. Here, randomized cases confirm that our four
formulas (recall, hit/success, reciprocal rank, precision) agree with the reference,
including the k cutoff and the paper-level gold derived from chunk ids.

Semantics worth knowing: ir-measures ranks by score, so every retrieved list gets strictly
descending scores; P@k divides by the cutoff k while ours divides by the number of DISTINCT
papers in the top-k, so lists are de-duplicated here and both agree.
"""

import random

import ir_measures
import pytest
from ir_measures import P, R, RR, Success

import evaluators as ev

N_CASES = 300
UNIVERSE = [f"p{i}" for i in range(12)]


def _oracle(ranked: list[str], gold: set[str], k: int | None = None) -> dict[str, float]:
    k = k or len(ranked)
    qrels = [ir_measures.Qrel("q", g, 1) for g in gold]
    run = [ir_measures.ScoredDoc("q", d, float(len(ranked) - i)) for i, d in enumerate(ranked)]
    scores = ir_measures.calc_aggregate([R @ k, Success @ k, RR @ k, P @ k], qrels, run)
    return {
        "recall": scores[R @ k],
        "hit": scores[Success @ k],
        "mrr": scores[RR @ k],
        "precision": scores[P @ k],
    }


def _ours(
    chunks: list[str],
    papers: list[str],
    reference: dict,
    level: str = "paper",
    k: int | None = None,
) -> dict[str, float]:
    return {
        "recall": ev.recall_at_k(chunks, papers, reference, level=level, k=k),
        "hit": ev.hit_rate(chunks, papers, reference, level=level, k=k),
        "mrr": ev.mrr(chunks, papers, reference, level=level, k=k),
        "precision": ev.precision_at_k(chunks, papers, reference, level=level, k=k),
    }


def _assert_close(ours: dict[str, float], theirs: dict[str, float], case):
    for name, value in ours.items():
        assert value == pytest.approx(theirs[name]), (name, case)


def test_paper_level_matches_oracle():
    rng = random.Random(0)
    for _ in range(N_CASES):
        gold = set(rng.sample(UNIVERSE, rng.randint(1, 4)))
        # several chunks per paper, as in the 500/0 corpus: duplicates must collapse
        picked = [rng.choice(UNIVERSE) for _ in range(rng.randint(1, 8))]
        chunks = [f"{p}:{i:03d}" for i, p in enumerate(picked)]
        papers = list(picked)
        ranked = list(dict.fromkeys(picked))  # first-occurrence rank, distinct papers
        case = (picked, gold)
        reference = {"gold_arxiv_ids": sorted(gold)}
        _assert_close(_ours(chunks, papers, reference), _oracle(ranked, gold), case)
        # same result when only chunk ids are provided (papers derived from them)
        _assert_close(_ours(chunks, [], reference), _oracle(ranked, gold), case)


def test_chunk_level_matches_oracle():
    rng = random.Random(1)
    chunk_universe = [f"{p}:{h}" for p in UNIVERSE for h in ("a", "b")]
    for _ in range(N_CASES):
        gold = set(rng.sample(chunk_universe, rng.randint(1, 4)))
        ranked = rng.sample(chunk_universe, rng.randint(1, 8))
        papers = [c.split(":")[0] for c in ranked]
        case = (ranked, gold)
        _assert_close(
            _ours(ranked, papers, {"gold_chunk_ids": sorted(gold)}, level="chunk"),
            _oracle(ranked, gold),
            case,
        )


def test_k_cutoff_with_paper_gold_derived_from_chunk_ids_matches_oracle():
    """The real answerable/persona case: the dataset stores only gold_chunk_ids (source
    chunk), the evaluator scores at paper level and cuts the retriever's top-k."""
    rng = random.Random(2)
    for _ in range(N_CASES):
        n_ranked = rng.randint(1, 8)
        ranked = rng.sample(UNIVERSE, n_ranked)
        gold = set(rng.sample(UNIVERSE, rng.randint(1, 4)))
        k = rng.randint(1, n_ranked)
        chunks = [f"{p}:h{i}" for i, p in enumerate(ranked)]
        reference = {"gold_chunk_ids": [f"{g}:src" for g in sorted(gold)]}
        _assert_close(_ours(chunks, [], reference, k=k), _oracle(ranked, gold, k), (ranked, gold, k))


def test_hit_curve_matches_oracle_success_at_k():
    rng = random.Random(3)
    for _ in range(N_CASES):
        ranked = rng.sample(UNIVERSE, rng.randint(1, 8))
        gold = set(rng.sample(UNIVERSE, rng.randint(1, 3)))
        chunks = [f"{p}:h{i}" for i, p in enumerate(ranked)]
        curve = ev.hit_at_ks(chunks, [], {"gold_arxiv_ids": sorted(gold)}, ks=(1, 3, 5))
        for k, value in curve.items():
            assert value == pytest.approx(_oracle(ranked, gold, k)["hit"]), (ranked, gold, k)
