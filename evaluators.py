"""Evaluator catalog — CKPT-0.6 (docs/roadmap/ckpt-0.6-plan.md §3).

Surface 1 (retrieval, code-based, reference-based): recall_at_k, hit_rate, mrr,
precision_at_k.

Anti-pooling-bias semantics (ADR-001/lesson L1): metrics count GOLD labels only;
retrieved docs outside the gold are unjudged, never auto-irrelevant. Examples with
empty gold or `retrieval_gold=pending-adjudication` return None (skipped), so a
miscomputed number can never masquerade as a measurement.
"""

from __future__ import annotations

from typing import Any


def _gold_chunk_ids(reference_outputs: dict[str, Any]) -> set[str]:
    return set(reference_outputs.get("gold_chunk_ids") or [])


def _gold_arxiv_ids(reference_outputs: dict[str, Any]) -> set[str]:
    return set(reference_outputs.get("gold_arxiv_ids") or [])


def is_retrieval_evaluable(reference_outputs: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Only examples with a judged retrieval gold participate in retrieval metrics.

    Skips: zero-gold slices by design (unanswerable/stale) and open-topic multi-doc
    still pending the EXP-0 mini-pooling adjudication.
    """
    if metadata.get("retrieval_gold") == "pending-adjudication":
        return False
    return bool(_gold_chunk_ids(reference_outputs) or _gold_arxiv_ids(reference_outputs))


def recall_at_k(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
) -> float | None:
    """Fraction of gold documents present in the retrieved top-k (FP2)."""
    gold = _gold_chunk_ids(reference_outputs) or _gold_arxiv_ids(reference_outputs)
    if not gold:
        return None
    retrieved = set(retrieved_chunk_ids) | set(retrieved_arxiv_ids)
    return len(gold & retrieved) / len(gold)


def hit_rate(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
) -> float | None:
    """1 if at least one gold document is in the retrieved top-k, else 0."""
    gold = _gold_chunk_ids(reference_outputs) or _gold_arxiv_ids(reference_outputs)
    if not gold:
        return None
    retrieved = set(retrieved_chunk_ids) | set(retrieved_arxiv_ids)
    return 1.0 if gold & retrieved else 0.0


def mrr(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
) -> float | None:
    """1/rank of the first gold document in the retrieved list (0 if absent)."""
    gold = _gold_chunk_ids(reference_outputs) or _gold_arxiv_ids(reference_outputs)
    if not gold:
        return None
    for rank, rid in enumerate([*retrieved_chunk_ids, *retrieved_arxiv_ids], 1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def precision_at_k(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
) -> float | None:
    """Fraction of the retrieved top-k that is gold-labeled (FP3 noise).

    Meaningful only where the gold is the complete relevant set (or adjudicated);
    by design we currently call it only on examples with non-empty gold.
    """
    gold = _gold_chunk_ids(reference_outputs) or _gold_arxiv_ids(reference_outputs)
    if not gold:
        return None
    retrieved = [*retrieved_chunk_ids] if retrieved_chunk_ids else [*retrieved_arxiv_ids]
    if not retrieved:
        return 0.0
    gold_ids = gold
    return sum(1 for rid in retrieved if rid in gold_ids) / len(retrieved)
