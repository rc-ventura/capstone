from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
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


def _arxiv_id_of_chunk(chunk_id: str) -> str:
    # utils.chunk_id() == f"{arxiv_id}:{sha1[:12]}"; kept as a string split so this
    # module stays free of utils/config imports (config needs API keys at import time).
    # tests/test_evaluators.py pins this against utils.chunk_id.
    return chunk_id.split(":", 1)[0]


def _ranked_and_gold(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
) -> tuple[list[str], set[str]] | None:
    """Pick ONE comparison level from the gold, then rank retrieved items at that level.

    - gold_chunk_ids present  -> chunk level: rank = order of retrieved chunk ids.
    - else gold_arxiv_ids     -> paper level: rank = first occurrence of each paper
      in the retrieved list (duplicate chunks of one paper collapse, order kept).
      Papers come from `retrieved_arxiv_ids`, or are derived from the chunk ids.
    - neither                 -> None (no judged gold; metric undefined).

    Never mixes chunk ids and arxiv ids in one list: that inflated MRR ranks by k and
    zeroed precision for paper-level gold (CKPT-0.6.1b, P1-P3).
    """
    gold_chunks = _gold_chunk_ids(reference_outputs)
    if gold_chunks:
        return list(dict.fromkeys(retrieved_chunk_ids)), gold_chunks
    gold_papers = _gold_arxiv_ids(reference_outputs)
    if gold_papers:
        papers = retrieved_arxiv_ids or [_arxiv_id_of_chunk(c) for c in retrieved_chunk_ids]
        return list(dict.fromkeys(papers)), gold_papers
    return None


def recall_at_k(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
) -> float | None:
    """Fraction of gold documents present in the retrieved top-k (FP2)."""
    level = _ranked_and_gold(retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs)
    if level is None:
        return None
    ranked, gold = level
    return len(gold & set(ranked)) / len(gold)


def hit_rate(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
) -> float | None:
    """1 if at least one gold document is in the retrieved top-k, else 0."""
    level = _ranked_and_gold(retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs)
    if level is None:
        return None
    ranked, gold = level
    return 1.0 if gold & set(ranked) else 0.0


def mrr(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
) -> float | None:
    """1/rank of the first gold document in the retrieved list (0 if absent)."""
    level = _ranked_and_gold(retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs)
    if level is None:
        return None
    ranked, gold = level
    for rank, rid in enumerate(ranked, 1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def precision_at_k(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
) -> float | None:
    """Fraction of the retrieved top-k (at the gold's level) that is gold-labeled (FP3 noise).

    Meaningful only where the gold is the complete relevant set (or adjudicated);
    by design we currently call it only on examples with non-empty gold. At paper
    level the denominator is the number of DISTINCT papers retrieved. Whether this
    metric should run at all for single-chunk gold is open (CKPT-0.6.1b, P6).
    """
    level = _ranked_and_gold(retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs)
    if level is None:
        return None
    ranked, gold = level
    if not ranked:
        return 0.0
    return sum(1 for rid in ranked if rid in gold) / len(ranked)


# =====================================================================
# Surface 2 helpers — format_validator (FP5, plan §4; P1 code+reference)
# =====================================================================

def _check_markdown_table(question: str, answer: str) -> tuple[bool, str]:
    m = re.search(r"columns? `?(\|[^`]*\|)`?", question)
    if not m:
        # "with columns A and B" style: columns named in backticks individually
        cols = re.findall(r"`([A-Za-z ]{2,})`", question)
        required = cols if cols else None
    else:
        required = [c.strip() for c in m.group(1).strip("|").split("|")]
    lines = [l for l in answer.strip().splitlines() if l.strip()]
    if len(lines) < 2 or not lines[0].lstrip().startswith("|"):
        return False, "no markdown table found"
    if not re.fullmatch(r"\|[\s\-:|]+\|", lines[1]):
        return False, "missing markdown separator row"
    if required:
        header_ok = all(c in lines[0] for c in required)
        if not header_ok:
            return False, f"header {lines[0]!r} missing required columns {required}"
    n_rows = len(lines) - 2
    n_cols = lines[0].count("|") - 1
    if any(l.count("|") - 1 != n_cols for l in lines[2:]):
        return False, "row column count differs from header"
    n_expected = re.search(r"(three|two|four) (?:papers|rows|objects)", question)
    if n_expected:
        want = {"two": 2, "three": 3, "four": 4}[n_expected.group(1)]
        if n_rows != want:
            return False, f"expected {want} data rows, got {n_rows}"
    return True, "ok"


def _check_json(question: str, answer: str) -> tuple[bool, str]:
    try:
        obj = json.loads(answer)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    keys = re.findall(r"`([a-z_]+)`", question)
    if "JSON array" in question:
        if not isinstance(obj, list):
            return False, "expected a JSON array"
        n = re.search(r"with (\w+) objects", question)
        if n and len(obj) != {"two": 2, "three": 3}.get(n.group(1), len(obj)):
            return False, f"expected {n.group(1)} objects, got {len(obj)}"
        if keys and any(set(o) != set(keys) for o in obj):
            return False, "object keys differ from required set"
    else:
        if not isinstance(obj, dict):
            return False, "expected a JSON object"
        if keys and set(obj) != set(keys):
            return False, f"keys {sorted(obj)} != required {sorted(keys)}"
    if "No other text" in question and answer.strip() != answer.strip().lstrip("{["):
        pass  # json.loads already enforces nothing extra parsed
    return True, "ok"


def _check_csv(question: str, answer: str) -> tuple[bool, str]:
    m = re.search(r"header `([^`]+)`", question)
    if not m:
        return False, "no header spec in question"
    rows = list(csv.reader(io.StringIO(answer.strip())))
    if not rows or rows[0] != m.group(1).split(","):
        return False, f"header mismatch: {rows[0] if rows else None}"
    ids = re.findall(r"2609\.\d{5}v\d", question)
    body = rows[1:]
    if ids and [r[0] for r in body] != ids:
        return False, "ids missing or out of order"
    return True, "ok"


def _check_yaml(question: str, answer: str) -> tuple[bool, str]:
    keys = re.findall(r"`([a-z_]+)`", question)
    present = {l.split(":")[0].strip() for l in answer.strip().splitlines() if ":" in l}
    if set(keys) - present:
        return False, f"missing keys {set(keys) - present}"
    return True, "ok"


def _check_bullets(question: str, answer: str) -> tuple[bool, str]:
    n = re.search(r"exactly (\d+) bullet points?", question)
    bullets = [l for l in answer.strip().splitlines() if l.strip().startswith("- ")]
    if n and len(bullets) != int(n.group(1)):
        return False, f"expected {n.group(1)} bullets, got {len(bullets)}"
    w = re.search(r"at most (\d+) words", question)
    if w:
        too_long = [b for b in bullets if len(b[2:].split()) > int(w.group(1))]
        if too_long:
            return False, f"{len(too_long)} bullet(s) over {w.group(1)} words"
    return True, "ok"


def _check_sentences_citation(question: str, answer: str) -> tuple[bool, str]:
    n = re.search(r"exactly (\d+) sentences?", question)
    cid = re.search(r"\[([0-9.]+v\d)\]", question)
    body = answer.strip()
    if cid and not body.endswith(f"[{cid.group(1)}]."):
        return False, f"does not end with citation [{cid.group(1)}]"
    if cid:
        body = body[: -len(f"[{cid.group(1)}]")]
    sentences = [s for s in re.split(r"(?<=[.!?]) ", body.strip()) if s.strip()]
    if n and len(sentences) != int(n.group(1)):
        return False, f"expected {n.group(1)} sentences, got {len(sentences)}"
    return True, "ok"


def format_validator(question: str, answer: str) -> dict[str, Any]:
    """Literal instruction-following check (FP5). Dispatches on the format the
    question demands; deterministic code, zero LLM cost."""
    if "markdown table" in question:
        ok, why = _check_markdown_table(question, answer)
    elif "JSON array" in question or "JSON object" in question:
        ok, why = _check_json(question, answer)
    elif "as CSV" in question:
        ok, why = _check_csv(question, answer)
    elif "As YAML" in question or "as YAML" in question:
        ok, why = _check_yaml(question, answer)
    elif "bullet point" in question:
        ok, why = _check_bullets(question, answer)
    elif "sentences" in question and "citation" in question:
        ok, why = _check_sentences_citation(question, answer)
    else:
        return {"score": 0, "reason": "no known format instruction detected"}
    return {"score": 1 if ok else 0, "reason": why, "key": "format_valid"}


# =====================================================================
# f1_summary_evaluator (P5 course pattern, adapted: token F1 vs gold)
# =====================================================================

def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _token_f1(pred: str, ref: str) -> float:
    p, r = Counter(_tokens(pred)), Counter(_tokens(ref))
    overlap = sum((p & r).values())
    if not overlap:
        return 0.0
    precision = overlap / max(sum(p.values()), 1)
    recall = overlap / max(sum(r.values()), 1)
    return 2 * precision * recall / (precision + recall)


def f1_summary_evaluator(
    outputs: list[dict[str, Any]], reference_outputs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summary evaluator: aggregated token-F1 of produced answers vs gold answers
    across the whole run (substantive-coverage proxy), plus abstention behavior
    vs should_abstain (FP1 guard).

    Signature follows langsmith's summary-evaluator contract (course
    module_2/summary_evaluators.ipynb): parallel lists `outputs` (target outputs,
    expected to carry "answer") and `reference_outputs` (golden-set outputs).
    Returns {"results": [{"key", "score"}, ...]}: langsmith builds each entry as an
    EvaluationResult (extra="forbid"), so any extra top-level key makes the whole
    evaluator fail (the exception is logged and every metric is dropped).
    """
    f1s, abstain_hits, abstain_total = [], 0, 0
    for run_out, ref in zip(outputs, reference_outputs):
        answer = run_out.get("answer", "")
        gold = ref.get("answer", "")
        if gold:
            f1s.append(_token_f1(answer, gold))
        wants_abstain = ref.get("should_abstain")
        if wants_abstain is not None:
            abstain_total += 1
            did_abstain = answer.strip().startswith(("I don't know", "No papers found"))
            if did_abstain == wants_abstain:
                abstain_hits += 1
    results = [{"key": "f1_summary", "score": sum(f1s) / len(f1s) if f1s else 0.0}]
    if abstain_total:
        results.append({"key": "abstention_accuracy", "score": abstain_hits / abstain_total})
    return {"results": results}
