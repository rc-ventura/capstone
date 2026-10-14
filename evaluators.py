from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import re
import statistics
import struct
from collections import Counter
from collections.abc import Callable
from pathlib import Path
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


def is_gold_complete(reference_outputs: dict[str, Any], metadata: dict[str, Any] | None) -> bool:
    """True only where the gold is the COMPLETE relevant set, so "not in gold" may be read as an error.

    Derived from metadata that already exists (no stored flag, no dataset migration; ADR-001
    avoids two sources of truth), roadmap P6 / D-8:
    - multi-doc known-item: by definition "relevant = the papers named in the question";
    - review.state == "adjudicated": a human judged the retrieved candidates (mini-pooling),
      so the gold is complete with respect to that pool only (roadmap Hole@k, D-10).
    Everything else (answerable/persona 1-paper gold, open-topic pending, no metadata) is False:
    other papers may also answer, and nobody judged them.
    """
    metadata = metadata or {}
    if not (_gold_chunk_ids(reference_outputs) or _gold_arxiv_ids(reference_outputs)):
        return False
    if (metadata.get("review") or {}).get("state") == "adjudicated":
        return True
    return metadata.get("slice") == "multi-doc" and not metadata.get("open_topic")


def _arxiv_id_of_chunk(chunk_id: str) -> str:
    # utils.chunk_id() == f"{arxiv_id}:{sha1[:12]}"; kept as a string split so this
    # module stays free of utils/config imports (config needs API keys at import time).
    # tests/test_evaluators.py pins this against utils.chunk_id.
    return chunk_id.split(":", 1)[0]


def _gold_papers(reference_outputs: dict[str, Any]) -> set[str]:
    """Paper-level gold: gold_arxiv_ids, else the papers of gold_chunk_ids."""
    papers = _gold_arxiv_ids(reference_outputs)
    if papers:
        return papers
    return {_arxiv_id_of_chunk(c) for c in _gold_chunk_ids(reference_outputs)}


def _ranked_and_gold(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
    *,
    level: str = "paper",
    k: int | None = None,
) -> tuple[list[str], set[str]] | None:
    """Rank retrieved items at ONE comparison level and pair them with the gold at that level.

    - level="paper" (default, roadmap D-1): gold = gold_arxiv_ids, or the papers of
      gold_chunk_ids. Rank = first occurrence of each paper in the retrieved list
      (duplicate chunks of one paper collapse, order kept). Papers come from
      `retrieved_arxiv_ids`, or are derived from the chunk ids.
    - level="chunk" (diagnostic: "did we retrieve the exact source chunk?"): gold =
      gold_chunk_ids only. Meaningless across chunking configs: chunk ids hash the text.
    - no gold at the requested level -> None (metric undefined; skipped).

    `k` cuts the RAW retrieved list (the retriever's top-k) before de-duplication.
    Never mixes chunk ids and arxiv ids in one list (CKPT-0.6.1b P1-P3).
    """
    if level not in ("paper", "chunk"):
        raise ValueError(f"level must be 'paper' or 'chunk', got {level!r}")
    if level == "chunk":
        gold = _gold_chunk_ids(reference_outputs)
        raw = list(retrieved_chunk_ids)
    else:
        gold = _gold_papers(reference_outputs)
        raw = list(retrieved_arxiv_ids or [_arxiv_id_of_chunk(c) for c in retrieved_chunk_ids])
    if not gold:
        return None
    if k is not None:
        raw = raw[:k]
    return list(dict.fromkeys(raw)), gold


def recall_at_k(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
    *,
    level: str = "paper",
    k: int | None = None,
) -> float | None:
    """Recall@k (coverage): fraction of the gold documents present in the retrieved top-k (FP2).
    With a single gold document it is identical to hit_rate (use hit + MRR there; roadmap D-2)."""
    ranked_gold = _ranked_and_gold(
        retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs, level=level, k=k
    )
    if ranked_gold is None:
        return None
    ranked, gold = ranked_gold
    return len(gold & set(ranked)) / len(gold)


def hit_rate(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
    *,
    level: str = "paper",
    k: int | None = None,
) -> float | None:
    """Hit@k (success): 1 if at least one gold document is in the retrieved top-k, else 0."""
    ranked_gold = _ranked_and_gold(
        retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs, level=level, k=k
    )
    if ranked_gold is None:
        return None
    ranked, gold = ranked_gold
    return 1.0 if gold & set(ranked) else 0.0


def mrr(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
    *,
    level: str = "paper",
    k: int | None = None,
) -> float | None:
    """Reciprocal rank (1 / position of the first gold document; 0 if absent). Averaged
    over examples it is the MRR (Mean Reciprocal Rank)."""
    ranked_gold = _ranked_and_gold(
        retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs, level=level, k=k
    )
    if ranked_gold is None:
        return None
    ranked, gold = ranked_gold
    for rank, rid in enumerate(ranked, 1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def precision_at_k(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
    *,
    level: str = "paper",
    k: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> float | None:
    """Top-k purity: fraction of the retrieved top-k (at the gold's level) that is gold-labeled.

    Reads "not in gold" as an error, so it is defined ONLY where the gold is complete
    (`is_gold_complete`, roadmap P6 / D-8); otherwise None. With a 1-paper gold and k=5 the
    ceiling is 1/5 = 0.2 even for perfect retrieval, because the other papers are unjudged,
    not wrong. Without `metadata` there is no evidence of completeness -> None.
    It is a diagnostic of noise reaching the generator (a cause of Barnett's FP4); it does NOT
    measure FP3, which needs a selection step between retrieval and prompt (roadmap C-3).
    At paper level the denominator is the number of DISTINCT papers retrieved.
    """
    if not is_gold_complete(reference_outputs, metadata):
        return None
    ranked_gold = _ranked_and_gold(
        retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs, level=level, k=k
    )
    if ranked_gold is None:
        return None
    ranked, gold = ranked_gold
    if not ranked:
        return 0.0
    return sum(1 for rid in ranked if rid in gold) / len(ranked)


def hit_at_ks(
    retrieved_chunk_ids: list[str],
    retrieved_arxiv_ids: list[str],
    reference_outputs: dict[str, Any],
    ks: tuple[int, ...] = (1, 3, 5),
    *,
    level: str = "paper",
) -> dict[int, float | None]:
    """Hit curve: hit@k for several cutoffs (how much each extra position buys; feeds CKPT-1's k sweep)."""
    return {
        k: hit_rate(retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs, level=level, k=k)
        for k in ks
    }


def primary_retrieval_metrics(reference_outputs: dict[str, Any]) -> list[str]:
    """Which retrieval metrics to REPORT for an example (roadmap D-2).

    1 gold paper  -> ["hit", "mrr"]            (recall would just repeat hit)
    2+ gold papers -> ["recall", "hit", "mrr"]  (recall = coverage of all needed papers)
    no gold        -> []                        (unanswerable/stale/open-topic: nothing to score)
    """
    n_gold = len(_gold_papers(reference_outputs))
    if n_gold == 0:
        return []
    return ["hit", "mrr"] if n_gold == 1 else ["recall", "hit", "mrr"]


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
# Abstention (FP1 guard; roadmap P7/D-5 detection, P8/D-6 reporting)
# =====================================================================

def _normalize_for_abstention(text: str) -> str:
    t = text.lower()
    for curly, straight in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
        t = t.replace(curly, straight)
    t = t.replace("'", "")
    return re.sub(r"\s+", " ", t).strip()


# Matched on the normalized text (lowercase, no apostrophes). Two families:
# - first person: the SPEAKER owns the ignorance ("i dont know", "i cannot find");
# - source-negative: the CONTEXT/CORPUS lacks the answer ("no papers found",
#   "not in the context", "the corpus doesnt...").
# Deliberately narrow: "dont know" alone is NOT a marker, so that an answer
# reporting the paper's own uncertainty ("the authors dont know whether...")
# is not read as a refusal.
_ABSTENTION_MARKERS: tuple[str, ...] = (
    "i dont know",
    "i do not know",
    "i dont have",
    "i do not have",
    "i have no",
    "i cant find",
    "i cannot find",
    "i couldnt find",
    "i could not find",
    "i did not find",
    "i didnt find",
    "im unable",
    "i am unable",
    "no papers found",
    "no paper found",
    "no papers were found",
    "no relevant papers",
    "no relevant abstracts",
    "no matching papers",
    "no documents found",
    "not in the context",
    "not in the corpus",
    "not in the retrieved",
    "not in the provided",
    "isnt in the",
    "is not in the",
    "not covered by",
    "not covered in",
    "none of the retrieved",
    "none of the papers in the corpus",
    "none of the abstracts",
    "the context doesnt",
    "the context does not",
    "the corpus doesnt",
    "the corpus does not",
    "the abstracts dont",
    "the abstracts do not",
    "doesnt say",
    "does not say",
    "doesnt mention",
    "does not mention",
    "doesnt cover",
    "does not cover",
    "dont cover",
    "beyond the scope of",
    "outside the scope of",
)


def abstained(answer: str) -> bool:
    """Did the answer REFUSE to answer? Cheap deterministic floor, not the truth (P7/D-5).

    PROMPT_V1 never fixes a refusal phrase, so the old
    `answer.strip().startswith(("I don't know", "No papers found"))` missed every
    paraphrase ("I'm sorry, the context doesn't say." scored as an answer -> 0.0).
    Here the text is normalized (case, typographic apostrophes, whitespace) and
    matched against two marker families (see _ABSTENTION_MARKERS).

    Known false positive (accepted, measured at EXP-0): a first-person hedge inside
    a real answer ("I don't know whether the paper tests this, but the results...").
    The `abstention_quality` judge (0.6.3, T=0, answer-only) is the ground truth,
    validated on ~50 real stratified answers; EXP-0 reports the heuristic x judge
    divergence (target >= 90% agreement; AbstentionBench C.3.1).

    Scope: **English only**. PROMPT_V1 fixes no answer language, but the golden set
    is English, so today's measurements never see other languages. A refusal in
    another language ("nao sei", "je ne sais pas") is NOT detected and reads as an
    answer — the same defect class as the two exact phrases, one axis over
    (paraphrase -> language). Deliberately NOT fixed by translating the marker list
    (an ever-growing list that never completes is the original trap again); the
    answers are the judge (natively multilingual) and, if the product ever serves
    non-English questions, a prompt-level language contract (CKPT-6) or
    judge-first detection. Pinned by test_abstained_is_english_scoped; ADR-007
    trade-offs.
    """
    t = _normalize_for_abstention(answer or "")
    return any(marker in t for marker in _ABSTENTION_MARKERS)


def wilson_ci(k: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion (k successes in n trials; P8/D-6).

    Why Wilson and not Wald or Student's t: with n ~ 10-25 the naive Wald interval
    (p +/- z*se) is too narrow and can fall outside [0, 1] near p=0/1. Student's t
    is the wrong tool here: it corrects the mean of CONTINUOUS data when the
    variance must be estimated from the sample (extra variability -> heavier
    tails, df = n-1). A proportion carries no separate variance to estimate
    (sigma^2 = p(1-p)), so the correct small-sample fix is to invert the score
    test — which is exactly what Wilson does — keeping the interval inside [0, 1]
    and honest for extreme p. Sanity anchors: 10/10 -> [0.72, 1.00]; 14/15 ->
    [0.70, 0.99] (one example swings the point estimate by 6.7 p.p., hence the CI
    in every reported metric). Caveat shared with any binomial interval: trials
    are assumed independent (candidates within one question are not; roadmap §6).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k must be within [0, n], got k={k}, n={n}")
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _slice_of(example: Any) -> str:
    if isinstance(example, dict):
        meta = example.get("metadata") or {}
    else:
        meta = getattr(example, "metadata", None) or {}
    return str(meta.get("slice") or "unknown")


def _abstention_metrics(rows: list[tuple[bool, bool]], suffix: str = "") -> list[dict[str, Any]]:
    """Three numbers from (should_abstain, did_abstain) pairs — never accuracy (P8/D-6).

    accuracy mixes two opposite errors: a system that refuses EVERYTHING scores
    100% on the gate and is useless (its over_refusal_rate is 100%); one that never
    refuses scores 75% here and hallucinates. RefusalBench: the two errors trade
    off (r = -0.78; GPT-4o refused 14.6x more than needed).
    """
    pos = [did for should, did in rows if should]
    neg = [did for should, did in rows if not should]
    n_pos, k_pos = len(pos), sum(pos)
    n_neg, k_neg = len(neg), sum(neg)
    out: list[dict[str, Any]] = []
    if n_pos:
        lo, hi = wilson_ci(k_pos, n_pos)
        out.append({
            "key": f"abstention_recall{suffix}",
            "score": k_pos / n_pos,
            "comment": f"correct refusals {k_pos}/{n_pos}; Wilson 95% CI [{lo:.2f}, {hi:.2f}]",
        })
    if n_neg:
        lo, hi = wilson_ci(k_neg, n_neg)
        out.append({
            "key": f"over_refusal_rate{suffix}",
            "score": k_neg / n_neg,
            "comment": f"unwarranted refusals {k_neg}/{n_neg}; Wilson 95% CI [{lo:.2f}, {hi:.2f}]",
        })
    if n_pos and n_neg:
        n_refused = k_pos + k_neg
        precision = k_pos / n_refused if n_refused else 0.0
        recall = k_pos / n_pos
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append({
            "key": f"abstention_f1{suffix}",
            "score": f1,
            "comment": "diagnostic only: a single number hides the recall/over-refusal trade-off",
        })
    return out


def abstention_summary(
    outputs: list[dict[str, Any]],
    reference_outputs: list[dict[str, Any]],
    examples: list[Any] | None = None,
) -> dict[str, Any]:
    """Summary evaluator: abstention as recall + over-refusal, per slice (roadmap P8/D-6).

    - abstention_recall: of should-abstain examples, the fraction that refused
      (the plan's gate, >= 90% on `unanswerable`);
    - over_refusal_rate: of should-answer examples, the fraction that refused
      (the guardrail: a recall gain must not be bought with over-refusal;
      regression rule in decisions.md);
    - abstention_f1: diagnostic only.
    Reported total and per metadata.slice (`__<slice>` suffix), each with raw
    counts + Wilson 95% CI in the comment. Uses abstained() from P7. The
    abstention_accuracy key was removed from f1_summary_evaluator (this replaced it;
    that evaluator is now token_f1_summary, P9a).

    LangSmith contract (P4, extra="forbid"): supported args are runs/examples/
    inputs/outputs/reference_outputs; langsmith maps outputs <- run.outputs,
    reference_outputs <- example.outputs, examples <- the Example objects
    (their .metadata carries the slice). Returns {"results": [...]}.
    """
    rows: list[tuple[bool, bool]] = []
    by_slice: dict[str, list[tuple[bool, bool]]] = {}
    for i, (run_out, ref) in enumerate(zip(outputs, reference_outputs)):
        should = ref.get("should_abstain")
        if should is None:
            continue
        did = abstained(run_out.get("answer", ""))
        row = (bool(should), did)
        rows.append(row)
        sl = "unknown"
        if examples is not None and i < len(examples):
            sl = _slice_of(examples[i])
        by_slice.setdefault(sl, []).append(row)
    results = _abstention_metrics(rows)
    for sl in sorted(by_slice):
        results.extend(_abstention_metrics(by_slice[sl], suffix=f"__{sl}"))
    return {"results": results}


# =====================================================================
# Lexical correctness signals: token-F1 and word Recall (roadmap P9a; ADR-008, ADR-010)
# =====================================================================

# Slices where "is the answer right?" is asked of the answer text (ADR-008 §1). Refusal
# (`unanswerable`, `stale`) has its own evaluator (abstention_summary) and `format` has
# format_validator.
CORRECTNESS_SLICES = ("answerable", "persona", "multi-doc")

# arXiv id with an optional version, anywhere in the text, any case ("V1" too). The prompt's
# example is `[2609.12345]` (no version) while the context prints `[2609.20822v1]`, and the
# gold answers also name papers by a bare id outside brackets (ADR-010 / D-15).
ARXIV_ID = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_text(text: str) -> list[str]:
    """Words of `text` after the cleaning rule fixed in ADR-010 (D-14/D-15).

    Order: remove arXiv ids -> lowercase -> every character that is not a letter, digit,
    underscore or whitespace becomes a SPACE (so "component-level" -> "component level" and
    the em dash vanishes; the literal SQuAD rule deletes punctuation and merges those words)
    -> drop the articles a/an/the -> split on whitespace. No stopword list (it would also
    delete "not"), no stemming.
    """
    text = ARXIV_ID.sub(" ", text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return _ARTICLES.sub(" ", text).split()


def _overlap(pred: str, ref: str) -> tuple[int, int, int]:
    """(shared words, words in the answer, words in the reference); shared words are a
    multiset intersection, so repeating a word in the answer never counts past its count
    in the reference."""
    p, r = Counter(normalize_text(pred)), Counter(normalize_text(ref))
    return sum((p & r).values()), sum(p.values()), sum(r.values())


def word_recall(pred: str, ref: str) -> float:
    """Share of the reference's words that appear in the answer (0.0 if the reference is empty).

    Does not penalize a longer answer (Adlakha 2024). Blind to negation, and an answer made
    only of the reference's function words still scores ~0.22 on the `multi-doc` golds
    (ADR-010 trade-off) — a level, not a quality. A diagnostic until calibrated (ADR-008).
    """
    shared, _, n_ref = _overlap(pred, ref)
    return shared / n_ref if n_ref else 0.0


def token_f1(pred: str, ref: str) -> float:
    """Harmonic mean of the answer's word precision and word Recall (0.0 if either is empty)."""
    shared, n_pred, n_ref = _overlap(pred, ref)
    if not shared:
        return 0.0
    precision, recall = shared / n_pred, shared / n_ref
    return 2 * precision * recall / (precision + recall)


def _correctness_rows(
    outputs: list[dict[str, Any]],
    reference_outputs: list[dict[str, Any]],
    examples: list[Any] | None,
) -> list[tuple[str, str, str]]:
    """(slice, answer, gold answer) of the examples the correctness signals apply to: a slice in
    `CORRECTNESS_SLICES` (read from `examples[i].metadata`, never guessed) and a non-empty gold answer."""
    rows: list[tuple[str, str, str]] = []
    for i, (run_out, ref) in enumerate(zip(outputs, reference_outputs, strict=True)):
        gold = ref.get("answer")
        sl = _slice_of(examples[i]) if examples is not None and i < len(examples) else "unknown"
        if gold and sl in CORRECTNESS_SLICES:
            rows.append((sl, run_out.get("answer", "") or "", gold))
    return rows


def _lexical_results(scores: list[tuple[float, float]], suffix: str = "") -> list[dict[str, Any]]:
    n = len(scores)
    note = f"mean over n={n} in-scope examples; diagnostic until calibrated (ADR-008)"
    return [
        {"key": f"token_f1_summary{suffix}", "score": sum(f for f, _ in scores) / n, "comment": note},
        {"key": f"word_recall_summary{suffix}", "score": sum(r for _, r in scores) / n, "comment": note},
    ]


def token_f1_summary(
    outputs: list[dict[str, Any]],
    reference_outputs: list[dict[str, Any]],
    examples: list[Any] | None = None,
) -> dict[str, Any]:
    """Summary evaluator: mean token-F1 and mean word Recall of the answers against the gold
    answers, total and per slice (`__<slice>` suffix) — roadmap P9a, ADR-008 §2, ADR-010.

    Renamed from `f1_summary_evaluator`. Scope: only `CORRECTNESS_SLICES` with a non-empty gold
    answer; the slice comes from `examples[i].metadata` (an example without one is skipped, never
    guessed). Both keys are DIAGNOSTICS, not "the quality of the answers", until the EXP-0
    calibration shows which signal passes the pass marks (ADR-008 §5). Cleaning rule: ADR-010.

    LangSmith contract (P4): `examples` is a supported arg name; returns {"results": [...]}
    (EvaluationResult is extra="forbid"). No in-scope example -> {"results": []}.
    """
    scores: list[tuple[float, float]] = []
    by_slice: dict[str, list[tuple[float, float]]] = {}
    for sl, answer, gold in _correctness_rows(outputs, reference_outputs, examples):
        row = (token_f1(answer, gold), word_recall(answer, gold))
        scores.append(row)
        by_slice.setdefault(sl, []).append(row)
    if not scores:
        return {"results": []}
    results = _lexical_results(scores)
    for sl in sorted(by_slice):
        results.extend(_lexical_results(by_slice[sl], suffix=f"__{sl}"))
    return {"results": results}


# =====================================================================
# Embedding cosine (roadmap P9; ADR-008 §2, ADR-010 D-15): the "middle rung" of the ladder
# =====================================================================

# Pinned on purpose and NOT read from RunConfig: the retrieval index may change embedding model
# at CKPT-2, but the cosine's scale is specific to its model (Cao 2025), so a change here
# invalidates its pass marks and forces a recalibration (ADR-008, ADR-010).
CORRECTNESS_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBED_BATCH = 256  # inputs per API request (the endpoint accepts up to 2048)

Embedder = Callable[[list[str]], list[list[float]]]


_EMPTY_HOLDER = re.compile(r"\[[\s,;]*\]|\([\s,;]*\)")


def strip_arxiv_ids(text: str | None) -> str:
    """The cleaning applied before embedding (ADR-010 D-15 as clarified on 2026-10-12): the arXiv ids are
    removed, and so are the brackets or parentheses that held ONLY ids (what is left empty, or with just
    separators, after the removal: `[2609.20804v1]` -> nothing, `[a, b]` -> nothing). Runs of spaces are
    collapsed. Case, punctuation, line breaks and wording are left alone because an embedding reads natural
    text. Without the holder removal a same-content answer scored a cosine of 0.976 on average (0.931 at
    worst) against its gold instead of 1.0, depending on how the id was written (ADR-010 live check)."""
    text = ARXIV_ID.sub(" ", text or "")
    text = _EMPTY_HOLDER.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def cosine_similarity(u: list[float], v: list[float]) -> float:
    """Cosine of the angle between two vectors; 0.0 if either has zero length."""
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    if not nu or not nv:
        return 0.0
    return sum(a * b for a, b in zip(u, v, strict=True)) / (nu * nv)


def openai_embed(texts: list[str], model: str = CORRECTNESS_EMBEDDING_MODEL) -> list[list[float]]:
    """Embed `texts` with the OpenAI API in batches, in input order. The client is created here,
    lazily, so importing this module needs neither the network nor an API key (offline tests)."""
    from openai import OpenAI

    client = OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        response = client.embeddings.create(model=model, input=texts[start:start + _EMBED_BATCH])
        vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
    return vectors


# Where the default embedder keeps its vectors between runs (git-ignored, see .gitignore).
DEFAULT_EMBEDDING_CACHE = Path(__file__).resolve().parent / "resources" / "answer_embeddings.jsonl"


def _to_float32(vector: list[float]) -> list[float]:
    """Round-trip through float32 (the precision the API works in) so that a vector read back from disk is
    bit-identical to the one that was kept in memory the first time."""
    return list(struct.unpack(f"{len(vector)}f", struct.pack(f"{len(vector)}f", *vector)))


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class CachedEmbedder:
    """Wraps a batch embedding function with a cache keyed by (model, text).

    Each distinct text is embedded once. With `cache_path` the cache also lives on disk, so the reference
    answers (the same 65 in every experiment) and the vectors used in a calibration survive the process and
    can be re-scored offline. The file is an append-only JSON-lines file, one `{"key", "dim", "vec"}` per
    text: `key` = sha256(model + text) so a different model never reuses a vector, `vec` = base64 of the
    float32 bytes (about 8 KB per text). Loaded lazily on the first call; a malformed or truncated line is
    skipped. Delete the file to invalidate it (for instance if the provider ever changes the model behind the
    pinned name). Only a hash of the text is stored, never the text."""

    def __init__(self, embed: Embedder, model: str = CORRECTNESS_EMBEDDING_MODEL, cache_path: Path | str | None = None):
        self._embed = embed
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, list[float]] = {}
        self._loaded = False

    def _load(self) -> None:
        self._loaded = True
        if not self.cache_path or not self.cache_path.exists():
            return
        for line in self.cache_path.read_text().splitlines():
            try:
                row = json.loads(line)
                raw = base64.b64decode(row["vec"])
                if len(raw) != 4 * int(row["dim"]):
                    continue
                self._cache[row["key"]] = list(struct.unpack(f"{row['dim']}f", raw))
            except (ValueError, KeyError, TypeError):
                continue  # a bad or half-written line is ignored, the text is simply embedded again

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not self._loaded:
            self._load()
        keys = [_cache_key(self.model, t) for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if k not in self._cache))
        if missing:
            fresh = {}
            for text, vector in zip(missing, self._embed(missing), strict=True):
                fresh[_cache_key(self.model, text)] = _to_float32(vector)
            self._cache.update(fresh)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as fh:
                    for key, vector in fresh.items():
                        raw = struct.pack(f"{len(vector)}f", *vector)
                        fh.write(json.dumps({"key": key, "dim": len(vector), "vec": base64.b64encode(raw).decode()}) + "\n")
        return [self._cache[k] for k in keys]


def make_embedding_cosine_summary(embed: Embedder) -> Callable[..., dict[str, Any]]:
    """Build the summary evaluator around an embedding function (a fake one in the offline tests).

    A factory because langsmith only accepts summary evaluators whose arguments are
    runs/examples/inputs/outputs/reference_outputs, so the embedder cannot be an extra argument."""

    def embedding_cosine_summary(
        outputs: list[dict[str, Any]],
        reference_outputs: list[dict[str, Any]],
        examples: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Summary evaluator: mean cosine between the embedding of each answer and of its gold
        answer, total and per slice (`__<slice>` suffix) — roadmap P9, ADR-008 §2, ADR-010 D-15.

        Same scope as `token_f1_summary` (CORRECTNESS_SLICES, non-empty gold answer). Only the
        arXiv ids are removed before embedding. An answer with no text left (empty, or only ids)
        scores 0.0 and is not sent to the API (the endpoint rejects empty input). A DIAGNOSTIC
        until calibrated: the cosine scale is specific to the embedding model, and it is expected
        to miss contradictions (Cao 2025; ADR-008 §6)."""
        rows = [(sl, strip_arxiv_ids(a), strip_arxiv_ids(g)) for sl, a, g in
                _correctness_rows(outputs, reference_outputs, examples)]
        if not rows:
            return {"results": []}
        vectors = embed([t for _, a, g in rows for t in (a, g) if t])
        lookup = dict(zip([t for _, a, g in rows for t in (a, g) if t], vectors, strict=True))
        by_slice: dict[str, list[float]] = {}
        scores: list[float] = []
        for sl, answer, gold in rows:
            score = cosine_similarity(lookup[answer], lookup[gold]) if answer and gold else 0.0
            scores.append(score)
            by_slice.setdefault(sl, []).append(score)

        def entry(values: list[float], suffix: str = "") -> dict[str, Any]:
            return {
                "key": f"embedding_cosine_summary{suffix}",
                "score": sum(values) / len(values),
                "comment": (f"mean over n={len(values)} in-scope examples; model "
                            f"{CORRECTNESS_EMBEDDING_MODEL} (pinned); diagnostic until calibrated (ADR-008)"),
            }

        results = [entry(scores)]
        results.extend(entry(by_slice[sl], f"__{sl}") for sl in sorted(by_slice))
        return {"results": results}

    return embedding_cosine_summary


# Default instance used by the runner: real embeddings, cached in memory and in DEFAULT_EMBEDDING_CACHE.
embedding_cosine_summary = make_embedding_cosine_summary(
    CachedEmbedder(openai_embed, cache_path=DEFAULT_EMBEDDING_CACHE))
