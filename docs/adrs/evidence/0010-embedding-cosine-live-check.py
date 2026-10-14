"""Live check of the embedding-cosine evaluator (P9; ADR-008 §2, ADR-010 D-15 as clarified, ADR-011 Prediction 3).

REQUIRES the network and OPENAI_API_KEY (read from .env). Cost: well under one US cent
(about 120 short texts with text-embedding-3-small). Run from the repository root:

    uv run python docs/adrs/evidence/0010-embedding-cosine-live-check.py

Recorded output: 0010-embedding-cosine-live-check.out.txt. The texts are the 15 `multi-doc` gold answers
of build_golden_dataset.py (project data, no personal data). The API's vectors agree between runs to about the
third decimal of the cosines reported here.

Questions:
  1. Does the evaluator work end to end with the real model (answer == gold must give cosine ~ 1.0)?
  2. D-15 removes the arXiv ids. If ONLY the ids go, the brackets that held them stay ("[ ]"). How much does that
     move the cosine of an answer with the SAME content? ("literal" = ids only, the first implementation;
     "as implemented" = ids plus the brackets/parentheses left empty, the clarification of 2026-10-12)
  3. Does the cosine of a same-content answer still depend on how the id is written?
  4. Prediction 3 of ADR-011: do canaries (a gold answer with ONE false edit) get a cosine close to a perfect
     answer's, i.e. is the cosine blind to them? Scale: the cosine between two UNRELATED golds.
  5. Does the in-memory cache avoid paying twice for the same text?
  6. Does the on-disk cache avoid paying again in a NEW process (a new CachedEmbedder reading the same file)?
"""
import re
import sys
import tempfile
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
import build_golden_dataset as b  # noqa: E402
import evaluators as ev  # noqa: E402

GOLDS = [a for _, _, a in b._MULTIDOC_EXAMPLES]
calls = {"api": 0, "texts": 0}


def counting_embed(texts):
    calls["api"] += 1
    calls["texts"] += len(texts)
    return ev.openai_embed(texts)


embedder = ev.CachedEmbedder(counting_embed)
summary = ev.make_embedding_cosine_summary(embedder)
literal = lambda t: ev.ARXIV_ID.sub(" ", t or "").strip()  # noqa: E731  ids only (first implementation)


def cos_with(clean, a, b_):
    va, vb = embedder([clean(a), clean(b_)])
    return ev.cosine_similarity(va, vb)


def cos(a, b_):  # as implemented
    return cos_with(ev.strip_arxiv_ids, a, b_)


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


hr("1. end to end: answer == gold through embedding_cosine_summary")
ex = [SimpleNamespace(inputs={}, outputs={}, metadata={"slice": "multi-doc"}) for _ in GOLDS]
out = summary([{"answer": g} for g in GOLDS], [{"answer": g} for g in GOLDS], ex)
for r in out["results"]:
    print(f"  {r['key']:36s} {r['score']:.4f}   {r['comment']}")
vec = embedder([ev.strip_arxiv_ids(GOLDS[0])])[0]
print(f"  vector length: {len(vec)} (text-embedding-3-small = 1536); model pinned: {ev.CORRECTNESS_EMBEDDING_MODEL}")

hr("2+3. same content, different id spelling -> cosine with the gold (literal ids-only vs as implemented)")
ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
SPELL = {
    "bare id written [id] (model style)": lambda t: re.sub(rf"(?<!\[)\b({ID})\b", r"[\1]", t),
    "citation without version": lambda t: re.sub(r"\b(\d{4}\.\d{4,5})v\d+\b", r"\1", t),
    "no citation: id AND its brackets gone": lambda t: re.sub(rf"\[\s*{ID}\s*\]|\b{ID}\b", " ", t),
    "no citation, written naturally (no space left before '.')": lambda t: re.sub(rf"\s*\[\s*{ID}\s*\]|\b{ID}\b\s*", "", t),
}
print(f"  {'case':58s} {'literal: mean / min':>22s} {'as implemented: mean / min':>28s}")
for name, fn in SPELL.items():
    lit = [cos_with(literal, fn(g), g) for g in GOLDS]
    cur = [cos(fn(g), g) for g in GOLDS]
    print(f"  {name:58s} {mean(lit):12.4f} / {min(lit):.4f} {mean(cur):16.4f} / {min(cur):.4f}")

hr("4. canaries: cosine between a canary and the gold it came from (a perfect answer = 1.0), as implemented")
NEG = {"fixes": "does not fix", "isolates": "does not isolate", "uses": "does not use", "shows": "does not show",
       "finds": "does not find", "proposes": "does not propose", "introduces": "does not introduce",
       "improves": "does not improve", "reduces": "does not reduce", "evaluates": "does not evaluate",
       "compares": "does not compare", "measures": "does not measure", "achieves": "does not achieve",
       "outperforms": "does not outperform", "studies": "does not study", "presents": "does not present"}
NUM = {"two": "eight", "three": "seven", "four": "nine", "five": "ten", "six": "twelve", "ten": "two"}
ANT = {"increase": "decrease", "increases": "decreases", "higher": "lower", "lower": "higher", "improve": "degrade",
       "improves": "degrades", "better": "worse", "worse": "better", "reduce": "increase", "reduces": "increases",
       "outperforms": "underperforms", "more": "fewer", "success": "failure", "gains": "losses",
       "positive": "negative", "faster": "slower", "larger": "smaller"}


def first_sub(text, mapping):
    for m in re.finditer(r"\b[A-Za-z]+\b", text):
        if m.group(0).lower() in mapping:
            return text[:m.start()] + mapping[m.group(0).lower()] + text[m.end():]
    return None


for name, mp in (("negation", NEG), ("number swap", NUM), ("direction reversed", ANT)):
    pairs = [(first_sub(g, mp), g) for g in GOLDS if first_sub(g, mp)]
    v = [cos(c, g) for c, g in pairs]
    print(f"  {name:20s} n={len(v):2d}  mean {mean(v):.4f}  min {min(v):.4f}  max {max(v):.4f}")
unrelated = [cos(GOLDS[i], GOLDS[j]) for i in range(len(GOLDS)) for j in range(i + 1, len(GOLDS))]
print(f"  scale: cosine between two UNRELATED golds ({len(unrelated)} pairs): mean {mean(unrelated):.4f}, "
      f"min {min(unrelated):.4f}, max {max(unrelated):.4f}")
print("  (the 2026-10-10 local test gave 0.59-0.84 for correct paraphrases and 0.94 for a negated answer)")

hr("5. in-memory cache: scoring the same texts again must not call the API")
before = dict(calls)
summary([{"answer": g} for g in GOLDS], [{"answer": g} for g in GOLDS], ex)
print(f"  API requests so far: {calls['api']}, texts embedded: {calls['texts']}; "
      f"new requests during the repeat run: {calls['api'] - before['api']}")

hr("6. on-disk cache: a NEW embedder (a new process) reading the same file must not call the API")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "answer_embeddings.jsonl"
    c1 = {"api": 0}

    def counted(texts):
        c1["api"] += 1
        return ev.openai_embed(texts)

    first = ev.make_embedding_cosine_summary(ev.CachedEmbedder(counted, cache_path=path))
    r1 = first([{"answer": g} for g in GOLDS], [{"answer": g} for g in GOLDS], ex)["results"][0]["score"]
    size = path.stat().st_size
    c2 = {"api": 0}

    def must_not_be_called(texts):
        c2["api"] += 1
        raise RuntimeError("the API was called although the cache file has every text")

    second = ev.make_embedding_cosine_summary(ev.CachedEmbedder(must_not_be_called, cache_path=path))
    r2 = second([{"answer": g} for g in GOLDS], [{"answer": g} for g in GOLDS], ex)["results"][0]["score"]
    print(f"  first run : {c1['api']} API request(s), score {r1:.6f}; cache file {size / 1024:.0f} KB "
          f"({len(path.read_text().splitlines())} vectors)")
    print(f"  new process: {c2['api']} API request(s), score {r2:.6f}; identical score: {r1 == r2}")
