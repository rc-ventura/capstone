"""Evidence for ADR-010 (text normalization of the lexical correctness signals).

Run from the repository root:

    uv run python docs/adrs/evidence/0010-normalization-synthetic-tests.py

Output of the run recorded in the ADR: 0010-normalization-synthetic-tests.out.txt

What this is: SYNTHETIC tests. Every "answer" below is built mechanically from one of
the 15 real `multi-doc` gold answers in build_golden_dataset.py (one change at a time),
so the correct score is known in advance. They show what a normalization rule CAN do to a
score. They do NOT show how often gpt-4o-mini produces these variants, and they do not
measure agreement with human labels (that is the calibration of ADR-008, after EXP-0).

Definitions used everywhere (same as evaluators._token_f1, minus the normalization):
    tokens(text)  = the normalized text split on whitespace; "token" == "word" here
    overlap       = sum over words of min(count in answer, count in reference)   (multiset)
    Recall        = overlap / number of tokens in the REFERENCE
    Precision     = overlap / number of tokens in the ANSWER
    F1            = 2 * Precision * Recall / (Precision + Recall)
Everything is deterministic: random draws use fixed seeds (0 and 1).
"""
import re
import string
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from random import Random
from statistics import mean

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import build_golden_dataset as b  # noqa: E402

REFS = [answer for _, _, answer in b._MULTIDOC_EXAMPLES]
N = len(REFS)

# ---------------------------------------------------------------- rules under test
ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
RE_ANY = re.compile(rf"\b{ID}\b", re.I)  # an arXiv id anywhere (case-insensitive: "V1" too)
RE_BRACKET = re.compile(rf"\[\s*{ID}(?:\s*[,;]\s*{ID})*\s*\]", re.I)  # only inside [ ]
RE_BARE = re.compile(rf"(?<!\[)\b{ID}\b(?!\])", re.I)  # outside [ ]
RE_BRACKET_ONE = re.compile(r"\[\d{4}\.\d{4,5}(?:v\d+)?\]")  # single id, used for part A only
ARTICLES = re.compile(r"\b(a|an|the)\b")


def tokens_current(text):
    """What evaluators._tokens does today (lowercase, \\w+), after dropping [id] citations."""
    return re.findall(r"\w+", RE_BRACKET_ONE.sub(" ", text).lower())


def n1a(text):
    """N1-a: SQuAD literal. lower -> DELETE ascii punctuation -> drop a/an/the -> split."""
    t = text.lower()
    t = "".join(c for c in t if c not in string.punctuation)
    return ARTICLES.sub(" ", t).split()


def n1b(text):
    """N1-b: like N1-a, but punctuation becomes a SPACE (any non-word char, incl. unicode dashes)."""
    t = re.sub(r"[^\w\s]", " ", text.lower())
    return ARTICLES.sub(" ", t).split()


def strip_ids(text, mode):
    """mode: 'all' (O1) | 'brackets' (O2) | 'keep' (O3)."""
    if mode == "all":
        return RE_ANY.sub(" ", text)
    if mode == "brackets":
        return RE_BRACKET.sub(" ", text)
    return text


def score(answer, reference, norm, ids_mode):
    p, r = norm(strip_ids(answer, ids_mode)), norm(strip_ids(reference, ids_mode))
    overlap = sum((Counter(p) & Counter(r)).values())
    recall = overlap / len(r) if r else 0.0
    precision = overlap / len(p) if p else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return recall, f1


# ------------------------------------------------------------------ NLTK stopword list
NLTK_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip"


def load_nltk_english():
    import io
    import zipfile

    with urllib.request.urlopen(NLTK_URL, timeout=30) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    return set(zf.read("stopwords/english").decode().split())


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ================================================================= PART A: how much do the rules remove
hr("PART A - what each rule does to the 15 multi-doc reference answers")
NLTK = load_nltk_english()
print(f"NLTK English stopword list: {len(NLTK)} entries; contains 'not','no','nor': "
      f"{all(w in NLTK for w in ('not', 'no', 'nor'))}")


def n1c(text):  # N2 candidate: N1-a + NLTK list (only used here, to size the floor)
    return [w for w in n1a(RE_BRACKET_ONE.sub(" ", text)) if w not in NLTK]


rules = {
    "A  current code (\\w+, lowercase)": tokens_current,
    "B  N1-a SQuAD literal": lambda t: n1a(RE_BRACKET_ONE.sub(" ", t)),
    "B2 N1-b punctuation->space": lambda t: n1b(RE_BRACKET_ONE.sub(" ", t)),
    "C  N1-a + NLTK list (=N2)": n1c,
}
base = [len(tokens_current(r)) for r in REFS]
for name, fn in rules.items():
    lens = [len(fn(r)) for r in REFS]
    removed = mean(1 - l / b0 for l, b0 in zip(lens, base))
    print(f"  {name:36s} mean tokens/reference = {mean(lens):5.1f}   removed vs current = {100 * removed:4.1f}%")
share_articles = mean(
    sum(w in ("a", "an", "the") for w in tokens_current(r)) / len(tokens_current(r)) for r in REFS
)
print(f"  share of tokens that are a/an/the alone (current tokens): {100 * share_articles:.1f}%")
print(f"  hyphens/en-dashes/em-dashes per reference (raw text), mean: "
      f"{mean(len(re.findall(r'[-–—]', r)) for r in REFS):.1f}")
print("  -> the 10% drop of rule B is NOT articles: most of it is hyphenated words and ids "
      "being merged by the punctuation deletion.")

print("\nWorked example, reference 0 (first 160 chars):")
print("  ", REFS[0][:160].replace("\n", " "))
for name, fn in rules.items():
    toks = fn(REFS[0])
    print(f"  {name:36s} {len(toks):3d} tokens: {toks[:9]} ...")

print("\nFloor of Recall for an answer with NO content = only the reference's own function words")
print("(probe = reference tokens that are in the NLTK list or a/an/the, in order):")
floors_b, floors_c = [], []
for i, r in enumerate(REFS):
    probe = " ".join(w for w in tokens_current(r) if w in NLTK or w in ("a", "an", "the"))
    rb = score(probe, RE_BRACKET_ONE.sub(" ", r), n1a, "keep")[0]
    rc = score(probe, RE_BRACKET_ONE.sub(" ", r), n1c, "keep")[0]
    floors_b.append(rb)
    floors_c.append(rc)
    if i < 3:
        ref_len = len(n1a(RE_BRACKET_ONE.sub(" ", r)))
        print(f"  ref{i}: probe = '{probe[:60]}'  Recall N1-a = {rb:.3f} "
              f"({round(rb * ref_len)}/{ref_len} tokens)   Recall with NLTK list = {rc:.3f}")
print(f"  mean over the 15 references: Recall N1-a = {mean(floors_b):.3f} "
      f"(min {min(floors_b):.2f}, max {max(floors_b):.2f}); with NLTK list = {mean(floors_c):.3f}")

floors_adopted = []
for r in REFS:
    clean = strip_ids(r, "all")
    probe = " ".join(w for w in tokens_current(clean) if w in NLTK or w in ("a", "an", "the"))
    floors_adopted.append(score(probe, r, n1b, "all")[0])
print(f"  same probe under the ADOPTED rule (N1-b, ids removed): mean {mean(floors_adopted):.3f} "
      f"(min {min(floors_adopted):.2f}, max {max(floors_adopted):.2f})")

neg_p, neg_q = "the method does not improve accuracy", "the method improves accuracy"
print("\nNegation under each rule:")
print("  N1-a :", n1a(neg_p), "vs", n1a(neg_q))
print("  +NLTK:", n1c(neg_p), "vs", n1c(neg_q))

# ================================================================= PART B: N1-a vs N1-b
hr("TEST 1 - N1-a (SQuAD literal) vs N1-b (punctuation -> space)")
print("Method: answer = the reference reformatted in ONE way; correct score = 1.0 (same content).")
print("Ids are removed everywhere on both sides (option O1) so only punctuation is being tested.\n")


def respell_hyphen(t):
    return re.sub(r"(?<=\w)-(?=\w)", " ", t)


def dash_to_comma(t):
    return re.sub(r"\s*[—–]\s*", ", ", t)


def shout(t):
    # space out commas and SENTENCE-ending periods only. A first version replaced every "."
    # and so split the dot inside ids ("2609 .20804V1"), which made the F1 of T3 look < 1
    # for a reason unrelated to the normalization rule (fixed 2026-10-11).
    return re.sub(r"\.(\s|$)", r" .\1", t.upper().replace(",", " ,"))


CASES1 = {
    "T0 identical copy": lambda t: t,
    "T1 hyphen -> space (component level)": respell_hyphen,
    "T2 em dash -> comma": dash_to_comma,
    "T3 UPPERCASE + loose punctuation": shout,
    "T4 T1 + T2 together": lambda t: dash_to_comma(respell_hyphen(t)),
}
print(f"{'case':40s} {'Recall a':>9s} {'Recall b':>9s} {'F1 a':>7s} {'F1 b':>7s}")
for name, fn in CASES1.items():
    ra = [score(fn(r), r, n1a, "all") for r in REFS]
    rb = [score(fn(r), r, n1b, "all") for r in REFS]
    print(f"{name:40s} {mean(x[0] for x in ra):9.3f} {mean(x[0] for x in rb):9.3f} "
          f"{mean(x[1] for x in ra):7.3f} {mean(x[1] for x in rb):7.3f}")
t4 = CASES1["T4 T1 + T2 together"]
print(f"\nworst single reference in T4: N1-a {min(score(t4(r), r, n1a, 'all')[0] for r in REFS):.3f}   "
      f"N1-b {min(score(t4(r), r, n1b, 'all')[0] for r in REFS):.3f}")

print("\nT5 - does the ORDER of answers change? Method: for each reference and each keep fraction")
print("f in {0.2,0.4,0.6,0.8,1.0}, draw 40 answers by keeping each whitespace-separated word with")
print(f"probability f (seed 0): {N} refs x 5 x 40 = {N * 5 * 40} answers. 'respell' = also turn hyphens")
print("into spaces and dashes into commas. Score = Recall. Spearman between Recall and f.")
rng = Random(0)
for respell in (False, True):
    xs, ya, yb, by = [], [], [], {}
    for r in REFS:
        for keep in (0.2, 0.4, 0.6, 0.8, 1.0):
            for _ in range(40):
                ans = " ".join(w for w in r.split() if rng.random() < keep)
                if respell:
                    ans = respell_hyphen(dash_to_comma(ans))
                a = score(ans, r, n1a, "all")[0]
                bb = score(ans, r, n1b, "all")[0]
                xs.append(keep)
                ya.append(a)
                yb.append(bb)
                by.setdefault(keep, ([], []))
                by[keep][0].append(a)
                by[keep][1].append(bb)
    print(f"\n  respell = {respell}")
    print("    keep fraction :", "  ".join(f"{k:.1f}" for k in sorted(by)))
    print("    mean Recall a :", "  ".join(f"{mean(by[k][0]):.2f}" for k in sorted(by)))
    print("    mean Recall b :", "  ".join(f"{mean(by[k][1]):.2f}" for k in sorted(by)))
    print(f"    Spearman(Recall, keep fraction): N1-a {spearmanr(xs, ya).statistic:.3f} | "
          f"N1-b {spearmanr(xs, yb).statistic:.3f}")

# ================================================================= PART C: ids
hr("TEST 2 - what to do with arXiv ids (rule N1-b underneath)")
corpus_ids = [m["arxiv_id"] for m in pd.read_parquet(
    ROOT / "resources/arxiv_text-embedding-3-small_abstract.parquet").metadatas]
with_ids = [r for r in REFS if RE_ANY.search(r)]
print(f"references with >=1 id: {len(with_ids)} of {N}; with >=1 id OUTSIDE [ ]: "
      f"{sum(1 for r in REFS if RE_BARE.search(r))} of {N}; "
      f"mean bare ids per reference: {mean(len(RE_BARE.findall(r)) for r in REFS):.1f}")
print("Options: O1 = remove ids anywhere, both sides | O2 = remove only [id], both sides | O3 = keep ids")
print("Method: answer = the reference with ONE change in how ids are written; Recall | F1 averaged "
      f"over the {len(with_ids)} references.\n")
rng2 = Random(1)


def model_style(t):  # the model always cites inside brackets
    return RE_BARE.sub(lambda m: f"[{m.group(0)}]", t)


def no_version(t):  # [2609.20804] instead of [2609.20804v1] (the PROMPT_V1 example has no version)
    return re.sub(rf"\b(\d{{4}}\.\d{{4,5}})v\d+\b", r"\1", model_style(t), flags=re.I)


def wrong_paper(t):
    mapping = {}

    def sub(m):
        i = m.group(0)
        if i not in mapping:
            mapping[i] = rng2.choice([c for c in corpus_ids if c.lower() != i.lower()])
        return mapping[i]

    return model_style(RE_ANY.sub(sub, t))


def ids_only(t):
    return " ".join(f"[{i}]" for i in RE_ANY.findall(t))


CASES2 = {
    "U1 same content, bare id written as [id] (model style)": model_style,
    "U2 same content, citation without version [2609.20804]": no_version,
    "U3 same content, cites the WRONG paper": wrong_paper,
    "U4 same content, no citation at all": lambda t: RE_ANY.sub(" ", t),
    "U5 ONLY the ids, no content": ids_only,
}
OPTS = [("O1", "all"), ("O2", "brackets"), ("O3", "keep")]
print(f"{'case':56s} " + " ".join(f"{o:>13s}" for o, _ in OPTS) + "   (Recall | F1)")
for name, fn in CASES2.items():
    cells = []
    for _, mode in OPTS:
        sc = [score(fn(r), r, n1b, mode) for r in with_ids]
        cells.append(f"{mean(x[0] for x in sc):5.2f} | {mean(x[1] for x in sc):4.2f}")
    print(f"{name:56s} " + " ".join(f"{c:>13s}" for c in cells))
u1 = [score(model_style(r), r, n1b, "brackets")[0] for r in with_ids]
print(f"\nO2 ceiling for a perfect model-style answer (U1): mean {mean(u1):.3f}, "
      f"min {min(u1):.3f}, max {max(u1):.3f}")
print("\nSame table with N1-a underneath (Recall only):")
for name, fn in CASES2.items():
    cells = [f"{mean(score(fn(r), r, n1a, m)[0] for r in with_ids):5.2f}" for _, m in OPTS]
    print(f"{name:56s} " + " ".join(f"{c:>13s}" for c in cells))
