"""Evidence for the ADR-011 addendum "Observations after the decision" (2026-10-12).

Run from the repository root:

    uv run python docs/adrs/evidence/0011-observations-after-decision.py

Recorded output: 0011-observations-after-decision.out.txt

Three independent pieces, each answering one question raised by the author after D-16..D-19 were taken:

  PART A (OWN SIMULATION, same model as 0011-label-supply-simulation.py)
      "Is the Spearman a property of the ruler, or of the mix of right and wrong answers? And what does a cut of
       0.50 instead of 0.60 demand?"
      Model: scores of right answers ~ Normal(0,1), of wrong ones ~ Normal(-d,1), d = sqrt(2)*Phi^-1(AUC);
      n = 118 labeled answers, p = share of wrong ones, 3000 repetitions per cell, numpy default_rng(0).
      AUC = probability that a random right answer scores above a random wrong one ("ruler quality").

  PART B (OFFLINE PRE-CHECK, no API, no simulation)
      "Can token-F1 / word Recall tell a canary from a perfect answer?" Each canary is a real gold answer
      (the 15 `multi-doc` references) with ONE minimal edit that makes it false. The canary is scored as the ANSWER
      against the original gold as the REFERENCE; a perfect answer would score 1.0. Uses the P9a functions
      (evaluators.word_recall / token_f1, i.e. the ADR-010 cleaning rule).
      Limits: automatic edits of gold answers, not of real answers; few cases per type (a verb or number is only
      found in some golds).

  PART C (ARITHMETIC with the repository's wilson_ci)
      "If the LLM judge catches 90% of the wrong answers, how imprecise is that figure for a given number of wrong
       answers?"
"""
import re
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import build_golden_dataset as b  # noqa: E402
import evaluators as ev  # noqa: E402


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ====================================================================== PART A
hr("PART A - the Spearman depends on the share of wrong answers, not only on the ruler (own simulation)")
rng = np.random.default_rng(0)
N, REPS = 118, 3000
PS = (0.10, 0.20, 0.30)
AUCS = (0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
results = {}
for p in PS:
    k = max(2, round(N * p))
    for auc in AUCS:
        d = np.sqrt(2) * norm.ppf(auc)
        rhos = np.empty(REPS)
        for i in range(REPS):
            score = np.r_[rng.normal(0, 1, N - k), rng.normal(-d, 1, k)]
            label = np.r_[np.ones(N - k), np.zeros(k)]
            rhos[i] = spearmanr(score, label).statistic
        results[(p, auc)] = rhos

print("\nA1. mean Spearman of the SAME ruler, by share of wrong answers (n = 118)\n")
print(f"{'wrong':>6s} | " + " ".join(f"AUC {a:.2f}".rjust(9) for a in AUCS))
for p in PS:
    print(f"{p:6.2f} | " + " ".join(f"{results[(p, a)].mean():9.2f}" for a in AUCS))

print("\nA2. chance of passing the cut 0.50 / the cut 0.60 (share of 3000 repetitions)\n")
print(f"{'wrong':>6s} | " + " ".join(f"AUC {a:.2f}".rjust(15) for a in (0.80, 0.85, 0.90, 0.95)))
for p in PS:
    cells = [f"{np.mean(results[(p, a)] >= 0.5):.2f} / {np.mean(results[(p, a)] >= 0.6):.2f}".rjust(15)
             for a in (0.80, 0.85, 0.90, 0.95)]
    print(f"{p:6.2f} | " + " ".join(cells))

# ====================================================================== PART B
hr("PART B - can token-F1 / word Recall detect a canary? (offline pre-check on the 15 multi-doc golds)")
GOLDS = [a for _, _, a in b._MULTIDOC_EXAMPLES]
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
    """Replace the first word of `text` found in `mapping`; None if no word matches."""
    for m in re.finditer(r"\b[A-Za-z]+\b", text):
        w = m.group(0)
        if w.lower() in mapping:
            return text[:m.start()] + mapping[w.lower()] + text[m.end():]
    return None


def swap_ids(text):
    ids = list(dict.fromkeys(ev.ARXIV_ID.findall(text)))
    if len(ids) < 2:
        return None
    a, c = ids[0], ids[1]
    return text.replace(a, "@@A@@").replace(c, a).replace("@@A@@", c)


TYPES = {
    "negation": lambda t: first_sub(t, NEG),
    "number swap": lambda t: first_sub(t, NUM),
    "direction reversed": lambda t: first_sub(t, ANT),
    "paper swap (ids only)": swap_ids,
}
print("canary = gold answer with ONE edit; scored as the answer against the original gold (perfect = 1.0)\n")
print(f"{'type':24s} {'n':>3s} {'mean Recall':>12s} {'mean F1':>9s} {'share Recall>=0.9':>19s} {'share F1>=0.9':>15s}")
for name, fn in TYPES.items():
    rows = []
    for g in GOLDS:
        c = fn(g)
        if c and c != g:
            rows.append((ev.word_recall(c, g), ev.token_f1(c, g)))
    if rows:
        print(f"{name:24s} {len(rows):3d} {mean(r for r, _ in rows):12.3f} {mean(f for _, f in rows):9.3f} "
              f"{mean(r >= 0.9 for r, _ in rows):19.2f} {mean(f >= 0.9 for _, f in rows):15.2f}")
n_swap_identical = sum(
    ev.normalize_text(swap_ids(g)) == ev.normalize_text(g) for g in GOLDS if swap_ids(g))
print(f"\nafter the ADR-010 cleaning, the paper-swap canary is IDENTICAL to its original in "
      f"{n_swap_identical} of {sum(1 for g in GOLDS if swap_ids(g))} cases")
print("for comparison, a CORRECT paraphrase scored F1 0.21 / Recall 0.25 in the 2026-10-10 local test (roadmap section 7)")
example = next((g, TYPES['negation'](g)) for g in GOLDS if TYPES['negation'](g))
print("\nexample negation canary:\n  original:", example[0][:110], "\n  canary  :", example[1][:110])

# ====================================================================== PART C
hr("PART C - how imprecise is 'the judge catches 90% of the wrong answers', by number of wrong answers")
print("Wilson 95% interval (evaluators.wilson_ci) for a judge that catches 90% of the wrong answers it sees\n")
print(f"{'wrong answers':>14s} {'caught':>8s}   {'95% interval':>14s}")
for k, n in [(11, 12), (18, 20), (27, 30), (36, 40), (45, 50)]:
    lo, hi = ev.wilson_ci(k, n)
    print(f"{n:14d} {f'{k}/{n}':>8s}   [{lo:5.0%}, {hi:5.0%}]")
