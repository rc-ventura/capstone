"""Evidence for ADR-011 (calibration design): how many WRONG answers the author must label for the
C1 pass mark (Spearman >= 0.60, ADR-008 D-13) to be reachable.

Run from the repository root:

    uv run python docs/adrs/evidence/0011-label-supply-simulation.py

Recorded output: 0011-label-supply-simulation.out.txt

This is an OWN SIMULATION, not a result from a paper. It answers one question:
    "If a correctness signal separates right answers from wrong ones this well, and the labeled set
     has this share of wrong answers, how often does Spearman(signal, label) reach 0.60?"

Model (deliberately simple, so each step can be checked by hand):
  * label: 1 = answer judged correct, 0 = incorrect; p = share of INCORRECT answers; n = labeled answers.
  * signal score of a correct answer ~ Normal(0, 1); of an incorrect one ~ Normal(-d, 1).
  * d is set from the AUC: the probability that a random correct answer scores higher than a random
    incorrect one. For two unit-variance normals, AUC = Phi(d / sqrt(2)), so d = sqrt(2) * Phi^-1(AUC).
    AUC 0.80 = "decent", 0.90 = "good" (right-vs-wrong pair ordered correctly 9 times in 10),
    0.95 = "very good".
  * Spearman is computed between the score and the 0/1 label (what Adlakha 2024 did).
  * 4000 repetitions per cell, numpy default_rng(0). The result is the share of repetitions with rho >= 0.60.
Limits: normal scores and independent answers. Real answers come in pairs (two EXP-0 rounds of the same
question) and real scores are not normal, so read the numbers as an order of magnitude.
"""
import numpy as np
from scipy.stats import norm, spearmanr

rng = np.random.default_rng(0)
REPS = 4000
PASS = 0.60


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------- PART 1: where the labeled answers come from
hr("PART 1 - how many answers can be labeled (counts from the roadmap, not recomputed here)")
ANSWERABLE, PERSONA = 40, 10          # golden set core slices (roadmap section 3)
MULTIDOC_KNOWN_ITEM, MULTIDOC_FOUND = 10, 9   # baseline hit@5 = 0.90 on the 10 known-item (roadmap P5 check)
MULTIDOC_OPEN_TOPIC = 5               # no gold paper yet (pending adjudication) -> not eligible
retrieval_hit_answerable_persona = 1.0        # hit@5 = 100% on both (roadmap P5 check)
eligible = ANSWERABLE + PERSONA + MULTIDOC_FOUND
print(f"eligible questions = {ANSWERABLE} answerable + {PERSONA} persona + {MULTIDOC_FOUND} multi-doc "
      f"(of {MULTIDOC_KNOWN_ITEM} known-item where the gold paper was retrieved) = {eligible}")
print(f"(the {MULTIDOC_OPEN_TOPIC} open-topic multi-doc have no gold paper until the mini-pooling adjudicates them)")
print(f"EXP-0 runs twice (docs/plan.md CKPT-0): answers = {eligible} x 2 = {eligible * 2}; "
      "each question appears twice, so the answers are NOT independent")

# ---------------------------------------------------------------- PART 2: the ceiling
hr("PART 2 - the maximum Spearman a PERFECT signal can reach, by share of wrong answers")
print("Derivation: with a perfectly separating score the ranks are uniform and the label is 1[rank > 1-p];")
print("Cov(rank, label) = p(1-p)/2 and Var(rank) = 1/12, Var(label) = p(1-p), so rho = sqrt(3 p (1-p)).")
print("Check by simulation (correct answers score in (1,2), wrong in (0,1), n = 1000):\n")
print(f"{'p wrong':>8s} {'formula':>8s} {'simulated':>10s}")
for p in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
    n = 1000
    k = round(n * p)
    score = np.r_[rng.uniform(1, 2, n - k), rng.uniform(0, 1, k)]
    label = np.r_[np.ones(n - k), np.zeros(k)]
    print(f"{p:8.2f} {np.sqrt(3 * p * (1 - p)):8.3f} {spearmanr(score, label).statistic:10.3f}")

# ---------------------------------------------------------------- PART 3: chance of passing C1
hr("PART 3 - chance that Spearman >= 0.60, by share of wrong answers (p), labeled set size (n) and signal quality")
print("cell = P(rho >= 0.60) over 4000 repetitions\n")
AUCS = (0.80, 0.90, 0.95)
print(f"{'n':>4s} {'p wrong':>8s} {'#wrong':>7s} | " + " ".join(f"AUC {a:.2f}".rjust(9) for a in AUCS))
for n in (118, 150, 200, 250):
    for p in (0.10, 0.20, 0.25, 0.30, 0.35, 0.40):
        k = max(2, round(n * p))
        cells = []
        for auc in AUCS:
            d = np.sqrt(2) * norm.ppf(auc)
            hits = 0
            for _ in range(REPS):
                good = rng.normal(0, 1, n - k)
                bad = rng.normal(-d, 1, k)
                rho = spearmanr(np.r_[good, bad], np.r_[np.ones(n - k), np.zeros(k)]).statistic
                hits += rho >= PASS
            cells.append(hits / REPS)
        print(f"{n:4d} {p:8.2f} {k:7d} | " + " ".join(f"{c:9.2f}" for c in cells))
    print()

# ---------------------------------------------------------------- PART 4: effort
hr("PART 4 - how many EXTRA answers from a worse system version are needed to reach the target share")
TARGET = 0.30
REAL_N = 118
print(f"Real answers: {REAL_N}. If k of them are wrong and the worse version is wrong a share e of the time,")
print(f"adding m answers gives (k + e*m) / ({REAL_N} + m) = {TARGET}  =>  m = ({TARGET}*{REAL_N} - k) / (e - {TARGET}).\n")
print(f"{'real wrong k':>12s} | " + " ".join(f"e={e:.1f}: m (total)".rjust(18) for e in (0.4, 0.5, 0.7)))
for k in (6, 12, 24):
    row = []
    for e in (0.4, 0.5, 0.7):
        m = (TARGET * REAL_N - k) / (e - TARGET)
        row.append(f"{m:5.0f} ({REAL_N + m:4.0f})".rjust(18))
    print(f"{k:12d} | " + " ".join(row))
print("\nThe more the worse version errs (e), the fewer extra answers are needed - and the less it resembles the")
print("real system. 'total' is the number of answers the author labels.")
