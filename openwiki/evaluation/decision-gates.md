---
type: governance model
title: Experiments & Decision Gates
description: The checkpoint-and-experiment governance model — the CKPT-0 through CKPT-8 ladder, the EXP series each ending in a PROMOTE/REJECT gate, the decision log, ADR conventions, and the rule that every failure point gets an evaluator, a slice, an experiment, and a production metric.
tags: [experiments, decision-gates, checkpoints, adr, governance, roadmap]
sources:
  - id: openwiki-source-a2371d6362e5db4bc834ad03
    resource: repo://CLAUDE.md
  - id: openwiki-source-8f04a5996c3b2c8616f1e345
    resource: repo://docs/adrs/0009-incomplete-gold-and-pool-bias-mini-pooling.md
  - id: openwiki-source-5e8cfa6ef22542c9ea8aaf4a
    resource: repo://docs/decisions.md
  - id: openwiki-source-0efdc1e16f1045bb19dad3ef
    resource: repo://docs/plan.md
  - id: openwiki-source-196170e31ff8ec60a116165b
    resource: repo://docs/README.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "claude-code", at: "2026-10-12T14:24:36.766Z" }
verified:
  - by: openwiki/0.6.0
    at: 2026-10-12T14:24:36.766Z
---

# Experiments & Decision Gates

Development is organized as **checkpoints** (CKPT-0 → CKPT-8), and every checkpoint centers on an **experiment** (EXP-0 → EXP-7) — a controlled change to exactly one configuration axis measured against the golden set. Each experiment ends in a **decision gate**: a PROMOTE or REJECT verdict on whether the change becomes the new reference configuration.

## The checkpoint contract

Every checkpoint follows the same template (`docs/plan.md` §5):

```
HYPOTHESIS → CONFIG CHANGE → TARGET SLICES → EVALUATORS → DECISION GATE → ARTIFACT
```

Two rules apply to all gates:

- **No regression on non-target slices** (Δfaithfulness > −2 pts, Δcost within budget) — a gain on the target slice cannot be bought by hurting others.
- **Statistical floors scale with the set**: core-set (~100 examples) gates require a **≥15% delta** to be credible (slices of n=10–15 have no power at 10%); the extended set (~200) is reserved for CKPT-7's pairwise A/B, where ≥10% or a ≥60% non-tie win rate suffices because pairwise judging has lower variance than absolute scores.

## The ladder

| CKPT | Experiment | Axis changed | Target slices | Gate |
|---|---|---|---|---|
| 0 | EXP-0 | none — baseline measurement, **run twice** to fix the judge noise floor | all | none; produces the baseline metrics table + judge variance report |
| 1 | EXP-1 | `k` ∈ {3, 5, 10} | retrieval | best recall@k under latency < 4 s and cost budget — pick the knee of the curve |
| 2 | EXP-2 | embedding model (3-small vs 3-large) | retrieval | promote large only if recall@5 gain ≥ 3 pts (it costs ~5× more) |
| 3 | EXP-3 | chunking (whole abstract vs 500/0, maybe 1000/200) | retrieval + faithfulness | best recall@k that doesn't regress faithfulness > 2 pts, compared at paper level |
| 4 | EXP-4 | LLM rerank (top-20 scored 0–10, cut to k) | `deep-hit` | adopt if MRR gain ≥ 15% AND added latency < 800 ms/query |
| 5 | EXP-5 | rewrite × decompose (4 arms) | `deep-hit`, `multi-doc` | each arm adopted independently if its target slice improves ≥15% without hurting `answerable` |
| 6 | EXP-6 | prompt versions v1/v2/v3 via Prompt Hub | `unanswerable`, `answerable`, `format`, `persona` | highest-scoring version keeping correct abstention ≥ 90% (release-blocking) |
| 7 | EXP-7 | final A/B: best config vs baseline, pairwise judge | extended set | ≥60% of non-tie comparisons → ship candidate |
| 8 | — | production: FastAPI server, online evals on 1–5% traffic, feedback queue, monitors, corpus refresh drill | live | monitors live + one user-feedback round-trip + one routed annotation queue |

EXP-0's "run twice" is foundational: it measures LLM-judge inter-run variance (target ≤10%, else `num_repetitions` is increased), which is the noise floor every later gate's delta must exceed.

## The decision log

`docs/decisions.md` is the gate ledger — one row per experiment: date, config change, metric before → after on the target slice, regressions checked, verdict (PROMOTE/REJECT). It also has an **Observations** section for non-gate findings that change the system anyway; load-bearing entries so far:

- **2026-10-04** — the date-scan corpus rebuild broke (snapshot pin skip budget exhausted), producing the manifest rebuild-by-id mechanism and a documented whitespace subtlety in reassembled abstracts.
- **2026-10-12** — the lexical correctness signals landed (P9a; ADR-008/010): `f1_summary_evaluator` became `token_f1_summary`, reporting mean token-F1 **and** word Recall scoped to `answerable`/`persona`/`multi-doc`, both diagnostics until EXP-0 calibration; the `f1_summary` key no longer exists.
- **2026-10-06** — the abstention-metrics redesign: `abstention_accuracy` was removed in favor of recall + over-refusal + Wilson CIs, and a new gate regression rule was added (*a gain in abstention_recall cannot come from an increase in over_refusal_rate*).
- **2026-09-27** — manual finding of chunk-level top-k diversity collapse (same paper 3× in top-5), which fed the ADR-005 retrieval-unit decision.

## ADR and documentation conventions

`docs/README.md` fixes the governance conventions:

- **Plan before code**: each checkpoint gets a `docs/roadmap/ckpt-N-plan.md` approved before implementation; mini-checkpoints (0.5 golden set, 0.6 evaluators, 0.6.1b metrics hardening) get their own plans.
- **ADR per architectural decision** (`docs/adrs/`): ten accepted — provenance data model, corpus snapshot pin, multi-doc hybrid slice, persona axis, retrieval unit, gold granularity/metrics, abstention detection/reporting, answer-correctness signal ordering (ADR-008), incomplete gold & pool-bias mini-pooling (ADR-009), and the text-normalization rule for lexical correctness signals (ADR-010). ADRs record context, rejected alternatives, consequences, and *conditions that invalidate the decision*.
- **Learning lessons** (`docs/learning-lessons/`): technical discoveries with auditable evidence, indexed in `CLAUDE.md`.
- **Living roadmap**: `docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md` is the continuously-updated metrics-hardening document — it holds the D-* decision numbering (D-1 paper level, D-2 reporting policy, D-5/D-6 abstention, D-8 gold completeness) that code comments reference, plus the P1–P11 proposal log marking which fixes are executed.

## The coverage rule

`docs/plan.md` §9 defines done: **every failure point** in the research review must have (a) a measuring evaluator, (b) a golden-set slice, (c) a mitigation experiment with a logged decision, and (d) a production metric watching it. The README's FP1–FP7 table is the current mapping; the `docs/research/fichamentos/` reading notes (21 papers) are the evidence base the evaluator design cites.

## Status

Done: CKPT-0 foundation (corpus, traced pipeline, baseline config), CKPT-0.5 golden set (100 examples, human review 100/100), CKPT-0.6 retrieval + format/summary evaluators, metrics hardening P1–P8 and P9a (lexical correctness signals — the embedding-cosine evaluator, canaries and calibration are still pending). Planned next: LLM judges (CKPT-0.6.3), experiment runner (CKPT-0.7), EXP-0 baseline (CKPT-0.8), then CKPT-1→8. The `experiments/` runner, `server.py`, `monitor.py`, `feedback.py`, and `prompt_management.py` in the plan's directory layout do not exist yet.
