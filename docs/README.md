# docs/ — navigation guide

Documentation structure of the capstone. The master plan organizes the work into
checkpoints (CKPT-0 → CKPT-8), each with a decision gate recorded in the log.

## Structure

```
docs/
├── README.md            ← this index
├── plan.md              ← MASTER PLAN: product, golden set, evaluators, checkpoints
├── decisions.md         ← decision log for the gates + observations (non-gates)
├── manual-usage.md      ← inspecting the system by hand
├── research/            ← bibliographic grounding
│   ├── rag_failure_modes_review.md
│   └── fichamentos/     ← 15 reading notes on the papers behind the evaluation design
├── roadmap/             ← detailed plans per checkpoint
│   ├── ckpt-0.5-plan.md, ckpt-0.5-review-plan.md
│   ├── ckpt-0.6-plan.md                      (evaluator catalog)
│   └── ckpt-0.6.1b-metrics-roadmap.md        (living roadmap: metrics hardening, decisions D-/L-)
├── adrs/                ← architectural decisions (one per ADR)
│   └── 0001 … 0006      (provenance, corpus pin, multi-doc slice, persona axis, retrieval unit, gold granularity)
├── learning-lessons/    ← technical learnings with auditable evidence
│   ├── golden_dataset_construction_and_human_calibration.md
│   └── retrieval_unit_and_gold_granularity.md
└── ui/                  ← plan for the Gradio web interface (not implemented)
```

## How to read (suggested order)

1. [`research/rag_failure_modes_review.md`](./research/rag_failure_modes_review.md) —
   why the project exists: the RAG failure modes from the literature.
2. [`plan.md`](./plan.md) — product, golden set (§2), evaluator catalog (§4),
   checkpoints with gates (§5).
3. [`roadmap/`](./roadmap/) — implementation plan for the checkpoint in progress
   (mini-checkpoints approved before each batch).
3b. [`research/fichamentos/`](./research/fichamentos/) — reading notes on the papers behind the evaluators and metrics.
4. [`adrs/`](./adrs/) — architectural decisions with context, rejected alternatives,
   and consequences.
5. [`decisions.md`](./decisions.md) — what each experiment decided (PROMOTE/REJECT).
6. [`learning-lessons/`](./learning-lessons/) — technical discoveries made during
   development, with sources.

## Conventions

- Each checkpoint gets a `docs/roadmap/ckpt-N-plan.md` file **before** the first
  line of code; implementation only after approval.
- Architectural decisions (design/behavior changes) become an ADR in `docs/adrs/`.
- Every experiment gate generates a line in `decisions.md`.
- Every relevant technical discovery generates a learning lesson (with auditable citations) and
  goes into the index of the root `CLAUDE.md`.
