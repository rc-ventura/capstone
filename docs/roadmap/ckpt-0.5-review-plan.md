# CKPT-0.5-review Plan — Golden Set Governance: provenance semantics, ADRs, human review, manual tutorial

> Status: **awaiting approval**. No code will be changed before the "go".
> Parent plans: `docs/plan.md` §2, `docs/plan/ckpt-0.5-plan.md`.

## 1. Context

The `core` split is complete (100 examples; `deep-hit` arrives in CKPT-0.8). During 0.5.3
we identified a **provenance honesty** problem: the `hand-written` label claimed human
authorship, but the 50 examples were drafted by AI and approved at batch level — without
item-by-item review. A golden set is the measuring instrument of the whole capstone: if its
metadata lies, every downstream comparison inherits the lie.

This governance checkpoint locks the semantics BEFORE the review, reviews the 100 examples with
the human, and records the architectural decisions made so far.

## 2. Decisions already made (in this thread, 2026-09-27)

1. **Two-axis data model** (origin × review), replacing the single `provenance` field.
2. **Three ADRs** in `docs/adrs/`.
3. **Human review of the 100 examples** in compact blocks in this conversation.

## 3. What to build (exact scope)

### 3.1 Data model — two orthogonal axes

Problem with the current model: a single `provenance` field mixes *who wrote it* with
*who reviewed it*, allowing contradictory states ("curated" without review). New model:

| Field | Values | Meaning |
|---|---|---|
| `metadata.authoring.origin` | `llm` \| `human` \| `llm+human` | who wrote it (`llm+human` = a human edited the draft) |
| `metadata.authoring.method` | `from-chunk` \| `topic-authored` \| `empirical` | how it was built (from a chunk / by topic / found in the baseline) |
| `metadata.review.state` | `draft` \| `spot_checked` \| `human_reviewed` \| `adjudicated` | depth of human verification |
| `metadata.review.reviewer` | `user` \| absent | who reviewed (auditability) |

Lifecycle:

```
origin=llm, state=draft              ← estado atual dos 100
   │  human review session (this checkpoint)
   ▼  if approved without edits      → origin=llm,        state=human_reviewed
      if approved with edits         → origin=llm+human,  state=human_reviewed
   │  EXP-0 mini-pooling (CKPT-0.8)
   ▼  state=adjudicated + gold_arxiv_ids atualizados
```

**Consistency rule** (a chimera is impossible by construction): `state ∈ {human_reviewed,
adjudicated}` requires `reviewer` to be present; `origin=human` is reserved for human-written text
(no example today); `origin=llm+human` requires at least one text field edited by the human.

The old `provenance` field is **removed** in the migration (not just renamed) so as not to leave
two sources of truth. `slice`, `base_id`, `audience`, `source_arxiv_id`, `corpus_snapshot`,
`open_topic`, `retrieval_gold`, `regenerated` remain untouched.

### 3.2 Human review of the 100 examples (in this conversation)

Mechanics:

1. I print the examples in blocks of ~10 in the format:
   `A07 [2609.19934v1] P: pergunta… | G: gabarito…` (+ `FONTE:` chunk excerpt for synthetic).
2. You reply in batch by id: `ok`, `edita: <texto novo>`, or `reprova: <motivo>`.
3. I apply via `update_example`: approved → `state=human_reviewed`; edited →
   `origin=llm+human` + your text; rejected → I remove it from the dataset and regenerate/rewrite (keeps the target count).
4. At the end, report: N approved / N edited / N regenerated.

Block order: `unanswerable`(15), `stale`(10), `format`(10), `multi-doc`(15), then
`persona`(10) and `answerable`(40) — the synthetic ones last, since they require checking against the source chunk.

### 3.3 ADRs in `docs/adrs/`

| ADR | Decision | Rejected alternatives |
|---|---|---|
| **ADR-001** | Two-axis data model (origin × review state) | single `provenance` field (curated-without-review chimera); boolean flag only |
| **ADR-002** | Corpus snapshot pin (`CORPUS_SNAPSHOT_DATE` 2026-09-17 + additive skip-budget) | multiplicative oversample (real failure: 766 papers post-pin, ~128/day); no pin (`stale` slice dies on the CKPT-2 rebuild) |
| **ADR-003** | Hybrid `multi-doc`: 10 known-item + 5 open-topic with gold pending adjudication | 15 pure known-item (unrealistic for production); 15 pure open (retrieval unmeasurable without annotation) |

Format: standard template (Status / Context / Decision / Consequences) with date and links to the
plan and to the learning lesson.

### 3.4 Field-semantics doc

`docs/plan/ckpt-0.5-plan.md` §4.3 gets the final data-model table (replacing the provisional
correction note) + the lifecycle diagram. `docs/README.md` lists `docs/adrs/`.

### 3.5 Manual-usage tutorial for the system

New: `docs/manual-usage.md` — how to inspect the system's behavior without the golden set:

```bash
uv run python main.py                 # pipeline completo (corpus → resposta traceada)
uv run python build_golden_dataset.py # sync idempotente do golden set
```

And REPL snippets for ad-hoc analysis:
- run only the retriever with a custom k on a free-form question (inspects recall by hand);
- run the pipeline with a free-form question and see the answer;
- where to look in LangSmith (project, filters by run_type/metadata, thread_id).

## 4. Out of scope

- `deep-hit` (empirical, CKPT-0.8), `evaluators.py` (0.6), wiring into `main.py` (0.5.4),
  mini-pooling (0.8). No changes to `app.py`.

## 5. Verification / acceptance criteria

1. 100/100 examples with the new schema (`provenance` absent; both axes present and valid).
2. Consistency rule verified by script (state→reviewer; origin=llm+human→edit exists).
3. After the review: verdict report saved in `results/` (approved/edited/regenerated per slice).
4. `python -m pytest`-free: validation by a single script running against the server (as in the previous batches).
5. ADRs render; the tutorial runs (`main.py` and a REPL snippet actually tested).

## 6. Batches (internal mini-checkpoints), each with its own commit

| Batch | Content |
|---|---|
| R1 | Data-model migration on the 100 examples + validators (script) |
| R2 | Human review: curated blocks (unanswerable/stale/format/multi-doc) |
| R3 | Human review: synthetic blocks (persona/answerable) + regenerations |
| R4 | ADR-001/002/003 + semantics doc in the plan + docs README |
| R5 | `docs/manual-usage.md` tested |

The order puts the migration before the review because the human verdict **writes** to the new fields;
ADRs after the review to incorporate any learning from it.
