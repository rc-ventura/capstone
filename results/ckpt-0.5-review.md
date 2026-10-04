# CKPT-0.5-review — Golden Set Human Review Report

**Date:** 2026-09-27
**Scope:** item-by-item review of the 100 examples of the `core` split (docs/plan/ckpt-0.5-review-plan.md).
**Reviewer:** user (product owner).

## Final state

| Metric | Value |
|---|---|
| Total examples (core split) | 100 |
| `review.state = human_reviewed` | **100/100** |
| `authoring.origin = llm` | 99 |
| `authoring.origin = llm+human` | 1 (F01, edited by the reviewer) |
| Slices | answerable 40 · unanswerable 15 · multi-doc 15 · format 10 · persona 10 · stale 10 |

## Verdicts per slice

| Slice | Approved | Edited (human) | Regenerated (AI) | Notes |
|---|---|---|---|---|
| unanswerable (15) | 15 | 0 | 0 | topics confirmed to be outside the corpus |
| stale (10) | 10 | 0 | 0 | temporal bait validated by running the retriever manually |
| format (10) | 9 | 1 (F01, "tool-augmented" bullet made literal) | 0 | content checked against abstracts |
| multi-doc (15) | 15 | 0 | 0 | 10 known-item + 5 open-topic; literal grounding per doc |
| persona (10) | 10 | 0 | 10 (pm/phd → lay/phd axis + language) | ADR-004 decision |
| answerable (40) | 40 | 0 | 16 | see "regenerations" |

## Regenerations in `answerable` (16)

| Reason | n |
|---|---|
| generic/definitional question ("what is the main focus / purpose / how does X improve…") | 14 |
| enumeration not supported by the chunk (A22: chunk only names "H1-H5", does not enumerate) | 1 |
| generic question resistant to the generator → hand-curated (A05, numbers) | 1 |

Generation prompt reinforced (banned forms + good/bad example + requirement of a distinctive term)
and regex `_GENERIC_Q_PAT` broadened to catch "designed to do / purpose of / how does X improve /
what method is introduced / main function of".

## Design decisions made during the review

1. **Two-axis data model** (origin × review.state) — ADR-001.
2. **Persona audience axis**: `pm/phd` → `lay/phd` (technical × lay contrast) — ADR-004.
3. **Language**: golds in English (language guard in generation).
4. **Fidelity**: each gold checked against the source chunk with literal citations.

## Out-of-scope findings (non-gates)

- **Chunk dedup in the top-k** (recorded in docs/decisions.md): the baseline retriever returns the
  same paper 3× in the top-5 → diversity collapse (FP3).

## Next steps

- `docs/adrs/` (ADR-001/002/003/004) — batch R4.
- `docs/manual-usage.md` — batch R5.
- `deep-hit` (15) is born in EXP-0 (CKPT-0.8) with the mini-pooling.
- `evaluators.py` (CKPT-0.6) + wiring in main.py (0.5.4).
