
## Learning Lessons

> Folder: `./docs/learning-lessons/`

- [Golden Dataset Construction for RAG: synthetic-by-construction vs. human adjudication vs. judge calibration](./docs/learning-lessons/golden_dataset_construction_and_human_calibration.md) — 2026-09-24
- [Retrieval Unit and Gold Granularity for RAG over Abstracts: Chunk vs. Paper vs. Whole Abstract](./docs/learning-lessons/retrieval_unit_and_gold_granularity.md) — 2026-10-03

## ADRs — Technical Decisions

> Folder: `./docs/adrs/`

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](./docs/adrs/0001-golden-set-provenance-data-model.md) | Golden set provenance: two-axis data model | Accepted | 2026-09-27 |
| [ADR-002](./docs/adrs/0002-corpus-snapshot-pin.md) | Corpus snapshot pin (frozen-at-date) + additive skip budget | Accepted | 2026-09-24 |
| [ADR-003](./docs/adrs/0003-hybrid-multidoc-slice.md) | Hybrid `multi-doc` slice (10 known-item + 5 open-topic) | Accepted | 2026-09-27 |
| [ADR-004](./docs/adrs/0004-persona-audience-axis.md) | Persona audience axis: lay vs phd (stark contrast) | Accepted | 2026-09-27 |
| [ADR-005](./docs/adrs/0005-retrieval-unit-whole-abstract.md) | Retrieval unit for the baseline index: one record per whole abstract | Accepted | 2026-10-03 |
| [ADR-006](./docs/adrs/0006-retrieval-gold-granularity-and-metrics.md) | Retrieval gold granularity and metric reporting policy | Accepted | 2026-10-03 |
