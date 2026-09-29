# Experiment Decision Log

> One entry per experiment gate. Format: date, experiment, config change, metric before → after (target slice), regressions checked, verdict (PROMOTE / REJECT), rationale.

## Observações (não-gates, achados fora do experimento)

- **2026-09-27 — dedup de chunks no top-k (achado manual).** Rodando o retriever baseline
  (k=5) em perguntas do slice `stale`, o top-5 devolveu o mesmo paper 3× (ex.: 3 chunks de
  2609.19425v1; 3 de 2609.18471v1). Colapso de diversidade — top-k dominado por um único paper.
  Verificado manualmente durante a revisão do golden set (CKPT-0.5-review). Impacto: FP3 (multi-doc
  sofre com top-k sem diversidade). Ação futura: `retrieval_precision_at_k` + métrica de diversidade
  no EXP-0; possível dedup por `arxiv_id` antes de cortar para k.

| Date | EXP | Change | Target slice | Metric before → after | Regressions | Verdict |
|------|-----|--------|--------------|-----------------------|-------------|---------|
| | EXP-0 | baseline (k=5, emb-small, chunk 500/0, prompt v1) | all | — | — | BASELINE |
| | EXP-1 | | | | | |
| | EXP-2 | | | | | |
| | EXP-3 | | | | | |
| | EXP-4 | | | | | |
| | EXP-5 | | | | | |
| | EXP-6 | | | | | |
| | EXP-7 | | | | | |
| | EXP-8 | | | | | |
