# docs/ — guia de navegação

Estrutura de documentação do capstone. O plano mestre organiza o trabalho em
checkpoints (CKPT-0 → CKPT-8), cada um com gate de decisão registrado no log.

## Estrutura

```
docs/
├── README.md            ← este índice
├── plan.md              ← PLANO MESTRE: produto, golden set, avaliadores, checkpoints
├── decisions.md         ← log de decisões dos gates + observações (não-gates)
├── research/            ← fundamentação bibliográfica
│   └── rag_failure_modes_review.md
├── roadmap/             ← planos detalhados por checkpoint (um arquivo por CKPT)
│   ├── ckpt-0.5-plan.md
│   └── ckpt-0.5-review-plan.md
├── adrs/                ← decisões arquiteturais (uma por ADR)
│   ├── 0001-golden-set-provenance-data-model.md
│   ├── 0002-corpus-snapshot-pin.md
│   ├── 0003-hybrid-multidoc-slice.md
│   └── 0004-persona-audience-axis.md
└── learning-lessons/    ← aprendizados técnicos com citações auditáveis
    └── golden_dataset_construction_and_human_calibration.md
```

## Como ler (ordem sugerida)

1. [`research/rag_failure_modes_review.md`](./research/rag_failure_modes_review.md) —
   por que o projeto existe: os modos de falha de RAG da literatura.
2. [`plan.md`](./plan.md) — produto, golden set (§2), catálogo de avaliadores (§4),
   checkpoints com gates (§5).
3. [`roadmap/`](./roadmap/) — plano de implementação do checkpoint em curso
   (mini-checkpoints aprovados antes de cada lote).
4. [`adrs/`](./adrs/) — decisões arquiteturais com contexto, alternativas rejeitadas
   e consequências.
5. [`decisions.md`](./decisions.md) — o que cada experimento decidiu (PROMOTE/REJECT).
6. [`learning-lessons/`](./learning-lessons/) — descobertas técnicas feitas durante o
   desenvolvimento, com fontes.

## Convenções

- Cada checkpoint ganha um arquivo `docs/roadmap/ckpt-N-plan.md` **antes** da primeira
  linha de código; implementação só após aprovação.
- Decisões arquiteturais (mudança de design/comportamento) viram um ADR em `docs/adrs/`.
- Toda gate de experimento gera uma linha em `decisions.md`.
- Toda descoberta técnica relevante gera uma learning lesson (com citações auditáveis) e
  entra no índice do `CLAUDE.md` raiz.
