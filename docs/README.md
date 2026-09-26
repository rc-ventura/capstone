# docs/ — guia de navegação

Estrutura de documentação do capstone. O plano mestre organiza o trabalho em
checkpoints (CKPT-0 → CKPT-8), cada um com gate de decisão registrado no log.

## Estrutura

```
docs/
├── README.md            ← este índice
├── plan.md              ← PLANO MESTRE: produto, golden set, avaliadores, checkpoints
├── decisions.md         ← log de decisões dos gates (uma linha por experimento)
├── research/            ← fundamentação bibliográfica
│   └── rag_failure_modes_review.md
├── plan/                ← planos detalhados por checkpoint (um arquivo por CKPT)
│   └── ckpt-0.5-plan.md
└── learning-lessons/    ← aprendizados técnicos com citações auditáveis
    └── golden_dataset_construction_and_human_calibration.md
```

## Como ler (ordem sugerida)

1. [`research/rag_failure_modes_review.md`](./research/rag_failure_modes_review.md) —
   por que o projeto existe: os modos de falha de RAG da literatura.
2. [`plan.md`](./plan.md) — produto, golden set (§2), catálogo de avaliadores (§4),
   checkpoints com gates (§5).
3. [`plan/`](./plan/) — plano de implementação do checkpoint em curso
   (mini-checkpoints aprovados antes de cada lote).
4. [`decisions.md`](./decisions.md) — o que cada experimento decidiu (PROMOTE/REJECT).
5. [`learning-lessons/`](./learning-lessons/) — descobertas técnicas feitas durante o
   desenvolvimento, com fontes.

## Convenções

- Cada checkpoint ganha um arquivo `docs/plan/ckpt-N-plan.md` **antes** da primeira linha
  de código; implementação só após aprovação.
- Toda gate de experimento gera uma linha em `decisions.md`.
- Toda descoberta técnica relevante gera uma learning lesson (com citações auditáveis) e
  entra no índice do `CLAUDE.md` raiz.
