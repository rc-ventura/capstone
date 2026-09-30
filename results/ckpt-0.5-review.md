# CKPT-0.5-review — Relatório de Revisão Humana do Golden Set

**Data:** 2026-09-27
**Escopo:** revisão item-a-item dos 100 exemplos do split `core` (docs/plan/ckpt-0.5-review-plan.md).
**Revisor:** usuário (proprietário do produto).

## Estado final

| Métrica | Valor |
|---|---|
| Total de exemplos (split core) | 100 |
| `review.state = human_reviewed` | **100/100** |
| `authoring.origin = llm` | 99 |
| `authoring.origin = llm+human` | 1 (F01, editado pelo revisor) |
| Slices | answerable 40 · unanswerable 15 · multi-doc 15 · format 10 · persona 10 · stale 10 |

## Vereditos por slice

| Slice | Aprovado | Editado (humano) | Regenerado (IA) | Observações |
|---|---|---|---|---|
| unanswerable (15) | 15 | 0 | 0 | temas confirmadamente fora do corpus |
| stale (10) | 10 | 0 | 0 | isca temporal validada rodando o retriever manualmente |
| format (10) | 9 | 1 (F01, bullet do "tool-augmented" tornado literal) | 0 | conteúdo conferido contra abstracts |
| multi-doc (15) | 15 | 0 | 0 | 10 known-item + 5 open-topic; fundamentação literal por doc |
| persona (10) | 10 | 0 | 10 (eixo pm/phd → lay/phd + idioma) | decisão ADR-004 |
| answerable (40) | 40 | 0 | 16 | ver "regenerações" |

## Regenerações no `answerable` (16)

| Motivo | n |
|---|---|
| pergunta genérica/definicional ("what is the main focus / purpose / how does X improve…") | 14 |
| enumeração não-suportada pelo chunk (A22: chunk só nomeia "H1-H5", não enumera) | 1 |
| pergunta genérica resistente ao gerador → curada à mão (A05, números) | 1 |

Prompt de geração reforçado (formas banidas + exemplo bom/ruim + exigência de termo distintivo)
e regex `_GENERIC_Q_PAT` ampliado para capturar "designed to do / purpose of / how does X improve /
what method is introduced / main function of".

## Decisões de design tomadas durante a revisão

1. **Data model de dois eixos** (origin × review.state) — ADR-001.
2. **Eixo de audiência do persona**: `pm/phd` → `lay/phd` (contraste técnico×leigo) — ADR-004.
3. **Idioma**: golds em inglês (guarda de idioma na geração).
4. **Fidelidade**: cada gold conferido contra o chunk-fonte com citações literais.

## Achados fora do escopo (não-gates)

- **Dedup de chunks no top-k** (registrado em docs/decisions.md): retriever baseline devolve o
  mesmo paper 3× no top-5 → colapso de diversidade (FP3).

## Próximos passos

- `docs/adrs/` (ADR-001/002/003/004) — lote R4.
- `docs/manual-usage.md` — lote R5.
- `deep-hit` (15) nasce no EXP-0 (CKPT-0.8) com o mini-pooling.
- `evaluators.py` (CKPT-0.6) + wiring em main.py (0.5.4).
