# CKPT-0.5-review Plan — Golden Set Governance: provenance semantics, ADRs, human review, manual tutorial

> Status: **aguardando aprovação**. Nenhum código será alterado antes do "go".
> Parent plans: `docs/plan.md` §2, `docs/plan/ckpt-0.5-plan.md`.

## 1. Contexto

O split `core` está completo (100 exemplos; `deep-hit` chega no CKPT-0.8). Durante o 0.5.3
identificamos um problema de **honestidade de proveniência**: o rótulo `hand-written` afirmava
autoria humana, mas os 50 exemplos foram redigidos por IA e aprovados em nível de lote — sem
revisão item a item. Um golden set é o instrumento de medida de todo o capstone: se seus
metadados mentem, todas as comparações downstream herdam a mentira.

Este checkpoint de governança trava a semântica ANTES da revisão, revisa os 100 exemplos com
o humano, e registra as decisões arquiteturais tomadas até aqui.

## 2. Decisões já tomadas (nesta thread, 2026-09-27)

1. **Data model de dois eixos** (origem × revisão), substituindo o campo único `provenance`.
2. **Três ADRs** em `docs/adrs/`.
3. **Revisão humana dos 100 exemplos** em blocos compactos nesta conversa.

## 3. O que construir (escopo exato)

### 3.1 Data model — dois eixos ortogonais

Problema do modelo atual: um único campo `provenance` mistura *quem escreveu* com
*quem revisou*, permitindo estados contraditórios ("curated" sem revisão). Modelo novo:

| Campo | Valores | Significado |
|---|---|---|
| `metadata.authoring.origin` | `llm` \| `human` \| `llm+human` | quem escreveu (`llm+human` = humano editou o rascunho) |
| `metadata.authoring.method` | `from-chunk` \| `topic-authored` \| `empirical` | como foi construído (de chunk / por tema / encontrado no baseline) |
| `metadata.review.state` | `draft` \| `spot_checked` \| `human_reviewed` \| `adjudicated` | profundidade da verificação humana |
| `metadata.review.reviewer` | `user` \| ausente | quem revisou (auditabilidade) |

Ciclo de vida:

```
origin=llm, state=draft              ← estado atual dos 100
   │  sessão de revisão humana (este checkpoint)
   ▼  se aprovado sem edição        → origin=llm,        state=human_reviewed
      se aprovado com edição        → origin=llm+human,  state=human_reviewed
   │  EXP-0 mini-pooling (CKPT-0.8)
   ▼  state=adjudicated + gold_arxiv_ids atualizados
```

**Regra de consistência** (impossível por construção ter quimera): `state ∈ {human_reviewed,
adjudicated}` exige `reviewer` presente; `origin=human` é reservado a texto escrito por humano
(nenhum exemplo hoje); `origin=llm+human` exige pelo menos um campo textual editado pelo humano.

O campo antigo `provenance` é **removido** na migração (e não apenas renomeado) para não deixar
duas fontes de verdade. `slice`, `base_id`, `audience`, `source_arxiv_id`, `corpus_snapshot`,
`open_topic`, `retrieval_gold`, `regenerated` permanecem intocados.

### 3.2 Revisão humana dos 100 exemplos (nesta conversa)

Mecânica:

1. Eu imprimo os exemplos em blocos de ~10 no formato:
   `A07 [2609.19934v1] P: pergunta… | G: gabarito…` (+ `FONTE:` trecho do chunk para synthetic).
2. Você responde em lote por id: `ok`, `edita: <texto novo>`, ou `reprova: <motivo>`.
3. Eu aplico via `update_example`: aprovado → `state=human_reviewed`; editado →
   `origin=llm+human` + seu texto; reprovado → removo do dataset e regenero/reescrevo (mantém contagem-alvo).
4. Ao final, relatório: N aprovados / N editados / N regenerados.

Ordem dos blocos: `unanswerable`(15), `stale`(10), `format`(10), `multi-doc`(15), depois
`persona`(10) e `answerable`(40) — os sintéticos por último, pois exigem conferir contra o chunk-fonte.

### 3.3 ADRs em `docs/adrs/`

| ADR | Decisão | Alternativas rejeitadas |
|---|---|---|
| **ADR-001** | Data model de dois eixos (origin × review state) | campo único `provenance` (quimera curated-sem-revisão); apenas flag booleana |
| **ADR-002** | Corpus snapshot pin (`CORPUS_SNAPSHOT_DATE` 2026-09-17 + skip-budget aditivo) | oversample multiplicativo (falha real: 766 papers pós-pin, ~128/dia); sem pin (slice `stale` morre no rebuild do CKPT-2) |
| **ADR-003** | `multi-doc` híbrido 10 known-item + 5 open-topic com gold pendente de adjudicação | 15 known-item puros (irreal p/ produção); 15 open puros (retrieval imensurável sem anotação) |

Formato: template padrão (Status / Context / Decision / Consequences) com data e links para o
plano e para a learning lesson.

### 3.4 Doc de semântica dos campos

`docs/plan/ckpt-0.5-plan.md` §4.3 ganha a tabela final do data model (substitui a nota de
correção provisória) + o diagrama de ciclo de vida. `docs/README.md` lista `docs/adrs/`.

### 3.5 Tutorial de uso manual do sistema

Novo: `docs/manual-usage.md` — como inspecionar o comportamento do sistema sem o golden set:

```bash
uv run python main.py                 # pipeline completo (corpus → resposta traceada)
uv run python build_golden_dataset.py # sync idempotente do golden set
```

E snippets REPL para análise pontual:
- rodar só o retriever com k custom em uma pergunta livre (inspeciona recall na mão);
- rodar o pipeline com pergunta livre e ver a resposta;
- onde olhar no LangSmith (projeto, filtros por run_type/metadata, thread_id).

## 4. Fora de escopo

- `deep-hit` (empírico, CKPT-0.8), `evaluators.py` (0.6), wiring no `main.py` (0.5.4),
  mini-pooling (0.8). Nenhuma mudança em `app.py`.

## 5. Verificação / critérios de aceite

1. 100/100 exemplos com o schema novo (`provenance` ausente; dois eixos presentes e válidos).
2. Regra de consistência verificada por script (state→reviewer; origin=llm+human→edição existe).
3. Após a revisão: relatório de vereditos salvo em `results/` (aprovação/editado/regenerado por slice).
4. `python -m pytest`-free: validação por script único rodando contra o servidor (como nos lotes anteriores).
5. ADRs renderizam; tutorial executa (`main.py` e um snippet REPL testados de verdade).

## 6. Lotes (mini-checkpoints internos), cada um com commit próprio

| Lote | Conteúdo |
|---|---|
| R1 | Migração do data model nos 100 exemplos + validadores (script) |
| R2 | Revisão humana: blocos curated (unanswerable/stale/format/multi-doc) |
| R3 | Revisão humana: blocos sintéticos (persona/answerable) + regenerações |
| R4 | ADR-001/002/003 + doc de semântica no plano + README docs |
| R5 | `docs/manual-usage.md` testado |

A ordem coloca a migração antes da revisão porque o veredito humano **escreve** nos campos novos;
ADRs depois da revisão para incorporar qualquer aprendizado dela.
