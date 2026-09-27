# CKPT-0.5 Plan — Golden Dataset (core split)

> Status: **aguardando aprovação**. Nenhum código será alterado antes do "go".
> Parent plan: `docs/plan.md` §2 (Golden Set) + §5 (CKPT-0 contract).

## 1. Contexto — onde estamos

CKPT-0 (Foundation & Baseline) está sendo entregue em lotes pequenos:

| Lote | Entrega | Commit |
|---|---|---|
| 0.1 | scaffold do projeto | `e787586` |
| 0.2 | `config.py` + `RunConfig` | `27e3130` |
| 0.3 | ingestão arXiv + vector store (parquet) + `main.py` | `6b14177` |
| 0.4 | pipeline RAG traceado (`app.py`, 3 métodos de tracing) | `bade39e` |
| **0.5** | **`build_golden_dataset.py` (split core)** | **este lote** |
| 0.6 | `evaluators.py` (catálogo §4) | próximo |
| 0.7 | `experiments/run_experiment.py` + EXP-0 rodada 1 | |
| 0.8 | EXP-0 rodada 2 (variância do juiz) + slice `deep-hit` + `results/EXP-0-baseline.md` | |

## 2. O que construir (escopo exato deste lote)

1. **`build_golden_dataset.py`** — constrói o dataset `arxiv-copilot-golden` (split `core`) com **6 slices e ~100 exemplos**.
2. **`utils.py`** — dois acréscimos pequenos:
   - `chunk_id(doc)` — identidade determinística de chunk (ver §4.1);
   - pin de snapshot do corpus (ver §4.2).
3. **`config.py`** — `CORPUS_SNAPSHOT_DATE` + contagens-alvo por slice.
4. **`main.py`** — ligar `build_golden_dataset()` no fluxo (o stub do 0.4 já anuncia isso).

**Fora de escopo (explícito)**: `evaluators.py`, runner de experimentos, slice `deep-hit` (definido empiricamente no EXP-0 — `docs/plan.md` §2), split `extended` (~200 ex; só usado no CKPT-7 — vira lote próprio antes dele), qualquer mudança em `app.py`.

Após este lote o split core terá 100 exemplos — número que fecha com o "~100" do §2 (a tabela do §2 soma 115 porque inclui os 15 de `deep-hit`, que só existem depois do EXP-0).

## 3. Por quê (fundamentação)

- **Contrato do plano**: CKPT-0 exige o golden set visível no LangSmith com todas as slices antes do EXP-0 (`docs/plan.md` §5, verificação). As slices existem porque cada FP do `docs/research/rag_failure_modes_review.md` precisa de (a) um avaliador, (b) uma slice, (c) um experimento — este lote entrega (b).
- **`deep-hit` adiado, propositalmente**: a tabela do §2 define-o como "found empirically: questions whose answer doc ranks low in baseline". Construí-lo agora seria chute. Ele nasce do EXP-0 (lote 0.8), preservando a honestidade do rótulo.
- **Split extended adiado**: §2/§8 reservam-no ao CKPT-7; gastar ~200 gerações LLM agora anteciparia custo sem consumidor.
- **Ground-truth map já definido** (`docs/plan.md` §2b): este lote só precisa materializar os rótulos que a tabela exige — nada além.

## 4. Decisões de design (com evidência do codebase)

### 4.1 `chunk_id` determinístico — sem rebuild de cache

**Problema**: o §2 exige "Source chunk ID" como gold retrieval label, mas o `SKLearnVectorStore` gera IDs UUID aleatórios por chunk (verificado: `ids` do parquet = `f665f189-…`). Rótulos de retrieval são inúteis sem identidade estável e reproduzível.

**Decisão**: `utils.chunk_id(doc) = f"{arxiv_id}:{sha1(page_content)[:12]}"`. Derivado de campos já persistidos (sem mudança de schema, sem rebuild do parquet). A mesma função é usada pelo construtor do dataset e, depois (0.6), pelos avaliadores de retrieval ao inspecionar docs recuperados — colisão prática zero (chunks distintos têm conteúdo distinto).

Alternativa rejeitada: estampar `chunk_id` no metadata em `chunk_documents` + rebuild do cache — mesma garantia, mas queima a snapshot atual (ver 4.2) e re-embeda 843 chunks sem ganho.

### 4.2 Pin de snapshot do corpus (`stale` slice)

**Evidência**: o corpus cacheado cobre papers publicados entre **2026-09-16 e 2026-09-17**; hoje (build do lote) é 2026-09-24. Ou seja, existe ~1 semana de papers reais publicados *depois* do conteúdo do corpus.

**Problema**: hoje `fetch_arxiv_corpus` sempre pega os 250 mais recentes. Qualquer rebuild futuro (CKPT-2 troca o embedding → novo parquet) puxaria papers novos e *invalidaria acidentalmente* o slice `stale`.

**Decisão** (recomendação ao capstone): `config.CORPUS_SNAPSHOT_DATE = "2026-09-17"` (= máx. `published` do cache atual). `fetch_arxiv_corpus` passa a filtrar client-side `published.date() <= snapshot`, com oversampling (busca ~2× os resultados, filtra, corta em 250). O cache existente **permanece válido** (seu max já é 17/09). Assim:

- **pré-refresh** (agora até CKPT-7): perguntas `stale` sobre papers ≥ 2026-09-18 → gold = abstention ("No papers found in this period");
- **refresh drill (CKPT-8)**: bump da data → re-index real → gold vira sumário dos papers novos. O drill é executável por comando, não depende de tempo real passar.

Alternativas rejeitadas: adiar o slice ao CKPT-8 (deixaria CKPT-0 sem cobertura do FP1′/R3 e do §2b); perguntas de nicho ausentes sem data (testa "fora do corpus", não *staleness* temporal — não exercita R3).

### 4.3 Construção por slice

| Slice | n | Proveniência | Como | Gold labels |
|---|---|---|---|---|
| `answerable` | 40 | sintético-por-construção (§2) | amostragem estratificada de chunks (seed fixo, máx. 1–2 perguntas por paper), 1 chamada LLM por chunk → `{question, answer}` JSON | `gold_chunk_ids=[chunk_id fonte]`, `answer` concisa |
| `unanswerable` | 15 | hand-written | temas fora do corpus (física quântica, genômica, astrobiologia, finanças, medicina…) | `gold_chunk_ids=[]`, `answer="I don't know…"`, `should_abstain=true` |
| `multi-doc` | 15 | **híbrido (decisão 2026-09-26)**: 10 known-item com IDs na pergunta (Features B/D do produto) + 5 open-topic sem IDs (queries realistas de produção) | pares/trios de papers reais do corpus sobre temas adjacentes; pergunta exige síntese conjunta | known-item: `gold_arxiv_ids=[2–3 ids]`; open-topic: `gold_arxiv_ids=[]` **pendente** até adjudicação do mini-pooling no EXP-0 (§8), flag `retrieval_gold=pending-adjudication` no metadata. Ambos: `answer` = síntese com os pontos esperados |
| `format` | 10 | hand-written | instruções explícitas de formato (tabela markdown com colunas X; JSON conforme schema; lista com exatamente N itens) | `answer` = saída formatada exemplar; sem gold de retrieval |
| `persona` | 10 | derivado de `answerable` | 5 perguntas-base × 2 audiências (`pm`, `phd`) = 10 exemplos gêmeos; gold calibrado por audiência via LLM | idem `answerable` + `audience` no input; pares ligados por `base_id` no metadata |
| `stale` | 10 | hand-written | "O que há de novo sobre \<tema presente no corpus\> publicado após 17/09/2026?" — tema do corpus para tentar o retriever a responder com material velho | `gold_chunk_ids=[]`, `answer="No papers found in this period"`, `should_abstain=true` |

**Schema de exemplo (kv, uniforme)**:
- `inputs`: `{"question": str}` (+ `"audience": "pm"|"phd"` só em `persona`);
- `outputs`: `{"answer": str, "gold_chunk_ids": list[str], "gold_arxiv_ids": list[str], "should_abstain": bool}`;
- `metadata`: `{"slice": str, "provenance": "synthetic"|"curated", "base_id"?: str}` — correção 2026-09-27: os slices "hand-written" do plano são na verdade redigidos por IA e **aprovados por humano** (spot-check/aprovação de lote); a âncora humana entra via revisão, adjudicação do mini-pooling e calibração de juízes, não via autoria. Renomeado para `"curated"` por honestidade de proveniência;
- `split=["core"]`.

Schema uniforme (campos sempre presentes, vazios quando N/A) evita branching nos avaliadores do lote 0.6.

**Geração de gold answers**: `GENERATION_MODEL` (gpt-4o-mini), temperatura 0, com spot-check humano de 5 amostras/slice antes da aprovação do commit — é a mitigação já prevista na tabela de riscos §8. Custo estimado: ~60 chamadas → < $0.05.

### 4.4 Idempotência

Contrato do §2: checar `client.has_dataset(...)` antes de escrever. Implementação: se o dataset existir, listar exemplos do split `core` uma vez, contar por `metadata.slice`, e criar **somente slices faltantes/incompletas**; relatório final com contagens por slice. Segunda rodada = noop (exceto relatório).

## 5. Verificação do lote (como saberemos que está pronto)

1. `python build_golden_dataset.py` cria o dataset; rodar 2× → 2ª rodada noop.
2. Dataset visível no projeto `capstone-arxiv-copilot` com `split=core` e as 6 slices (verificação da §5 CKPT-0).
3. Contagens por slice batem com a tabela 4.3 (total 100).
4. **Spot-check humano** (você aprova antes do merge do lote): 5 amostras impressas por slice sintética/derivada (`answerable`, `persona`) — perguntas respondíveis de fato pelo chunk-fonte, gold answers corretas.
5. Cache do baseline intocado (parquet não regenera) e `python main.py` continua rodando de ponta a ponta — a mudança de fetch não pode quebrar o caminho quente.
6. Sem mudança de comportamento do pipeline (`app.py` não é tocado).

## 6. Commits esperados do lote (lotes ainda menores)

1. `chore: pin corpus snapshot date (2026-09-17) + deterministic chunk_id` — config + utils.
2. `feat: golden dataset builder, core split — 6 slices (~100 examples)` — build_golden_dataset.py + wiring em main.py + spot-check aprovado.

## 7. Riscos específicos deste lote

| Risco | Mitigação |
|---|---|
| Perguntas sintéticas fáceis demais ("too easy", §8) | amostragem com máx. 1–2/paper + spot-check; diversidade medida no EXP-0 |
| Perguntas `multi-doc`/`stale` hand-written com viés meu | provenance no metadata; você revisa no spot-check |
| Pin de data mudar o corpus de futuros configs | skip-budget aditivo (≥ ~1 semana de margem) garante 250 papers ≤ snapshot; guarda explode explícito se encurtar — ver `config.py` |
| `chunk_id` por hash divergir do que o avaliador computa | única função em `utils.py` é a fonte de verdade (dataset builder + 0.6 avaliadores importam dela) |

## 8. Adendo — adjudicação humana e calibração de juízes (pesquisa 2026-09-24)

Fundamentação completa com citações auditáveis:
`docs/learning-lessons/golden_dataset_construction_and_human_calibration.md`. Decisões derivadas:

1. **Mini-pooling no EXP-0 (CKPT-0.8)**: para cada pergunta golden, rodar retrieval top-k, listar
   candidatos fora do gold para adjudicação humana; docs julgados relevantes entram em
   `gold_arxiv_ids`. TREC-pooling em miniatura — golds por construção não são revisados exaustivamente.
2. **Semântica das métricas de retrieval (vale para o 0.6)**: em `answerable`/`multi-doc`, docs
   recuperados fora do gold são **não-julgados**, nunca contados como irrelevantes (viés de pooling
   documentado na literatura); recall/hit/MRR contam apenas gold. `precision@k` plena só em
   `unanswerable`/`stale`, cujo gold é "nada deve casar".
3. **Spot-check humano (§5 item 4) passa a ter dupla função**: validação de qualidade do dataset E
   semente do calibration set dos juízes LLM (ARES/RAGAS-Align/TruLens: ~100–200 labels humanas
   alinham o juiz — não o dataset inteiro).
