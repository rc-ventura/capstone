# CKPT-0.6 Plan — `evaluators.py` (catálogo de avaliadores)

> Status: **aguardando aprovação**. Nenhum código será alterado antes do "go".
> Parent plan: `docs/plan.md` §2b (ground-truth map) + §4 (catálogo).
> Dependências já prontas: golden set revisado (100 ex), `utils.chunk_id`, ADR-001.

## 1. Contexto

O EXP-0 (CKPT-0.8) precisa de avaliadores para medir as duas superfícies do plano §3:
**retrieval** (recall@k, hit@k, MRR, precision@k) e **geração** (faithfulness,
citation accuracy, completeness, format, abstention, specificity, relevance). Sem eles,
o golden set não produz métrica nenhuma — é o instrumento de medida do capstone.

## 2. Contrato de dados (o que o runner do 0.7 vai produzir)

Cada exemplo avaliado chega ao avaliador com:

- `outputs` (do target/runner): `{"answer": str, "retrieved_chunk_ids": list[str],
  "retrieved_context": list[str], "retrieved_arxiv_ids": list[str]}`
  (`app.arxiv_copilot` hoje retorna só `answer`; o runner (0.7) expõe os docs recuperados)
- `reference` (do golden set): `outputs.{answer, gold_chunk_ids, gold_arxiv_ids,
  should_abstain}` + `metadata.slice`

## 3. Escopo exato — o que cada avaliador mede e por quê

### 3.1 Superfície 1 — Retrieval (code, reference-based)

**`retrieval_recall_at_k`** — *cobertura dos documentos necessários.*
O que é: fração dos documentos-gold que aparecem no top-k recuperado.
Mede: responder completamente exige os docs certos presentes; se o sistema nunca os
recupera, a geração está condenada antes de começar.
Por quê (FP2 — retrieval loss / R5 ranking fraco): é a métrica central dos gates
CKPT-1/2/3 ("o k/modelo/chunk novo cobre mais os docs necessários?").

**`retrieval_hit_rate`** — *achou pelo menos um?*
O que é: 1 se ≥1 doc-gold está no top-k, senão 0.
Mede: casos em que parte da evidência basta (fallback), e dá uma métrica binária barata
de leitura.
Por quê: complemento do recall — distingue "não achou nada" de "achou parte".

**`retrieval_mrr`** — *quão alto no ranking o primeiro doc-gold aparece.*
O que é: Mean Reciprocal Rank = 1/(posição do 1º doc-gold). MRR=1 se gold em 1º,
1/2 se em 2º, etc.
Mede: posição, não apenas presença — docs no fundo da página tendem a ser ignorados
pelo gerador.
Por quê (R5 — ranking fraco): é o alvo direto do gate do reranker (CKPT-4) e do
slice `deep-hit` (docs que ranqueiam baixo).

**`retrieval_precision_at_k`** — *fração do top-k que é gold.*
O que é: dos k recuperados, quantos são gold-label.
Mede: ruído — quanta coisa irrelevante compete pela atenção do gerador.
Por quê (FP3 — retorno irrelevante/noise): expõe inject de ruído (ex.: o colapso de
dedup notado em decisions.md). **Restrição de honestidade (viés de pooling)**: só
computado onde há gold não-vazio adjudicável; em exemplos sem julgamento de
irrelevância dos outros docs, tratar não-gold como erro seria falso.

### 3.2 Superfície 2 — Geração

**`faithfulness_judge`** (LLM, ref-free) — *a resposta é sustentada pelo contexto
recuperado?*
O que é: juiz verifica claim a claim da resposta contra `retrieved_context` (não contra
o gabarito).
Mede: alucinação-em-resposta-do-RAG — o modo mais perigoso quando o sistema parece
"citar" mas inventa.
Por quê (FP4): métrica estilo RAGAS; independente de gabarito certo — mesmo com
gold errado, uma resposta fiel ao contexto é comportamento correto do componente.

**`citation_accuracy`** (code + LLM, ref-free) — *a citação existe E sustenta a claim?*
O que é: regex extrai `[2609.xxxxvN]` da resposta (code); para cada citação, juiz
checa se o chunk daquele `arxiv_id` no contexto recuperado sustenta a frase citada.
Mede: misgrounding — o sistema cita uma fonte que não diz o que a resposta afirma
(Magesh et al. 2025, §4).
Por quê (FP4 em sua forma mais específica): produto promete "cited Q&A" — citação
falsa quebra a proposta de valor.

**`completeness_judge`** (LLM, híbrido) — *todas as partes da pergunta foram cobertas?*
O que é: ref-free nos demais slices (todas as sub-perguntas respondidas?);
reference-based em `multi-doc` (todos os pontos do gold synthesis presentes?).
Mede: consolidação incompleta — respondeu parcialmente.
Por quê (FP7): é o disco-alvo dos experimentos de decomposição (CKPT-5).

**`abstention_quality`** (LLM, ref-based) — *absteve quando devia / respondeu quando podia?*
O que é: juiz compara o comportamento da resposta com `should_abstain` do gold.
Mede: hedge de segurança vs utilidade — abster demais é tão ruim quanto alucinar.
Por quê (FP1): abstenção correta em `unanswerable`/`stale` é release-blocking
(plano §5 CKPT-6: gate de 90%).

**`specificity_judge`** (LLM, ref-based) — *nível de detalhe bate com a audiência pedida?*
O que é: juiz compara a resposta com o gold da audiência requerida (`lay` vs `phd`).
Mede: calibração de densidade — nem vago demais nem denso demais.
Por quê (FP6): mira diretamente no produto Feature C.

**`answer_relevance`** (LLM, ref-free) — *a resposta endereça a pergunta feita?*
O que é: juiz avalia relevância/on-topic da resposta vs a pergunta (estilo RAGAS).
Mede: deriva — resposta correta mas sobre outra coisa.
Por quê: métrica de monitoramento transversal (CKPT-8 usa em produção).

### 3.3 Formatos e agregados

**`format_validator`** (code, ref-based) — *a saída obedece ao formato pedido?*
O que é: validação determinística por tipo — tabela markdown (colunas certas), JSON
(chaves exatas), CSV parseável, YAML, nº exato de bullets/palavras/frases+citação.
Mede: aderência a instrução de formato (sem juiz — código puro).
Por quê (FP5): Feature B/D prometem saídas estruturadas (tabelas de comparação).

**`f1_summary_evaluator`** (code, summary, ref-based) — *agregado substantivo×alucinado.*
O que é: evaluator de resumo (P5 do curso): computa F1 agregado entre o conjunto de
respostas substantivas vs. alucinadas/abstenções sobre as labels gold coletadas do
experimento inteiro.
Mede: balanço global entre utilidade e segurança num experimento.
Por quê: uma leitura única de "qualidade substantiva vs. risco" por configuração de
experimento — usada no relatório EXP-0-baseline.

---

**Matriz evaluador × slice** (quem roda onde — do §2b ground-truth map):

| Slice | Retrieval | Geração |
|---|---|---|
| `answerable` | recall, hit, MRR, precision | faithfulness, citation, relevance |
| `unanswerable` | (n/a — gold vazio por design) | abstention, faithfulness |
| `multi-doc` known | recall, hit, precision | completeness, faithfulness, citation |
| `multi-doc` open | **skip até adjudicação** | completeness, faithfulness, citation |
| `format` | n/a | format_validator |
| `persona` | recall (fonte compartilhada) | specificity, faithfulness |
| `stale` | (gold vazio pré-refresh) | abstention |

**Semântica de retrieval (decisão já documentada no adendo da revisão / lesson L1):**
recall/hit/MRR contam apenas rótulos-gold; docs recuperados fora do gold são
**não-julgados**, nunca irrelevantes automáticos — elimina o viés de pooling.
`precision_at_k` só computável onde o gold é não-vazio. Exemplos `open_topic`
(`retrieval_gold=pending-adjudication`) são **pulados** dos retrieval metrics até o
mini-pooling do EXP-0.

## 4. Juízes LLM — configuração

- Modelo do juiz: `GENERATION_MODEL` (gpt-4o-mini), `temperature=0`, saída estruturada
  (JSON com `{"score": bool|float, "reason": str}`). Barato; variância será medida no
  EXP-0 rodada 2 (pra isso o prompt precisa ser determinístico).
- Cada juiz = 1 função `(question, answer, context, reference) -> EvaluationResult`.
- `citation_accuracy`: passo 1 code (extrai `[2609.xxxxvN]` da resposta via regex),
  passo 2 juiz por citação (o chunk `arxiv_id` suporta a claim?).

## 5. Mini-checkpoints internos (um commit cada, você revisa entre eles)

| Lote | Conteúdo | Teste |
|---|---|---|
| **0.6.1** | 4 avaliadores code de retrieval | testes offline puros (sem nenhuma chamada LLM): slices de exemplos sintéticos com golds controlados, cobrindo edge cases (gold vazio, open-topic skip, dedup, top-k parcial) |
| **0.6.2** | `format_validator` + `f1_summary_evaluator` | testes offline; dogfood contra os 10 golds do slice `format` (devem passar 10/10) |
| **0.6.3** | 6 juízes LLM | testes com casos-canais pass/fail ambos (ex.: resposta fiel × resposta alucinada) — ~12 chamadas LLM no total |

## 6. Fora de escopo

- Runner (`experiments/run_experiment.py`) → 0.7. Rodada real → EXP-0 (0.8).
- Calibração formal dos juízes com concordância humana (kappa) → 0.8, antes de congelar
  o baseline (usa suas labels da revisão).

## 7. Verificação

1. Suite de testes offline: `python -m pytest tests/test_evaluators.py -q` (ou script
   único se preferir não adicionar pytest ao projeto — decisão abaixo).
2. Cada juiz validado num caso pass E num caso fail reais (falha = score errado é bug).
3. `format_validator` dogfood: 10/10 dos golds do slice `format` passam.
4. Zero chamadas rastreadas de juiz fora dos testes (controle de custo).

## 8. Decisão aberta (pequena)

- **Testes**: adicionar `pytest` como dev-dependency, ou manter o padrão atual de
  scripts de validação inline? Recomendo pytest (o catálogo vai crescer até CKPT-8).
