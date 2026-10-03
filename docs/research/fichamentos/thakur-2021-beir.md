# Thakur et al. 2021 — BEIR: benchmark heterogêneo de recuperação zero-shot (e o estudo de viés de anotação)

- **Referência completa e link:** Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS 2021, Datasets and Benchmarks Track. arXiv:2104.08663 — <https://arxiv.org/abs/2104.08663>. Código: <https://github.com/UKPLab/beir>.
- **Confiança da leitura:** **integral.** Li o PDF `arxiv.org/pdf/2104.08663` — confirmei que é a **v4** (24 págs., 21/out/2021; o cabeçalho ainda diz "Preprint. Under review"), incluindo Tabelas 1, 2 e 4 (extraídas com layout) e os apêndices D, F, G e H. Citações de página referem-se a essa v4. Não li o código do BEIR nem o leaderboard; números de versões anteriores (v1, 21 págs.) podem diferir.

> Convenção: **(F)** = o que a fonte afirma; **(I)** = minha interpretação.

## 1. Problema que a fonte ataca
(F, §1, p.1–2) Modelos neurais de recuperação eram avaliados em cenários homogêneos (mesmo dataset de treino e teste, p.ex. MS MARCO/NQ), o que não diz nada sobre **generalização fora do domínio**. Os benchmarks existentes (MultiReQA, KILT) cobrem uma tarefa ou domínio só. BEIR propõe um benchmark **zero-shot** com 18 datasets de 9 tarefas e compara 10 sistemas de 5 arquiteturas. Como segundo achado (e o que mais nos interessa), o artigo mostra que os datasets têm **viés lexical de anotação** que desfavorece retrievers não-lexicais (§6).

## 2. Método (com as definições exatas)
(F)
- **Escopo (§2, §3):** "document" é usado como "cover term for text of any length" (§2); queries também de qualquer comprimento. Datasets selecionados por quatro critérios: tarefas diversas, domínios diversos, dificuldade suficiente, estratégias de anotação diversas (§3). Formato único (corpus, queries, qrels). Os 18 zero-shot + MS MARCO (in-domain, fora da média).
- **Estatísticas (Tab. 1, p.4):** p.ex. SciFact — corpus de 5.183 abstracts, 300 queries, **1,1 relevante/consulta**; SCIDOCS — 25.657 documentos, 1.000 queries, 4,9 relevantes/consulta, tarefa "citation prediction"; TREC-COVID — 171.332 docs, 50 queries, **493,5** relevantes/consulta (3 níveis); NQ — 1,2; ArguAna — 1,0.
- **Sistemas (§4):** léxico (BM25, Anserini, k1=0,9 b=0,4); esparsos (DeepCT, SPARTA, docT5query); densos (DPR, ANCE, TAS-B, **GenQ** = TAS-B ajustado em consultas **sintéticas geradas dos próprios documentos** com T5, 5 por doc, teto de 100 mil docs); late-interaction (ColBERT); re-ranking (BM25 + cross-encoder MiniLM top-100). Documentos truncados a **512 word pieces** nos modelos neurais.
- **Métrica principal (§4, p.5):** **nDCG@10** (normalized Discounted Cumulative Gain: premia relevantes nas primeiras posições e aceita graus de relevância; no trec_eval, ganho/log₂(posição+1), normalizado pela ordem ideal), calculada pela interface Python do trec_eval oficial. Justificativa literal do autor: "Decision support metrics such as Precision and Recall which are both rank unaware are not suitable. Binary rank-aware metrics such as MRR and MAP fail to evaluate tasks with graded relevance judgements."
- **Recall@k (Apêndice G, p.20):** `R@k = (1/|Q|) Σ_i |top-k(A_i) ∩ A*_i| / |A*_i|` (de todos os relevantes da consulta, que fração apareceu no top-k) e **Capped Recall@k**: denominador `min(k, |A*_i|)` — evita que consultas com mais de k relevantes (ex.: 500) tenham teto baixo (500 relevantes ⇒ R@100 máx. 0,2). **BEIR não define hit rate/success@k nem MRR@k como métrica reportada**; o software "provides all IR-based metrics from Precision, Recall, MAP, MRR to nDCG" (§3.2, p.5).
- **Estudo de viés de anotação (§6, p.8–9):** os autores calculam o **Hole@10** de cada sistema no TREC-COVID — fração dos 10 primeiros resultados **sem julgamento** (nunca vistos pelos anotadores) — e **anotam manualmente os buracos**, sem saber qual sistema os recuperou ("to avoid a preference bias"), seguindo as diretrizes originais; recalculam nDCG@10.

## 3. Principais contribuições
- Benchmark zero-shot unificado, com formato padrão e pacote Python (integra Sentence-Transformers, Anserini, DPR, ColBERT, Elasticsearch…).
- Evidência de que o desempenho in-domain **não prediz** generalização; BM25 é baseline forte.
- Comparação de 5 famílias de retriever com custo (latência e tamanho de índice, §5.1, Tab. 3).
- **Quantificação do viés de seleção/lexical dos julgamentos** (Tab. 4) — a contribuição mais relevante para nós.
- Padrão de métrica: nDCG@10 como métrica única comparável entre tarefas, e Capped Recall@k como recall comparável.
- Análise de preferência de comprimento (TAS-B prefere docs curtos; Apêndice H: efeito da similaridade cosseno × produto interno).

## 4. Resultados-chave
- **Tab. 2 (p.7) — nDCG@10 zero-shot** (BM25 / BM25+CE / TAS-B / ANCE / docT5query): SciFact 0,665 / 0,688 / 0,643 / 0,507 / 0,675; SCIDOCS 0,158 / 0,166 / 0,149 / 0,122 / 0,162; TREC-COVID 0,656 / 0,757 / 0,481 / 0,654 / 0,713; NQ 0,329 / 0,533 / 0,463 / 0,446 / 0,399. Linha "Avg. Performance vs. BM25" (Tab. 2, na ordem das colunas DeepCT, SPARTA, docT5query, DPR, ANCE, TAS-B, GenQ, ColBERT, BM25+CE): −27,9%, −20,3%, +1,6%, −47,7%, −7,4%, −2,8%, −3,6%, +2,5%, +11%.
- **Achados enumerados (§5, p.6–7):** (1) in-domain ≠ generalização: BM25 perde 7–18 pontos no MS MARCO mas é forte no BEIR; (2) re-ranking BM25+CE supera o BM25 em **16/18** datasets; (3) docT5query supera o BM25 em 11/18; (4) GenQ ajuda em domínios especializados (científico, finanças) e piora em Wikipedia; (5) TAS-B recupera documentos curtos (mediana 10 palavras no TREC-COVID vs 160 do ANCE, Fig. 4).
- **Tab. 4 (p.9) — Hole@10 no TREC-COVID:** BM25 **6,4%**; docT5query **2,8%**; BM25+CE 1,6%; ColBERT 12,4%; SPARTA 12,4%; ANCE **14,4%**; DeepCT 19,4%; DPR 30,6%; TAS-B **31,8%**. Foram anotados **980 pares consulta-documento**.
- **nDCG@10 antes → depois de anotar os buracos (Tab. 4):** BM25 0,656 → 0,668; docT5query 0,713 → 0,714; BM25+CE 0,757 → 0,760; **ANCE 0,654 → 0,735** (de ligeiramente abaixo para 6,7 pontos acima do BM25); **ColBERT 0,677 → 0,735** (+5,8); DPR 0,332 → 0,445; TAS-B 0,481 → 0,555; DeepCT 0,406 → 0,472; SPARTA 0,538 → 0,624.
- **Conclusão do autor (§6):** "Even though many systems contributed to the TREC-COVID annotation pool, the annotation pool is still biased towards lexical approaches."

## 5. Limitações
- **Do autor:** necessidade de "better unbiased datasets" e de pools diversos (§6–§7); apenas inglês; truncamento a 512 word pieces; GenQ limitado a 100 mil docs por restrição de recursos (§4).
- **Minhas (I):**
  1. O estudo de buracos cobre **um dataset** (TREC-COVID, 50 consultas), **só top-10** e só uma rodada; não há teste de significância nem repetição — os valores são ilustrativos, não uma lei.
  2. A recomendação "nDCG@10 em vez de P/R" parte de um benchmark com relevância graduada e centenas de relevantes em alguns datasets; **não vale automaticamente** para um top-5 que vai inteiro ao prompt de um LLM (onde a ordem interna do top-k pesa menos).
  3. O BEIR não discute **granularidade** (passagem × documento): cada dataset usa a unidade que veio pronta.
  4. Nenhum dataset do BEIR tem gold **sintético gerado do próprio documento**; GenQ gera consultas sintéticas mas só para *treinar* retrievers, não para o gold de avaliação. Portanto o BEIR **não responde** à nossa preocupação "gold sintético ⇒ viés lexical"; ele apenas mostra o efeito análogo para gold feito com busca lexical.
  5. Tabela 4 nesta versão não inclui a coluna GenQ.

## 6. Reflexões ancoradas no NOSSO projeto

| # | Ponto / arquivo | Reflexão | A fonte… |
|---|---|---|---|
| 1 | **P5 / D-2**, `hit_rate` vs `recall_at_k` (`evaluators.py`) | Com |gold|=1: `recall@k = |top-k ∩ gold|/1 = 1[gold∈top-k] = hit@k` (derivação minha). O BEIR **não reporta hit rate**; define Recall@k e o **Capped Recall@k** (denominador `min(k,|A*|)`), este último irrelevante para nós (gold ≤3 e k=5 ⇒ cap = |gold|). Para gold único, o BEIR usa nDCG@10 — que, com 1 relevante binário, vira `1/log₂(posição+1)` (derivação minha: posição 1 → 1,0; 2 → 0,63; 3 → 0,50; 5 → 0,39 vs MRR 1; 0,5; 0,33; 0,2): função monótona da posição como o MRR, só com desconto mais suave. Logo, apoia manter **hit@k + MRR** (ou nDCG@k) em gold único e recall só em multi-gold. | **(i) APOIA** (P5) |
| 2 | **Rank-unaware (P/R) vs rank-aware** — escolha de métricas | O BEIR rejeita P e R por serem "rank unaware". Isso **desafia** apresentar `recall@5`/`precision@5` como métrica principal; mas (I) nosso consumidor é o LLM, que recebe os 5 chunks de uma vez — a ordem dentro do top-5 importa menos que no ranking de 10+ resultados do BEIR. Adotaríamos: hit@k (barra de "o contexto tem a resposta?") + MRR (barra de "está no topo?") — sem empilhar recall; nDCG só quando houver graus de relevância (CKPT-4). | **(ii) DESAFIA** parcialmente |
| 3 | **D-1 / buracos no gold de 1 chunk**, `precision_at_k` (P6) | A Tab. 4 mostra o mecanismo exato: quando o sistema avaliado recupera algo **que ninguém julgou**, o score cai sem que o sistema tenha errado — e o efeito é **desigual**: BM25 +0,012, ANCE +0,081. Nosso gold sintético é um pool de **1 item**; o top-5 com irmãos do mesmo paper tem "Hole@5" potencialmente alto. Adotaríamos: reportar **Hole@5** (fração do top-5 que não é gold nem julgado) e **julgar os buracos** no EXP-0 (BEIR anotou 980 pares para 50 consultas ≈ 20/consulta; para nós ≤ ~200–250 itens, estimativa minha: ~50 consultas com gold de chunk × irmãos do mesmo paper). | **(i) APOIA** (P6, mini-pooling do EXP-0) |
| 4 | **Viés lexical do gold sintético** (lesson L1, ADR do golden set) | O BEIR é evidência **análoga e indireta**: gold derivado de busca lexical (BioASQ, Signal-1M, TREC-COVID) favorece retrievers lexicais (§6). Nosso gold nasce de um LLM que leu o chunk ⇒ provável sobreposição de vocabulário (I, não testado em nenhuma fonte). Risco concreto: em CKPT-1/4, um retriever denso ou um reranker pode parecer pior que BM25/híbrido no golden set sintético por causa do gold, não do retriever. Mitigação (minha): comparar BM25 × denso nos mesmos exemplos e acompanhar o Hole@k de cada um; incluir perguntas parafraseadas/humanas (slice `persona` já ajuda). | **(ii) DESAFIA** a neutralidade do gold sintético (por analogia) |
| 5 | **D-1** (nível chunk × paper) | O BEIR usa **a unidade que o dataset fornece** (abstracts em SciFact/SCIDOCS, sem decompor em passagens); **não discute** passagem × documento nem multi-nível. Fonte **neutra**; os dois datasets de abstracts científicos (SciFact: 1,1 relevante/consulta; SCIDOCS: 4,9) são os mais parecidos com nosso corpus (cs.AI abstracts) e mostram nDCG@10 baixos mesmo para os melhores sistemas (SCIDOCS ≤ 0,166), um aviso de que métricas absolutas em abstracts são difíceis de interpretar sem baseline BM25 ao lado. | **(iii) NEUTRA** (lacuna) |
| 6 | **L-3** — ranx como oráculo | O BEIR calcula suas métricas pelo **trec_eval oficial via interface Python** (§4, p.5). Isso sustenta usar uma implementação derivada do trec_eval (ranx ou `pytrec_eval`) como oráculo: é o padrão de fato do campo. Não decide entre elas. Ver `bassani-2022-ranx.md`. | **(i) APOIA** (conceito de oráculo = trec_eval) |
| 7 | **Slice `unanswerable`/`stale`** | O BEIR não tem consultas sem resposta nos datasets; fonte **neutra** para abstenção. | **(iii) NEUTRA** |

**O que adotaríamos do BEIR:** (1) baseline BM25 em paralelo a qualquer retriever novo, para detectar viés lexical do gold; (2) taxa de buracos (Hole@k) e julgamento dos buracos no EXP-0; (3) pelo menos uma métrica rank-aware (MRR já existe; nDCG@k quando houver graus); (4) o padrão trec_eval como referência de correção.
**O que NÃO adotaríamos:** (1) nDCG@10 como métrica única — nosso k=5 e relevância binária tornam-no redundante com MRR; (2) Capped Recall — não tem efeito no nosso regime; (3) a extrapolação de rankings de modelos do BEIR para nosso domínio (abstracts cs.AI) — o próprio artigo diz que in-domain/generalização não se transferem.

### Respostas às perguntas do pedido (parte BEIR)
- **(a)** O BEIR **não define hit rate/success@k**; define Recall@k (e Capped) e adota **nDCG@10** como métrica única, recusando P/R por serem rank-unaware e MRR/MAP por não lidarem com relevância graduada. Com 1 relevante, hit@k e recall@k são numericamente iguais (derivação minha); o BEIR não discute esse caso.
- **(b)** Não trata perguntas geradas do próprio documento; mostra viés lexical de qrels derivados de busca por termos (Tab. 4, §6).
- **(c)** Postura: **tratar não-julgado como irrelevante é a prática**; os autores propõem **julgar os buracos** e medir Hole@10, não usar bpref/condensed lists.
- **(d)** Sem discussão de granularidade; unidade = documento "de qualquer tamanho" (§2).

## 7. Citações úteis
1. §4 (p.5): "Decision support metrics such as Precision and Recall which are both rank unaware are not suitable."
2. §6 (p.8): "All other unseen documents are assumed to be irrelevant. This is a source for selection bias [39]: A new retrieval system might retrieve vastly different results than the system used for the annotation."
3. §6 (p.9): "Even though many systems contributed to the TREC-COVID annotation pool, the annotation pool is still biased towards lexical approaches."
