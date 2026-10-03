# Abstenção e recusa seletiva — AbstentionBench (Kirichenko et al. 2025), RefusalBench (Muhamed et al. 2025) e extras (UAEval4RAG, OR-Bench)

> Ficha única, uma seção por artigo + comparação. Convenção: **[Artigo]** = o que o texto afirma (com seção/tabela/figura);
> **[Minha leitura]** = interpretação minha, não do artigo. Texto lido a partir do PDF (arXiv v1/v3 indicadas abaixo),
> extraído com `pdftotext`; **números que só existem dentro de figuras-gráfico foram lidos apenas quando também aparecem em texto/legenda/tabela**.

## Índice
1. AbstentionBench — leitura integral
2. RefusalBench — leitura integral
3. Comparação AbstentionBench × RefusalBench × projeto
4. Extra A: UAEval4RAG (Peng et al. 2025) — leitura parcial
5. Extra B: OR-Bench (Cui et al. 2025) — leitura parcial
6. Síntese para o roadmap (P7, P8, P11)

---

# 1. Kirichenko et al. 2025 — AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions

- **Referência completa e link:** Polina Kirichenko\*, Mark Ibrahim\*, Kamalika Chaudhuri, Samuel J. Bell\* (FAIR at Meta). *AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions*. arXiv:2506.09038v1 [cs.AI], 10 Jun 2025. <https://arxiv.org/abs/2506.09038> · código <https://github.com/facebookresearch/AbstentionBench>
- **Confiança da leitura:** **integral** (corpo §1–5 + Apêndices A–F, incl. prompt do juiz, Tabelas 1–6). Ressalvas: (i) gráficos (Fig. 2–8, S1–S16) lidos por legenda e texto, não por valor de barra; (ii) é a v1 — pode ter revisões posteriores; (iii) no Apêndice C.3.3 o número de anotadores por par aparece como "[NUMBER REDACTED]" no PDF.

## 1. Problema que o artigo ataca
**[Artigo]** LLMs precisam saber *quando não responder* (underspecified, mal-postas, sem resposta conhecida), mas "abstention remains understudied, without a systematic evaluation framework for modern LLMs" (Abstract). Os datasets existentes cobrem um único tipo de problema isolado (§2). Pergunta central: o fine-tuning de raciocínio (o1, R1, s1) ajuda ou atrapalha a abstenção?

## 2. Método (passo a passo)
**[Artigo]**
1. **Definição de abstenção (§3, p.3):** "a response that refrains from directly answering the question, such as by expressing a lack of knowledge, communicating uncertainty or caveats, or highlighting unanswerable aspects of the prompt. This can include simple statements such as 'I don't know'… but can also include detailed responses providing partial answers". *(= definição ampla e semântica; NÃO é um prefixo/regex.)*
2. **Coleta (§3.1, App. C.2.1):** busca no Semantic Scholar ("LLM abstention/abstain/uncertainty") → 183 artigos → 82 datasets candidatos → revisão manual (abstenção desejável em ≥1 amostra, público, licença ok) → **17 datasets**; + 3 variantes próprias de raciocínio (GSM8K-Abstain, GPQA-Abstain, MMLU-Math-Abstain: duplicam a pergunta e **removem o contexto** para torná-la irrespondível) + UMWP. Abstract diz "20 datasets"; o apêndice soma 17+3+UMWP (contagem não bate exatamente — **[Minha leitura]** divergência menor de contagem, não afeta conclusões). Cada dataset limitado a 3.500 amostras (C.2.2). Mais de **35k perguntas** (Fig. 1).
3. **Cada amostra** = prompt (pergunta + contexto opcional) + rótulo binário `should_abstain` + respostas de referência opcionais (C.2.2). Ou seja, há perguntas respondíveis *e* irrespondíveis na mesma base.
4. **Seis cenários (§3.2):** Answer Unknown, False Premise, **Stale**, Subjective, Underspecified Context, Underspecified Intent. "Stale" = "questions regarding recent events that occurred after model pretraining, such that answers contained in the training data may be stale".
   - Implementação de Stale em FreshQA (C.2.2): comparam dois snapshots (v10282024, antes do pré-treino, e v12182024, depois do cutoff máximo dos modelos); respostas que **mudaram** entre eles são "should abstain".
5. **Detecção de abstenção (§3.4, C.3):** **LLM-as-judge** = **Llama 3.1 8B Instruct**, prompt adaptado de Brahman et al. (CoCoNot), saída "Yes"/"No", **decodificação gulosa (T=0), "crucial for high performance"** (C.3.1). Rejeitam similaridade de embeddings (usada em trabalhos anteriores) por não capturar a diversidade de abstenções. O prompt (C.3.2) tem uma descrição por cenário de "abstenção apropriada" × "NOT an abstention" e **recebe também o `[GROUND TRUTH ABSTENTION LABEL]` e as respostas de referência** ("they can be noisy, so mostly rely on the [QUESTION]…"); "accuracy or verbosity of the answer does not matter".
6. **Juiz de correção** (só para respostas não-abstentoras com gabarito): também Llama 3.1 8B, prompt de Thakur et al.; saída "correct"/"incorrect"; respostas inválidas são filtradas (C.3.5).
7. **Validação do juiz (C.3.3–C.3.4):** 300 pares prompt–resposta (3 prompts por benchmark de domínio geral × {Llama 3.1 70B, GPT-4o}), amostragem **estratificada** por `should_abstain` × predição do juiz de primeira passada; autores rotulam independentemente (abstenção total / parcial / nenhuma); não-unânimes discutidas → consenso; 50/50 validação/teste (validação para iterar o prompt, teste para o número final).
8. **Geração:** máx. 4k tokens, **temperatura 0,8**, top-p 0,95, seed fixa (C.1); o1 com T=1,0. Reasoning: 4k tokens de "thinking" + 4k de resposta final; só a resposta final é avaliada por padrão (§4.3).

**Métricas (§3.4, p.5)** — rótulo do juiz comparado ao `should_abstain` da amostra:
- **Abstention recall** = proporção das amostras *irrespondíveis* em que o modelo de fato se absteve. *(Em linguagem simples: das perguntas em que devia dizer "não sei", em quantas disse.)* Citado como "the proportion of responses where the model correctly abstained".
- **Abstention precision** "to account for over-abstention" — das vezes em que se absteve, quantas deviam. *(Mede a recusa indevida.)*
- **Abstention F1** — média harmônica de ambas.
- **Response accuracy** — correção (pelo juiz de correção) nas respostas que *não* se abstiveram.
- Decisão: "as we find that models generally exhibit high abstention precision, we focus on abstention recall."
- Não há fórmulas escritas; as definições são verbais.

## 3. Principais contribuições
1. Benchmark único de abstenção, 20 datasets, 6 cenários, >35k perguntas (Fig. 1).
2. Achado: **abstenção não escala com o tamanho do modelo** (Fig. 3b; Llama 8B/70B/405B ≈ iguais) e **melhor acurácia ≠ melhor abstenção** (Fig. 4).
3. Achado: **reasoning fine-tuning piora abstenção em média −24%** (Abstract/§1; Fig. 6a), inclusive em matemática/ciência; mais "reasoning budget" melhora acurácia mas não ajuda/piora abstenção (Fig. 7).
4. **RLVR** (etapa final do Tülu 3) degrada abstenção (Fig. 5c); SFT/DPO ajudam, exceto em *underspecified context*.
5. Cadeias de raciocínio *contêm* incerteza, mas a resposta final é definitiva (Fig. 8a, Fig. S15).
6. Um *system prompt* que descreve os cenários eleva a abstenção sem degradar muito a precision (Fig. 8b, S16) — "unlikely to fundamentally address a lack of reasoning about uncertainty" (§4.4).
7. Juiz de abstenção validado contra humanos (Tab. 1) e *fast subset* de 100 perguntas/benchmark (App. E, Tab. 6).

## 4. Resultados-chave
- **Juiz (Tab. 1, C.3.4):** Llama 3.1 8B — acc 0,88 / F1 0,85 / precision 0,86 / recall 0,83; Llama 3.3 70B — 0,88 / 0,83 / 0,94 / 0,75; GPT-4o — 0,89 / 0,85 / 0,96 / 0,77. Escolheram o 8B por custo e por ter o melhor recall (§3.4 cita só "88% accuracy").
- **Média por modelo (Tab. 4, App. D.1; acurácia média / recall médio de abstenção):** DeepSeek-R1-Distill-Llama-70B 0,81 / **0,46**; o1 0,80 / 0,66; s1.1-32B 0,80 / **0,43**; Llama 3.3 70B Instruct 0,78 / 0,66; GPT-4o 0,75 / 0,69; Qwen2.5-32B 0,75 / **0,71** (melhor recall médio); Llama 3.1 8B Instruct 0,70 / 0,66. Texto: "reasoning models… are the top 3 performing LLMs [em acurácia]. At the same time, DeepSeek R1 Distill and s1.1 are close to the worst models in terms of abstention".
- **Precision (App. D.1, Fig. S1):** "on most datasets the precision is close to 1 for most models—i.e., models rarely over-abstain"; ranking por F1 "mostly agrees" com o ranking por recall (Fig. S2). Exceções de over-abstenção: precision degradada em *stale* após o pós-treino do Tülu (App. D.3, Fig. S5) e quando o juiz vê a cadeia de raciocínio (Fig. S15).
- **Reasoning budget (Fig. 7, §4.3):** s1.1 com 512→4096 tokens: acurácia sobe; recall de abstenção "either not improving (GSM8K-Abstain) or worsening (UMWP)".
- **Stale (Fig. 4, §4.1; Tab. 6):** em FreshQA, "improving correctness correlates with degraded abstention". Tab. 6 (Llama 3.1 8B Instruct): recall *stale* = 0,74 no *fast subset* vs 0,95 no full — o próprio autor diz que stale está "saturated" e é a exceção onde o subset não reproduz o full.
- **System prompt (§4.4, App. D):** sobe recall em modelos padrão e de raciocínio "without a significant degradation in abstention precision".
- **Por cenário (§4.1, Fig. 2/3c):** quase perfeito em BIG-Bench Known Unknowns, quase zero em MediQ; "underspecification, subjectivity, and false assumptions are key challenges"; "Answer Unknown" é o cenário em que os modelos vão bem ("with the exception of questions with unknown answers, frontier LLMs struggle across all other abstention scenarios", §1).

## 5. Limitações
**[Artigo] (App. A/B):** só inglês; cenários podem não estar todos cobertos; **vazamento** (train split do CoCoNot está no Tülu → confunde OLMo/Tülu); conjunto finito de modelos; "any LLM judge will be imperfect"; dados públicos → risco de inflação com o tempo (sem teste privado).
**[Minha leitura]**
- O juiz foi validado em 300 pares **só de benchmarks de domínio geral**, sobre 2 modelos geradores (Llama 70B, GPT-4o); o uso do mesmo juiz nos datasets de matemática/ciência e em reasoning models não foi validado nominalmente.
- O prompt do juiz **recebe o rótulo `should_abstain`** (mesmo dito "ruidoso"): isso pode *inflar* a concordância com o rótulo-gabarito e dilui a independência entre "juiz" e "gabarito". O artigo não mede essa influência.
- Abstenção "parcial" (responde, mas com caveat) conta como abstenção na definição ampla; isso é mais frouxo que nosso gabarito literal ("I don't know.").
- Geração a T=0,8, uma amostra por pergunta, e **não vi intervalos de confiança/bootstrap** nem testes de significância nas figuras/tabelas lidas.
- Foco explícito em *recall* porque a precision é alta **no conjunto deles**, onde a maioria das amostras irrespondíveis é desenhada para provocar abstenção; a proporção respondível/irrespondível por dataset varia e não é reportada de forma agregada que permita calcular acurácia "ingênua".

## 6. Reflexões ancoradas no NOSSO projeto
| # | Ponto / arquivo | Veredito | Reflexão |
|---|---|---|---|
| 1 | **P8** — `f1_summary_evaluator` reporta *acurácia* de abstenção | **APOIA o espírito, matiza o formato** | O artigo **não usa acurácia** como métrica de abstenção; usa recall (primária), precision e F1. Isso sustenta abandonar acurácia. Mas **não sustenta "F1 como métrica principal"**: eles *focam em recall* porque precision ≈ 1. Nosso gate do plano ("abstenção correta ≥ 90% em `unanswerable`") **é um recall** — alinhado com a prática deles. |
| 2 | **P8** — over/under-refusal | **APOIA** | Eles tratam over-abstention via *precision* (Fig. S1, S5, S15) — ou seja, mantêm o lado "recusar o respondível" no relatório, ainda que secundário. Recomendação: gate = recall em `unanswerable` + **precision/over-refusal como guarda-corpo (regressão não pode piorar)**, especialmente ao mexer no prompt (§4.4: prompt restritivo "não degrada muito" a precision *neles*; no nosso caso precisa ser medido). |
| 3 | **P7** — `startswith(("I don't know","No papers found"))` | **DESAFIA** | O artigo gastou um apêndice inteiro justificando por que *não* usar match de string/embedding (C.3.1) e adotou juiz LLM + validação humana (300 pares). Nosso gabarito ("I don't know."/"No papers found in this period.") e nosso prompt (`PROMPT_V1`: "say you don't know" sem frase fixa) tornam o `startswith` frágil exatamente como eles descrevem. **Adotaríamos:** juiz (já previsto em `abstention_quality`, 0.6.3) com validação em amostra humana estratificada por `should_abstain × predição` (receita do C.3.3, cabe nos nossos 25 should-abstain + amostra de respondíveis). **Não adotaríamos:** juiz de 8B sem validação própria — o 88% deles é em inglês geral/benchmarks diversos, não em respostas curtas de um RAG de abstracts. |
| 4 | **P7** — juiz T=0 | **APOIA** | "greedy decoding… crucial" (C.3.1) coincide com o plano 0.6 §4 (`temperature=0`). |
| 5 | **Slice `unanswerable`** (15, fora do corpus) | **APOIA (≈ Answer Unknown)** | É o cenário em que os modelos *vão melhor* no benchmark (§1) → esperar recall alto; se o nosso baseline ficar baixo, pode ser o *prompt*/detecção (P7) e não capacidade. |
| 6 | **Slice `stale`** ("What's new in <mês>?") | **NEUTRO/DESAFIA a analogia** | O "Stale" do AbstentionBench = a *resposta mudou depois do cutoff do modelo* (FreshQA, dois snapshots). O nosso `stale` é diferente: o corpus RAG é fixo (250 abstracts) e a pergunta pede um período sem documentos → é **ausência no corpus** (mais próximo de Out-of-Database/Answer Unknown, ver UAEval4RAG abaixo) que de "dado desatualizado do modelo". Não dá para reaproveitar os números de stale deles como referência. Lição transferível: em FreshQA, acertar mais correlacionava com abster menos (Fig. 4) — um RAG que *recupera* algo vagamente do mês e responde pode "parecer correto". |
| 7 | **P11** — juiz = gerador (gpt-4o-mini) | **NEUTRO/APOIA fracamente** | O juiz deles (Llama 8B) não é o modelo gerador principal, mas também avalia o Llama 8B como gerador (Fig. 3b/S4) — não discutem auto-preferência. Não há evidência aqui a favor/contra; a evidência está no RefusalBench (seção 2). |
| 8 | **P8/gate** — tamanho amostral | **DESAFIA (lacuna)** | 15 `unanswerable` → um erro = 6,7 p.p.; gate de 90% ≈ aceita 1 erro (93,3%) e reprova com 2 (86,7%). O artigo não oferece guia para amostras pequenas (sem ICs). Nossa decisão: reportar contagem bruta + IC (Wilson/bootstrap) — **decisão de engenharia, não da literatura**. |
| 9 | Prompt do sistema para abstenção (CKPT-6) | **APOIA** | §4.4: um prompt descrevendo cenários eleva recall; mas "unlikely to fundamentally address" a falta de raciocínio sobre incerteza. Não adotar como "solução", apenas como alavanca medida com guarda-corpo de precision. |

**O que adotaríamos:** (a) recall como gate; (b) precision/over-refusal como guarda-corpo; (c) juiz LLM com T=0 + amostra humana estratificada para validar (inclusive medir quanto o `startswith` diverge do juiz); (d) relatar por cenário/slice, não só média.
**O que NÃO adotaríamos:** (a) a média simples de recall entre cenários como número único; (b) passar o rótulo `should_abstain` ao juiz sem testar o efeito (para o nosso gabarito literal, o juiz deveria julgar só a *resposta*); (c) transpor "Stale" como equivalente do nosso `stale`.

## 7. Citações úteis (máx. 3)
1. "We define abstention as a response that refrains from directly answering the question, such as by expressing a lack of knowledge, communicating uncertainty or caveats, or highlighting unanswerable aspects of the prompt." — §3, p.3.
2. "However, as we find that models generally exhibit high abstention precision, we focus on abstention recall." — §3.4, p.5.
3. "We use greedy decoding (i.e. temperature = 0) for judge inference following prior works, and found this to be crucial for high performance of the judge." — App. C.3.1.

---

# 2. Muhamed et al. 2025 — RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models

- **Referência completa e link:** Aashiq Muhamed, Leonardo F. R. Ribeiro, Markus Dreyer, Virginia Smith, Mona T. Diab (CMU; Amazon AGI). *RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models*. arXiv:2510.10390v1 [cs.CL], 12 Oct 2025. <https://arxiv.org/abs/2510.10390>
- **Confiança da leitura:** **integral** (corpo §1–6, Apêndice de definição de métricas, validação humana C.2, análises F, prompts I). Ressalvas: (i) figuras/heatmaps (Fig. 2, 5–11, 18–30) extraídos como texto desordenado; usei só números que aparecem também em texto corrido/legenda; (ii) a ordem de colunas do PDF vem misturada — numeração de seções do apêndice inferida pelo conteúdo; (iii) é v1 (preprint de out/2025, sem revisão por pares até onde li).

## 1. Problema que o artigo ataca
**[Artigo]** Em sistemas RAG, o modelo deve **recusar seletivamente** quando o contexto recuperado tem defeitos (ambíguo, contraditório, informação ausente…), mas "falha sistematicamente": ou recusa "over 60%" do respondível, ou responde com confiança apesar de defeitos (§1). Benchmarks estáticos sofrem de contaminação, overfitting adaptativo e saturação (§3.1) → propõem **avaliação generativa**: criar instâncias novas por perturbação controlada de pares QA respondíveis.

## 2. Método (passo a passo)
**[Artigo]**
1. **Taxonomia de 6 incertezas (§3.2)** e o código de recusa esperado: P-Ambiguity→`REFUSE_AMBIGUOUS`; P-Contradiction→`REFUSE_CONTRADICTORY`; P-MissingInfo→`REFUSE_MISSING`; P-FalsePremise→`REFUSE_FALSE_PREMISE`; P-GranularityMismatch→`REFUSE_GRANULARITY`; P-EpistemicMismatch→`REFUSE_NONFACTUAL`.
2. **Motor de perturbação (§3.3):** **176 "levers"** linguísticos (~10 por combinação categoria×intensidade; 6×3=18). **3 intensidades:** LOW = incerteza sutil que "a competent model should resolve and answer correctly" (testa recusa *excessiva*); MEDIUM = deficit claro → deve recusar; HIGH = defeito severo → deve recusar. "The expected behavior is to answer correctly at LOW intensity and refuse appropriately at MEDIUM and HIGH."
3. **Pipeline gerador–verificador (§3.4):** 4 LLMs geram (Claude-4-Sonnet, DeepSeek-R1, GPT-4o, Nova-Pro) e todos verificam todos (7 critérios); só passa **consenso unânime**.
4. **Bases (§4.1):** *RefusalBench-NQ* (100 instâncias-base de NaturalQuestions+KILT, só perguntas que todos os modelos de fronteira acertaram sem perturbação → 1.600 amostras balanceadas) e *RefusalBench-GaRAGe* (100 bases multi-documento, 20 por 5 domínios → 1.506 amostras, desbalanceado).
5. **Protocolo de avaliação (§4.1, App. D/I):** cada modelo é instruído a **responder OU emitir "only the corresponding refusal code"** (prompt I.1: `REFUSE_AMBIGUOUS_QUERY`, `REFUSE_CONTRADICTORY_CONTEXT`, `REFUSE_INFO_MISSING_IN_CONTEXT`, `REFUSE_FALSE_PREMISE_IN_QUERY`, `REFUSE_GRANULARITY_MISMATCH`, `REFUSE_NONFACTUAL_QUERY`, `REFUSE_OTHER`). Um **LLM-juiz (Claude-4-Sonnet)** classifica a saída em `answer_attempt` ou um `REFUSE_*` ("Look for refusal codes even if they appear with additional text") e dá nota 1–5 às respostas (≥4 = correta, para NQ; GaRAGe usa o RAF score oficial). *(Ou seja: a detecção de recusa é por **marcador estruturado emitido pelo modelo**, com juiz como parser tolerante e avaliador de qualidade — não por regex de frase livre.)*

**Métricas (App. D, "Core Behavioral Metrics" e "Other Refusal Analysis Metrics")** — com explicação simples:
- **Answer Accuracy** — dos respondíveis, fração respondida *e* correta.
- **Refusal Accuracy** — dos irrespondíveis, fração recusada **com a categoria certa** ("correctly refused with appropriate categorization"). *(Mais exigente que "recusou".)*
- **Correct Refusal Rate** — dos irrespondíveis, fração que recusou (sem exigir categoria).
- **False Refusal Rate (FRR)** — dos *respondíveis*, fração recusada: "measuring over-cautious behavior" = **over-refusal**.
- **Missed Refusal Rate (MRR)** — dos *irrespondíveis*, fração respondida: "potentially harmful over-confidence" = **under-refusal**. *(Atenção: MRR aqui ≠ Mean Reciprocal Rank do nosso roadmap — sigla homônima.)*
- **Refusal Detection F1** — F1 binário (recusar × responder). *(Mede só "decidiu certo?", sem categoria.)*
- **Category Accuracy** — dado que recusou corretamente, acertou o *motivo*?
- **Hierarchical Refusal Score** = Detection F1 × Category Accuracy.
- **Calibrated Refusal Score (CRS)** = média aritmética de Answer Accuracy e Refusal Accuracy ("our primary balanced metric"); **Hybrid Score (GaRAGe)** = média ponderada pela proporção no dataset.
- **ECE** (Expected Calibration Error, 5 bins), global/respostas/recusas. *(Mede se a confiança declarada bate com a taxa de acerto.)*

## 3. Principais contribuições
1. Metodologia **generativa** (perturbação de QA respondível → irrespondível) com prova de limite de erro sob contaminação (Teorema 3.1, App. B).
2. Taxonomia de 6 incertezas × 3 intensidades; 176 levers; dois benchmarks e o framework liberados.
3. Estudo de **30+ modelos**: recusa = **duas habilidades separáveis** (detectar *quando* × categorizar *por quê*) (RQ2, Fig. 6).
4. Escala e raciocínio estendido **não** melhoram recusa (RQ3); recusa é "trainable, alignment-sensitive" (DPO > SFT; Claude forte).
5. Evidência de **viés de auto-avaliação** e baixíssimo acordo entre verificadores LLM (Fig. 3, 14) → justifica consenso unânime.

## 4. Resultados-chave
- **Refusal accuracy (RQ2, §4.2):** melhor no NQ = **73,0%** (Claude-4-Sonnet); no GaRAGe o melhor é DeepSeek-R1 com **47,4%**; Claude-4-Sonnet cai de 73,0% → 36,1%. "No frontier model achieves >80% on both dimensions" (Fig. 5). Resposta errada é rara (<3,0% NQ, <3,4% GaRAGe; App. F.8): "the decision of whether to answer dominates model failures".
- **Over × under-refusal (App. F.3, Fig. 22):** GPT-4o = **FRR 62,8% / MRR 4,3%** no NQ (refusa 14,6× mais do que o necessário); o4-mini = FRR 17,8% / MRR 21,5%; Claude: FRR 32–42%, MRR ≲11%; no GaRAGe Nova-Premier MRR 53,7%. Correlação **r = −0,78** (NQ) entre falsa e perdida: "models face a fundamental trade-off" (RQ1).
- **Detecção ≠ categorização (§4.2, F.6–F.7):** Claude-3.5-Sonnet recusa corretamente 88,2% dos irrespondíveis (NQ); GPT-4o recusa 88,4% mas acerta a categoria em apenas **54,1%**. Modelos usam `REFUSE_INFO_MISSING` como "catch-all" (25% das predições; Fig. 8). `REFUSE_GRANULARITY` ≈ "nearly unsolvable".
- **CRS (F.7):** Claude-4-Sonnet 65,3% no NQ (refusal 73,0%, answer 57,7%) → 51,7% no GaRAGe.
- **Intensidade:** refusal rate cresce monotonicamente de LOW a HIGH; GPT-4o já recusa 62,8% no LOW (F.5).
- **Escala/raciocínio (RQ3):** Qwen salta de 13,0% a 56,1% de answer acc entre 4B e 7B, mas refusal acc fica <17% em todos os tamanhos; até 4096 tokens de thinking dá <1 p.p. de ganho em refusal acc (Fig. 32).
- **Calibração (§4.2):** todos mal calibrados; ECE 0,286 (Claude-4-Sonnet, melhor) até 0,546 (GPT-4.1); ">73% das predições em confiança máxima".
- **Qualidade do dataset:** pass-rate humano 93,1% (NQ) e 88,3% (GaRAGe), 180 perturbações por benchmark, **1 anotador especialista** (§4.1, C.2); auto-avaliação de verificadores 91,0% vs 82,1% cross; κ entre verificadores 0,061–0,442 (NQ), 0,116–0,230 (GaRAGe) (App. E.1, Fig. 14).

## 5. Limitações
**[Artigo] (Limitations):** perturbações programáticas podem não capturar a complexidade orgânica de fontes reais; qualidade arbitrada por LLMs (viés compartilhado não eliminado pelo consenso); risco de os provedores treinarem nos levers (overfitting ao padrão, não à instância).
**[Minha leitura]**
- A validação humana (1 anotador, 180 itens) valida a **qualidade das perturbações**, **não** o juiz que classifica resposta × recusa e categoria; **não encontrei no texto lido uma validação humana desse juiz** (o juiz principal, Claude-4-Sonnet, é também um dos modelos avaliados, um dos geradores e um dos verificadores — risco de auto-preferência que o próprio artigo documenta em Fig. 3).
- FRR é medido, em grande parte, nas instâncias **LOW** (perturbações "sutis" cuja resposta correta é responder) — ou seja, o "respondível" é um respondível *quase-ambíguo*, não uma pergunta natural limpa; os 62,8% de GPT-4o não equivalem a "recusa 62,8% de perguntas normais". Cuidado ao comparar com over-refusal em perguntas respondíveis comuns.
- Contexto sempre dado (grounded) e **o defeito está no contexto/pergunta fornecido**; não cobre o caso "o retrieval falhou e o contexto é irrelevante".
- Cenário NQ/GaRAGe (Wikipedia/web); sem domínio acadêmico/abstracts.
- Sem intervalos de confiança nas tabelas que li.

## 6. Reflexões ancoradas no NOSSO projeto
| # | Ponto / arquivo | Veredito | Reflexão |
|---|---|---|---|
| 1 | **P8** — acurácia × P/R/F1, over/under-refusal | **APOIA fortemente** | É o artigo mais direto sobre a nossa proposta: separa **FRR (over-refusal)** de **MRR (under-refusal)**, reporta **Detection F1** e mostra o trade-off (r=−0,78; GPT-4o 62,8%/4,3%). O trade-off sustenta exigir *ambos* os lados num mesmo relatório. **Mas:** a métrica "principal" que eles propõem é o **CRS = média simples de duas acurácias**, não F1 de classe; e o F1 deles é diagnóstico. |
| 2 | **P8** — `f1_summary_evaluator` | **APOIA** | Mapeamento direto: nosso recall de abstenção ≈ *Correct Refusal Rate* (1−MRR); nosso over_refusal_rate ≈ FRR; F1 ≈ Refusal Detection F1. Eles confirmam que "high detection F1 doesn't guarantee good categorization" — relevante só se um dia categorizarmos o motivo (não é requisito do plano). |
| 3 | **P7** — detecção por `startswith` | **DESAFIA (e dá alternativa concreta)** | Eles **eliminam a ambiguidade na origem**: o modelo é instruído a emitir *só* um código `REFUSE_*`; o juiz apenas lê ("even if they appear with additional text"). É exatamente a opção 3 do P7 ("marcador estruturado / campo JSON"). **Adotaríamos** (em experimento CKPT-6, não no baseline): marcador explícito + parser tolerante + juiz como fallback. **NÃO adotaríamos** no EXP-0: mudar o prompt do produto só para medir (o baseline deve medir o prompt real). |
| 4 | **P8** — nossos dois gabaritos de recusa ("I don't know." / "No papers found in this period.") | **NEUTRO/DESAFIA** | A taxonomia deles (6 motivos) distingue *por que* recusar. O nosso projeto tem 2 motivos (fora do corpus; sem docs no período). Não precisamos de 6 categorias; mas reportar por slice (`unanswerable` × `stale`) é o análogo de reportar por categoria — e eles mostram que as categorias diferem muito (INFO_MISSING fácil, GRANULARITY quase impossível). |
| 5 | **Gate "≥90% em unanswerable"** | **DESAFIA (cautela)** | Nenhum modelo de fronteira passa de 73% de refusal accuracy **com categoria**; sem categoria (detecção) Claude-3.5 chega a 88,2% e GPT-4o a 88,4% **à custa de FRR enorme**. Conclusão: um gate de recall isolado é **gameável por over-refusal** → reforça tornar o guarda-corpo de over-refusal obrigatório junto do gate (P8, registro no `decisions.md`). |
| 6 | **Slices `unanswerable`/`stale` × cenários deles** | **PARCIAL** | Nosso `unanswerable` (fora do corpus) ≈ P-MissingInfo/"informação ausente no contexto" (a mais tratável: 76–98% de refusal acc por modelo no NQ, App. F.4) — porém lá o contexto *tem* defeito dado; no nosso, o retrieval devolve **contexto irrelevante** e o modelo deve notar. `stale` (pedir novidades de um mês sem papers) é análogo a MissingInfo temporal; **não há categoria "stale" no RefusalBench** (diferente do AbstentionBench). |
| 7 | **P11** — juiz = gerador | **APOIA** | Evidência empírica (Fig. 3, §4.1): auto-avaliação 91,0% vs 82,1% cross; Claude-4-Sonnet com viés negativo (75,7% self vs 97,3% cross) — o viés **tem sinal variável**; κ entre juízes tão baixo quanto 0,061 → "models apply fundamentally different quality criteria". Sustenta separar `JUDGE_MODEL` e medir κ/sensibilidade a 2 juízes. (Nota: o viés estudado é de *geração de perturbações*, não de julgar respostas — extrapolação minha; ainda assim consistente com Zheng 2023/Wataoka 2024.) |
| 8 | **P7/gate** — tamanho amostral | **DESAFIA (lacuna)** | 100 bases/benchmark, n por célula 100–200+; eles tampouco reportam ICs que li. Nosso n=15/10 exige reportar contagens. |

**O que adotaríamos:** FRR + MRR (ou seus nomes em PT "over/under-refusal rate") com *Detection F1* como diagnóstico; marcador estruturado de recusa como experimento de produto; **separar juiz do gerador**; reportar por slice.
**O que NÃO adotaríamos:** CRS (média simples) como gate — esconde o trade-off; a taxonomia de 6 categorias; o desenho de perturbação programática (o nosso `unanswerable` é humano/sintético *by construction* do corpus — ADR-001/003).

## 7. Citações úteis (máx. 3)
1. "The expected behavior is to answer correctly at LOW intensity and refuse appropriately at MEDIUM and HIGH intensities." — §3.3.
2. "On all types, we find that models face a fundamental trade-off: the strong negative correlation (r = −0.78 on NQ) between false and missed refusals forces models to choose between being overly cautious or overly permissive." — §4.2, RQ1.
3. "…it refuses 14.6 times more often than necessary to avoid harmful outputs." (sobre GPT-4o, FRR 62,8% vs MRR 4,3%) — App. F.3.

---

# 3. Comparação AbstentionBench × RefusalBench × nosso projeto

| Dimensão | AbstentionBench (2506.09038) | RefusalBench (2510.10390) | Nosso projeto |
|---|---|---|---|
| Unidade | pergunta (+contexto opcional) com `should_abstain` | (pergunta, contexto) perturbado, com código de recusa esperado | pergunta + RAG de abstracts; `should_abstain` por slice |
| Origem do irrespondível | datasets existentes + variantes (remoção de contexto) | **perturbação programática** de QA respondível | construção sintética/humana (ADR-001/003) |
| Detecção de abstenção | **LLM-judge** Llama 3.1 8B, "yes/no", T=0; recebe o rótulo | **modelo emite `REFUSE_*`**; LLM-judge (Claude-4-Sonnet) classifica | `startswith` (P7) → juiz `abstention_quality` previsto |
| Validação humana | 300 pares, autores, estratificada, 50/50 val/teste; acc 0,88 | 180 perturbações, **1** especialista (valida o dataset, não o juiz) | 100/100 exemplos revisados; juiz ainda sem κ |
| Métrica principal | **recall de abstenção** | **Refusal Accuracy** (com categoria) e **CRS** (média Ans+Ref acc) | acurácia de abstenção (→ P8) |
| Over-refusal | via **precision** (secundária; ≈1) | **FRR** explícito + Detection F1 | não medido hoje |
| Under-refusal | 1 − recall | **MRR** explícito (homônimo de MRR de ranking!) | não medido hoje |
| P/R/F1 ou acurácia? | P/R/F1 (foco em R); **não** acurácia | FRR/MRR/F1 + acurácias por classe; **CRS** é média de acurácias | acurácia (a mudar) |
| "Stale" | **sim**, cenário nomeado (FreshQA; resposta mudou pós-cutoff **do modelo**) | não | `stale` = corpus fixo sem papers no período |
| Sustenta P8? | **Parcialmente** (P/R/F1; mas foco em recall) | **Sim** (over/under + F1 + trade-off) | — |

**[Minha leitura] Veredito P8:** os dois artigos **sustentam abandonar a acurácia isolada** e medir over e under-refusal separadamente. **Nenhum prescreve "F1 como headline"**: AbstentionBench prioriza *recall* (= nosso gate), RefusalBench prioriza acurácias por classe + CRS. Portanto a proposta do P8 (recall/precision/F1 + over/under) é **compatível e conservadora**, mas o artigo não é fonte para afirmar "a literatura manda F1".

---

# 4. Extra A — Peng et al. 2025 — Unanswerability Evaluation for Retrieval Augmented Generation (UAEval4RAG)

- **Referência:** Xiangyu Peng, Prafulla Kumar Choubey, Caiming Xiong, Chien-Sheng Wu (Salesforce Research). *Unanswerability Evaluation for Retrieval Augmented Generation*. arXiv:2412.12300v3, 21 Abr 2025. <https://arxiv.org/abs/2412.12300> · código <https://github.com/SalesforceAIResearch/Unanswerability_RAGE>
- **Confiança da leitura:** **parcial** — corpo §1–4 lido (taxonomia, pipeline, métricas, Tabelas 1–5, validação humana); **não li** o apêndice completo (prompts A.1–A.8, Tabelas 6–18, Tabela 16 Llama-Guard); tabelas extraídas desordenadas → uso só números reconstituíveis com segurança.

**1. Problema.** Benchmarks de RAG medem só perguntas respondíveis; benchmarks de unanswerable testam LLMs *sem* base, de modo que "rejection often stems from the inability to retrieve relevant context rather than a true understanding that the request should not be fulfilled" (§1).

**2. Método.** Taxonomia de **6 categorias** (§3.1): Underspecified, False-presupposition, Nonsensical, Modality-limited, Safety-concerned e **Out-of-Database** (relevante ao domínio, mas sem resposta na base). Pipeline que sintetiza pedidos irrespondíveis **para qualquer base de conhecimento** com geração + verificação por LLM (§3.2); para Out-of-Database, crawl de notícias recentes → pergunta → recupera chunks → verifica que nenhum contém a resposta (Fig. 3). Métricas LLM (§3.3): **Acceptable Ratio** (resposta aceitável por critérios por categoria), **Unanswered / Answered / Ask-for-Clarification Ratio**, e **Joint Score = w1·Correctness + w2·Acceptable Ratio** com w1=0,7, w2=0,3 ("no universal weight").

**3. Contribuições.** Taxonomia para RAG; geração automática de unanswerables ancorados na base; análise de 27 combinações de componentes (embedding, retriever, reranker, rewriting, 3 LLMs, 3 prompts) em 4 datasets.

**4. Resultados.** Validação do pipeline: 3 autores, 92% de acurácia (TriviaQA e MuSiQue), concordância 0,85/0,88 (§4.1). Juiz LLM × 150 pares rotulados por 3 autores (concordância 0,76 e 0,83): acurácia 81–84% e F1 76–86% nos três juízes (GPT-4o, Claude 3.5, DeepSeek-R1) (Tab. 1). "No single configuration" é ótima em todos os datasets (§4.3). **Prompts restritivos** sobem muito o Acceptable/Unanswered mas **reduzem** a correção nas respondíveis (Tab. 3: ex. TriviaQA, prompt default→#2: acceptable 53,2%→83,0% e correct 88,0%→74,8%; MuSiQue default→#2: acceptable 61,7%→88,0% e correct 49,0%→16,0%). "Underspecified" é a categoria mais difícil; "False Presupposition" e "Out-of-Database" as mais fáceis (§4.3).

**5. Limitações.** *[Artigo]* não há seção própria nas partes lidas; *[Minha leitura]* sintético+LLM-juiz sem ICs; domínio Wikipedia/trivia; a ponderação 0,7/0,3 é arbitrária (admitido); a ordem das colunas de Tab. 2–5 no PDF extraído é ambígua, não reproduzi números de célula.

**6. Reflexões ancoradas no projeto.**
- **P8 — APOIA** a necessidade de olhar os **dois lados**: o *trade-off* answerable × unanswerable é o resultado central (prompt restritivo ↑ rejeição, ↓ correção; Tab. 3). Sustenta o guarda-corpo de over-refusal; **desafia** só em um ponto: eles resumem num **Joint Score** ponderado (0,7/0,3), não em F1 de classe. Para nós, um Joint Score seria um segundo agregado possível, mas o peso é arbitrário → **não adotaríamos como gate**.
- **Slice `unanswerable` e `stale` — APOIA a taxonomia:** *Out-of-Database* (pergunta relevante ao domínio sem resposta na base) é o análogo mais fiel do nosso `unanswerable` fora do corpus, e a sua construção usa **notícias recentes** (≈ nosso `stale`: "o que há de novo em <mês>?"). A lição: o benchmark deles mostra que, para RAG, **Out-of-Database é fácil** (aceitável alto em Tab. 4) — logo um baseline baixo no nosso `unanswerable` apontaria para o prompt/detecção (P7) mais que para capacidade.
- **P7 — NEUTRO/DESAFIA:** usam juiz LLM com exemplos in-context por categoria e validam com 150 pares humanos (acc ≈ 82–84%) — mesma receita do AbstentionBench; **concordância humano×humano de 0,76** mostra que "o que é abstenção/aceitável" é ambíguo mesmo para humanos → ceticismo com metas de 90% de concordância heurística×juiz (alvo do P7).
- **O que adotaríamos:** distinção "rejeitado" × "pediu clarificação" × "respondeu" (três estados, não dois); critério por categoria. **O que não:** Joint Score com pesos fixos como release gate.

**7. Citação útil.** "…rejection often stems from the inability to retrieve relevant context rather than a true understanding that the request should not be fulfilled." — §1, p.1 *(= o risco de medir só por "recuperou algo / não recuperou nada")*.

---

# 5. Extra B — Cui et al. 2025 — OR-Bench: An Over-Refusal Benchmark for Large Language Models

- **Referência:** Justin Cui, Wei-Lin Chiang, Ion Stoica, Cho-Jui Hsieh. *OR-Bench: An Over-Refusal Benchmark for Large Language Models*. ICML 2025 (PMLR 267); arXiv:2405.20947v5, 15 Jun 2025. <https://arxiv.org/abs/2405.20947>
- **Confiança da leitura:** **parcial** — li abstract, §1–3 (definições, pipeline de geração/moderação) e o início de §4 (setup, avaliação, Tabela 1); **não li** §4.3 em diante, Tabelas 2/6/7, defesas de jailbreak e a maior parte dos apêndices (F, Q, V). Escopo: **recusa por segurança** (prompts quase-tóxicos), *não* abstenção por falta de conhecimento.

**1. Problema.** Alinhamento de segurança causa **over-refusal**: recusar prompts inofensivos. Falta um benchmark em larga escala (só havia o XSTest, 250 prompts manuais) (§1).

**2. Método.** *Over-refusal* definido como "when a model refuses to provide a helpful response, even when a safe and plausible answer is possible" (§3.1). Pipeline: (1) sementes tóxicas por Mixtral-8×7B em 10 categorias; (2) reescrita em prompts "borderline" benignos (5 por semente, com 5 exemplos few-shot); (3) **moderação por ensemble de 3 LLMs** (GPT-4-turbo, Llama-3-70b, Gemini-1.5-pro; voto de maioria) → OR-Bench-80K; subconjunto **Hard-1K** (rejeitados por ≥3 dos maiores modelos de cada família) e **OR-Bench-Toxic** (600 prompts tóxicos "para evitar respostas indiscriminadas"). **Detecção de recusa:** **keyword matching** no conjunto de 80K e **GPT-4** no Hard-1K e Toxic; "keyword matching closely approximates GPT-4 evaluations… discrepancies of 2.4% for GPT-3.5-turbo-0125 and 1.2% for Llama-3-70b on sampled datasets" (§4.1; App. F, não lido).

**3. Contribuições.** Pipeline automatizado; OR-Bench-80K/Hard-1K/Toxic; avaliação de 32 modelos de 8 famílias.

**4. Resultados.** Moderador ensemble vs especialista humano: TP 94,7/96,0, FN 5,3/4,0, TN 92,0/84,0, FP 8,0/16,0, Acc 94,0/93,0 (Tab. 1; colunas "humano/ensemble" pela ordem do PDF extraído) → "over 98% of the performance level of human experts" (§3.2.3). Resultado principal: **Spearman 0,89** entre rejeição de tóxicos (segurança) e over-refusal (Fig. 1) — "most models simply trade over-refusal for safety". Modelo maior ≠ melhor equilíbrio; Claude mais seguro e mais over-refusing; GPT-3.5-turbo reduz over-refusal nas versões mais novas.

**5. Limitações.** *[Artigo]* não lidas. *[Minha leitura]* domínio de segurança (frases de recusa padronizadas: "I cannot…", "I'm sorry…") — **o sucesso do keyword matching é específico desse tipo de recusa** e não se transfere automaticamente a abstenção epistêmica ("I don't know", "the context doesn't say").

**6. Reflexões ancoradas no projeto.**
- **P7 — APOIA com ressalva:** valida empiricamente que **heurística barata pode aproximar um juiz forte** (≤2,4% de divergência) *quando medida contra o juiz*, exatamente o desenho do P7 (heurística como pré-filtro + concordância heurística×juiz no EXP-0). **Ressalva:** em recusa de segurança as frases são previsíveis; na nossa abstenção epistêmica com `PROMPT_V1` sem frase fixa, a concordância tem que ser **medida**, não presumida.
- **P8 — APOIA:** só faz sentido reportar over-refusal **pareado** com a taxa de aceitação indevida (toxic) — eles incluem o conjunto Toxic justamente para que um modelo que "recusa tudo" não pareça bom. É o análogo exato do nosso risco: abstenção alta comprada com recusa indiscriminada. Sustenta **guarda-corpo de over-refusal** no gate.
- **O que adotaríamos:** par de métricas inseparável (recall de abstenção + over-refusal); medir divergência heurística×juiz. **O que não:** transpor a taxa de keyword matching como evidência para nosso caso.

**7. Citação útil.** "…most models simply trade over-refusal for safety, with few breaking the trade-off." — §4.2 (resultado do Spearman 0,89 sobre Fig. 1).

---

# 6. Síntese para o roadmap

| Ponto | O que a leitura integral permite afirmar | O que continua decisão de engenharia (não da literatura) |
|---|---|---|
| **P7** | Match de string/embedding é considerado inadequado (AbstentionBench C.3.1); duas alternativas validadas: **juiz LLM T=0 + amostra humana estratificada** (AbstentionBench, Tab. 1) e **marcador estruturado `REFUSE_*`** (RefusalBench §4.1); heurística barata ≈ juiz em recusa de segurança (OR-Bench). Nem humano×humano concorda 100% (UAEval4RAG 0,76/0,83). | Meta de concordância heurística×juiz ≥90%; qual juiz usar; se mudar o prompt do produto. |
| **P8** | Separar over e under-refusal é prática comum (RefusalBench FRR/MRR; AbstentionBench precision; OR-Bench pareado com Toxic; UAEval4RAG trade-off). **Acurácia isolada não é usada por nenhum.** | Que F1 seja a métrica de resumo (eles priorizam recall ou médias de acurácia); IC/Wilson para n=15/10; regra "over-refusal não sobe mais que X p.p.". |
| **P11** | Auto-preferência e κ baixo entre juízes LLM (RefusalBench Fig. 3/14). | Qual modelo usar como juiz. |
| Dado pequeno | Nenhum dos artigos oferece guia para n≈15. | Reportar contagens brutas + IC. |
