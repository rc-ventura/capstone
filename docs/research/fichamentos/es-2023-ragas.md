# Es et al. 2023 — RAGAS (avaliação de RAG sem gabarito)

- **Referência completa:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert. *Ragas: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217 (versão lida: v2, 28/04/2025). <https://arxiv.org/abs/2309.15217> · Código: <https://github.com/explodinggradients/ragas> · Dataset: <https://huggingface.co/datasets/explodinggradients/WikiEval>
  *(O paper é um artigo curto de demonstração; o venue de publicação — EACL 2024, demos — vem do meu conhecimento prévio e **não foi verificado** nesta leitura.)*
- **Confiança da leitura:** **integral.** Li o texto bruto completo da versão HTML do arXiv (v2): resumo, Seções 1–6, Tabela 1 e Apêndice A (Tabelas 2–4 de exemplos). O artigo é curto (≈ 34 mil caracteres). **Não vi:** figuras (não há figuras relevantes no texto extraído). O PDF não pôde ser decodificado pela ferramenta de leitura; usei o HTML.

> Convenção: **[Artigo]** = afirmado no texto; **[Minha leitura]** = interpretação minha.

---

## 1. Problema que o artigo ataca

**[Artigo]** Avaliar RAG exige olhar várias dimensões: a capacidade do retriever de achar passagens relevantes *e focadas*, a capacidade do LLM de usá-las *fielmente*, e a qualidade da geração em si (Resumo). Avaliações comuns têm limites: perplexidade nem sempre prevê desempenho downstream e exige probabilidades que modelos fechados não expõem; QA costuma usar datasets de resposta curta/extrativa, pouco representativos (Seção 1). Objetivo: um conjunto de métricas **reference-free** — "sem depender de anotações humanas de verdade" — para acelerar ciclos de avaliação (Resumo, Seção 3: "we focus on metrics that are fully self-contained and reference-free").

## 2. Método (passo a passo, fórmulas exatas)

Setting (Seção 3): dada uma pergunta *q*, o sistema recupera contexto *c(q)* e gera resposta *a_s(q)*. Três aspectos de qualidade, todos medidos **por prompting de LLM** (gpt-3.5-turbo-16k; embeddings: text-embedding-ada-002). Não há uso de gabarito.

### 2.1 Faithfulness (fidelidade) — Seção 3

*(mede se tudo o que a resposta afirma pode ser deduzido do contexto recuperado; é o antídoto contra alucinação)*

1. LLM decompõe a resposta em um conjunto de **statements** *S(a_s(q))* (prompt: "Given a question and answer, create one or more statements from each sentence in the given answer").
2. Para cada statement *sᵢ*, o LLM verifica se pode ser inferido do contexto (função *v(sᵢ, c(q))*; prompt de verificação com veredito Yes/No e explicação breve por statement).
3. **F = |V| / |S|**, em que |V| = nº de statements suportados e |S| = total de statements.

### 2.2 Answer relevance (relevância da resposta) — Seção 3, Eq. (1)

*(mede se a resposta trata exatamente da pergunta feita, sem ser incompleta nem redundante; **não** avalia se é verdadeira)*

1. Dado *a_s(q)*, o LLM gera **n perguntas** *qᵢ* plausíveis para aquela resposta (prompt: "Generate a question for the given answer").
2. Embeddings (ada-002) de todas as perguntas; **sim(q, qᵢ)** = cosseno entre a pergunta original e cada pergunta gerada.
3. **AR = (1/n) · Σᵢ sim(q, qᵢ)**.

O artigo diz explicitamente que essa avaliação "não leva em conta factualidade, mas penaliza respostas incompletas ou com informação redundante".

### 2.3 Context relevance (relevância do contexto) — Seção 3, Eq. (2)

*(mede o quanto o contexto recuperado é **focado**: quanta fração dele é realmente necessária para responder; penaliza contexto com informação redundante/irrelevante)*

1. O LLM extrai de *c(q)* o subconjunto de sentenças **S_ext** "cruciais" para responder *q* (prompt pede sentenças relevantes sem alterá-las; se não houver, retorna "Insufficient Information").
2. **CR = nº de sentenças extraídas / nº total de sentenças em c(q)**.

**[Minha leitura]** Isto **não** é recall nem precision de retrieval no sentido de IR: não compara com documentos-gold, não considera a posição no ranking e não mede se *toda* a informação necessária está presente — mede só a *proporção do contexto entregue que o LLM julga útil*. Um contexto que contém uma única sentença útil e nada mais teria CR = 1,0 mesmo faltando o resto da resposta.

### 2.4 Validação: WikiEval (Seção 4)

- **Construção:** 50 páginas da Wikipedia sobre eventos desde o início de 2022 (além do corte do modelo), priorizando páginas com edições recentes. Para cada página, o ChatGPT sugere uma pergunta respondível pela seção introdutória e a responde com essa seção como contexto.
- **Anotação:** duas pessoas anotaram as três dimensões; concordância ~**95%** (faithfulness e context relevance) e ~**90%** (answer relevance); divergências resolvidas por discussão.
- **Tarefas construídas (pareadas):**
  - *Faithfulness:* comparar a resposta padrão com a que o ChatGPT gerou **sem** contexto; o humano diz qual é mais fiel à página.
  - *Answer relevance:* comparar a resposta padrão com uma resposta deliberadamente **incompleta** (ChatGPT instruído a responder de forma incompleta).
  - *Context relevance:* comparar o contexto original com um contexto **ampliado** com sentenças de backlinks da Wikipedia (ou completado pelo ChatGPT, nas páginas sem backlinks).

### 2.5 Baselines (Seção 5)

- **GPT Score:** pede ao ChatGPT uma nota de 0 a 10 por dimensão (prompt com a definição); empates desfeitos aleatoriamente.
- **GPT Ranking:** pede ao ChatGPT para escolher a resposta/contexto preferido, com a definição no prompt.

Métrica de concordância: **acurácia** = fração de instâncias em que o candidato preferido pela métrica coincide com o preferido pelos anotadores.

## 3. Principais contribuições

1. Três métricas **reference-free** definidas operacionalmente (F=|V|/|S|; AR por perguntas geradas + cosseno; CR por extração de sentenças) — Seção 3.
2. **WikiEval**, dataset público com julgamentos humanos nas três dimensões — Seção 4.
3. Integração com llama-index e LangChain (Seção 1), que explica a adoção prática da biblioteca.
4. Evidência de que, no WikiEval, as métricas concordam mais com humanos que dois baselines de LLM-judge (Tabela 1).

## 4. Resultados-chave

**Tabela 1 (Seção 5)** — acurácia de concordância com humanos em comparações pareadas, WikiEval:

| Método | Faithfulness | Answer Relevance | Context Relevance |
|---|---|---|---|
| **Ragas** | **0.95** | **0.78** | **0.70** |
| GPT Score | 0.72 | 0.52 | 0.63 |
| GPT Ranking | 0.54 | 0.40 | 0.52 |

Texto da Seção 5: para faithfulness a concordância é "em geral altamente precisa"; para answer relevance é menor, "em grande parte porque as diferenças entre as duas respostas candidatas são muitas vezes muito sutis"; context relevance foi "a dimensão mais difícil" — o ChatGPT "often struggles" ao selecionar as sentenças cruciais, "especialmente para contextos mais longos".

**Apêndice A (Tabelas 2–4):** exemplos. Em particular, a Tabela 3 mostra como "baixa answer relevance" uma resposta que diz que data/hora "não foram fornecidas" e depois acrescenta conteúdo genérico.

**Corroboração externa (outro artigo, para contexto):** no RAGChecker (Tabela 5 do apêndice), métricas do RAGAS correlacionam fraco com preferência humana de correção/completude: Faithfulness Pearson 8.22 / 4.90 / 7.83 e Answer Relevance 11.59 / 9.39 / 10.27 (correctness / completeness / overall). **Ressalva:** o alvo humano ali (correção/completude vs gabarito) é diferente do que essas métricas pretendem medir.

## 5. Limitações

**[Artigo]** As da Seção 5: answer relevance tem concordância menor por diferenças sutis; context relevance é a mais difícil (erro de seleção de sentenças em contextos longos). Não há seção de limitações formal.

**[Minha leitura]**
- **Amostra pequena e tarefa fácil:** n = 50 perguntas, todas geradas e respondidas pelo ChatGPT, com **contrastes construídos** (resposta com × sem contexto; completa × incompleta; contexto limpo × contexto poluído). A acurácia de 0.95 em faithfulness é em uma escolha binária pareada com contraste grande — não equivale à acurácia de um juiz *absoluto* sobre respostas reais, que são quase todas "parcialmente fiéis".
- **Mesma família gerando e julgando:** respostas do ChatGPT, julgadas por gpt-3.5-turbo; o artigo não discute autopreferência (self-enhancement). Relevante para P11.
- **Sem validação do uso real:** nenhuma análise de variância entre rodadas, sensibilidade ao prompt ou custo.
- **Answer relevance e abstenção:** pela definição (gerar perguntas a partir da resposta), uma resposta "não sei" gera perguntas genéricas e pontua baixo; o exemplo "low answer relevance" da Tabela 3 contém exatamente "have not been provided". O artigo não discute abstenção.
- **Context relevance e conteúdo ausente:** como CR é razão de sentenças úteis sobre total, não detecta *falta* de informação — o oposto de um recall.
- **Dependência de segmentação em sentenças:** com chunks de ~420 caracteres (2–3 sentenças), a granularidade seria grosseira. *(Inferência: o artigo não discute tamanho de chunk.)*

## 6. Reflexões ancoradas no NOSSO projeto

**R1 — Context relevance NÃO substitui as métricas por ID (P1–P3, P5, P6, D-1; `hit_rate`, `recall_at_k`, `mrr`, `precision_at_k`).** *NEUTRO/DESAFIA a ideia de usá-las como alternativa.* CR não usa gabarito (bom: dispensa `gold_chunk_ids`), mas por isso mesmo **não responde** "o chunk que responde estava no top-k?" nem "em que posição?". Em nosso corpus, com gold de 1 chunk e top-5, um contexto ideal teria 1 chunk útil entre 5: CR tende a ter um teto próximo de 1/k — o mesmo problema do P6 (teto 1/k de `precision_at_k`), *[Minha leitura; é inferência, não medido]*. Logo CR herdaria o defeito do P6 sem entregar a vantagem de ranking do MRR.

**R2 — Onde adotaríamos CR (ckpt-0.6-plan §3.1; fatia `multi-doc open-topic`, 5 exemplos com `retrieval_gold=pending-adjudication`).** *APOIA como complemento.* Para esses 5 exemplos, o plano os **pula** das métricas de retrieval até o mini-pooling do EXP-0. CR (ou melhor, uma variante por entailment — ver ficha do RAGChecker) daria um **sinal provisório de foco do contexto** sem gold. Também serve como **diagnóstico de ruído** (FP3) em todos os slices: "quanto do contexto entregue ao gerador é inútil" — relevante para a escolha de k no CKPT-1 e para o colapso de dedup (mesmo paper 3× no top-5). Marcado como diagnóstico, **não** como gate.

**R3 — `faithfulness_judge` (ckpt-0.6-plan §3.2, FP4).** *APOIA.* O plano já descreve "juiz verifica claim a claim da resposta contra `retrieved_context`" — é exatamente F = |V|/|S| em dois passos (extrair statements; verificar cada um). A diferença a decidir é o formato: RAGAS usa **proporção** de statements suportados; o plano fala em saída `{"score": bool|float, ...}`. Recomendo o float (proporção) por ser mais informativo que bool e permitir agregações por slice. Para `unanswerable`/`stale`, a faithfulness da resposta de abstenção é trivialmente alta (nada afirmado) — não confundir com acerto de abstenção (P8).

**R4 — `answer_relevance` (ckpt-0.6-plan §3.2).** *APOIA parcialmente; DESAFIA aplicar em todos os slices.* A definição por perguntas geradas + cosseno é barata e determinística (sem juiz de nota 1–10), uma alternativa a um juiz LLM livre. **Mas** penaliza abstenção correta (ver Limitações). A matriz do plano já restringe `answer_relevance` a `answerable`, o que é consistente; reforço: **nunca** aplicá-la em `unanswerable`/`stale`, senão uma abstenção correta aparece como "resposta irrelevante" e distorce P8. Adotaríamos o cálculo por embeddings como **baseline barato** e compararíamos com o juiz LLM na calibração (0.8).

**R5 — Validação do juiz (P11, P9; calibração 0.8).** *DESAFIA a força da evidência.* O RAGAS mostra que métricas ref-free *podem* concordar com humanos, mas sob condições favoráveis (n=50, pares com contraste forte, 2 anotadores). Não serve como prova de que um juiz gpt-4o-mini funciona no **nosso** domínio (abstracts de cs.AI, chunks de ~420 caracteres). Reforça a decisão de calibrar com κ no nosso conjunto e **não** confiar em "o RAGAS já foi validado". Também mostra o risco P11 em forma pura: o mesmo ChatGPT gera e julga.

**R6 — Reference-free × nosso gold (ckpt-0.6-plan §3.2; `completeness_judge`, `abstention_quality`).** *NEUTRO.* Nosso golden set **tem** gabarito (resposta, gold IDs, `should_abstain`); para correção, completude e abstenção usamos referência. O RAGAS cobre só o que dispensa referência (fidelidade, relevância, foco) — por isso é complemento do nosso catálogo, não alternativa.

**R7 — P9 (token-F1).** *NEUTRO.* O RAGAS não traz métrica de correção contra gabarito (só *answer relevance*, que ignora factualidade). Não ajuda a decidir token-F1 × juiz de correção; ver fichas do RAGChecker e do QA-judge.

**O que adotaríamos:** (1) a decomposição em statements + verificação para `faithfulness_judge`, com score = proporção; (2) CR como diagnóstico de ruído e sinal provisório para `open-topic`; (3) AR por embeddings como baseline barato de `answer_relevance`, só em `answerable`. **O que NÃO adotaríamos:** (1) CR como substituto de hit/recall/MRR ou como gate de retrieval; (2) AR nos slices de abstenção; (3) a acurácia de 0.95 como justificativa de confiabilidade do juiz (R5).

## 7. Citações úteis

1. *"We say that the answer a_s(q) is faithful to the context c(q) if the claims that are made in the answer can be inferred from the context."* — Seção 3, "Faithfulness".
2. *"our assessment of answer relevance does not take into account factuality, but penalises cases where the answer is incomplete or where it contains redundant information."* — Seção 3, "Answer relevance".
3. *"We found context relevance to be the hardest quality dimension to evaluate. In particular, we observed that ChatGPT often struggles with the task of selecting the sentences from the context that are crucial, especially for longer contexts."* — Seção 5 (após a Tabela 1).
