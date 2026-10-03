# Yu et al. 2024 — Survey de avaliação de RAG (Auepora)

- **Referência completa:** Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, Zhaofeng Liu. *Evaluation of Retrieval-Augmented Generation: A Survey.* arXiv:2405.07437v2 [cs.CL], 03/07/2024. <https://arxiv.org/abs/2405.07437> · Repositório do artigo: <https://github.com/YHPeter/Awesome-RAG-Evaluation>
- **Confiança da leitura:** **integral**, com ressalvas. Li o texto bruto da versão HTML do arXiv (v2): resumo, Seções 1–5, Tabelas 1 e 2 e a lista de benchmarks. **Não li:** o Apêndice 0.A (descrição da estrutura de um RAG), a lista de referências e as figuras (Fig. 1 e 2 vistas só pela legenda). O PDF não pôde ser decodificado pela ferramenta; o resumo automático do HTML continha erros, por isso usei o texto bruto. Números e definições abaixo vêm do texto e das tabelas; a localização é a seção/tabela do próprio artigo.
- **Natureza do artigo:** survey descritivo (sem experimento próprio, sem números comparativos de desempenho). Corte da literatura: junho/2024 (legenda da Tabela 1).

> Convenção desta ficha: **[Artigo]** = o que o artigo afirma; **[Minha leitura]** = interpretação minha, não afirmada pelos autores.

---

## 1. Problema que o artigo ataca

**[Artigo]** Avaliar um RAG é difícil porque ele é um sistema híbrido (recuperação + geração) e depende de bases de conhecimento dinâmicas (Seção 1 e Seção 2). Os autores listam três frentes de dificuldade (Seção 2):

- **Retrieval:** bases vastas e mutáveis; relevância que muda no tempo; fontes enganosas ou de baixa qualidade. Frase-chave: *"The traditional evaluation indicators for retrieval, such as Recall and Precision, cannot fully capture the nuances of RAG retrieval systems"* (Seção 2).
- **Generation:** avaliar fidelidade (faithfulness) e correção da resposta, mais relevância e coerência; tarefas abertas tornam "correto" subjetivo.
- **Sistema como um todo:** o desempenho não se entende avaliando cada componente isolado; latência e consultas ambíguas também importam.

**[Artigo]** Contribuições declaradas (Seção 1): (1) classificar os desafios de avaliar RAG por componente; (2) propor o framework de análise **Auepora**; (3) analisar os benchmarks existentes com ele.

Lacuna que o survey aponta: *"Most prior benchmarks predominantly tackle one or several aspects of the RAG assessment but lack a comprehensive, holistic analysis"* (Seção 2, "Conclusion").

## 2. Método (passo a passo, com definições exatas)

O "método" é analítico: os autores reúnem os frameworks/benchmarks de avaliação de RAG e os classificam por **Target** (o que avaliar), **Dataset** (como avaliar) e **Metric** (como medir). A ideia central é enumerar todos os pares possíveis entre **"Evaluable Outputs" (EOs, saídas avaliáveis)** e **"Ground Truths" (GTs, verdades de referência)** (Seção 3, Fig. 1).

### 2.1 Targets (Seção 3.1)

**Retrieval** — EO = documentos relevantes recuperados:

- **Relevance (Relevant Documents ↔ Query)** — *(mede se o que veio bate com a necessidade de informação expressa na pergunta)*. O artigo diz que mede "a precisão e especificidade do processo de recuperação".
- **Accuracy (Relevant Documents ↔ Documents Candidates)** — *(mede se o sistema consegue pontuar documentos relevantes acima dos irrelevantes dentro do conjunto de candidatos; é a face de ranking)*.

**Generation** — EO = texto gerado e conteúdo estruturado:

- **Relevance (Response ↔ Query)** — *(a resposta trata do que foi perguntado?)*
- **Faithfulness (Response ↔ Relevant Documents)** — *(a resposta só reflete o que está nos documentos relevantes? consistência entre gerado e fonte)*
- **Correctness (Response ↔ Sample Response)** — *(a resposta bate com uma resposta de referência, tomada como verdade)*

**Requisitos adicionais** (Seção 3.1, sem GT no sentido acima): *Latency* (tempo de resposta), *Diversity* (variedade/amplitude do recuperado e do gerado), *Noise Robustness* (lida com informação irrelevante sem piorar a resposta), *Negative Rejection* (abster-se quando a informação é insuficiente), *Counterfactual Robustness* (identifica e ignora informação incorreta mesmo avisada), e "mais": readability, toxicity, perplexity.

### 2.2 Datasets (Seção 3.2, Tabela 2)

Estratégias de construção: datasets existentes (parte do KILT — NQ, HotpotQA, FEVER —, SuperGLUE — MultiRC, ReCoRD), WikiEval (RAGAs), e conjuntos **gerados por LLM a partir de notícias** (RGB, MultiHop-RAG, CRUD-RAG, CDQA) ou de páginas de admissão universitária (DomainRAG). **[Artigo]** O survey nota que datasets estáticos "não resolvem" cenários dinâmicos e que LLMs permitem gerar pares pergunta/gabarito em resolução diária para evitar "cola" (Seções 3.2 e 4).

### 2.3 Metrics (Seção 3.3) — definições literais do artigo

**Não baseadas em rank:**
- **Accuracy** — *(proporção de resultados verdadeiros — positivos e negativos — entre todos os casos)*.
- **Precision = TP / (TP + FP)** — *(dos itens recuperados, que fração é relevante)*.
- **Recall@k = |RD ∩ Top_k_d| / |RD|** — *(dos documentos relevantes RD, que fração apareceu no top-k)*.

**Baseadas em rank:**
- **MRR = (1/|Q|) · Σᵢ 1/rankᵢ**, onde rankᵢ é a posição do **primeiro** documento relevante da consulta i — *(quão no topo está o primeiro correto, média sobre consultas)*.
- **MAP = (1/|Q|) · Σ_q [ Σ_k P(k)·rel(k) ] / |relevantes_q|** — *(média da precisão acumulada nas posições em que há relevante; premia relevantes cedo e muitos)*.

**Geração:** **ROUGE** *(sobreposição de n-gramas/subsequências com a referência, orientada a recall)*; **BLEU** *(precisão de n-gramas com penalidade de brevidade; o artigo reconhece que não capta fluência/gramática)*; **BERTScore** *(similaridade semântica por embeddings contextuais, mais robusto a paráfrase)*; **LLM as a Judge** *(um LLM dá nota por coerência, relevância, fluência etc.; pode ser zero/few-shot ou ajustado em julgamentos humanos; escalas detalhadas de 1 a 5 padronizam a avaliação)*.

**Requisitos adicionais:** *Single Query Latency* (tempo médio por consulta, recuperação + geração); *Cosine similarity/distance* para diversidade (menor similaridade ⇒ mais diverso); *Rejection Rate* ("taxa em que o sistema se abstém de gerar resposta") para negative rejection; *Error Detection Rate* ("razão de afirmações contrafactuais detectadas") para robustez contrafactual; *Misleading Rate* e *Mistake Reappearance Rate* (do benchmark RECALL) para ruído.

### 2.4 Tabela 1 (resumo do que li)

Classifica 3 *tools* (TruEra RAG Triad, LangChain Bench., Databricks Eval), 11 *benchmarks* (RAGAs, RECALL, ARES, RGB, MultiHop-RAG, CRUD-RAG, MedRAG, FeB4RAG, CDQA, DomainRAG, ReEval) e 2 trabalhos de pesquisa (FiD-Light — latência; Diversity Reranker — diversidade). Exemplos de como cada um mede: RAGAs = Context Relevance via "LLM as a Judge", Answer Relevance via "LLM Gen + CosSim", Faithfulness via LLM-judge; ARES = os três via "LLM + Classifier"; MultiHop-RAG = Retrieval Quality via "MAP, MRR, Hit@K" e Response Correctness via LLM-judge. **[Minha leitura]** Contei 16 linhas na Tabela 1; a introdução diz "12 distinct evaluation frameworks" (Seção 1) — discrepância menor de contagem (talvez 12 = benchmarks + datasets), sem impacto no conteúdo.

## 3. Principais contribuições

1. Taxonomia de **desafios** por componente (retrieval, generation, sistema) — Seção 2.
2. Framework **Auepora** (Target × Dataset × Metric, via pares EO↔GT) — Seção 3.
3. Mapeamento de **16 trabalhos** em alvos e métricas (Tabela 1) e de datasets (Tabela 2).
4. Lista de **requisitos adicionais** (latência, diversidade, robustez a ruído/contrafactual, rejeição negativa) com métricas associadas — Seções 3.1 e 3.3.
5. Agenda de lacunas (Seção 4): benchmarks RAG-específicos, métricas além do QA tradicional, relação entre métricas de retrieval e qualidade final.

## 4. Resultados-chave (com localização)

Survey sem experimentos; os "resultados" são achados descritivos:

- **Tabela 1:** a maioria dos frameworks mede relevância de contexto e fidelidade/correção da resposta; poucos cobrem robustez e rejeição (RGB) ou diversidade/latência (trabalhos de pesquisa).
- **Seção 3.1 (texto):** *tools* só especificam alvos (flexíveis); *benchmarks* trazem dados + alvos; RAGAs e ARES avaliam relevância do contexto recuperado, RGB e MultiHop-RAG priorizam acurácia contra GT.
- **Seção 3.3:** métricas de ranking tradicionais (MAP@K, MRR@K, F1 por tokens) aparecem em MultiHop-RAG e CDQA; o survey cita ([17]) que essa metodologia baseada em ranking "não é inadequada" para RAG mas deveria haver métricas de retrieval mais específicas.
- **Seção 4:** em QA com LLMs fortes é difícil distinguir o efeito do retrieval; LLM-judge é tendência, com os problemas listados abaixo.

## 5. Limitações

**[Artigo] — as do autor (Seção 4):**
- Datasets estáticos não cobrem cenários dinâmicos; dificuldade de um dataset universal (alvos específicos por benchmark).
- LLM-as-a-Judge: dificuldade de alinhar com humanos, de definir escala; *"there's no universally applicable grading scale and prompting text"*.
- Custo/recursos de usar LLM para gerar e validar dados.
- Falta pesquisa sobre *"the relationship and analysis between retrieval metrics and final generation outputs"*.

**[Minha leitura] — as que identifico:**
- Sem números: nenhum resultado comparativo de métricas, concordância com humanos ou custo; não dá para "calibrar escolhas" com este artigo.
- **Não discute viés de pooling / julgamentos incompletos** (0 ocorrências de "pooling") nem como tratar documento recuperado mas não julgado. Precision é definida só como TP/(TP+FP), o que implicitamente trata não-julgado como falso positivo.
- **Não discute vieses do LLM-judge** (autopreferência, posição, verbosidade) nem calibração (κ, IC); só diz que alinhar é difícil.
- **nDCG não aparece** em nenhuma ocorrência do texto; "Hit@K" aparece uma única vez (Tabela 1, MultiHop-RAG) sem definição formal. A cobertura de métricas de ranking é mais fraca do que o catálogo padrão de IR.
- *Rejection Rate* é unilateral (só mede o quanto se absteve), sem a face de **over-refusal**.
- Cobertura até jun/2024: ficam de fora trabalhos posteriores (ex.: RAGChecker é de ago/2024 e não está aqui).

## 6. Reflexões ancoradas no NOSSO projeto

**R1 — Fórmulas de retrieval (P1–P3, L-3; `evaluators.py: recall_at_k, mrr`).** *APOIA.* O survey define Recall@k = |RD ∩ Top_k|/|RD| e MRR com o **primeiro** relevante — idênticas ao que o P1–P3 consolidou. *NEUTRO* quanto a D-1 (nível chunk × paper): o survey fala em "documentos" sem tratar granularidade. *NEUTRO* sobre L-3 (`ranx` como oráculo): o survey não cita `ranx`; a decisão continua sendo de engenharia, dependendo da ficha do Bassani.

**R2 — Taxonomia: o que o nosso catálogo não cobre (ckpt-0.6-plan §3).** Mapeando Targets do survey ↔ nossos avaliadores: Retrieval Relevance/Accuracy ↔ hit/recall/MRR/precision ✔; Generation Relevance ↔ `answer_relevance` ✔; Faithfulness ↔ `faithfulness_judge`, `citation_accuracy` ✔; Correctness ↔ `f1_summary_evaluator` (fraco, P9) e `completeness_judge` parcial; Negative Rejection ↔ `abstention_quality` ✔ (e é mais rico que o do survey, ver R4); Diversity ↔ `distinct_papers@k` (planejada no P6) ✔. **Não cobertos:** (a) **latência e custo** (o survey trata como alvo; para o CKPT-4/rerank e CKPT-1/k isso é decisão de produto — recomendo registrar tempo/tokens por exemplo, barato); (b) **noise robustness** (ruído irrelevante no contexto) e **counterfactual robustness** — a fatia `stale` só cobre parcialmente (resposta desatualizada, não informação *incorreta injetada*); (c) **MAP e nDCG** — nDCG já está no roadmap (glossário), mas o survey **não** fornece definição, então não é "exigência da literatura de survey"; (d) readability/toxicity — fora de escopo para o capstone. *APOIA* a cobertura geral; *DESAFIA* levemente a ausência de latência/custo.

**R3 — Precision e pooling (P6, D-1; `precision_at_k`).** *NEUTRO/aponta lacuna.* O survey define Precision = TP/(TP+FP) sem discutir julgamento incompleto — exatamente o ponto em que o nosso P6 já foi além (retornar `None` sem gold completo). Consequência prática: **não dá para citar o survey como respaldo** da decisão de P6; o respaldo vem de Buckley & Voorhees/BEIR. O survey também dá razão à preocupação geral (Seção 2: Recall/Precision "não capturam" nuances do RAG), o que **apoia** complementar com métricas de conteúdo (ver ficha do RAGChecker).

**R4 — Abstenção (P7, P8; `f1_summary_evaluator`, `abstention_quality`).** *APOIA a necessidade de medir; DESAFIA reportar só taxa.* O survey inclui *Negative Rejection* como requisito (Seção 3.1) e mede com **Rejection Rate** — fração em que o sistema se abstém. Isso é, na prática, o nosso `abstention recall` e ignora over-refusal; o nosso plano P8 (P/R/F1 + over/under-refusal por slice) vai **além** do survey. Não adotaríamos a métrica do survey isolada: um sistema que sempre se abstém teria Rejection Rate máximo.

**R5 — LLM-judge e calibração (P11, P9; `faithfulness_judge` etc.).** *APOIA* medir/calibrar, sem fornecer números. A Seção 4 diz que não há escala/prompt universal e que alinhar com humanos é difícil → sustenta o nosso passo "κ juiz×humano por juiz antes de congelar o baseline" (ckpt-0.6-plan §6). **Não** sustenta o argumento específico de autopreferência (juiz = gerador): o survey não menciona. Para P11 precisamos de Zheng et al./Wataoka (outras fichas).

**R6 — Format (P10; `format_validator`).** *NEUTRO/apoio fraco.* DomainRAG introduz "Structural Output" como alvo (Seção 3.1/Tabela 1), mostrando que aderência a formato é considerada alvo de RAG; não discute validação programática.

**R7 — Relação retrieval ↔ geração (D-2, P5; relatório EXP-0).** *APOIA* uma análise que o nosso plano ainda não tem: a Seção 4 pede estudar "a relação entre métricas de retrieval e saídas finais". Isso sugere reportar, no EXP-0, a qualidade da resposta **condicionada** a hit@k (acertou/errou o retrieval) — informação nova e barata (ver ficha do RAGChecker: *context utilization*).

**O que adotaríamos:** (1) o vocabulário Auepora (Target × EO↔GT) como índice do relatório/ADR do catálogo; (2) registrar latência/tokens por exemplo; (3) a lista de lacunas (noise/counterfactual robustness) como backlog consciente, não como escopo do CKPT-0.6. **O que NÃO adotaríamos:** Rejection Rate unilateral (R4); BLEU/ROUGE/BERTScore como métricas de qualidade (o próprio survey reconhece limitações; a ficha do RAGChecker mostra correlações com humanos ≤ 43,10 Pearson para BLEU/ROUGE/BERTScore); citar o survey como evidência para decisões de pooling ou de viés de juiz — ele é silencioso nesses pontos.

## 7. Citações úteis

1. *"The traditional evaluation indicators for retrieval, such as Recall and Precision, cannot fully capture the nuances of RAG retrieval systems"* — Seção 2, "Retrieval".
2. *"there's no universally applicable grading scale and prompting text, complicating the standardization of 'LLM as a Judge'"* — Seção 4 ("Discussion"), parágrafo sobre métricas.
3. *"research is needed on performance changes involving intermediate outputs and retrieved documents, as well as the relationship and analysis between retrieval metrics and final generation outputs"* — Seção 4 ("Discussion"), parágrafo de abertura.
