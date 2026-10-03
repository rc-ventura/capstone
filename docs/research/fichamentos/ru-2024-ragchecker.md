# Ru et al. 2024 — RAGChecker (diagnóstico de RAG em nível de claim)

- **Referência completa:** Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Jiayang Cheng, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, Zheng Zhang. *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation.* arXiv:2408.08067. <https://arxiv.org/abs/2408.08067> · Código: <https://github.com/amazon-science/RAGChecker>
  *(Lista de autores e venue — NeurIPS 2024, Datasets & Benchmarks — vêm do meu conhecimento prévio/roadmap e **não foram verificados** no texto lido; confirmar antes de citar numa banca.)*
- **Confiança da leitura:** **integral.** Li o texto bruto da versão HTML do arXiv: Seções 1–5, Tabelas 1–3, e os Apêndices A–I (benchmark, fórmulas, meta-avaliação, setup, resultados por dataset — Tabelas 4–12 vistas até a Recreation —, diagnóstico, limitações). **Não vi:** figuras (Fig. 1 e 4–10 apenas pelas legendas), as Tabelas 13–15 (demais domínios) e a Tabela 16 (validação do RefChecker com Llama3). O PDF não pôde ser decodificado; usei o HTML.

> Convenção: **[Artigo]** = afirmado no texto/tabelas; **[Minha leitura]** = interpretação minha.

---

## 1. Problema que o artigo ataca

**[Artigo]** Três desafios (Seção 1): **(1) complexidade modular** — RAG tem retriever e gerador, e é preciso entender a origem dos erros; **(2) limitação das métricas** — recall@k e MRR "dependem de chunks anotados e de uma abordagem rígida de chunking, perdendo o escopo semântico completo da base"; BLEU/ROUGE/BERTScore e LLM-judges "funcionam bem com respostas concisas mas falham em distinções finas em respostas longas"; **(3) confiabilidade das métricas** pouco explorada (alinhamento com humanos).

Resposta: avaliação **em nível de claim** (afirmação atômica) com extração e checagem de entailment, gerando métricas **gerais**, do **retriever** e do **gerador**.

## 2. Método (passo a passo, fórmulas exatas)

### 2.1 Formulação e entradas (Seção 3, 3.1)

RAG = {R, G}. Dada uma consulta q e documentos D: recupera top-k chunks {chunkⱼ} = R(q, D, k) e gera a resposta m = G({chunkⱼ}, q). Cada amostra do benchmark é ⟨q, D, gt⟩: consulta, documentos (segmentados em chunks de mesmo tamanho em tokens) e **resposta-gabarito completa e correta, em formato longo**.

**Duas personas** (Seção 3, "Design Principle"): o *usuário* que quer um número único para comparar sistemas; o *desenvolvedor* que quer achar a **causa** de erros — erro de retrieval (contexto incompleto/irrelevante) ou erro de gerador (não identifica/usa a informação relevante).

### 2.2 Avaliação por claims (Seção 3.2)

Dois componentes: **(1) extrator texto→claims** (decompõe um texto T em claims {cᵢ}); **(2) checador de entailment** (decide se um claim c está *entailed* (∈) ou *não* (∉) em um texto de referência *Ref*). Implementação: **Llama3-70B** como extrator e checador, via **RefChecker** (Seção 4.1; validado no benchmark do RefChecker — Apêndice G; valores da Tabela 16 não lidos).

### 2.3 Métricas — definições (Seção 3.3 e fórmulas do Apêndice B)

Notação: m = resposta do modelo; gt = resposta-gabarito; {chunkⱼ} = chunks recuperados. Um chunk é **relevante (r-chunk)** se **contém ao menos um claim do gt** (∃i: cᵢ(gt) ∈ chunkⱼ); os demais são **irrelevantes (irr-chunk)**.

**Métricas gerais (visão do usuário; 3.3.1):**
- **Precision** = |{cᵢ(m) | cᵢ(m) ∈ gt}| / |{cᵢ(m)}| — *(dos claims que a resposta fez, que fração é confirmada pelo gabarito)*.
- **Recall** = |{cᵢ(gt) | cᵢ(gt) ∈ m}| / |{cᵢ(gt)}| — *(dos claims do gabarito, que fração a resposta cobriu)*.
- **F1** = média harmônica dos dois — *(nota geral única)*.

**Métricas do retriever (3.3.2):**
- **Claim Recall** = |{cᵢ(gt) | cᵢ(gt) ∈ {chunkⱼ}}| / |{cᵢ(gt)}| — *(dos claims necessários para a resposta-gabarito, que fração está **presente** nos chunks recuperados; mede se o retriever trouxe a informação necessária, **independente de ID ou posição**)*.
- **Context Precision** = |{r-chunkⱼ}| / k — *(dos k chunks recuperados, quantos têm ao menos um claim útil; definida em nível de **chunk**, não de claim, por interpretabilidade — o artigo nota que um chunk pode misturar informação útil e enganosa, então o teto de precision em nível de claim é < 100%)*.

**Métricas do gerador (3.3.3) — todas sobre os claims da resposta, exceto Context Utilization:**
- **Faithfulness** = |{cᵢ(m) | cᵢ(m) ∈ {chunkⱼ}}| / |{cᵢ(m)}| — *(fração dos claims da resposta suportada pelos chunks; fidelidade ao contexto)*.
- **Relevant Noise Sensitivity** = |{cᵢ(m) | cᵢ(m) ∉ gt **e** cᵢ(m) ∈ {r-chunkⱼ}}| / |{cᵢ(m)}| — *(fração de claims **errados** que vieram de um chunk **relevante**: o gerador se deixou enganar por ruído misturado com informação útil)*.
- **Irrelevant Noise Sensitivity** = idem com ∈ {irr-chunkⱼ} — *(claims errados vindos de chunk irrelevante)*.
- **Hallucination** = |{cᵢ(m) | cᵢ(m) ∉ gt **e** cᵢ(m) ∉ {chunkⱼ}}| / |{cᵢ(m)}| — *(claims errados que **não** estão em nenhum chunk: invenção do gerador)*.
- **Self-knowledge** = |{cᵢ(m) | cᵢ(m) ∈ gt **e** cᵢ(m) ∉ {chunkⱼ}}| / |{cᵢ(m)}| — *(claims **corretos** que não vieram de nenhum chunk: o gerador acertou usando o próprio conhecimento; menor é melhor quando se espera dependência só do contexto)*.
- **Context Utilization** = |{cᵢ(gt) | cᵢ(gt) ∈ {chunkⱼ} **e** cᵢ(gt) ∈ m}| / |{cᵢ(gt) | cᵢ(gt) ∈ {chunkⱼ}}| — *(dos claims do gabarito que o retriever **conseguiu trazer**, quantos a resposta realmente incorporou: "achou e usou")*.

### 2.4 Benchmark e sistemas (Seção 4.1; Apêndices A e D)

- **Benchmark:** **4.162 consultas em 10 domínios** (Tabela 1): ClapNQ 300, NovelQA 280, RobustQA Writing/BioASQ(511)/Finance/Lifestyle/Recreation/Science/Technology (500 cada, BioASQ 511) e KIWI 71 (papers de NLP, 429 documentos). Respostas curtas dos datasets foram convertidas em **respostas longas** (GPT-4 + controle de qualidade com RefChecker) — Apêndice A.
- **8 sistemas** = {BM25, E5-Mistral} × {GPT-4, Mixtral-8x7B, Llama3-8B, Llama3-70B}. Chunks de **300 tokens**, overlap **0,2**, **k = 20**, temperatura 0 (Apêndice D).

### 2.5 Meta-avaliação com humanos (Seção 4.2; Apêndice C)

- **280 instâncias** = 28 pares de sistemas × 10 domínios; cada instância é um **par de respostas** de dois sistemas à mesma pergunta.
- Dois anotadores por instância, que escolhem entre 5 opções (significativamente melhor … significativamente pior) em **correctness, completeness e overall**; 10 anotadores no total (7 internos, 3 estudantes de pós; US$ 255 no total — Apêndice C). Os anotadores veem críticas geradas pelo GPT-4 comparando cada resposta com o gabarito.
- **Concordância humana:** proporção com |hᵢ − hᵢ′| ≤ 1 = **90,95%**.
- Métrica: correlação (Pearson/Spearman) entre a diferença humana de preferência e a diferença normalizada do score da métrica.
- Baselines: 10 métricas de TruLens, RAGAS, ARES e CRUD-RAG (+ BLEU, ROUGE-L, BERTScore); **backbone Llama3-70B-Instruct** "when applicable" para todos (embeddings: o backbone padrão de cada um).

## 3. Principais contribuições

1. Framework **claim-level** com métricas separadas por componente (retriever × gerador) + métrica geral — Seção 3.
2. Métricas novas para o gerador: **context utilization, relevant/irrelevant noise sensitivity, self-knowledge** (além de faithfulness e hallucination) — Seção 3.3.3.
3. **Meta-avaliação humana** (280 instâncias) mostrando correlação superior às demais métricas — Seção 4.2/Tabela 2.
4. **Benchmark** de 4.162 consultas, 10 domínios, respostas longas — Tabela 1/Apêndice A.
5. Achados diagnósticos (trade-offs retriever↔gerador; efeito de k, chunk, overlap, prompt) — Seções 4.3–4.4/Apêndice F.

## 4. Resultados-chave (com localização)

**Tabela 2 (Seção 4.2) / Tabela 5 (Apêndice C)** — correlação com preferência humana (Pearson; Correctness / Completeness / Overall):

| Métrica | Correctness | Completeness | Overall |
|---|---|---|---|
| **RAGChecker** (mesma métrica que o humano) | **49.66** | **60.67** | **61.93** |
| RAGAS Answer Similarity | 41.07 | 53.16 | 48.31 |
| ROUGE-L | 31.75 | 47.88 | 43.10 |
| CRUD-RAG Recall | 30.93 | 45.11 | 41.25 |
| BLEU-avg | 38.89 | 32.13 | 35.14 |
| TruLens Answer Relevance | 35.01 | 37.24 | 35.15 |
| BERTScore | 30.34 | 37.93 | 33.51 |
| TruLens Groundedness (Tab. 5) | 21.11 | 14.01 | 19.45 |
| ARES Answer Relevance | 18.63 | 20.13 | 17.81 |
| ARES Answer Faithfulness (Tab. 5) | 9.46 | 10.25 | 8.80 |
| RAGAS Faithfulness (Tab. 5) | 8.22 | 4.90 | 7.83 |
| RAGAS Answer Relevance (Tab. 5) | 11.59 | 9.39 | 10.27 |
| *Humano × humano (teto)* | *63.67* | *71.91* | *70.09* |

Spearman (overall): RAGChecker 60.90; RAGAS Answer Similarity 57.23; humano 68.89 (Tabela 5).

**Tabela 3 (Seção 4.3)** — médias nos 10 domínios (Prec / Rec / F1 | CR / CP | CU / NS(I) / NS(II) / Hallu / SK / Faith | #claims):

| Sistema | Prec | Rec | F1 | CR | CP | CU | NS(I) | NS(II) | Hallu | SK | Faith | #claims |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25_GPT-4 | 61.0 | 49.7 | 50.3 | 74.0 | 52.3 | 61.4 | 26.2 | 4.1 | 8.7 | 3.4 | 87.9 | 12 |
| E5-Mistral_GPT-4 | 62.0 | 53.0 | **52.7** | 83.5 | 61.8 | 60.4 | 28.9 | 3.5 | 5.7 | 1.4 | 92.9 | 12 |
| E5-Mistral_Llama3-8b | 53.8 | 48.3 | 45.0 | 83.5 | 61.8 | 55.0 | 33.5 | 5.5 | 6.6 | 0.8 | 92.7 | 11 |
| E5-Mistral_Llama3-70b | 60.6 | 50.4 | 50.2 | 83.5 | 61.8 | 57.6 | 31.7 | 4.3 | 3.3 | 0.8 | 95.9 | 10 |
| BM25_Llama3-70b | 59.1 | 44.9 | 46.3 | 74.0 | 52.3 | 56.2 | 30.4 | 5.3 | 5.1 | 1.7 | 93.2 | 9 |

(Demais 3 sistemas na Tabela 3 original; omitidas aqui.) Insights textuais (Seção 4.3): (i) o retriever importa de forma consistente; (ii) Llama3-70B > Llama3-8B em todas as métricas do gerador; (iii) **context utilization correlaciona fortemente com o F1 geral**; (iv) contexto mais informativo melhora faithfulness e reduz hallucination/self-knowledge; (v) **claim recall do retriever tem trade-off com noise sensitivity do gerador** (mais recall → mais ruído acompanhando a informação útil); (vi) **ruído relevante** pesa mais que irrelevante — "geradores demonstram faithfulness em nível de chunk"; (vii) modelos open-source "são fiéis mas tendem a confiar cegamente no contexto".

**Seção 4.4 / Apêndice F** — ajustes de configuração (3 sistemas × 3 domínios: Writing, Finance, KIWI):
- **k de 5→20:** claim recall 61,5→77,6; faithfulness 88,1→92,2; noise sensitivity 34,0→35,4; F1 51,7→53,4. **Chunk size 150→300:** claim recall 70,3→77,6; F1 52,6→53,4.
- **Overlap:** context precision 69,3→71,1, mas claim recall 77,8→78,1 — mais chunks com a *mesma* informação; overlap "não importa muito".
- **Prompt com requisitos explícitos:** faithfulness 92,2→93,6; context utilization 59,2→63,7, **mas** noise sensitivity 35,4→38,1 — "trilema" entre context utilization, noise sensitivity e faithfulness.

## 5. Limitações

**[Artigo] (Apêndice H):** (1) as métricas do **retriever** são "menos informativas" que as do gerador (focam em recall de claims e precision do contexto; não captam densidade, diversidade, coerência); (2) **não distinguem Neutral de Contradiction** nos resultados do checker; (3) benchmark **somente texto e inglês**, reaproveitado de datasets existentes.

**[Minha leitura]**
- **Validação só das métricas gerais:** a meta-avaliação (Seção 4.2) compara correlação com preferência de *correção/completude/overall*; as métricas **diagnósticas** (claim recall, context precision, context utilization, noise sensitivity, hallucination, self-knowledge) **não** foram validadas contra humanos. Fatos a confiar com cautela.
- **Comparação não "like-for-like":** todos os baselines usam Llama3-70B; o **ARES** aqui não é o juiz DeBERTa ajustado por domínio do artigo original, e métricas de *faithfulness* (RAGAS/ARES/TruLens) não foram feitas para prever "correção vs gabarito" (alvo humano) — as correlações baixíssimas (7,83; 8,80) refletem em parte essa **incompatibilidade de alvo**, não necessariamente falha da métrica.
- **Dependência do gabarito longo:** a base de todas as métricas é a lista de claims do `gt`; gabaritos curtos ou incompletos limitam claim recall/precision. O gabarito longo foi **gerado por GPT-4** a partir de respostas curtas (Apêndice A.2), com filtro do próprio RefChecker.
- **Extrator/checador = LLM:** erros de extração/entailment se propagam; a validação do RefChecker com Llama3 (Apêndice G, Tabela 16) não foi lida por mim.
- **Metas de custo:** o checador roda por claim × chunk (k = 20 por consulta) — caro em escala.
- **Abstenção ausente:** não há consultas sem resposta; com gt vazio, claim recall/recall/CU ficam indefinidos (0/0). *(Inferência minha.)*
- Humanos veem críticas geradas pelo GPT-4 (Apêndice C) — possível ancoragem do julgamento.

## 6. Reflexões ancoradas no NOSSO projeto

**R1 — Claim recall / context precision × `gold_chunk_ids` (D-1, P6, P1–P3; `evaluators.py: precision_at_k, recall_at_k`).** *DESAFIA o gold por ID e APOIA a semântica anti-pooling.* O RAGChecker rotula a **relevância dos chunks pelo conteúdo** (um chunk é relevante se *entails* ≥1 claim do gt), em vez de supor "não-gold = irrelevante". Isso ataca exatamente o problema do P6 — chunks **irmãos** do mesmo paper que também respondem — **sem** adjudicar manualmente todos os chunks, e é uma resposta automatizada à "gold de chunk ampliado" da D-1. O artigo também critica explicitamente recall@k/MRR por dependerem de "chunks anotados" (Seção 1). **Condição:** o checador (LLM) precisaria ser **validado numa amostra nossa** (p.ex., 30–50 pares chunk×claim rotulados por humano), porque o artigo **não** valida as métricas diagnósticas contra humanos. *Proposta, não decisão:* adotar como **diagnóstico paralelo** às métricas por ID no EXP-0, medindo a discordância (nº de chunks "irmãos" que o checador marca como relevantes e o gold por ID marca como erro).

**R2 — Fragilidade dos `gold_chunk_ids` a mudanças de chunking (CKPT-3; `utils.chunk_id`).** *DESAFIA nosso desenho por ID de chunk.* O plano prevê gates sobre "chunk novo" (CKPT-3). Em `utils.py`, `chunk_id(doc) = f"{arxiv_id}:{sha1(page_content)[:12]}"` — é um **hash do conteúdo**; qualquer re-chunking muda o ID e invalida `gold_chunk_ids`, enquanto o prefixo `arxiv_id` sobrevive. O RAGChecker é robusto a isso (relevância por claim) e a Seção 4.4 testa justamente tamanho/overlap de chunk. **Implicação para D-1:** para os experimentos de chunking, usar **nível paper** (ou claim-level) como medida comparável entre configurações; manter chunk-ID só para o baseline. Isso favorece a opção "dois níveis" **com a comparação entre configurações de chunking feita no nível paper/claim**.

**R3 — Precision inflada por redundância (P1–P3 dedup, P6 `distinct_papers@k`; `decisions.md`: mesmo paper 3× no top-5).** *APOIA.* No Apêndice F, aumentar overlap sobe context precision (69,3→71,1) **sem** subir claim recall (77,8→78,1), porque se recuperam "mais chunks com o mesmo segmento útil". É o análogo do nosso colapso observado (mesmo paper repetido). Reforça: (i) deduplicar por paper antes de medir; (ii) tratar `precision_at_k` isolada como enganosa; (iii) reportar `distinct_papers@k` junto.

**R4 — *Context utilization*: separar "não achou" de "achou e não usou" (D-2, P5; relatório EXP-0, matriz do plano §3).** *APOIA e acrescenta um diagnóstico que nos falta.* O artigo mostra (Seção 4.3) que context utilization correlaciona fortemente com o F1 geral. **Versão barata, sem extração de claims:** reportar a qualidade da resposta (correctness/completeness) **condicionada** a hit@k = 1 versus hit@k = 0. Isso responde à pergunta do survey (relação retrieval ↔ geração) e dá sentido ao par hit/recall quando |gold| = 1 (D-2): em vez de reportar hit e recall como duas "evidências", reportar hit **e** a taxa de acerto da resposta dado hit. **Versão completa** (claims) só se o custo permitir (ver R7).

**R5 — Hallucination × Self-knowledge refinam `faithfulness_judge` e a leitura de abstenção (P7, P8, `unanswerable`/`stale`).** *APOIA/refina.* O `faithfulness_judge` do plano trata "não suportado pelo contexto" como um bloco. O RAGChecker separa **claim não suportado e errado (hallucination)** de **claim não suportado mas correto (self-knowledge)**. Para `unanswerable`, uma resposta não-abstentora com claims não suportados é *under-refusal*; saber se são falsos (hallucination) ou verdadeiros (self-knowledge) distingue "inventou" de "respondeu do conhecimento paramétrico" — informação útil para o gate release-blocking de abstenção. Limitação: precisa de gt para decidir "correto", e `unanswerable` tem gt vazio (R6).

**R6 — Abstenção fica fora do RAGChecker (P7, P8).** *NEUTRO.* Sem consultas sem resposta, as métricas não foram definidas para gt vazio; **não** substituem nosso P/R/F1 de abstenção nem o `abstention_quality`. Usar o RAGChecker apenas nos slices com gt (answerable, multi-doc, persona).

**R7 — Custo e P11 (juiz ≠ gerador; `config.py`).** *DESAFIA parcialmente.* Cada claim é checado contra cada chunk (k = 20 no artigo; o nosso k é menor, ~5, com ~100 exemplos: viável). Mas o RAGChecker usa **Llama3-70B** para extrair/checar, diferente dos geradores; no nosso plano o juiz seria o **mesmo** gpt-4o-mini do gerador (P11). Se adotarmos claim-level, vale usar um modelo **distinto** do gerador para o checador; o artigo (Apêndice G) valida o Llama3-70B no benchmark do RefChecker — não gpt-4o-mini.

**R8 — P9 (token-F1 × julgamento semântico; `f1_summary_evaluator`).** *APOIA a troca, com ressalva.* Na Tabela 2/5, as métricas lexicais/embedding vs gabarito correlacionam fraco com humanos: BLEU 35,14, ROUGE-L 43,10, BERTScore 33,51 (Overall Pearson), contra RAGChecker 61,93 e teto humano 70,09. Isso é evidência (em **respostas longas**, 10 domínios) a favor de substituir token-F1 por um juízo semântico/claim-level. **Ressalvas:** (a) o gabarito do RAGChecker é longo e composto de vários claims; os nossos gold answers são curtos (1–3 frases, prompt limita a 3 frases), onde a diferença entre token-F1 e claim-F1 é menor e o efeito é incerto; (b) falta calibrar com **as nossas** labels (P9 passo 4). O **Overall F1 de claims** é, em essência, a versão semântica do que o nosso token-F1 tenta ser — candidata natural a substituí-lo, mas só após validação local.

**R9 — Trilema CU/NS/Faithfulness e o CKPT-6 (prompt; P8 over/under-refusal).** *APOIA por analogia.* O artigo mostra que um prompt que pede mais fidelidade e uso do contexto melhora CU (59,2→63,7) mas piora noise sensitivity (35,4→38,1) — "difícil melhorar todos ao mesmo tempo". Isso reforça nossa regra do P8 (registrar over-refusal ao melhorar recall de abstenção) como instância do mesmo princípio: **toda mudança de prompt deve reportar as métricas opostas junto**. Não é evidência sobre abstenção em si.

**R10 — Escolha de k (CKPT-1).** *APOIA o desenho do experimento.* k 5→20 melhora claim recall e faithfulness, com custo de mais noise sensitivity e saturação (Seção 4.4). O nosso gate de k deve olhar também para a **resposta**, não só hit@k; o artigo usa dados com respostas longas e k = 20, então os valores **não se transferem** ao nosso caso (abstracts de ~420 caracteres, k ≈ 5).

**O que adotaríamos:** (1) context utilization em versão barata (resposta condicionada a hit@k) — R4; (2) hallucination × self-knowledge como refinamento do `faithfulness_judge` — R5; (3) **claim recall / context precision por entailment como diagnóstico paralelo** às métricas por ID, após validação numa amostra nossa — R1/R2; (4) nível paper/claim para comparar configurações de chunking — R2. **O que NÃO adotaríamos:** (1) o pipeline completo RAGChecker como métrica principal agora (custo, gabarito longo, diagnósticas não validadas, checador ≠ nossa escolha de modelo); (2) correlações de faithfulness do RAGAS/ARES da Tabela 2 como prova de que são ruins (alvo humano diferente); (3) transferir os valores numéricos de k/chunk do artigo (domínio e tamanho de chunk distintos).

## 7. Citações úteis

1. *"traditional metrics like recall@k and MRR for retrievers depend on annotated chunks and a rigid chunking approach, missing out on the full semantic scope of the knowledge base."* — Seção 1, desafio (2) "metric limitation".
2. *"A retrieved chunk is called relevant chunk (r-chunk), if any ground-truth claim is entailed in it."* — Seção 3.3.2 (Retriever Metrics).
3. *"the diagnostic metrics for the retriever component are less insightful compared to those for the generator."* — Apêndice H (Limitations).
