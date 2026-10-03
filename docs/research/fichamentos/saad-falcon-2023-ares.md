# Saad-Falcon et al. 2023 — ARES (juízes treinados + PPI)

- **Referência completa:** Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia. *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* arXiv:2311.09476 (versão lida: v2, 31/03/2024). <https://arxiv.org/abs/2311.09476> · Código/datasets no GitHub (link no artigo).
  *(O venue — NAACL 2024 — vem do meu conhecimento prévio e **não foi verificado** nesta leitura.)*
- **Confiança da leitura:** **integral.** Li o texto bruto da versão HTML do arXiv (v2): resumo, Seções 1–7, Tabelas 1–6 (com os números das células) e o Apêndice A.1–A.6 (configuração de fine-tuning e prompts). **Não vi:** as Figuras 1–3 (só as legendas), a Tabela 7 (exemplos de pares positivos/negativos; vi só parte) e a lista de referências. O PDF não pôde ser decodificado; usei o HTML.

> Convenção: **[Artigo]** = afirmado no texto/tabelas; **[Minha leitura]** = interpretação minha.

---

## 1. Problema que o artigo ataca

**[Artigo]** Avaliar RAG tradicionalmente exige anotação manual de perguntas, passagens a recuperar e respostas — cara e específica de domínio (Seção 1). Avaliação por LLM *out of the box* (ex.: RAGAS) usa prompts fixos escritos à mão, com "pouca adaptabilidade a vários contextos de avaliação e nenhuma garantia de qualidade" (Seção 1). ARES propõe (i) **juízes leves treinados por domínio** com dados sintéticos, e (ii) **intervalos de confiança estatísticos** via *prediction-powered inference* (PPI), usando um **pequeno conjunto de anotações humanas** (Resumo, Seção 1).

## 2. Método (passo a passo, definições exatas)

Entradas obrigatórias (Seção 1 e 3): (a) um **conjunto de passagens do domínio**, (b) um **conjunto de validação com preferência humana de ≈150 pontos anotados ou mais** (positivos e negativos para as três métricas), (c) **≥5 exemplos few-shot** de perguntas e respostas do domínio (usados para gerar dados sintéticos).

### 2.1 As três dimensões (Seção 3.2) — rótulos **binários** por tripla (pergunta, passagem, resposta)

- **Context relevance** — *"Is the passage returned relevant for answering the given query?"* (a passagem recuperada é relevante para responder a pergunta?)
- **Answer faithfulness** — *"Is the answer generated faithful to the retrieved passage, or does it contain hallucinated or extrapolated statements beyond the passage?"* (a resposta é fiel à passagem ou extrapola/alucina?)
- **Answer relevance** — *"Is the answer generated relevant given the query and retrieved passage?"* (a resposta é relevante dada a pergunta e a passagem?)

**[Minha leitura]** Diferente do RAGAS (score contínuo por extração/razão de sentenças), aqui cada dimensão é um **classificador binário**; a métrica do *sistema* é a média dos rótulos previstos sobre amostras de suas triplas (Seção 3.3: "By averaging the individual predicted labels…").

### 2.2 Dados sintéticos (Seção 3.1)

1. FLAN-T5 XXL gera, a partir de passagens do corpus e dos few-shots, uma pergunta e uma resposta por passagem (prompts em A.5/A.6).
2. **Filtro de qualidade:** descarta consultas que *não recuperam sua passagem original como primeiro resultado* com o retriever (técnica citada de trabalhos anteriores).
3. **Negativos** (mesmo número que positivos):
   - *Fracos:* para context relevance, passagens aleatórias do domínio; para faithfulness/relevance, respostas sintéticas de **outras** passagens.
   - *Fortes:* para context relevance, passagens **do mesmo documento** da passagem-gold (se o dataset não tem várias passagens por documento, amostra do top-10 BM25); para faithfulness/relevance, resposta **contraditória** gerada pelo FLAN-T5 com few-shot.

### 2.3 Juízes (Seção 3.2, 4.1, A.1)

Três modelos **DeBERTa-v3-Large** (304M), um por métrica, com cabeça de classificação binária (camada linear, dropout 0,1 sobre o [CLS]). Perda: entropia cruzada com Adam; lr **5e-6**, batch **32**, warmup/decay linear (A.1). Early stopping: parar após **3 épocas sem melhora na perda**, avaliada no conjunto de validação humano (Seção 3.2). Baseline de in-context learning: gpt-3.5-turbo-16k (versão 10/23), com 8 few-shots (A.2–A.4).

### 2.4 Ranking de sistemas com intervalos de confiança — PPI (Seção 3.3)

*(PPI = técnica estatística que combina um conjunto pequeno de rótulos humanos com previsões do juiz em um conjunto grande sem rótulo, para produzir um intervalo de confiança mais estreito do que só usar os rótulos humanos)*

1. Os juízes rotulam uma amostra das triplas de cada sistema; a média dá a nota "bruta".
2. Sobre o conjunto humano, o PPI aprende uma **função retificadora** (estima o erro do juiz) e a usa para construir um **conjunto de confiança** da métrica verdadeira do sistema (ex.: sua taxa de context relevance), com α padrão de **95%**.
3. O **ponto médio** do intervalo é usado para ranquear os sistemas.

### 2.5 Protocolo experimental (Seção 4)

- **Datasets (4.2):** KILT (NQ, HotpotQA, FEVER, WoW) e SuperGLUE (MultiRC, ReCoRD, versões *open-domain*). Faithfulness **não** foi avaliada nesses (sem alucinações anotadas por humanos) — só em AIS (Seção 5.2).
- **Sistemas "mock" (4.2):** a partir do conjunto de validação de cada dataset, criam **nove splits** com taxa de sucesso de 70% a 90% em passos de **2,5 p.p.** (70,0; 72,5; …; 90,0). Positivos = exemplos originais; negativos = passagens/respostas amostradas do mesmo documento ou de documento aleatório. Como a taxa real é conhecida, o ranking correto é conhecido.
- **Métrica (4.3): Kendall's τ** = (pares concordantes − discordantes) / total de pares, entre o ranking correto e o do ARES; "sucesso" se τ > 0,9. *(Detalhe: o texto diz que o τ "varia de 0,0 a 1,0" e define pares empatados como discordantes; o τ clássico varia de −1 a 1.)*
- **Baselines:** RAGAS v0.0.18; juiz GPT-3.5 few-shot; "sampled annotations" (150 labels por sistema mock).

## 3. Principais contribuições

1. Juízes **ajustados por domínio** por métrica, treinados só com dados sintéticos — Seção 3.1–3.2.
2. **PPI** aplicado à avaliação de RAG para dar **IC** às métricas — Seção 3.3.
3. Negativos **fortes** e **fracos** específicos por métrica — Seção 3.1.
4. Evidência em 8 tarefas (6 em mocks, AIS, sistemas reais) de ranking melhor que RAGAS e juiz GPT-3.5 — Seção 5.
5. Estudos de **quantidade de anotações** (Tabela 3), **GPT-4 no lugar de humanos** (Tabela 4) e **generalização entre domínios** (Tabela 6).

## 4. Resultados-chave (com localização)

**Tabela 1 (Seção 5.1)** — sistemas mock; PPI com **300** anotações humanas. Acurácia do juiz por dataset (CR = context relevance; AR = answer relevance):

| | NQ CR | NQ AR | HotpotQA CR | HotpotQA AR | WoW CR | WoW AR | FEVER CR | FEVER AR | MultiRC CR | MultiRC AR | ReCoRD CR | ReCoRD AR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RAGAS | 31.4 | 71.2 | 17.2 | 76.0 | 36.4 | 77.8 | 23.7 | 69.2 | 16.1 | 75.0 | 15.0 | 72.8 |
| Juiz GPT-3.5 | 73.8 | 95.5 | 75.3 | 71.6 | 84.3 | 85.2 | 60.4 | 59.6 | 72.4 | 60.3 | 81.0 | 65.8 |
| **ARES** | **79.3** | **97.2** | **92.3** | **81.3** | **85.7** | **96.1** | **88.4** | **78.5** | **85.8** | **82.7** | **67.8** | **92.3** |

- Nem toda célula favorece o ARES: em **ReCoRD CR** o juiz GPT-3.5 (81,0%) supera o ARES (67,8%); o ganho sobre os baselines é na média, não uniforme. *(Observação minha a partir das células.)*
- Kendall's τ: ARES é, em média, **+0,065** (CR) e **+0,132** (AR) acima do RAGAS; +0,06 acima do juiz GPT-3.5 (Seção 5.1). "Sampled annotations" (150 por sistema × 9 = **1.350** labels) fica **0,08 de τ abaixo** do ARES, que usa **78% menos anotações** (Seção 5.1).
- **Inconsistência interna:** a introdução diz que o ARES supera o RAGAS em **59,3** p.p. (CR) e 14,4 p.p. (AR) na acurácia; a Seção 5.1 diz **59,9** (CR) e 14,4 (AR). **[Minha leitura]** Calculei a média das seis colunas de CR da Tabela 1: (79,3+92,3+85,7+88,4+85,8+67,8)/6 − (31,4+17,2+36,4+23,7+16,1+15,0)/6 = **59,9**; AR: **14,35 ≈ 14,4**. O número da tabela é 59,9; 59,3 é o da introdução (provável typo). **O roadmap do projeto (`ckpt-0.6.1b…` §B, linha "Calibração humana") cita 59,3 e deveria usar 59,9** (ou citar a introdução explicitamente).

**Tabela 3 (apêndice) — nº de anotações humanas para o PPI × τ** (NQ / MultiRC / ReCoRD; CR e AR):

| Labels | NQ CR | NQ AR | MultiRC CR | MultiRC AR | ReCoRD CR | ReCoRD AR |
|---|---|---|---|---|---|---|
| 400 | 1.0 | 1.0 | 0.89 | 0.94 | 0.89 | 0.94 |
| 300 | 0.89 | 1.0 | 0.94 | 0.89 | 0.83 | 0.89 |
| 200 | 0.83 | 1.0 | 0.83 | 0.94 | 0.83 | 0.83 |
| 150 | 0.72 | 1.0 | 0.83 | 0.89 | 0.72 | 0.83 |
| 100 | 0.44 | 1.0 | 0.67 | 0.67 | 0.67 | 0.83 |
| 50 | 0.44 | 0.94 | 0.61 | 0.44 | 0.56 | 0.67 |
| 25 | 0.44 | 0.89 | 0.56 | 0.44 | 0.44 | 0.56 |

Legenda da Tabela 3: *"below about 100-150 datapoints in the human preference validation set, ARES cannot meaningfully distinguish between the alternate RAG systems"*. **[Minha leitura]** Mesmo com 150 labels o τ de **context relevance** é 0,72–0,83, abaixo do limiar de "sucesso" (τ > 0,9, Seção 4.3) nos três datasets; só em *answer relevance* o ARES atinge ~0,83–1,0 com 150. Os "≈150" são, portanto, um **piso** para answer relevance, não uma garantia para context relevance.

**Tabela 4 — GPT-4 no lugar dos humanos** (500 labels GPT-4 como set de validação): colunas NQ / ReCoRD / MultiRC, cada uma com CR e AR. τ com rótulos GPT-4: 0,78 / 1,0 · 0,78 / 0,72 · 0,89 / 0,78; τ com rótulos humanos: 0,94 / 1,0 · 0,83 / 0,89 · 0,94 / 0,89. A legenda diz que o τ cai "0,05 a 0,30 na maioria dos casos"; pelas células que li as quedas vão de 0,00 a 0,17 (maiores: ReCoRD AR, −0,17; NQ CR, −0,16). Conclusão do artigo (Seção 5.1): GPT-4 é menos bom que humanos nesse papel, mas a ideia "arguably has promise".

**Tabela 2 (Seção 5.2)** — AIS (faithfulness real), 200 labels humanos: WoW — juiz **62,5%** de acurácia (previsão 0,478 vs correto 0,458); CNN/DM — **84,0%** (0,835 vs 0,859). Texto: "dentro de 2,5 pontos" do valor correto.

**Tabela 5 (Seção 5.3)** — sistemas reais (3 retrievers: BM25, Ada, ColBERTv2 × 3 geradores: MPT-7b-Instruct, GPT-3.5, GPT-4, mais RAG da Facebook), **1 passagem recuperada por sistema**; PPI com 300 labels: τ médio ARES **0,91** (CR) e **0,97** (AR); +0,16/+0,15 sobre RAGAS. IC do PPI cobriu a média verdadeira em **>95%** dos casos; largura média **7,4** p.p. (CR) e **6,1** p.p. (AR). Melhor retriever: ColBERTv2; melhor gerador: GPT-4.

**Tabela 6 (Seção 5.4)** — generalização entre domínios: bom ao trocar tipo de query/doc (ex.: NQ→FEVER, NQ→MultiRC, NQ→ReCoRD: τ de 0,78 a 1,0); **falha** em mudanças drásticas — idioma (XGLUE: τ **0,33**), texto→código (CodeSearchNet: **0,28**), extração de entidades (T-Rex: **0,38**). Cada mudança exige passagens do domínio e few-shots para reconfigurar.

## 5. Limitações

**[Artigo] (Seção 7):** depende de **150–300 anotações** de humanos "familiarizados com o domínio" (domínios especializados — direito, medicina, finanças — exigem especialistas); exige **GPU** (~32 GB para DeBERTa-v3-Large e FLAN-T5-XXL; horas de ajuste/geração); só **inglês**.

**[Minha leitura]**
- **Faithfulness mal validada:** não foi avaliada nos 6 datasets de mock (4.2); só em AIS com 200 labels, e em WoW o juiz acerta 62,5% — modesto.
- **Sistemas mock têm negativos fáceis:** negativos são passagens aleatórias ou do mesmo documento; não reproduzem erros sutis de sistemas reais. Tabela 5 (reais) usa só 1 passagem por sistema — longe de um RAG com top-k.
- **Comparação com RAGAS opaca:** o RAGAS produz score **contínuo** (razão de sentenças); o texto lido não explica como foi convertido em "acurácia" (RAGAS com 15–36% em CR, abaixo do acaso de uma classificação binária, sugere um artefato de conversão). Os 59,9 p.p. devem ser lidos com essa cautela.
- **Filtro de consultas "recupera sua passagem no top-1"** enviesa os dados de treino para o que o retriever **já** acerta (circularidade).
- Dependência de GPU/treino limita a replicação por um projeto pequeno.
- Critério de sucesso τ > 0,9 raramente é atingido com 150 labels em CR (Tabela 3).

## 6. Reflexões ancoradas no NOSSO projeto

**R1 — Tamanho e *tipo* das nossas 100 labels humanas (P11, calibração 0.8; `ckpt-0.6-plan.md` §6 "Calibração formal… usa suas labels da revisão").** *DESAFIA o plano.* (a) **Tamanho:** 100 < 150. A Tabela 3 mostra que a 100 labels o τ de CR é 0,44–0,67 ("não distingue sistemas"). (b) **Tipo (ponto mais importante):** o conjunto de validação do ARES rotula **saídas de sistema** — triplas (pergunta, passagem recuperada, resposta gerada) com positivo/negativo em cada dimensão. Pelo roadmap (§1 "revisão humana do golden set 100/100"; plano §6), as nossas labels validam **o golden set** (a pergunta e o gabarito estão corretos?), não **respostas geradas pelo RAG**. Para calibrar `faithfulness_judge`/`citation_accuracy` precisamos de labels humanas sobre **respostas reais** do sistema (p.ex., ≥100 respostas × fiel/não fiel, + abstenção). *[Minha leitura; confirmar com o que foi de fato rotulado na revisão.]* Consequência: reservar tempo no CKPT-0.8 para rotular ~100–150 respostas reais, em vez de assumir que as 100 labels do golden servem.

**R2 — PPI e intervalos de confiança (P12 do roadmap, "teste de significância"; `decisions.md`).** *APOIA como evolução opcional.* O roadmap já registra "PPI formal fica como evolução opcional". O ARES mostra o ganho concreto: IC de 7,4/6,1 p.p. de largura **com 300 labels** (Seção 5.3) e τ melhor que anotar 1.350 amostras. Com 100 labels o IC seria mais largo *(inferência minha; o artigo não mediu a 100 na Tabela 5)*. Adotaríamos o conceito de **reportar IC** nas métricas de juiz (bootstrap simples primeiro; PPI se tivermos ≥150 labels de respostas reais). **Não** adotaríamos o pipeline completo.

**R3 — Fine-tuning de juízes DeBERTa + dados sintéticos (P11; `config.py`).** *NÃO adotaríamos.* Exige GPU de ~32 GB e horas; foi validado em Wikipedia/notícias em inglês (Tabela 6 mostra queda em mudanças drásticas de domínio); nossos abstracts de cs.AI são um domínio técnico diferente e o corpus tem 843 chunks. O custo-benefício é pior que usar um LLM-juiz bem calibrado com κ. Também não resolve P11 (juiz ≠ gerador) por si só — deslocaria o problema para a qualidade do dado sintético.

**R4 — Nosso golden é "synthetic-by-construction" como o ARES (ADR-001/003; `build_golden_dataset.py:518`).** *NEUTRO + alerta.* Em ambos, a pergunta nasce do próprio chunk/passagem. O ARES acrescenta um **filtro** (a consulta deve recuperar sua passagem no top-1). **Não adotaríamos esse filtro** no golden: ele elimina justamente as perguntas que o retriever erra, tornando o hit@k artificialmente alto e anulando o slice `deep-hit`, concebido para expor documentos que ranqueiam baixo (viés de seleção — o mesmo risco de pooling da lesson L1). Se quisermos medir "perguntas mal formuladas", fazê-lo como diagnóstico separado, não como filtro.

**R5 — Negativos fortes = chunks do mesmo documento (P6, D-1, L1/anti-pooling-bias; `precision_at_k`).** *DESAFIA a premissa deles para o nosso corpus, e APOIA o nosso gold apertado só onde a pergunta é gerada de 1 chunk.* O ARES trata passagens do mesmo documento como **negativas** (Seção 3.1) — a mesma suposição "não-gold = irrelevante" que o P6 e a lesson L1 buscam evitar. Em abstracts fragmentados em 2–5 chunks de ~420 caracteres, um chunk irmão pode perfeitamente responder à pergunta. O ARES só pode fazer isso porque o objetivo é *treinar* um juiz com negativos plausíveis (e os mocks têm rótulos conhecidos por construção). Para *avaliar retrieval* no nosso caso, manter `None`/dedup por paper (P6) continua mais honesto. Isso informa D-1: para o slice `answerable`, o gold de chunk é "apertado" por construção (ARES tem a mesma limitação), reforçando a opção de **gold ampliado** (chunks do paper que também respondem, adjudicados) como candidata — em vez de "tudo em paper".

**R6 — Avaliar *sistemas*, não exemplos (gates do plano: deltas ≥15% core / ≥10% extended; CKPT-1..4; P11/P12).** *APOIA uma mudança de ênfase.* O ARES mede se o juiz **ranqueia corretamente sistemas** (Kendall's τ sobre sistemas separados por 2,5 p.p.), que é o que os nossos gates exigem (comparar k, modelo, chunk, rerank). O κ por exemplo (plano 0.8) mede outra coisa. Proporia reportar, além do κ: (i) concordância no **nível de sistema/configuração** (o juiz ordena as configurações como o humano?), e (ii) o **IC** da diferença entre configurações. Com ~100 exemplos e deltas de 10–15%, a Tabela 3 sugere cautela: diferenças pequenas podem não ser resolvíveis.

**R7 — Uso do juiz GPT-3.5/4 como substituto de humano (P11; `JUDGE_MODEL`).** *APOIA parcialmente.* Tabela 4: rótulos de GPT-4 no lugar de humanos reduzem τ em 0,05–0,30 — útil como **pré-triagem barata** para ampliar o conjunto de validação, mas não substitui humanos. Para nós, isso sustenta: usar o juiz forte (≠ gerador) apenas para *auxiliar*, com amostra humana final. O ARES **não** estuda autopreferência (juiz da mesma família do gerador) — P11 continua dependendo de Zheng/Wataoka.

**R8 — Context relevance/faithfulness como alternativa às métricas por ID (P1–P3, P5, P6).** *NEUTRO/DESAFIA a ideia.* O "context relevance" do ARES é um rótulo binário por **passagem** (a passagem é relevante para responder?) — é uma *relevância por julgamento* (por LLM), não um substituto de gold por ID: não dá ranking nem recall; e, como o juiz é treinado com "mesmo documento = negativo", herda o viés da R5. Útil como **segunda opinião** em amostras (ex.: validar o gold apertado), não como métrica principal.

**O que adotaríamos:** (1) rotular **respostas reais** (~100–150) para calibrar juízes (R1); (2) reportar **IC** e concordância em nível de sistema (R2, R6); (3) usar juiz forte como pré-triagem, com amostra humana final (R7); (4) corrigir 59,3 → 59,9 no roadmap (Seção 4). **O que NÃO adotaríamos:** fine-tuning DeBERTa/dados sintéticos (R3); filtro "recupera no top-1" no golden (R4); "mesmo documento = negativo" como definição de irrelevância (R5).

## 7. Citações úteis

1. *"below about 100-150 datapoints in the human preference validation set, ARES cannot meaningfully distinguish between the alternate RAG systems based on their accuracies in context relevance and answer relevance"* — Tabela 3 (legenda, apêndice).
2. *"For context relevance negatives, we randomly sample in-domain passages from the same document as the gold passage."* — Seção 3.1 ("Strong Negative Generation").
3. *"ARES relies on a small set of annotations in the human preference validation set (roughly 150-300 datapoints but more is better). These annotations often require an annotator familiar with the RAG system's domain application."* — Seção 7 (Limitations).
