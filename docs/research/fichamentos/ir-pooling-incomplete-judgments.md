# Buckley & Voorhees 2004 · Buckley et al. 2007 (NIST) · Büttcher et al. 2007 — pooling, julgamentos incompletos e viés

- **Referências completas e links**
  1. Buckley, C. & Voorhees, E. M. (2004). *Retrieval Evaluation with Incomplete Information*. SIGIR '04, Sheffield, 25–29 jul. 2004 (o PDF lido não traz numeração de página do volume — cito seções/tabelas/figuras).
     PDF lido: <https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469> · índice: <https://www.semanticscholar.org/paper/Retrieval-evaluation-with-incomplete-information-Buckley-Voorhees/6878cdc5b632e018f827a9d1520e7353d8502d25>
  2. Buckley, C., Dimmick, D., Soboroff, I., Voorhees, E. (2007). *Bias and the Limits of Pooling for Large Collections*. Documento NIST datado "July 17, 2007" (16 págs.; **não confirmei o venue** de publicação — tratar como relatório/preprint NIST). PDF lido: <https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=51236> · página: <https://www.nist.gov/publications/bias-and-limits-pooling-large-collections>
  3. Büttcher, S., Clarke, C. L. A., Yeung, P. C. K., Soboroff, I. (2007). *Reliable Information Retrieval Evaluation with Incomplete and Biased Judgements*. SIGIR '07, Amsterdã. DOI <https://dl.acm.org/doi/10.1145/1277741.1277755>. **PDF lido:** versão hospedada pelo 1º autor, <http://stefan.buettcher.org/papers/buettcher_2007_reliable_evaluation.pdf> (8 págs.; traz o texto-modelo de copyright da ACM, então pode diferir em detalhes da versão final).
- **Confiança da leitura:**
  - B&V 2004: **integral** (8 págs.). Gráficos (Figs. 1–5) lidos só pelas legendas e pelo texto que os descreve.
  - NIST 2007: **integral** do texto; a Tabela 1 (características das coleções) e as figuras não foram extraídas — números dessa tabela **não** são citados.
  - Büttcher 2007: **integral do texto**; as Tabelas 2, 3, 5, 7 e 8 perderam os números na extração do PDF, então só uso valores que o próprio texto cita. O acesso à versão ACM foi bloqueado (Cloudflare).

> Convenção: **(F)** = o que a fonte afirma; **(I)** = minha interpretação. Os três trabalhos têm uma seção (A, B, C) com os itens 1–5 do formato; a **Síntese** traz os itens 6 e 7 e as respostas às perguntas (a)–(e) do pedido.

---

## A. Buckley & Voorhees 2004 — *Retrieval Evaluation with Incomplete Information*

### 1. Problema que a fonte ataca
(F, §1) O paradigma Cranfield assume que os julgamentos de relevância são **completos** (todo documento julgado para todo tópico). Em coleções grandes feitas por *pooling* essa suposição é só aproximadamente verdadeira. O artigo investiga a robustez das medidas a violações **grosseiras** dessa suposição, com duas metas: (i) julgamentos *incompletos* porém não enviesados e (ii) julgamentos *imperfeitos* (julgam documentos que já saíram da coleção, §5). Motivação: coleções TREC têm ~800 mil docs e pools de 1.000–2.000 docs/tópico; a coleção cresce e o esforço de anotação não.

### 2. Método
(F, §2.1) Medidas comparadas:
- **P(10)** — precisão nos 10 primeiros (de cada 10 recuperados, quantos são relevantes). Fácil de interpretar, mas "só importa um relevante entrar ou sair do top-10" e tem margem de erro grande.
- **R-precision** — precisão após R documentos recuperados, onde R é o nº de relevantes do tópico (o ponto em que precisão = recall).
- **MAP** — média, sobre os relevantes, da precisão no momento em que cada relevante aparece (zero se não recuperado). (Mede a qualidade do ranking inteiro; difícil de interpretar.)
- **bpref** (proposta do artigo, "binary preference") —
  `bpref = (1/R) · Σ_r [ 1 − |n acima de r| / R ]`, onde `r` é um relevante recuperado, `n` varre os **R primeiros não-relevantes *julgados*** recuperados pelo sistema, e R é o nº de relevantes. (Em palavras: para cada relevante, penaliza quantos não-relevantes *explicitamente julgados* vieram antes dele; documentos sem julgamento são **ignorados**.)
- **bpref-10** — variante usada nos experimentos: denominador `10 + R` e `n` entre os `10 + R` primeiros não-relevantes julgados; garante ≥10 pares de documentos, porque o bpref puro é "excessivamente grosseiro" com 1–2 relevantes (§2.1).
(F) Justificativa do desenho: N+R julgamentos binários geram N×R preferências; 12 variantes ingênuas de bpref foram testadas e "average poorly".

Estabilidade (§2.2): ranking de sistemas comparado por **Kendall τ**; τ>0,9 ≈ rankings equivalentes, τ<0,8 = diferenças notáveis.

Dados (§2.3, Tab. 1): TREC-8 (528 mil docs, 50 tópicos, 94,6 relevantes/tópico, 124 runs de 38 grupos), TREC-10 (1,7 M docs web, 50 tópicos, 67,3, 77 runs/26 grupos), TREC-12/Robust (528 mil, 100 tópicos, 60,7, 73 runs/16 grupos). Runs que recuperaram <95% do máximo foram excluídos.

Experimento de incompletude (§4): para cada tópico, listas aleatórias de relevantes e de não-relevantes julgados; criam-se 16 qrels reduzidos (90, 80, … 5, 4, 3, 2, 1% do original), com no mínimo 1 relevante e 10 não-relevantes por tópico. Cada qrels menor é subconjunto do maior. **A amostragem é aleatória ⇒ sem viés contra sistemas** (F, §4: "the reduced qrels are also unbiased with respect to systems").

Experimento de imperfeição (§5): qrels completo, mas a coleção é reduzida a 90–50% dos documentos (aleatório), avaliando só os 400 primeiros recuperados.

### 3. Principais contribuições
- Primeira demonstração sistemática de que MAP, P(10) e R-prec **não são robustas** a incompletude massiva (§4, Fig. 3).
- Proposta do **bpref**, que avalia só documentos julgados.
- Mostra que com julgamentos completos bpref-10 e MAP concordam (τ ≥ 0,9) — logo é substituto razoável (§3, Tab. 3).
- Distingue julgamentos *incompletos* de *imperfeitos* e mostra que o segundo é menos grave (§5).
- Explicita a **suposição de validade do bpref** (§6): a chance de um documento estar no pool deve ser independente de ele ser relevante ou não.

### 4. Resultados-chave
- **Tab. 3 (julgamentos completos, τ contra o ranking por MAP, TREC-8/10/12):** bpref-10 (1000 docs) 0,934/0,895/0,942; R-prec 0,916/0,851/0,899; P(10) 0,813/0,729/0,721.
- **Tab. 2:** diferença δ necessária para 95% de confiança (TREC-8): MAP 0,040 (9,7% do melhor score), P(10) 0,093 (12,9%), R-prec 0,039 (8,9%), bpref-10 0,041 (9,3%). TREC-10 é a coleção mais ruidosa (δ relativos 15,8–18,2%).
- **Regime de poucos relevantes (§4):** com 5% do qrels, a fração de tópicos com um único relevante é 26% / 34% / 40% (TREC-8/10/12); com 10%: 8/16/18%; com 1%: 78/92/90%. O autor avisa que, nesse extremo, "absolute number effects" dominam a incompletude.
- **Fig. 2:** MAP, P(10) e R-prec caem monotonicamente quando o qrels encolhe; bpref-10 sobe só levemente (exceto qrels minúsculos) — "scores consistentes" entre tópicos com graus diferentes de incompletude.
- **Fig. 3:** τ de bpref-10 permanece >0,9 até o qrels de 50% (TREC-10, a coleção mais ruidosa) e até 25% (TREC-8). P(10) cai cedo e depois fica plano (instável: dominado por tópicos com muitos relevantes).
- **§5 (Fig. 5):** nenhuma medida é fortemente afetada por documentos julgados que saíram da coleção; τ só passa de <0,9 quando ~metade da coleção some.
- **§6 (conclusão):** "Adding additional unjudged documents to a retrieved set can have no effect on that set's bpref score, but can have significant influence on the other measures' scores."

### 5. Limitações
- **Do autor (§6):** bpref só é válido se pertencer ao pool for independente da relevância; um sistema "suficientemente novo" pode recuperar mais não-relevantes fora do pool. Evidência indireta (só 70 dos 124 runs do TREC-8 contribuíram ao pool e o bpref concordou com MAP em todos) — mas "more investigation needs to be done before concluding that bpref can fairly evaluate novel systems that did not contribute to the judgment pool".
- **Minhas (I):**
  1. A incompletude simulada é **aleatória**, portanto não testa o caso que mais nos importa (viés sistemático contra um tipo de retriever). Esse caso é testado por Büttcher 2007 (seção C) e discutido por NIST 2007 (seção B).
  2. A métrica de sucesso é a correlação entre rankings de **sistemas** (Kendall τ), não o erro do score por consulta; com poucas consultas pode haver trocas de ordem que o τ não revela.
  3. Os tópicos têm em média 60–95 relevantes; **não** é o nosso regime (1 chunk relevante). O próprio artigo diz que o bpref puro é grosseiro com 1–2 relevantes e que *todas* as medidas são instáveis com poucos relevantes.
  4. bpref **exige não-relevantes julgados**; o nosso golden set só tem positivos (ver reflexões).

---

## B. Buckley, Dimmick, Soboroff, Voorhees — *Bias and the Limits of Pooling for Large Collections* (NIST, 2007)

### 1. Problema que a fonte ataca
(F, abstract/§1) *Pooling* assume que, ao juntar o top-λ de vários sistemas, os relevantes encontrados formam uma amostra **não enviesada** do conjunto relevante verdadeiro. Com λ constante e coleção crescente, a hipótese de julgamentos aproximadamente completos falha — e a pergunta é se a de **não-viés** também falha. Resposta: sim, em favor de documentos relevantes que contêm **palavras do título do tópico**.

### 2. Método
(F, §3–§5)
- **`titlestat`** — para um tópico T e um conjunto de docs C: `titlestat_T = (1/|T|) Σ_{t∈T} |C_t| / min(|C|, df_t)`, onde `t` são palavras do título, `C_t` = docs de C que contêm `t`, `df_t` = frequência do termo na coleção. (Mede "que fração dos documentos contém as palavras do título"; 1,0 = todos contêm todas; 0 = nenhum.) `titlestat_rel` = idem restrito aos relevantes; `titlestat_rank` = idem restrito aos docs recuperados num dado rank por um conjunto de runs.
- **Comparação controlada:** mesmos 50 tópicos em duas coleções — Disks4&5 (528.542 docs) e AQUAINT (1.033.461 docs; TREC 2005 HARD/Robust; 50 runs no pool; profundidade do pool = 55 vs ≥100 em Disks4&5).
- **Teste LOU ("leave out uniques", Zobel):** reavalia cada run do pool sem os documentos que só ele contribuiu.
- **Estudo de caso `sab05ror1`:** run de *routing* (consultas construídas a partir dos relevantes de Disks4&5, **sem** usar palavras do título), que contribuiu com uma anomalia: muitos relevantes únicos.

### 3. Principais contribuições
- Demonstração empírica de **viés de pooling dependente do tamanho da coleção** (não do nº de relevantes por tópico).
- Medida `titlestat` para detectá-lo.
- Argumento de contagem: sistemas razoáveis colocam primeiro os documentos com palavras do título; com coleção grande, esses documentos "enchem" o pool e expulsam outros tipos de relevante.
- Discussão de alternativas: pools mais profundos (move-to-front), amostragem aleatória/estratificada, diversidade de runs, "engenharia do conjunto de julgamentos" + bpref (§6).

### 4. Resultados-chave
- **§3.2:** `titlestat_rel` = **0,588** (Disks4&5) vs **0,719** (AQUAINT); maior em AQUAINT em **48 de 50 tópicos** (p = 6,25·10⁻¹⁰, t-test pareado).
- **`sab05ror1` (§3.2):** 405 dos 2.750 documentos que ele pôs no pool eram relevantes **únicos** — recorde histórico do TREC em razão únicos/contribuídos. LOU: MAP cai de 0,266 para 0,202 (−23%) sem suas contribuições; o segundo maior efeito foi −12%, os demais ≤ 8%.
- **§4:** `titlestat` do run = 0,388 vs 0,600 nos demais runs; os relevantes únicos que ele trouxe têm `titlestat` 0,530 (vs 0,719 do conjunto): relevantes sem palavras do título **existem** e o pool os perdeu. Fig. 4: o viés **não** se relaciona com o nº de relevantes do tópico.
- **§5 (Terabyte):** `titlestat_rank` ≈ 0,8 no rank 100 e ≈ 0,68 no rank 1000; `titlestat_rel` 0,889 (2004) e 0,898 (2005). LOU médio 9,6% (máx. 45,5%) em 2004 e 3,9% (máx. 17,7%) em 2005 — pequenas porque todos os runs do pool miravam termos do título. Em coleções TREC "pós-TREC-4" ~6% dos julgados são relevantes; nas coleções Terabyte/AQUAINT é bem maior.
- **Controle (§5):** na coleção "TREC-8 small web" (`titlestat_rel` alto, 0,850) uma busca ativa de relevantes novos achou **<1 relevante novo por tópico** — ou seja, `titlestat_rel` alto sozinho não prova viés; o indício é a *diferença* entre coleções com os mesmos tópicos.
- **§6:** move-to-front pooling recupera 79% dos relevantes do pool de profundidade 50 julgando só 48% dos não-relevantes.
- **§7:** o problema é "conservador" — um run afetado só é prejudicado ("unjudged relevant documents will be counted as non-relevant"); comparações entre sistemas *dentro* do pool não são afetadas; o risco é para sistemas **diferentes** (p.ex. baseados em conceitos semânticos), que recuperam relevantes sem as palavras do título.

### 5. Limitações
- **Do autor:** o viés é demonstrado só para palavras do título; para Terabyte a evidência é "circunstancial" (§5/§7); nenhuma solução comprovada ("it is currently unknown how to construct a fair sample", §6); relevância binária textual.
- **Minhas (I):** (1) o artigo é um relatório de 2007 com sistemas lexicais dominantes — hoje o contraste que importa é lexical × denso, que BEIR (seção D) mede; (2) `titlestat` mede presença de palavras, não equivalência semântica; (3) o efeito é estudado em coleções de milhões de documentos, enquanto nosso corpus tem 843 chunks (pool relativamente grande — o fator que o artigo identifica como crítico, "tamanho do pool vs tamanho da coleção", é benigno para nós).

---

## C. Büttcher, Clarke, Yeung, Soboroff — *Reliable IR Evaluation with Incomplete and Biased Judgements* (SIGIR 2007)

### 1. Problema que a fonte ataca
(F, §1–§2) Pooling é inerentemente enviesado contra sistemas que não contribuíram ao pool; a literatura de bpref/infAP estudou só incompletude **sem viés**. Exemplo do autor (§2): se só 50% do top-10 de um sistema foi julgado, seu P@10 não passa de 0,5 mesmo que mais da metade seja relevante. O artigo estuda o caso **enviesado** e testa se se pode **reconstruir** um conjunto de julgamentos não enviesado a partir do enviesado.

### 2. Método
(F, §2–§5)
- Medidas: P@k; AP; nDCG@20; **bpref** (`1 − Σ_r |n acima de r| / (|R|·min{|R|,|N|})`, com R relevantes conhecidos e N os |R| não-relevantes conhecidos mais bem ranqueados; docs julgados ausentes do ranking entram em rank ∞); **bpref-10**; **RankEff** (Grönqvist: `Σ_r |n acima de r| / (|R|·(|J|−|R|))`, com J = todos os julgados — usa *todos* os não-relevantes julgados); infAP (Yilmaz & Aslam); e a nova **P@k(j)** = precisão entre os k primeiros **julgados** (opção `-J` do trec_eval).
- **Dados (§4, Tab. 1):** TREC 2006 Terabyte/GOV2, 50 tópicos, 42 runs "no pool" (11 manuais, 31 automáticos), profundidade 50, ~640 docs julgados/tópico (118 relevantes, 522 não-relevantes). Avaliação sobre os 10.000 primeiros recuperados.
- **Experimentos de viés (§5.2):** (i) *leave-one-group-out*: remover do qrels os documentos que só um grupo contribuiu e reavaliar os runs desse grupo; (ii) *automatic-only*: remover todos os documentos que só runs manuais contribuíram (aproxima "um novo método melhor e diferente").
- **Correção proposta (§3, §5.3):** treinar um classificador nos julgados e **prever a relevância dos não-julgados** que o sistema avaliado recupera; usa dois classificadores — um baseado em divergência KL entre o modelo de linguagem do documento e o dos relevantes (limiar tal que precisão=recall no treino) e um SVM (SVMlight, vetores TF-IDF de 10⁶ termos). Depois avalia-se normalmente sobre o qrels "completado".
- Erro avaliado com Kendall τ e RMS do score (§4, Eq. 3).

### 3. Principais contribuições
- Primeiro estudo, nos moldes de bpref/AP, do impacto de **viés** (e não só incompletude aleatória) nos julgamentos.
- Evidência de que **bpref não é imune** a viés: onde AP subestima o sistema de fora do pool, o bpref o superestima.
- A medida P@k(j) e a demonstração de que ignorar não-julgados é uma má aproximação.
- Método de **completar julgamentos com um classificador**.
- Comparação com RankEff, que se mostra a medida mais robusta entre as de "só julgados".

### 4. Resultados-chave
- **§5.1 (incompletude aleatória):** reproduz a literatura — AP, P@k e nDCG@k correlacionam mal com o ranking original; bpref melhor; RankEff ligeiramente melhor (Fig. 1b).
- **§5.2 leave-one-group-out:** remover as contribuições únicas de um grupo tira em média **22 julgados por tópico**; o run do grupo perde em média **2,1 posições** (até 12 em P@20). AP muda ~1,5 posição; **bpref ~2 posições**, com 90,5% das diferenças estatisticamente significativas (t pareado, p<0,05). AP tende a **subestimar**, bpref a **superestimar**.
- **P@20(j):** após a remoção, o run tem em média **3,4 não-julgados** no top-20, dos quais só **0,3 são relevantes** (segundo o qrels original); por isso ignorar não-julgados "implicitly assumes that they exhibit the same proportion of relevant and non-relevant documents as the judged documents — an assumption that is simply wrong". P@20(j) superestima por até 22 posições.
- **RankEff:** a posição do run muda em média só **0,857** (pior caso 4).
- **Automatic-only (Tab. 4/5):** julgados 31.984 → 23.099 (−28%), relevantes 5.893 → 4.373 (−26%); 27,8% do pool só-manual é relevante vs 18,9% do só-automático. Kendall τ <0,9 para todas as medidas; o pior é P@20 com τ = 0,8281; bpref τ = 0,8676 (sobe o score de todos os runs, mais ainda nos manuais, p.ex. "zetaman" 0,398 → 0,475).
- **§5.3:** os classificadores sozinhos têm F1 ≈ 0,5 (Tab. 6) — "surprisingly poor" — mas no leave-one-out o SVM atinge precisão 0,7979 e recall 0,6872 (KLD: 0,6532/0,6642) e reduz o deslocamento de rank a <1 posição. No automatic-only, com SVM: τ de P@20 **0,8281 → 0,9512**; de bpref **0,8676 → 0,9164**; o deslocamento médio de um run manual em P@20 cai de 6,4 para 2,0.

### 5. Limitações
- **Do autor (§6):** classificar documentos *além da profundidade do pool* pode induzir viés no sentido oposto; há risco de os sistemas passarem a ser otimizados para o classificador e não para o julgamento humano ("only a short-term" solução); usaram o SVM indutivo (o transdutivo seria melhor com treino pequeno).
- **Minhas (I):** uma só coleção (GOV2/TB06) e relevância textual; o método depende de haver centenas de julgados por tópico (aqui ~118 relevantes) — com 1 relevante por consulta não há dados para treinar um classificador por consulta; tabelas com números ausentes na minha extração (as conclusões usadas acima estão no texto).

---

## Síntese

**O que as três fontes dizem em conjunto (F + I):**
1. Tratar não-julgado como irrelevante dá uma **cota inferior** do desempenho (conservador); ignorar o não-julgado (bpref, P@k(j), condensed lists) **não é neutro** — superestima quando os não-julgados têm menor proporção de relevantes que os julgados (Büttcher §5.2).
2. O viés vem de **como o pool foi formado** (quais sistemas contribuíram, quais características lexicais os relevantes tinham), não só do tamanho do pool: NIST §3–§4.
3. A saída "limpa" é **julgar os buracos** do sistema avaliado (que é o que o BEIR fez para o TREC-COVID — ver `thakur-2021-beir.md`), e não uma correção estatística.
4. Todas as medidas são instáveis com **poucos relevantes por consulta** (B&V §4); bpref puro é grosseiro com R = 1–2 (§2.1).

### 6. Reflexões ancoradas no NOSSO projeto

| # | Ponto (roadmap) / arquivo | Reflexão | A fonte… |
|---|---|---|---|
| 1 | **P6**, `precision_at_k` (`evaluators.py`) | Em `answerable`/`persona` o gold é **1 chunk** e o top-5 frequentemente traz **outros chunks do mesmo paper** (`decisions.md`, 2026-09-27); contá-los como "ruído" é exatamente tratar não-julgado como irrelevante. NIST §7 e Büttcher §5.2 mostram que isso é um viés **conservador** (piso). Isso **apoia** a decisão de P6 de não usar `precision_at_k` como medida de ruído com gold de 1 chunk (retornar `None`/trocar por precisão em nível de paper + `distinct_papers@k`). | **(i) APOIA** |
| 2 | **Guardrail "bpref-style tolerance"** (lesson `golden_dataset_construction_…`, "Design consequence" §1) e glossário | A lesson sugere "bpref-style tolerance" para o extra recuperado. **Desafio:** (a) bpref exige **não-relevantes julgados** (`min{|R|,|N|}` no denominador; B&V §2.1); nosso gold só tem positivos ⇒ indefinido (o ranx dividiria por zero, ver `bassani-2022-ranx.md`); (b) Büttcher §5.2 mostra que bpref **superestima** sistemas fora do pool (aqui: um retriever diferente do que gerou as perguntas); (c) com R=1 o próprio B&V diz que é grosseiro. Corrigir o texto da lesson/roadmap: a tolerância que adotamos é "não penalizar não-julgado", **não** bpref. | **(ii) DESAFIA** a formulação |
| 3 | **D-1** (um nível vs dois níveis) / `_ranked_and_gold` | O gold de chunk único é um **conjunto de relevantes incompleto** (irmãos do mesmo paper podem responder) ⇒ hit/recall/MRR em nível de chunk são **pisos**. Nível de paper é um **teto** (qualquer chunk do paper conta). As fontes sustentam reportar a faixa [piso, teto] e **julgar os buracos** em vez de escolher um nível. Estimativa (minha): ~50 exemplos com gold de chunk × top-5, só irmãos do mesmo paper (≤4 por paper) ⇒ ≤ ~200–250 julgamentos — viável como mini-pooling do EXP-0. | **(i) APOIA** manter 2 níveis (como faixa) e **DESAFIA** escolher só um |
| 4 | **P5 / D-2**, `hit_rate` e `recall_at_k` | Com |gold|=1 as duas coincidem; as fontes não discutem isso, mas medem aqui sobre qrels com muitos relevantes. B&V §4 alerta que com 1 relevante "absolute number effects" dominam: a métrica vira função binária por consulta, com muito ruído de amostragem em 40–50 exemplos. | **(iii) NEUTRA** |
| 5 | **Risco de viés lexical do gold sintético** (lesson L1; ADR de golden set) | Perguntas geradas **do próprio chunk** tendem a compartilhar vocabulário com o chunk. NIST 2007 (viés para palavras do título) e BEIR §6 são evidência **análoga** (julgamentos derivados de busca lexical favorecem retrievers lexicais); **nenhuma das três fontes estuda perguntas sintéticas** — é extrapolação minha. Consequência prática: ao comparar retriever denso × BM25/híbrido (CKPT-1/4), o gold sintético pode favorecer o lexical; cuidado antes de aceitar/rejeitar um experimento por <10–15 p.p. | **(ii) DESAFIA** a premissa de que gold sintético é "neutro por construção" (apenas por analogia) |
| 6 | **EXP-0 mini-pooling** (lesson, CKPT-0.8) | O desenho "rodar o retrieval, listar candidatos não julgados e adjudicar" é **o que a literatura recomenda** (NIST §6/§7, Büttcher conclusão, BEIR §6). Adotaríamos: relatar também uma "taxa de buraco" (fração do top-k nem-gold-nem-julgado; BEIR "Hole@k"). Para a escolha de qual retriever fornece candidatos: unir os top-k de **mais de um tipo** (denso + BM25), pois NIST §6: "run diversity is not a complete solution" mas ajuda a detectar viés. | **(i) APOIA** |
| 7 | **Dedup por paper** (L-1) | Nenhuma das três fontes trata duplicatas por documento no ranking; a regra é uma decisão de engenharia nossa (já marcada assim no roadmap §D). | **(iii) NEUTRA** |
| 8 | **Gates ≥15%/≥10%** (roadmap §C) e `ranx.compare` | B&V Tab. 2 mostra δ para 95% de confiança de ~8–18% do melhor score (7,7–18,2% conforme medida/coleção) com **50–100 tópicos** e muitos relevantes. Com ~50 exemplos de gold único e métrica binária, o δ necessário tende a ser **maior** (I). Sugere que os gates de 10–15% precisem de intervalo de confiança/teste pareado (ver `bassani-2022-ranx.md` para as opções). | **(iii) NEUTRA** (informa a decisão) |

**O que adotaríamos:** (a) manter "não-julgado ≠ irrelevante" e `precision_at_k` condicionada a gold completo (P6); (b) julgar os irmãos do mesmo paper no EXP-0 e reportar chunk-estrito como piso e paper-nível como teto; (c) reportar uma taxa de buracos; (d) citar Büttcher 2007 para justificar **não** usar P@k(j)/condensed lists.
**O que NÃO adotaríamos:** (a) **bpref** — requer não-relevantes julgados que não temos e é sensível a viés; (b) a correção por **classificador** de Büttcher — exige centenas de julgados por tópico e introduziria um segundo modelo a calibrar; (c) conclusões de magnitude (τ > 0,9 etc.) como limiares para o nosso caso: estão calibradas em coleções de 50–100 tópicos com dezenas de relevantes.

### Respostas às perguntas do pedido (parte pooling)
- **(a)** B&V/NIST/Büttcher não discutem a equivalência hit≡recall para 1 relevante; o que dizem é que P(10) e outras medidas "de limiar" têm ruído maior. Ver `thakur-2021-beir.md` e `bassani-2022-ranx.md` para como BEIR e ranx definem as métricas.
- **(b)** Nenhum dos três estuda consultas geradas do documento; a evidência é análoga (viés lexical dos julgamentos — NIST §3–§5, Tab. de `titlestat`); ver reflexão 5.
- **(c)** Documentos não julgados: **bpref** (ignora; B&V) só vale com não-relevantes julgados e sem viés de pool; **condensed lists/P@k(j)** (Büttcher) são instáveis e superestimam; **tratar como irrelevante** é conservador (piso) e é seguro para *comparar sistemas dentro do pool*; o que a literatura realmente recomenda é **julgar os buracos** do sistema avaliado. Para gold de 1 chunk com irmãos possivelmente relevantes: piso (chunk) + teto (paper) + adjudicar os irmãos.
- **(d)** As três fontes operam num único nível (documento); não discutem passagem × documento ⇒ **lacuna**. A leitura que ofereço é (I): dois níveis, reportados separadamente e como faixa, sem média combinada.

### 7. Citações úteis
1. Büttcher et al. 2007, §5.2: "Ignoring the unjudged documents in a system's ranking implicitly assumes that they exhibit the same proportion of relevant and non-relevant documents as the judged documents — an assumption that is simply wrong."
2. Buckley et al. 2007 (NIST), abstract: "This phenomenon is wholly dependent on the collection size and does not depend on the number of relevant documents for a given topic."
3. Buckley & Voorhees 2004, §6: "Adding additional unjudged documents to a retrieved set can have no effect on that set's bpref score, but can have significant influence on the other measures' scores."
