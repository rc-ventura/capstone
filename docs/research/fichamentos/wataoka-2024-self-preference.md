# Wataoka et al. 2024 — Viés de auto-preferência em LLM-as-a-Judge

- **Referência completa e link:** Wataoka, K.; Takahashi, T.; Ri, R. (SB Intuitions). *Self-Preference Bias in LLM-as-a-Judge*. arXiv:2410.21819 (v2, 21/06/2025; primeira versão out/2024). <https://arxiv.org/abs/2410.21819> · Venue de publicação: **não consta** no texto lido (só o carimbo arXiv).
- **Confiança da leitura:** **integral** do PDF v2 (9 páginas de corpo + referências). **Ressalva:** os valores por modelo da Fig. 1b (viés pela Definição 4.1) e as curvas das Figs. 3–6 estão só em imagem; **o único valor numérico da Def. 4.1 que o texto dá é o do GPT-4 (0,520)**. Os demais valores numéricos citados abaixo vêm da Tab. 2 (outra definição de viés). Não inventei valores de figura.
- **Convenção:** **[Artigo]** = afirmado/medido; **[Interpretação]** = leitura minha.

## 1. Problema que o artigo ataca

LLMs usados como juízes tendem a **superestimar as próprias saídas** (self-preference bias), o que pode embutir estilo/ideologia do juiz nos sistemas avaliados (§1). Faltavam (a) uma **métrica quantitativa** do viés comparável a humanos (trabalhos anteriores, como Panickssery et al., relacionam auto-reconhecimento mas "sem comparação com avaliadores humanos"; Xu et al. e Stureborg et al. usam nota absoluta, difícil de ancorar) e (b) uma **explicação da causa**. Os autores escolhem avaliação **pairwise** porque ela permite rótulo humano mais consistente (§1).

## 2. Método (passo a passo)

**2.1 Métrica (§3–4.1)**
- Base: **Equal Opportunity** (Hardt et al. 2016, Eq. 1): um classificador é justo se tem **o mesmo recall** em grupos com atributos sensíveis diferentes: P(Y'=1|S=1,Y=1) = P(Y'=1|S=0,Y=1) (mede se o classificador acerta igualmente bem nos dois grupos).
- **Definição 4.1 / Eq. 2 (self-preference bias do avaliador f):** `Bias = P(Y'=1 | S=1, Y=1) − P(Y'=1 | S=0, Y=1)`, em que, para um par de respostas (y0, y1): Y = índice da resposta preferida pelo **humano**; Y' = índice da preferida pelo **juiz f**; S = índice da resposta **gerada pelo próprio f**. Ou seja: (**quanto o juiz concorda com o humano quando a resposta do próprio juiz é a que o humano prefere**) − (**quanto concorda quando é a resposta do outro modelo que o humano prefere**). 0 = sem viés; próximo de 1 = alto viés; −1 = viés reverso (subestima as próprias respostas).
- **Ambiguidade que registro [Interpretação]:** a Eq. 2 literal condiciona ambos os termos a Y=1, mas a prosa do §4.1 e o cálculo do §4.3 tratam o 2º termo como "juiz favorece a si mesmo dado que o **humano o avaliou desfavoravelmente**"; no exemplo do GPT-4 (§4.3, Fig. 2): recall nos casos em que o humano **prefere** o GPT-4 = 1852/(108+1852) ≈ **0,945**; recall nos casos em que o humano **prefere o outro** = 118/(160+118) ≈ **0,425**; diferença = **0,520** (valor da Fig. 1b). A leitura prática é: *o juiz acerta 94,5% quando o humano dá razão ao GPT-4, mas só 42,5% quando o humano dá razão ao concorrente*.
- **Variante sem rótulo humano (§6, Eq. 4; Demographic Parity):** `Bias = P(Y'=1|S=1) − P(Y'=1|S=0)` (diferença entre a taxa com que o juiz escolhe a própria resposta e a taxa com que escolhe a do outro; **não** controla qualidade, por isso os autores avisam que não prova injustiça).

**2.2 Experimento (§4.2)**
- Dados: **Chatbot Arena** (Zheng et al.), **33.000 diálogos** com 1 pergunta humana + 2 respostas de LLMs diferentes, rótulo humano `model_a`/`model_b`/`tie`.
- **8 juízes**, todos modelos presentes na Arena: GPT-4, GPT-3.5-Turbo, Vicuna-7b, Vicuna-13b, oasst-pythia-12b, dolly-v2-12b, Koala-13b, stablelm-tuned-alpha-7b.
- **Prompt de Zheng et al.** (saída `[[A]]`/`[[B]]`/`[[C]]` após explicação). O juiz **não é informado** de qual resposta é a dele.
- **Score contínuo (Eq. 3):** `score_A = p(A|context) / Σ_{w∈{A,B}} p(w|context)`, usando a distribuição de probabilidade do token que segue "[[" (normaliza entre modelos com distribuições diferentes; segue Schick et al. 2021). **Mitigação de posição:** trocar A/B e **média** dos dois scores. Respostas fora do formato foram **excluídas**.

**2.3 Hipótese da perplexidade (§5)**
- Hipótese: o juiz prefere textos **mais familiares**, i.e. de **menor perplexidade** (perplexidade = quão "surpreendente" o texto é para um modelo; menor = mais provável/familiar), e textos próprios tendem a ter menor perplexidade.
- Procedimento: calcular a perplexidade de cada resposta **condicionada ao prompt**; tomar a **diferença de perplexidade A−B**; dividir em **bins**; em cada bin, medir a taxa com que cada juiz declara A vencedora **e** a taxa humana (Fig. 3). Depois separar os casos "par contém resposta do próprio juiz" × "par não contém" (Fig. 4), comparar a log-perplexidade média de respostas próprias vs alheias (Fig. 5) e repetir com Llama-2/3/3.1 (Fig. 6).
- **Modelo usado para calcular a perplexidade:** o texto não detalha explicitamente; Fig. 5 fala em "perplexity for each LLM evaluator" e Fig. 3 "LLMs conditioned on perplexity", o que sugere a perplexidade sob o próprio juiz — **[Interpretação]**, não confirmado no texto.
- **GPT-4 e GPT-3.5-Turbo foram excluídos** desta análise "porque os valores de perplexidade não puderam ser obtidos" (§5).

## 3. Principais contribuições

1. **Métrica quantitativa** de auto-preferência derivada de Equal Opportunity, comparável a rótulos humanos (Def. 4.1).
2. **Medição em 8 LLMs** na Chatbot Arena: **GPT-4 exibe viés significativo**.
3. **Hipótese e evidência de que a causa é a perplexidade** (familiaridade), e não a autoria em si.
4. Proposta (não testada) de **ensemble de juízes** e de ponderação por perplexidade.

## 4. Resultados-chave

- **GPT-4 tem o maior viés pela Def. 4.1: 0,520** (§4.3, Fig. 1b; reconstrução pela matriz de confusão da Fig. 2: 0,945 − 0,425). Seguem **Vicuna-13b e Koala-13b** ("também exibiram viés significativo"); os demais ficaram "relativamente próximos de zero", e **oasst-pythia-12b, dolly-v2-12b e stablelm-tuned-alpha-7b** negativos (viés reverso) (§4.3). Valores individuais: só na imagem.
- **Pela Demographic Parity (Tab. 2, p. 9):** GPT-4 **0,749**; Vicuna-13b 0,382; GPT-3.5-turbo 0,191; Koala-13b 0,175; Vicuna-7b 0,052; oasst-pythia-12b 0,006; stablelm −0,032; dolly-v2-12b −0,069. A ordem GPT-4 > Vicuna-13b "coincide com a Def. 4.1".
- **Exemplo qualitativo (Tab. 1):** para "liste os itens azuis", o GPT-4 abre com "As a language model AI, I can't see the physical objects…"; humanos e GPT-3.5 preferiram a resposta direta do GPT-3.5, e o GPT-4 escolheu a própria. Os autores: o viés **não é sobre erro factual**, mas sobre **estilo/política** do modelo (§4.3, §6) — "GPT-4 … pode usar o grau de alinhamento com suas políticas como critério decisivo em respostas de qualidade comparável".
- **Perplexidade (Fig. 3, §5):** "todos os modelos, exceto stablelm-tuned-alpha-7b, mostram uma clara tendência a dar notas mais altas a respostas de **menor perplexidade**"; a tendência é **mais forte que a dos humanos** em vicuna-13b, vicuna-7b, koala-13b e oasst-pythia-12b.
- **Autor vs. outros (Fig. 4):** as taxas de vitória dos juízes para texto próprio e alheio, no mesmo bin de perplexidade, foram **similares** (exceto dolly-v2-12b e stablelm) → "o fator que influencia o juiz **não é se a resposta é dele, mas a perplexidade**" (§5). Fig. 5: a log-perplexidade média é menor para as respostas do próprio avaliador em **todos** os modelos.
- **Llama-2/3/3.1 (Fig. 6, §5):** "todos os modelos mudaram suas avaliações mais que os humanos conforme a diferença de perplexidade" (sem rótulo humano para as respostas desses modelos na Arena, só comparação com os humanos existentes).
- **Mitigação sugerida (§6):** ensemble de vários modelos; reduzir o peso do juiz em amostras em que ele tem baixa perplexidade. A métrica proposta serviria para avaliar essas mitigações. **Não foi testada.**

## 5. Limitações

**Do autor (explícitas ou implícitas):** sem perplexidade para GPT-4/GPT-3.5 (justamente os mais vieses/competitivos), o que obrigou a testar a hipótese em outros modelos (§5); a Demographic Parity "não leva em conta a qualidade intrínseca" e não prova injustiça (Tab. 2); casos fora do formato foram excluídos.

**Que eu identifico [Interpretação]:**
- **Correlação, não causalidade:** o artigo mostra associação nota×perplexidade e que ela independe de autoria *dentro de bins*, mas não intervém (ex.: reescrever o texto para mudar a perplexidade e ver o veredito mudar). A conclusão "a essência do viés está na perplexidade" (Resumo; §7) é mais forte do que a evidência.
- **A explicação não cobre o caso mais forte:** o GPT-4 (maior viés) não entra na análise de perplexidade, e o próprio autor atribui o viés do GPT-4 a **estilo/política** (§6), não a perplexidade. Isso enfraquece "perplexidade explica o viés" para modelos de ponta.
- **Rótulos humanos da Arena** são de votantes anônimos, com ruído; empates e erros de formato são descartados (seleção de amostra).
- **Pairwise apenas:** nada sobre nota absoluta de resposta única (nosso caso), nem sobre juízes com gabarito (reference-based).
- **Modelos de 2023**, vários fracos (oasst-pythia, dolly, stablelm, que mal seguem o formato). Os "8 modelos" misturam juízes bons e ruins; os viés negativos de modelos fracos podem ser só ruído.
- **Mesma família:** o artigo **não testa explicitamente "mesma família"** (ex.: GPT-4 julgando GPT-4-turbo). O que ele mostra é viés para a *própria* saída e para texto *familiar* (baixa perplexidade) em geral.
- Métrica sensível ao **balanço** dos pares (recalls condicionados à preferência humana) e sem intervalo de confiança.
- Eq. 2 e prosa não são perfeitamente consistentes (ver §2.1).

## 6. Reflexões ancoradas no NOSSO projeto

**R1 — `JUDGE_MODEL` ≠ `GENERATION_MODEL` (P11; `config.py`; `ckpt-0.6-plan.md` §4: todos os 6 juízes com `gpt-4o-mini` = gerador).** Posição: **(i) APOIA fortemente.** O artigo mostra que o viés de auto-preferência é **real e grande no GPT-4 (0,520)**, aparece **mesmo sem o juiz saber a autoria**, e é mediado pela **familiaridade (perplexidade)**. No nosso desenho, o gerador **é** o juiz: as respostas do `gpt-4o-mini` são, por construção, de baixíssima perplexidade *para o `gpt-4o-mini`*. **Pergunta que o artigo responde só em parte:** *se o viés persiste quando juiz e gerador são o mesmo modelo/família?* — **o artigo não testa o par (mesmo modelo) em condição controlada nem "mesma família"**; ele mostra que o efeito existe para respostas próprias (Fig. 5) e que a perplexidade, não a autoria, move o veredito (Fig. 4). **[Interpretação]** → *família* importa só na medida em que modelos irmãos compartilham dados/alinhamento e portanto estilo familiar; um juiz de **outra família** é a escolha mais segura; apenas "outro tamanho da mesma família" reduz mas **não elimina** o risco. **O que o desenho exige:** (a) `JUDGE_MODEL` em **família diferente** do gerador (ou, no mínimo, rodada de sensibilidade com 2 juízes de famílias distintas — já no roadmap); (b) registrar `judge_model` e `generation_model` na metadata; (c) reportar deltas entre braços de experimento **com o mesmo juiz** (o viés é aproximadamente constante entre braços do mesmo gerador, então afeta mais o *nível absoluto* que o *delta*; **[Interpretação]**, não testado pelo artigo).

**R2 — Quais juízes são mais expostos? (`specificity_judge`, `completeness_judge`, `answer_relevance` vs `faithfulness_judge`, `citation_accuracy`).** Posição: **(iii) NEUTRO, com extrapolação minha.** O artigo diz que o viés de GPT-4 vem de **estilo/política** e não de erro factual. **[Interpretação]:** os juízes **subjetivos** (specificity, relevance, completeness) estão mais expostos que os de **verificação** contra o contexto (faithfulness, citação), que têm base observável. Isso é hipótese minha: o artigo só mede preferência pairwise geral. **Adotaríamos** priorizar o juiz de outra família **primeiro** nos juízes subjetivos.

**R3 — Como medir o viés no nosso setup (etapa 0.8; "100 labels humanas").** Posição: **(i) APOIA a necessidade de medir; (ii) DESAFIA a viabilidade imediata.** A Def. 4.1 exige, para cada par, **rótulo humano da preferência** e **conhecimento de qual resposta é do juiz**. As "100 labels humanas" do golden set (100/100 revisados) são, até onde entendo, **revisão do gabarito/pergunta**, não julgamentos de respostas geradas — **confirmar**. Se for isso, **não** dá para calcular a Def. 4.1; alternativas: (a) rotular ~30–50 pares de respostas (gerador A vs gerador B) para um juiz de cada família (barato: nossos 100 exemplos); (b) usar a **versão sem rótulo humano (Eq. 4)** apenas como *diagnóstico* (nota média do juiz para a própria família vs outra), tendo o cuidado do autor de que ela **não controla qualidade**. **Não adotaríamos** a Eq. 4 como critério de aceite (confunde viés e qualidade real).

**R4 — Perplexidade como diagnóstico (P11).** Posição: **(iii) NEUTRO / não adotar.** Perplexidade do prompt não está disponível para os modelos da OpenAI (o próprio artigo não conseguiu para GPT-4/3.5, §5); **[Interpretação]:** o mesmo vale para `gpt-4o-mini` pela API de chat (só *logprobs* de tokens gerados). **Não adotaríamos** a análise por bins de perplexidade; **adotaríamos** apenas a conclusão operacional (diversificar o juiz).

**R5 — Mitigação por ensemble (P11, rodada de sensibilidade).** Posição: **(i) APOIA, com custo baixo.** A sugestão do §6 (vários juízes) casa com a "rodada de sensibilidade com 2 juízes" já prevista; com 100 exemplos × 6 juízes × 2 modelos, o custo é pequeno. **Atenção:** o artigo **não testou** o ensemble; ver a ficha `qa-eval-judge-vs-f1.md` (Ho et al.) para evidência em contrário no regime extrativo.

**R6 — Posição e probabilidade do token (juízes single-answer).** Posição: **(iii) NEUTRO.** A normalização `p(A)/(p(A)+p(B))` e a média sobre trocas de posição (Eq. 3) valem para pairwise; nossos juízes são single-answer. **[Interpretação]:** usar os *logprobs* do token de veredito (score contínuo) poderia reduzir a variância dos juízes booleanos entre rodadas — **ideia opcional**, não mencionada pelo artigo para single-answer.

**R7 — Cruzamento com a literatura.** Zheng et al. (ficha própria) é **inconclusivo** em auto-enhancement; **Ho et al. (2504.11972)** reporta **ausência** de auto-preferência em QA extrativo curto. Esses resultados **desafiam** a generalização de Wataoka para o nosso caso, mas o regime é diferente (gabarito curto e verificável vs. conversa aberta). **[Interpretação]:** o risco é maior onde o julgamento é de **estilo** (specificity, answer_relevance) do que onde é de **correção contra gabarito**.

**Resumo — o que adotaríamos:** juiz de outra família (ou 2 juízes na sensibilidade); metadata `judge_model`; rotular um conjunto de **respostas geradas** para medir o viés; priorizar juízes subjetivos. **O que NÃO adotaríamos:** análise de perplexidade (inviável na API); Eq. 4 como critério de aceite; conclusão causal "perplexidade = causa" como fato; transferência literal dos números de GPT-4/Vicuna (2023, pairwise) para `gpt-4o-mini` single-answer.

## 7. Citações úteis

1. "LLMs assign significantly higher evaluations to outputs with lower perplexity than human evaluators, regardless of whether the outputs were self-generated." — Resumo (p. 1).
2. "In these experiments, we were unable to obtain perplexity values from GPT-4 and GPT-3.5-Turbo, resulting in a lack of analysis on the competitive LLMs." — §5 (p. 6).
3. "the bias was often not related to clear factual errors but rather to differences in response styles" — §6 (p. 7).
