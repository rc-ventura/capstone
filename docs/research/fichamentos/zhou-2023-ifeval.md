# Zhou et al. 2023 — IFEval: avaliação de seguimento de instruções com "instruções verificáveis"

- **Referência completa:** Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., Hou, L. (Google; Zhou também Yale) (2023). *Instruction-Following Evaluation for Large Language Models*. arXiv:2311.07911v1 [cs.CL], 14/11/2023 (PDF datado de 15/11/2023). Link: <https://arxiv.org/abs/2311.07911> (PDF: <https://arxiv.org/pdf/2311.07911>). Dataset: <https://huggingface.co/datasets/google/IFEval>. Código: <https://github.com/google-research/google-research/tree/master/instruction_following_eval>.
- **Confiança da leitura:** **integral para o artigo (v1)**: texto extraído do PDF com `pdftotext` e lido do resumo ao apêndice (Tabelas 1–3; Figura 1 e Figura 2 só por legenda/texto, sem ler valores de barra; o apêndice com a lista de prompts foi só percorrido, não lido prompt a prompt). O artigo é curto (~7 páginas + apêndice). **Não li versões posteriores** (se houver). **O esquema `instruction_id_list` + `kwargs` NÃO está no artigo:** vem do dataset card/dados no HuggingFace; conferi esse ponto no card (`README.md`) e na API de linhas do dataset em 03/10/2026 (541 linhas, campos `key`, `prompt`, `instruction_id_list`, `kwargs`). **Não li o código-fonte dos checadores** do repositório do Google, então qualquer afirmação sobre como cada checador é implementado está marcada "a verificar". Convenção: **[Artigo]** = afirmado no texto; **[HF/código]** = vem do dataset/repositório, fora do artigo; **[Minha leitura]** = interpretação minha.

## 1. Problema que o artigo ataca

**[Artigo]** Seguir instruções em linguagem natural é capacidade central de LLMs, mas sua avaliação "não é padronizada" (abstract). Os autores listam três famílias de avaliação, cada uma com defeito (§1): (1) **avaliação humana** é cara, lenta e pouco reprodutível; (2) **avaliação baseada em modelo** (LLM-juiz) depende da correção do avaliador, que "não é garantida"; (3) benchmarks quantitativos existentes. Instruções como "escreva com tom engraçado" têm critério "muito pouco claro" (§1). A proposta: focar em **"instruções verificáveis"**, definidas como "instruções passíveis de verificação objetiva de conformidade" (§1), por exemplo "escreva 450 a 500 palavras", "toda a saída deve estar em JSON", "inclua um título entre colchetes duplos". Assim a avaliação seria totalmente automática e objetiva.

**[Artigo]** Ressalva dos próprios autores (§1): "very few instructions are 100% verifiable objectively and automatically" (poucas instruções são 100% verificáveis); sempre haverá casos de borda. É por causa disso que existe o modo "loose" (§2.2).

**[Minha leitura]** É um artigo de **benchmark**, não de método: ele não testa se o verificador concorda com humanos. O argumento de que verificar por código é "unbiased" (§4) é uma afirmação, não um resultado medido.

## 2. Método (passo a passo)

### 2.1 O que o artigo afirma

**Instruções verificáveis (Tabela 1).** 25 tipos, em 9 grupos (conferi a contagem linha a linha: Keywords 4; Language 1; Length Constraints 4; Detectable Content 2; Detectable Format 6; Combination 2; Change Cases 3; Start/End 2; Punctuation 1 = 25). Exemplos: *Number Words* ("pelo menos / cerca de / no máximo N palavras"), *Number Bullets* ("exatamente N bullets markdown"), *JSON Format* ("toda a saída em JSON"), *No Commas*, *End Checker* ("termine com esta frase exata"), *Title* (`<<título>>`). Os autores dizem que escolheram esses tipos porque são "fáceis de verificar ou comuns em aplicações reais" e que a lista "pode ser expandida trivialmente", citando XML como exemplo de adição futura (legenda da Tabela 1). Cada instrução é "atômica" e verificável por "um programa simples, interpretável e determinístico" (§1).

**Prompts (§1, §2.1).** **541 prompts**, cada um com 1 a 3 instruções verificáveis. (O **abstract diz "around 500"**; o §1 diz 541; o dataset no HF tem 541 linhas. Uso 541.) Cada instrução tem variantes de **parâmetro** (450–500 vs 350–400 palavras) e de **redação** (§1). Síntese em 4 passos (§2.1): (1) gerar prompts-base com 1–3 instruções sorteadas anexadas ao final; (2) *few-shot prompting* para identificar e remover prompts ilógicos (instruções conflitantes, ex.: 5 parágrafos vs menos de 20 palavras); (3) outro *few-shot* para reescrever e aumentar a diversidade de redação; (4) revisão e edição manual, um a um. Motivo declarado: evitar conflitos entre instruções e evitar não saber se o modelo segue a instrução ou só uma redação particular dela (§2.1).

**Verificação estrita (strict) (§2.2, Eq. 1).** `is_followed(resp, inst)` devolve verdadeiro/falso. A acurácia calculada com essa função é chamada **"strict"**.

**Verificação frouxa (loose) (§2.2, Eq. 2).** Existe porque a estrita gera **falso-negativo**. Exemplo do artigo: instrução "termine o e-mail com: P.S. I do like the cake"; o modelo escreve "P.S. **I do like the cake**" (com negrito markdown); o casamento exato de string reprova. A versão loose é `is_followed_loose = Any(is_followed(transform_t(resp), inst))`: a resposta passa se **qualquer** das transformações, aplicada antes da checagem, fizer a instrução ser seguida. As transformações base são 3: (1) remover marcadores de fonte do markdown, "especialmente `*` e `**`"; (2) remover a **primeira linha** (pular introduções tipo "Sure, here it is:"); (3) remover a **última linha** (pular despedidas tipo "Hope it helps."). Combinando "cada duas e as três", mais a transformação identidade, há **8 transformações no total** (3 isoladas + 3 pares + 1 trio + identidade = 8). Custo, dito pelos autores: o loose "provavelmente introduz **falso-positivo**" (ex.: uma resposta que viola a contagem de palavras passa se a 1ª linha removida a trouxer para dentro do limite); por isso o loose é tratado como "complemento" ao critério original, não substituto.

**As 4 métricas (§3), em linguagem simples:**
- **Prompt-level strict** (acurácia por prompt, estrita): *percentual de prompts em que TODAS as instruções do prompt foram seguidas*, checando de forma exata. É a mais dura: um prompt com 3 instruções e 1 falha conta como erro inteiro.
- **Inst-level strict** (acurácia por instrução, estrita): *percentual de instruções individuais seguidas*, checando de forma exata. Um prompt com 3 instruções e 1 falha conta como 2 acertos de 3.
- **Prompt-level loose**: igual à primeira, mas a checagem de cada instrução aceita qualquer das 8 transformações.
- **Inst-level loose**: igual à segunda, com a checagem frouxa.

**Modelos avaliados (§3).** GPT-4 (respostas coletadas em novembro/2023) e PaLM 2 Small (agosto/2023), via API. Os autores avisam que **os dois modelos não são diretamente comparáveis** pela grande diferença de tamanho (legenda da Tabela 3).

### 2.2 O que vem do dataset/código no HuggingFace (FORA do artigo)

**[HF/código]** Cada linha do dataset (541) tem: `key` (inteiro), `prompt` (texto), **`instruction_id_list`** (lista de ids como `punctuation:no_comma`, `detectable_format:number_highlighted_sections`, `length_constraints:number_words`) e **`kwargs`** (uma lista, uma entrada por instrução, com os parâmetros). Conferi no card e na API: o esquema de `kwargs` é um **registro "união" com todos os campos possíveis** (`num_words`, `relation`, `num_bullets`, `num_sections`, `keywords`, `forbidden_words`, `end_phrase`, `language`, `letter`, `let_frequency`, `capital_relation`, `postscript_marker`, `first_word`, `nth_paragraph`, `prompt_to_repeat`, etc.), em que **a maioria vem `None`** em cada instrução (a entrada de uma instrução só preenche os campos que ela usa). A checagem por código (um checador por id, que lê os `kwargs`) está no repositório do Google; **como cada checador é implementado: a verificar** (não li o código). O artigo **não menciona** `instruction_id` nem `kwargs` em nenhum ponto.

## 3. Principais contribuições

1. Definição operacional de "instrução verificável" e uma lista de 25 tipos (Tabela 1, §1–§2).
2. Um conjunto de 541 prompts com 1–3 instruções cada, com método de síntese que tenta evitar conflito e redação única (§2.1).
3. Um par de critérios (strict e loose) e quatro métricas (prompt/inst × strict/loose) (§2.2, §3).
4. Baselines de dois modelos (Tabela 3) e quebra por categoria de instrução (Figura 2, só gráfico).
5. Código e dados abertos (abstract, §1).

## 4. Resultados-chave

**[Artigo] Tabela 3 (conferida contra o texto do PDF):**

| Modelo | Prompt strict (%) | Inst strict (%) | Prompt loose (%) | Inst loose (%) |
|---|---|---|---|---|
| GPT-4 | 76,89 | 83,57 | 79,30 | 85,37 |
| PaLM 2 S | 43,07 | 55,76 | 46,95 | 59,11 |

- **Inst-level ≥ prompt-level** em todos os casos, como esperado pela definição (um prompt só conta se todas as instruções passam).
- **Loose ≥ strict** em todos os casos. **[Cálculo meu, a partir da Tabela 3]** O ganho do loose sobre o strict: GPT-4 +2,41 p.p. (prompt) e +1,80 p.p. (inst); PaLM 2 S +3,88 p.p. (prompt) e +3,35 p.p. (inst). O artigo **não** decompõe quanto desse ganho é falso-negativo recuperado e quanto é falso-positivo introduzido; **não há validação humana** do critério loose.
- Figura 2 mostra a acurácia strict por categoria de instrução para os dois modelos; li só a legenda (valores por barra **não** lidos, a verificar se for citar).
- O artigo **não reporta** intervalos de confiança, número de execuções, temperatura/decodificação, nem o total de instruções individuais usado no nível "inst" (a verificar no código/dataset).
- **[Cálculo meu]** Com 541 prompts, o IC de Wilson 95% de um acerto de 76,89% (~416/541) é ~[73,2%; 80,2%]; de 43,07% (~233/541) é ~[39,0%; 47,3%]. É estreito o suficiente para separar GPT-4 de PaLM, mas o artigo não o apresenta.

## 5. Limitações

**Do autor:**
- "Poucas instruções são 100% verificáveis" (§1); o loose tenta mitigar falso-negativo, ao custo de falso-positivo (§2.2).
- A implementação "pode ser melhorada em muitas frentes" (§4): aumentar diversidade e quantidade de instruções; estender para multimodal; aproximar das aplicações reais (§4). A lista de 25 é admitida como incompleta (legenda da Tabela 1).
- Os dois modelos não são diretamente comparáveis (legenda da Tabela 3).

**Minhas (interpretação):**
- Só **2 modelos**, cada um uma vez; sem IC, sem repetições, sem análise de sensibilidade à redação.
- **Verificar ≠ concordar com humano.** O artigo não mede se o checador (strict ou loose) coincide com um juízo humano em casos de borda; a premissa "verificável = correto" não é testada.
- As **8 transformações são cegas**: removem a primeira/última linha *sempre*, sem saber se aquela linha é "introdução" ou conteúdo. O próprio artigo reconhece o falso-positivo (§2.2) mas não o quantifica.
- As instruções são **sintéticas e de estilo** (vírgulas, maiúsculas, número de palavras); o único formato estruturado é "JSON Format" (Tabela 1), descrito como "entire output should be wrapped in JSON format", sem checagem de chaves/esquema pelo que o texto diz (a verificar no código).
- Texto livre em inglês; nada sobre RAG, fundamentação em fontes ou citações.

## 6. Reflexões ancoradas no NOSSO projeto

Referência: `docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md` (P10, tabela de fontes). Estado atual: `evaluators.py::format_validator(question, answer)` despacha por substring da pergunta ("markdown table", "JSON array/object", "as CSV", "As YAML", "bullet point", "sentences"+"citation") e extrai colunas/chaves/contagens por **regex sobre o texto da pergunta** (`_check_markdown_table`, `_check_json`, `_check_csv`, `_check_yaml`, `_check_bullets`, `_check_sentences_citation`). Só há critério estrito. Formato não detectado devolve `{"score": 0, "reason": ...}` sem `key`. O slice `format` tem **N=10** (`build_golden_dataset.py::_FORMAT_EXAMPLES`), com 1–2 requisitos por pergunta (ex.: "exatamente 3 bullets de no máximo 12 palavras").

**R1 — `instruction_id` + `kwargs` como modelo para `metadata.format_spec` (P10). APOIA.**
O P10 propõe gravar uma spec estruturada (`{"type":"json_object","keys":[...]}`) em vez de regex na pergunta. O padrão do IFEval é exatamente "**tipo + parâmetros**" verificáveis por checador determinístico, e o artigo (§1) justifica a escolha pelo mesmo motivo do roadmap: objetividade e automação. **Atenção: o formato `instruction_id_list` + `kwargs` é do dataset no HF, não do artigo [HF/código]**; o artigo só afirma o princípio "instrução atômica verificável por programa simples e determinístico" (§1). **Afeta:** `build_golden_dataset.format_examples` (gravar a spec) e `evaluators.format_validator` (ler a spec, não a pergunta).
**O que adotaríamos:** (a) uma **lista** de requisitos atômicos por exemplo (como `instruction_id_list`), cada um com seu checador registrado por id (ex.: `table:columns`, `table:row_count`, `json:exact_keys`, `bullets:count`, `bullets:max_words`, `csv:header`, `sentences:count+final_citation`); (b) um **registro id→checador**, para que a pergunta possa ser reescrita (inclusive pelo eixo persona/ADR-004 e `regenerated: v2`) sem quebrar a validação; (c) reportar **por requisito** além do total do exemplo.
**O que NÃO adotaríamos:** o esquema "união" de `kwargs` (todos os campos possíveis, quase todos `None` [HF/código]). Para nós é melhor `params` específico por tipo (`{"id":"bullets:count","n":3}`), mais legível e validável.

**R2 — Prompt-level vs instruction-level: reportar os dois, com o prompt-level como gate. APOIA (adaptação minha).**
Nossas perguntas têm 1–2 requisitos; hoje `format_validator` devolve um único `format_valid` por exemplo, que equivale ao **prompt-level strict** (tudo ou nada). Com a lista de requisitos de R1, passaria a ser possível o **instruction-level** (fração de requisitos cumpridos), que dá diagnóstico fino (a tabela estava certa mas com 2 linhas em vez de 3). **Afeta:** `format_validator` (retornar também a fração por requisito) e o relatório do slice `format`. **Adotaríamos:** prompt-level como número principal (é o que o usuário sente: o formato está certo ou não) e inst-level como diagnóstico. Com 1–2 requisitos por pergunta a diferença entre os dois será pequena; **a verificar** se compensa o custo antes de implementar.

**R3 — strict vs loose: o artigo DESAFIA nosso desenho "só strict".**
O IFEval mostra que o strict gera falso-negativo atribuído ao modelo (§2.2) e que isso é um problema prático. O nosso validator tem o mesmo risco: `_check_json` usa `json.loads(answer)` na resposta inteira, então "Here is the JSON: {...}" ou uma cerca ```` ```json ```` reprova; `_check_markdown_table` exige que a primeira linha comece com `|`, então uma frase de introdução antes da tabela reprova; `_check_csv` idem. Em parte isso é **desejável** (a pergunta diz "No other text") e em parte é o falso-negativo que o IFEval descreve. **Afeta:** `format_validator` e o slice `format`. **Adotaríamos:** uma variante loose **limitada e explícita** por tipo (aceitar cerca de código; aceitar 1 linha de introdução antes da tabela quando a pergunta **não** disse "No other text"), **reportando strict e loose lado a lado** (o IFEval faz isso e justifica por causa do falso-positivo, §2.2). **Não adotaríamos** loose como substituto do strict nem como gate único.

**R4 — As 8 transformações cegas: NÃO adotaríamos. DESAFIA copiar o mecanismo.**
Remover **a primeira ou a última linha** de qualquer resposta (transformações 2 e 3, §2.2) seria destrutivo nos nossos formatos: a primeira linha de uma tabela markdown é o **cabeçalho** (as colunas pedidas), a primeira linha de um CSV é o **header**, e a última linha da pergunta com citação ("terminando com `[2609.xxxxxv1]`") é a **linha de citação**. Removê-las poderia (i) fazer uma resposta errada passar (falso-positivo: sem o cabeçalho, uma tabela quebrada pode "parecer" correta) ou (ii) fazer uma resposta certa falhar. O artigo reconhece esse tipo de falso-positivo (§2.2) para contagem de palavras. **Afeta:** qualquer desenho de modo loose para `format_validator`. **Alternativa:** normalização **específica por tipo** (tirar cerca ```` ``` ````, tirar `**`, aceitar preâmbulo de 1 linha *só* se a resposta restante parseia).

**R5 — N=10 no slice `format`: o artigo NÃO ajuda e o dataset dele é ~54 vezes maior. NEUTRO, com ressalva importante.**
O IFEval tem 541 prompts; nós temos 10 exemplos em `format`. **[Cálculo meu]** IC de Wilson 95%: 10/10 → [72%; 100%]; 9/10 → [60%; 98%]; 8/10 → [49%; 94%]; 7/10 → [40%; 89%]. Portanto, mesmo com 10/10 não dá para afirmar mais do que "≥ ~72%" com 95% de confiança; e uma diferença de 1–2 acertos entre duas versões do sistema **não é distinguível de ruído**. O artigo não discute amostra pequena (usa 541 sem IC). **Afeta:** o gate do slice `format` e o `test_format_golds_dogfood_all_pass` (10/10 em ouro, que valida os *checadores*, não o sistema). **Adotaríamos:** reportar `k/N` e o IC em vez de só a porcentagem; tratar o slice `format` como **teste de regressão** (qualquer falha investigada à mão), não como estimativa de taxa. O ganho de passar para requisitos atômicos (R1/R2) é também estatístico: mais itens (ex.: ~15–20 requisitos em 10 perguntas) para a versão inst-level, **mas esses requisitos não são independentes** (vêm das mesmas perguntas), então o IC não encolhe tanto quanto o N sugere (minha interpretação).

**R6 — Código determinístico em vez de LLM-juiz para formato. APOIA.**
O artigo defende checagem por código contra a "avaliação baseada em modelo" por causa do viés/limite do avaliador (§1). Para formato (FP5 do Barnett) isso bate com a nossa escolha: zero custo de LLM e resultado reprodutível. É **neutro** para as métricas que exigem julgamento semântico (fidelidade, completude, especificidade), em que não existe equivalente verificável. **Afeta:** `format_validator` (manter determinístico); P11 (juiz≠gerador) não muda.

**R7 — Fallback de formato desconhecido (`score 0` sem `key`). APOIA a correção do P10.**
No IFEval, uma instrução só entra na conta se houver um checador para o seu id (por construção, Tabela 1). O nosso fallback pontua **0** quando não reconhece o formato, o que contamina o slice. O IFEval não trata esse caso (todo prompt tem checador), então o apoio é só por analogia: com spec estruturada, spec ausente ⇒ `None` (pulado), como propõe o P10. **Afeta:** `format_validator` (linha `return {"score": 0, ...}`).

**O que adotaríamos:** (1) `format_spec` = lista de requisitos atômicos com id + parâmetros, e registro id→checador; (2) reportar prompt-level (principal) e inst-level (diagnóstico); (3) strict como gate, loose por tipo apenas como diagnóstico, sempre lado a lado; (4) `k/N` com IC no slice `format`; (5) manter checagem por código.
**O que NÃO adotaríamos:** (1) as 8 transformações genéricas (R4); (2) o esquema "união" de `kwargs` com campos `None` (R1); (3) os números do IFEval (76,89 etc.) como referência para o nosso sistema: tarefa, domínio e modelos diferentes; (4) tratar o loose como substituto do strict.

## 7. Citações úteis (trechos literais curtos)

1. "very few instructions are 100% verifiable objectively and automatically" — §1 (Introdução).
2. "Although this loose instruction-following verification process reduces false negatives, it is likely to introduce false positives." — §2.2 (IFEval Metrics).
3. "atomic instructions for which one can use a simple, interpretable, and deterministic program to verify" — §1 (Introdução).
