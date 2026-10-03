# Zheng et al. 2023 — LLM-as-a-Judge, MT-Bench e Chatbot Arena

- **Referência completa e link:** Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E. P.; Zhang, H.; Gonzalez, J. E.; Stoica, I. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023, Datasets and Benchmarks Track. arXiv:2306.05685 (v4, 24/12/2023). <https://arxiv.org/abs/2306.05685> · código/dados: <https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge>
- **Confiança da leitura:** **integral.** Texto completo do PDF v4 (corpo + Apêndices A–F, incluindo prompts das Figs. 5–10 e Tabelas 9–15). Figuras (curvas de win-rate) lidas só pelas legendas e pelo texto que as comenta. Localização: seção/tabela; página só quando clara.
- **Convenção desta ficha:** **[Artigo]** = o que o paper afirma/mede. **[Interpretação]** = leitura minha, não do paper.

## 1. Problema que o artigo ataca

Benchmarks clássicos (MMLU, HELM etc.) medem conhecimento em perguntas fechadas e **não distinguem** modelos alinhados (RLHF/instruction-tuning) de modelos-base, embora humanos os prefiram claramente (§1, Fig. 1, Tab. 8). Avaliar preferência humana em perguntas abertas é caro e lento; métricas de sobreposição (BLEU/ROUGE) "são ineficazes" para respostas abertas sem gabarito (§3). A pergunta: **um LLM forte pode substituir o humano como avaliador de chatbots, e com que vieses?** O artigo estuda vieses (posição, verbosidade, auto-enhancement, raciocínio limitado), mitigações, e mede o **acordo** juiz×humano em dois benchmarks novos (MT-bench e Chatbot Arena).

## 2. Método (passo a passo)

**2.1 Dados novos (§2, §4.1, Apêndice C)**
- **MT-bench:** 80 perguntas multi-turno (8 categorias × 10: writing, roleplay, extraction, reasoning, math, coding, STEM, humanities), escritas à mão. 6 modelos respondem (GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, LLaMA-13B). **58 anotadores "expert"** (maioria pós-graduandos, US$ 20 por 20 questões), cada um julga ≥20 questões → ~3K votos.
- **Chatbot Arena:** votação anônima em batalhas; ~30K votos em 1 mês; amostra aleatória de **3K votos single-turn** (2.114 IPs únicos) para o estudo de acordo.

**2.2 Três modos de juiz (§3.1)**
- **Pairwise comparison** (o juiz vê pergunta + duas respostas e diz A, B ou empate; prompt Fig. 5).
- **Single-answer grading** (nota 1–10 para uma resposta; "explicação curta primeiro, depois `[[rating]]`"; prompt Fig. 6).
- **Reference-guided grading** (o juiz recebe uma solução de referência; prompt Fig. 8).
- Trade-offs [Artigo]: pairwise cresce quadraticamente com o nº de modelos; single-answer "pode não discernir diferenças sutis" e as notas absolutas "flutuam mais" se o juiz muda.

**2.3 Como cada viés foi medido (§3.3 e Apêndice D.1)**
- **Viés de posição** (preferir a 1ª ou a 2ª resposta, independentemente do conteúdo). Construção: para cada pergunta de 1º turno do MT-bench, duas respostas *quase idênticas* (GPT-3.5 chamado 2× com T=0,7); cada juiz avalia nas duas ordens. **Consistência** = % de casos em que o juiz dá o mesmo veredito ao trocar a ordem (mede estabilidade; 100% = sem viés de posição). Também "Biased toward first/second" e "Error" (formato inválido). Prompt "rename" renomeia "Assistant A/B" para separar viés de posição de viés de nome.
- **Viés de verbosidade** (preferir resposta mais longa mesmo sem ser melhor). Ataque "**repetitive list**": 23 respostas do MT-bench que contêm lista numerada; o GPT-4 reescreve a lista sem acrescentar informação e o resultado é *prefixado* à lista original (5 itens → 10). O ataque "funciona" se o juiz julga a nova versão melhor. **Taxa de falha** = % de ataques bem-sucedidos. Calibração: juízes dão empate para duas respostas idênticas.
- **Auto-enhancement** (preferir respostas do próprio modelo). Medido apenas de forma **estatística**: win-rate (sem empates) de 6 modelos sob cada juiz e sob humanos (Fig. 3b).
- **Capacidade limitada em matemática/raciocínio:** 10 questões de matemática, LLaMA-13B vs Vicuna-13B, posições trocadas (20 casos). **Falha** = o GPT-4 diz que uma resposta *incorreta* está correta (Tab. 4).

**2.4 Mitigações (§3.4)**
- *Swapping positions:* chamar o juiz 2× trocando a ordem; só declarar vitória se a mesma resposta ganhar nas duas ordens, senão empate (abordagem "conservadora", usada nos experimentos). Alternativa "agressiva": posição aleatória, que funciona em escala.
- *Few-shot judge:* 3 exemplos (A melhor / B melhor / empate) gerados com GPT-3.5, Vicuna e julgados pelo GPT-4.
- *CoT judge:* o juiz resolve a questão por conta própria antes de julgar (prompt Fig. 7).
- *Reference-guided judge:* o juiz gera sua resposta de forma independente e ela é inserida como referência no prompt (Fig. 8).
- *Fine-tuning de juiz:* Vicuna-13B ajustado com 20K votos da Arena (Apêndice F).
- *Multi-turn:* mostrar a conversa completa num único prompt (e não um prompt por turno) evita erros de referência (§3.5, Fig. 16).

**2.5 Protocolo de acordo juiz×humano (§4.1, Apêndice D.3)**
- **Acordo** = probabilidade de dois indivíduos *distintos* de cada tipo (ex.: GPT-4 e um humano sorteado) concordarem no voto de uma questão sorteada (mede quanto o juiz coincide com um humano típico).
- **S1:** inclui votos não-empate, empate e "inconsistente" (por viés de posição, contado como empate); **acordo de dois juízes aleatórios R = 33%**.
- **S2:** apenas votos não-empate; **R = 50%**.
- Também: **human-majority** (voto majoritário dos humanos; teto de acordo do GPT-4 é o acordo humano-majoritário×humano; empate de votos conta ½) e *win-rate* médio por modelo.
- [Artigo, App. D.3] o acordo humano×humano pode ser subestimativa porque três humanos "A, A, B" concordam entre si só em 1/3 dos pares.

## 3. Principais contribuições

1. Estudo sistemático de LLM-as-a-judge, com taxonomia de 3 modos (pairwise / single / reference-guided) e quatro limitações nomeadas.
2. **MT-bench** (80 questões multi-turno, 3K votos de experts) e **dados da Arena** (30K conversas com preferência humana) públicos.
3. Evidência de que **GPT-4 atinge >80% de acordo** com humanos, no nível do acordo humano×humano (S2).
4. Mitigações simples e testadas (troca de posição, few-shot, CoT, reference-guided) e um juiz open-source ajustado (Vicuna-13B).
5. Argumento por avaliação **híbrida**: benchmarks de capacidade (MMLU etc.) + benchmarks de preferência com LLM-juiz (§5, Tab. 8: "nenhum benchmark único determina a qualidade do modelo").

## 4. Resultados-chave

**Viés de posição (Tab. 2, §3.3; 80 casos)**

| Juiz | Prompt | Consistência | Favorece 1ª | Favorece 2ª | Erro |
|---|---|---|---|---|---|
| Claude-v1 | default | 23,8% | 75,0% | 0,0% | 1,2% |
| Claude-v1 | rename | 56,2% | 11,2% | 28,7% | 3,8% |
| GPT-3.5 | default | 46,2% | 50,0% | 1,2% | 2,5% |
| GPT-3.5 | rename | 51,2% | 38,8% | 6,2% | 3,8% |
| GPT-4 | default | 65,0% | 30,0% | 5,0% | 0,0% |
| GPT-4 | rename | 66,2% | 28,7% | 5,0% | 0,0% |

- "Só o GPT-4 produz resultados consistentes em mais de 60% dos casos" (§3.3). Claude-v1 mostra também **viés de nome** (favorece "Assistant A", Tab. 2 rename).
- Outros prompts (Apêndice D.1, Tab. 9): "score" (duas notas absolutas) eleva a consistência do GPT-3.5 (55,0%) mas **reduz** a do Claude-v1 (20,0%) e a do GPT-4 (51,2%); "short" (sem instruções anti-viés): GPT-4 62,5%.
- Por categoria (Tab. 10, GPT-4 default): consistência **menor** em writing (42,0%), STEM (44,0%), humanities (36,0%); **maior** em math e coding (86,0%). Por par de modelos (Tab. 11): quase some quando os modelos diferem muito (GPT-3.5 vs LLaMA-13B: 98,8%) e é maior quando são próximos (GPT-3.5 vs Claude-v1: 67,5%).
- Few-shot (Tab. 12): consistência Claude-v1 23,8→63,7%; GPT-3.5 46,2→55,0% (o viés migra da 1ª para a 2ª posição: 28,7% favorecem a 2ª); **GPT-4 65,0→77,5%**. Mas "alta consistência pode não implicar alta acurácia", prompts longos tornam a API ~**4× mais cara** (§3.4) e o acordo com humanos do GPT-4 few-shot ficou **similar** ao zero-shot (Apêndice D.2).

**Viés de verbosidade (Tab. 3):** taxa de falha sob o ataque *repetitive list* em 23 respostas — Claude-v1 **91,3%**, GPT-3.5 **91,3%**, GPT-4 **8,7%**.

**Auto-enhancement (§3.3, Fig. 3b):** GPT-4 favorece a si mesmo com win-rate **10% maior** que o dos humanos; Claude-v1, **25% maior**; **GPT-3.5 não se favorece**. Os próprios autores: "devido a poucos dados e pequenas diferenças, nosso estudo não consegue determinar se os modelos exibem viés de auto-enhancement" e admitem que um estudo controlado é difícil.

**Matemática/raciocínio (Tab. 4):** taxa de falha com 10 questões × 2 posições = 20 julgamentos: **default 14/20 (70%) → CoT 6/20 (30%) → reference-guided 3/20 (15%)**. O texto (§3.4) cita "de 70% para 15%". Mesmo com CoT, o juiz repete o erro das respostas apresentadas (Fig. 15): "pode continuar sendo enganado pelo contexto".

**Acordo juiz×humano (Tab. 5, MT-bench, 1º turno; Tab. 13 no Apêndice D.3)**

| Par | S1 (R=33%) | S2 (R=50%) |
|---|---|---|
| GPT-4 pairwise × humano | 66% | **85%** |
| GPT-4 single × humano | 60% | **85%** |
| humano × humano | 63% | **81%** |
| GPT-4 pairwise × GPT-4 single | 70% | 97% |

2º turno (Tab. 5b): GPT-4 pair×humano S2 85%; humano×humano S2 **82%**. Arena (Tab. 6): GPT-4×humano S1 **64%**, S2 **87%**; humano×humano não é medido na Arena. Contra o voto majoritário dos humanos (Tab. 13): GPT-4 pair S2 85% (1º e 2º turno). O acordo **sobe de ~70% para ~100%** conforme aumenta a diferença de win-rate entre os modelos comparados (Fig. 2, §4.2). Em desacordos, humanos acharam o julgamento do GPT-4 razoável em **75%** e mudaram seu voto em **34%** (§4.2).

**Observação:** Tab. 6 também mostra que GPT-4 produz muito mais votos não-empate que GPT-3.5/Claude ("mais afirmativo, menos afetado por viés de posição"), com acordo não-empate parecido.

**Vicuna-13B como juiz (Apêndice F, Tab. 15):** zero-shot, consistência 11,2–16,2% e erro de formato 22,5–78,8%; ajustado em 20K votos da Arena (3 classes A/B/empate): consistência **65,0%**, erro 0%, acordo **56,8%** (S1) e **85,5%** (S2), vs GPT-4 66% e 87%.

**Benchmarks complementares (Tab. 8):** o MT-bench (nota GPT-4 single-answer, escala 10) separa Vicuna-7B-selected (5,95; MMLU 37,3) de LLaMA-13B (2,61; MMLU 47,0), mostrando que MMLU e preferência medem coisas distintas.

## 5. Limitações

**Do autor (§6, §3.3, §3.4):**
- Foca em *helpfulness*, quase não trata de segurança/honestidade; várias dimensões (acurácia, relevância, criatividade) ficam **fundidas numa nota única**.
- Auto-enhancement **inconclusivo** (poucos dados, sem experimento controlado).
- Few-shot pode introduzir novos vieses e custa ~4×.
- Origem do viés de posição só conjecturada (dados de treino ou arquitetura causal).
- Soluções propostas são "preliminares".

**Que eu identifico [Interpretação]:**
- **Teste de posição artificial:** pares *quase idênticos* de uma única família (GPT-3.5 a T=0,7) — o próprio texto diz que o teste é "desafiador" e que o viés é menor quando há diferença grande (Tab. 11). A consistência de 23,8%–65,0% **não é** uma estimativa do viés em pares reais.
- **Amostras pequenas:** 80 casos em Tab. 2; 23 em Tab. 3; **10 questões** (20 julgamentos) em Tab. 4. Sem intervalos de confiança.
- **Métrica de acordo sem correção formal por prevalência**, apenas a referência "R = 33% / 50%". Não há κ de Cohen. O acordo S2 exclui empates e inconsistências, o que **infla** o número (a Tab. 5 mostra G4-Pair×humano S1 = 66%, bem abaixo dos 85%).
- A intro cita "§4.2, Table 4" para o acordo, mas a tabela é a **5** (descuido editorial).
- Referência para matemática foi **gerada pelo próprio juiz** (reference-guided), logo ainda herda erros do juiz; com gabarito humano o efeito pode ser maior.
- Anotadores humanos são estudantes de pós-graduação (viés de amostra) e, na Arena, votantes anônimos não controlados.
- Juízes de 2023 (GPT-4, GPT-3.5, Claude-v1); os números **não** se transferem a `gpt-4o-mini`.

## 6. Reflexões ancoradas no NOSSO projeto

**R1 — Juiz = gerador (P11; `config.py`, 6 juízes de `ckpt-0.6-plan.md` §4).** Posição do artigo: **(iii) NEUTRO-cauteloso**, com uma pitada de (i). Os números (+10% GPT-4, +25% Claude-v1) *sugerem* auto-enhancement, mas o próprio artigo diz que não consegue concluir, e o GPT-3.5 não se favorece. Portanto **Zheng sozinho não sustenta nem derruba** a separação juiz/gerador; quem sustenta é Wataoka (ver ficha dedicada). **O que adotaríamos:** `JUDGE_MODEL` separado, registrado na metadata (já em P11). **O que NÃO faríamos:** citar "GPT-4 favorece a si em 10%" como prova de viés (é afirmação do artigo com ressalva do autor).

**R2 — Calibração contra humano na etapa 0.8 (κ de Cohen, `ckpt-0.6-plan.md` §6; glossário "κ").** Posição: **(i) APOIA** calibrar. Zheng não prescreve κ; usa **acordo S1/S2 contra baseline aleatório** e **humano×humano como teto**. **Adotaríamos:** (a) reportar acordo bruto *e* κ por juiz (nossos rótulos são desbalanceados: ~25 `should_abstain` vs ~90 responsíveis, P8, onde acordo bruto engana); (b) incluir um **baseline trivial** (juiz que sempre responde "pass") ao lado do aleatório; (c) medir o **teto humano×humano** (se houver 2 anotadores em alguma amostra) — sem ele não dá para dizer se κ=0,6 é "bom". **Alerta [Interpretação]:** o roadmap fala em calibrar contra "as 100 labels humanas"; se essas labels forem da *revisão do golden set* (qualidade de pergunta/gabarito) e não **julgamentos de respostas geradas**, o protocolo de Zheng exige rotular uma amostra de **saídas do sistema**. Isso precisa ser confirmado antes do 0.8.

**R3 — Formato de saída dos juízes `{"score": ..., "reason": ...}` (`ckpt-0.6-plan.md` §4; todos os 6 juízes).** Posição: **(ii) DESAFIA em parte** o desenho. Os prompts de Zheng (Figs. 5–7) mandam **explicar antes de dar a nota** ("Begin your evaluation by providing a short explanation… then rate"). Num JSON com `score` antes de `reason`, o modelo decide o veredito *antes* de raciocinar, anulando o efeito de CoT. **[Interpretação]:** colocar `reason` **antes** de `score` no schema. **Ressalva:** Wang et al. 2023 (ficha `qa-eval-judge-vs-f1.md`, Tab. 7) acharam que "dar razões" *piorou* o GPT-3.5 em QA-eval, e CoT melhorou só nas respostas longas — o efeito depende do juiz/tarefa; **testar na rodada de sensibilidade**, não assumir.

**R4 — Reference-guided e o juiz de qualidade vs gabarito (P9; `correctness_judge` planejado; `completeness_judge` em multi-doc).** Posição: **(i) APOIA**. Tab. 4: dar a referência derruba a falha de 70% para 15%. Nosso caso é *mais favorável* que o do artigo (o gabarito é revisado por humano, 100/100, e não gerado pelo juiz) — mas os `answerable`/`persona` são *synthetic-by-construction* (ADR-001/003), então o gabarito reflete a formulação do LLM que leu o chunk; juiz com gabarito pode punir respostas corretas e diferentes. **Adotaríamos:** passar gold answer ao juiz de `completeness`/`correctness`; **não** passar ao `faithfulness` (ref-free por desenho, contra `retrieved_context`).

**R5 — Verbosidade (`completeness_judge`, `specificity_judge`, P9).** Posição: **(ii) DESAFIA** o plano de usar um modelo pequeno. No ataque *repetitive list*, Claude-v1 e GPT-3.5 falharam 91,3%; só o GPT-4 resistiu (8,7%). `gpt-4o-mini` é o tipo de modelo de capacidade menor para o qual a literatura mostra fragilidade (mas **não foi testado** aqui). Nosso prompt limita a resposta a 3 frases, o que ajuda, porém completeness/specificity são exatamente os critérios que "recompensam" mais detalhe. **Adotaríamos:** um teste de sanidade barato inspirado em §3.3 — pegar ~10 respostas, acrescentar uma paráfrase redundante e verificar que a nota **não sobe**. **Não adotaríamos** o ataque exato (listas numeradas não são nosso formato).

**R6 — Viés de posição e juízes pairwise (curso `module_2/pairwise_experiments.ipynb`, gates ≥15%/≥10% do plano §2).** Posição: **(iii) NEUTRO** para os 6 juízes planejados (todos **single-answer**: a posição não se aplica). **Se** formos comparar baseline×variante com juiz *pairwise* (como no curso, com gpt-4o), o artigo manda **trocar a ordem e exigir consistência** (§3.4) — o curso não faz isso (conferir). Sem isso, a diferença entre braços pode ser artefato de posição (favorece a 1ª em 30–75% dos casos dependendo do juiz, Tab. 2). Single-answer é suficiente para o nosso desenho; Zheng diz que ele "casa bem" com humanos (Tab. 5) mas "flutua mais" se o juiz mudar — **relevante para a rodada de sensibilidade com 2 juízes** (P11): notas absolutas de dois juízes **não são comparáveis em valor**, só em ranking/concordância.

**R7 — Dimensões separadas (6 juízes).** Posição: **(i) APOIA**. A limitação nº 1 do autor é a nota única que mistura dimensões; nosso catálogo (faithfulness, citation, completeness, abstention, specificity, relevance) já as separa.

**R8 — Juiz enganado pelo contexto (faithfulness/citation).** Posição: **(iii) NEUTRO / alerta.** §3.3: o GPT-4 resolve a questão sozinho mas erra ao *julgar* quando as respostas apresentadas contêm erros convincentes. **[Interpretação]:** no `faithfulness_judge`, uma resposta alucinada porém fluente pode ser aceita; por isso a validação deve incluir casos negativos (já previsto em 0.6.3: "pass e fail").

**Resumo — o que adotaríamos:** `JUDGE_MODEL` ≠ `GENERATION_MODEL`; gold no prompt de correctness/completeness; `reason` antes de `score` (a validar); acordo + κ + baseline trivial; teto humano×humano; troca de ordem se algum dia usarmos pairwise. **O que NÃO adotaríamos:** few-shot como padrão (4× o custo, sem ganho de acordo, Apêndice D.2); fine-tuning de juiz (Apêndice F; escala errada para 100 exemplos); a nota "10 pontos" como nota absoluta comparável entre juízes diferentes; a premissa de "GPT-4 ≈ humano" para `gpt-4o-mini`.

## 7. Citações úteis

1. "GPT-4 favors itself with a 10% higher win rate; Claude-v1 favors itself with a 25% higher win rate. However, they also favor other models and GPT-3.5 does not favor itself." — §3.3 (Self-enhancement bias), p. 5.
2. "Due to limited data and small differences, our study cannot determine whether the models exhibit a self-enhancement bias." — §3.3, p. 5.
3. "A conservative approach is to call a judge twice by swapping the order of two answers and only declare a win when an answer is preferred in both orders." — §3.4 (Swapping positions), p. 6.
