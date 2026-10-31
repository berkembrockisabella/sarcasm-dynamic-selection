# Seleção Dinâmica de Classificadores para Detecção Multimodal de Sarcasmo

Projeto da disciplina Experiência Criativa: Projeto Transformador II (PUCPR).

Detecta sarcasmo em falas de séries de comédia usando **texto**, **contexto** (falas anteriores), **áudio** e **rosto**. Investiga se a **incongruência emocional entre essas modalidades** ajuda a escolher, fala a fala, qual classificador deve decidir.

- **Dataset:** [MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus) — 1202 falas (601 sarcásticas, 601 não), 27 falantes.
- **Protocolo final:** 10 splits por falante, testes estatísticos entre splits, ablação do roteamento, sensibilidade a parâmetros, análise de erros e custo computacional.

---

## Perguntas de pesquisa

| | Pergunta | Resposta nos 10 splits |
|---|---|---|
| **Q1** | A seleção dinâmica supera classificadores estáticos multimodais? | **Não.** O SVM estático e o voto dos 3 estáticos tiveram o maior F1-macro (80,1%); o melhor método dinâmico (KNORA-U) teve 79,2%. Nenhum par de métodos difere pelo pós-teste de Nemenyi. |
| **Q2** | Rotear a busca de vizinhos pela incongruência (grupo A/B) melhora a seleção dinâmica? | **Não de forma significativa.** Maior ganho no KNORA-E: +0,5 ponto contra sem roteamento e +0,4 contra roteamento aleatório (p ajustado = 1,0). |
| **Q3** | A seleção por competência local se compara a uma combinação aprendida (Mixture of Experts)? | **Empate.** Soft-MoE 78,3%, igual a OLA e KNORA-E, abaixo do KNORA-U (diferença não significativa). |

---

## Visão geral do sistema

```
                    MUStARD++ (1202 falas)
                             |
      10 splits por falante: TREINO ~70% | DSEL ~15% | TESTE ~15%
                             |
          +------------------+-------------------+
          |                                      |
   Representacoes                         Valencia emocional
   (BART, wav2vec2, rosto)                de cada modalidade
          |                                      |
   Selecao da pool (CV no TREINO)         Grupo A (as 4 concordam)
   45 candidatos -> 7 especialistas       ou B (pelo menos uma difere)
          |                                      |
   Pool congelada (treinada no TREINO)           |
          |                                      |
          +-----------+--------------+-----------+
                      |              |
         Selecao dinamica         Soft-MoE
         (DSEL como consulta)     (gate MLP treinado no DSEL)
         OLA | KNORA-E | KNORA-U
         sem | A/B | aleatorio
                      |              |
                      +------+-------+
                             |
          Avaliacao no TESTE x estaticos (SVM, RL, RF, voto)
                             |
          Agregacao dos 10 splits + testes estatisticos
```

### Papel de cada conjunto

| Conjunto | Papel |
|---|---|
| **Treino** | Escolher e treinar os classificadores. Depois eles ficam congelados. |
| **DSEL** | Base de consulta da seleção dinâmica e treino do gate do MoE. Fica separado do treino porque os classificadores acertariam mais em falas que já viram. |
| **Teste** | Só mede o resultado final. Nenhum parâmetro é escolhido com ele. |

A divisão é sempre **por falante**: todas as falas de um personagem ficam no mesmo conjunto.

---

## 1. Splits

O experimento final usa **10 splits** (`data/splits/split_00` a `split_09`), gerados por `src/data/create_splits_repetidos.py`.

- **Seeds percorridos em ordem** (0, 1, 2, …); ficam os 10 primeiros que passam no critério de aceitação: DSEL e teste com **120 a 240 falas** e **40% a 60% de sarcasmo** cada.
- **Por que o critério:** o MUStARD++ tem só 27 falantes, muito desiguais (CHANDLER sozinho tem 157 falas, 76% sarcásticas). Sem controle, os seeds 0 a 9 geravam DSEL de 49 a 365 falas e teste com 27% a 60% de sarcasmo.
- **Seeds aceitos:** 8, 18, 30, 45, 46, 48, 70, 96, 100 e 111 (112 seeds testados).

| | Treino | DSEL | Teste |
|---|---|---|---|
| Falas por split | 744 a 880 | 136 a 230 | 126 a 228 |
| Falantes por split | 18 | 4 | 5 |
| Grupo A por split | 95 a 135 | 16 a 38 | 16 a 49 |

O split único original (`data/splits/train.csv`, `dsel.csv`, `test.csv`, escolhido como o mais balanceado entre 1000 seeds por `create_splits.py`) continua no repositório. Sem `--split`, todos os scripts rodam nele.

## 2. Representações (entrada dos classificadores)

| Modalidade | Representação | Dimensão |
|---|---|---|
| Texto (fala) | `facebook/bart-base`, média dos estados ocultos | 768 |
| Contexto (falas anteriores) | `facebook/bart-base`, média dos estados ocultos | 768 |
| Áudio | `facebook/wav2vec2-base-960h`, média no tempo | 768 |
| Rosto | Média e desvio das 7 emoções faciais, nº de rostos e indicador de ausência | 16 |

## 3. Valência e grupo de incongruência

Para cada modalidade, um classificador de emoções pré-treinado dá a distribuição sobre as emoções. A **valência** é `P(positivas) − P(negativas)`, de −1 a +1. A emoção predominante define a polaridade:

- **positiva:** happy
- **negativa:** angry, disgust, fear, sad
- **neutra:** neutral, surprise ou sem sinal (entrada vazia ou rosto ausente)

| Modalidade | Classificador de emoções |
|---|---|
| Texto e contexto | `Tanneru/Emotion-Classification-DeBERTa-v3-Large` |
| Áudio | `iic/emotion2vec_plus_large` |
| Rosto | Emoções faciais por quadro (`visual_emotions.csv`; os scripts de extração não estão no repositório) |

- **Grupo A:** as 4 modalidades têm a mesma polaridade (concordam).
- **Grupo B:** pelo menos uma difere (discordam).

A incongruência **não entra como feature** dos classificadores. Ela só define o grupo usado no roteamento. Detalhes em [`src/incongruence/README.md`](src/incongruence/README.md).

## 4. Pool de 7 especialistas

Refeita em cada split, usando só o treino ([`src/pool_selection/README.md`](src/pool_selection/README.md)):

1. **45 candidatos:** 15 combinações de modalidades × 3 classificadores (SVM, regressão logística, random forest).
2. **Validação cruzada** estratificada por falante (5 folds) com busca em grade de hiperparâmetros e F1-macro como critério.
3. **Escolha gulosa** entre os 15 melhores: `0,7 × desempenho normalizado + 0,3 × discordância média` com os já escolhidos, até 7 membros.
4. **Treino final** dos 7 no treino inteiro ([`src/pool_training/README.md`](src/pool_training/README.md)).

Composição dos 70 membros (7 por split × 10):

| Modalidade | % dos membros | Classificador | % dos membros |
|---|---|---|---|
| Contexto | 98,6 | SVM | 52,9 |
| Áudio | 64,3 | Regressão logística | 31,4 |
| Texto | 61,4 | Random forest | 15,7 |
| Rosto | 52,9 | | |

## 5. Seleção dinâmica

Para cada fala de teste, procuram-se os **k = 7 vizinhos** mais parecidos no DSEL e verifica-se quais membros acertaram esses vizinhos ([`src/selection/README.md`](src/selection/README.md)). O espaço de vizinhança concatena as 4 modalidades, cada uma padronizada (ajustada no treino) e normalizada (L2).

| Método | Quem decide |
|---|---|
| **OLA** | O membro que mais acertou os vizinhos |
| **KNORA-E** | Votam só os que acertaram **todos** os vizinhos (se nenhum, reduz a vizinhança) |
| **KNORA-U** | Todos votam, com peso = nº de vizinhos acertados |

| Roteamento | Onde procura os vizinhos | O que testa |
|---|---|---|
| **Sem roteamento** | No DSEL inteiro | Linha de base |
| **A/B** | Só no DSEL do mesmo grupo da fala de teste | Efeito da incongruência |
| **Aleatório** | Em grupos sorteados do DSEL com o mesmo tamanho de A e B (10 sorteios; vale a média) | Separa a incongruência do simples fato de restringir a vizinhança |

## 6. Soft-MoE

Combina as probabilidades dos mesmos 7 membros com pesos dados por um **gate** ([`src/moe/README.md`](src/moe/README.md)):

- Gate: MLP com entrada das 4 modalidades (2320 dimensões), camada de 32 neurônios, ReLU, dropout 0,2 e softmax sobre os 7 membros (cerca de 74,5 mil parâmetros).
- Treinado no DSEL com entropia cruzada binária, Adam (taxa 0,001, weight decay 0,0001), lotes de 32, até 200 épocas e paciência de 20.
- O gate não recebe a incongruência.

## 7. Classificadores estáticos (referência)

Treinados uma vez no treino e aplicados igual a todas as falas, com **as 4 modalidades juntas**: SVM, regressão logística, random forest (hiperparâmetros da validação cruzada da pool) e o voto dos 3.

---

## Protocolo experimental

| | Experimento | Como | Estatística |
|---|---|---|---|
| **E0** | Base comum | 10 splits; k = 7 e peso de desempenho 0,7 fixados antes de olhar o teste | Média ± desvio, pior e melhor split |
| **E1** | Comparação principal (Q1, Q3) | Estáticos, pior/melhor membro, voto da pool, OLA/KNORA-E/KNORA-U sem roteamento, Soft-MoE, oráculo | Friedman + pós-teste de Nemenyi (Demšar, 2006) |
| **E2** | Ablação do roteamento (Q2) | Sem roteamento × A/B × aleatório, nos 3 métodos | Wilcoxon pareado + correção de Holm (6 testes) |
| **E3** | Sensibilidade | k ∈ {3, 5, 7, 11, 15}; peso do desempenho de 0,0 a 1,0 (avaliado no DSEL) | Descritiva |
| **E4** | Análise de erros | Erro do seletor (algum membro acertaria) × erro da pool (nenhum acerta), por grupo e classe; diversidade da pool | Descritiva |
| **E5** | Custo computacional | Tempo, CPU e memória por etapa; nº de treinos e de parâmetros | Descritiva |

O McNemar com correção de Holm e o bootstrap (IC 95%) por split (`src/statistical_comparison.py`) ficam como complemento.

---

## Resultados (10 splits, teste)

Valores em %, média ± desvio entre splits. Todos os números vêm de `data/processed/final/`.

### E1 · Comparação principal

| Método | F1-macro | Pior–melhor split | Acurácia | Rank médio |
|---|---|---|---|---|
| Estático voto dos 3 | 80,1 ± 5,6 | 69,3–85,4 | 80,4 | 3,80 |
| Estático SVM | 80,1 ± 5,8 | 67,8–85,4 | 80,3 | 4,50 |
| KNORA-U sem roteamento | 79,2 ± 6,0 | 66,8–86,0 | 79,5 | 4,40 |
| Voto da pool | 79,2 ± 5,9 | 66,8–86,0 | 79,5 | 4,75 |
| Melhor membro (pelo DSEL) | 78,6 ± 6,8 | 63,7–85,4 | 78,8 | 6,75 |
| Soft-MoE | 78,3 ± 5,3 | 67,8–86,6 | 78,6 | 5,65 |
| OLA sem roteamento | 78,2 ± 6,9 | 63,7–85,4 | 78,4 | 6,35 |
| KNORA-E sem roteamento | 78,2 ± 6,7 | 64,8–87,2 | 78,5 | 6,15 |
| Estático random forest | 77,7 ± 6,5 | 65,0–84,9 | 78,1 | 7,95 |
| Estático regressão logística | 77,4 ± 5,0 | 66,8–83,8 | 77,7 | 7,35 |
| Pior membro (pelo DSEL) | 75,8 ± 6,3 | 65,0–84,1 | 76,1 | 8,35 |
| *Oráculo (algum membro acerta)* | — | — | *91,0 ± 4,8* | — |

- **Friedman:** χ² = 21,32, p = 0,019 (11 métodos, 10 splits). **Nemenyi:** distância crítica de 4,77; a maior diferença de rank é 4,55 (voto dos 3 × pior membro), então **nenhum par é significativo**.
- **A variação entre splits (desvio de ~6 pontos) é maior que a diferença entre métodos (1 a 2 pontos).** O `split_06` é o pior split para todos os métodos.
- Os métodos dinâmicos ficam perto do melhor membro e do voto da pool (78,2 a 79,2), longe do oráculo.

### E2 · Ablação do roteamento

| Método | Comparação | Δ F1 (pp) | Vitórias / empates / derrotas | p | p (Holm) |
|---|---|---|---|---|---|
| OLA | A/B × sem roteamento | 0,0 | 6 / 1 / 3 | 0,820 | 1,0 |
| OLA | A/B × aleatório | 0,0 | 8 / 0 / 2 | 0,432 | 1,0 |
| KNORA-E | A/B × sem roteamento | +0,5 | 7 / 0 / 3 | 0,275 | 1,0 |
| KNORA-E | A/B × aleatório | +0,4 | 7 / 0 / 3 | 0,375 | 1,0 |
| KNORA-U | A/B × sem roteamento | 0,0 | 1 / 7 / 2 | 1,000 | 1,0 |
| KNORA-U | A/B × aleatório | 0,0 | 1 / 4 / 5 | 0,438 | 1,0 |

F1-macro por grupo no teste (exploratório; o grupo A tem cerca de 27 falas por split):

| Método | Grupo A: sem / A/B / aleatório | Grupo B: sem / A/B / aleatório |
|---|---|---|
| OLA | 77,7 / 80,8 / 78,5 | 78,2 / 77,7 / 78,0 |
| KNORA-E | 79,0 / 80,9 / 79,1 | 78,1 / 78,3 / 78,1 |
| KNORA-U | 81,1 / 81,2 / 80,9 | 78,8 / 78,8 / 78,9 |

No grupo A, o roteamento A/B tende a ficar 2 a 3 pontos acima no OLA e cerca de 2 pontos acima no KNORA-E. Os desvios são de 9 a 11 pontos, e essa diferença não foi testada.

### E3 · Sensibilidade

- **k:** entre 3 e 15, o F1-macro médio varia no máximo 0,7 ponto em cada método e roteamento (ex.: KNORA-U sem roteamento de 79,0 a 79,3).
- **Peso do desempenho na pool:** de 0,0 a 1,0, o F1-macro do voto da pool no DSEL fica entre 78,2 e 78,6, enquanto a discordância entre membros cai de 17,6% para 12,6%. Os 11 pesos geram de 2 a 6 pools diferentes por split.

### E4 · Análise de erros

| Método | Acerto | Erro do seletor | Erro da pool |
|---|---|---|---|
| Estático voto dos 3 | 79,9 | 10,9 | 9,2 |
| Estático SVM | 79,8 | 11,0 | 9,2 |
| KNORA-U sem roteamento | 79,1 | 11,5 | 9,5 |
| KNORA-E com roteamento | 78,6 | 12,0 | 9,5 |
| Soft-MoE | 78,2 | 12,3 | 9,5 |
| OLA sem roteamento | 78,0 | 12,5 | 9,5 |

% das 1788 falas de teste dos 10 splits. Para os estáticos, "erro da pool" significa que nenhum membro da pool acerta a fala.

- **O erro se concentra nas falas sem sarcasmo:** o erro da pool é de 15,6% nelas e de 2,9% nas sarcásticas. A pool tende a prever sarcasmo (recall de 83 a 88%, precisão de 73 a 77%).
- **Pool pouco diversa:** kappa médio entre pares de membros de 0,67 a 0,83 por split; discordância de 8,7% a 16,0%.
- **Espaço para o seletor:** cerca de 12% das falas seriam acertadas por algum membro, mas o método escolhe errado.

### E5 · Custo computacional

| Etapa (por split) | Tempo | CPU | Pico de memória |
|---|---|---|---|
| Seleção da pool (CV no treino) | 1469 s (~24,5 min) | 21027 s | 3544 MB |
| Treino da pool congelada | 7,7 s | 14,7 s | 195 MB |
| Seleção dinâmica (12 roteamentos, 3 métodos) | 52,1 s | 69,7 s | 304 MB |
| Soft-MoE (treino do gate + teste) | 8,4 s | 8,8 s | 1239 MB |
| Estáticos (reaproveitam a CV da pool) | 4,4 s | 7,3 s | 199 MB |

CPU e memória são aproximados (amostrados a cada 0,5 s, somando processos filhos).

| Abordagem | Treinos na busca de hiperparâmetros | Treinos finais | Parâmetros treináveis extras |
|---|---|---|---|
| Estáticos | 378 | 3 | 0 |
| Pool + seleção dinâmica | 5670 | 7 | 0 |
| Pool + Soft-MoE | 5670 | 7 | 74503 |

A decisão da seleção dinâmica custa cerca de 0,05 ms por fala.

---

## Limitações

- **Poucos falantes:** o MUStARD++ tem só 27 falantes, muito desiguais; mesmo com o critério de aceitação, o split pesa mais que o método.
- **Poder estatístico:** com 10 splits, os testes têm pouco poder para diferenças de 1 a 2 pontos. "Não significativo" não prova que os métodos são equivalentes.
- **Grupo A pequeno:** 16 a 49 falas no teste por split; os resultados por grupo são descritivos.
- **Pool dominada pelo contexto:** 98,6% dos membros usam o contexto, o que limita a complementaridade que a seleção dinâmica pode explorar.
- **Reprodutibilidade do split único:** ao retreinar a pool no split original, as predições da seleção dinâmica mudaram em 1 a 2 das 147 falas (ex.: KNORA-U sem roteamento de 70,5% para 69,1% de F1-macro) e o Soft-MoE passou de 70,1% para 66,4%. A definição da pool é a mesma e os estáticos deram resultados idênticos; a causa provável é o retreino da pool e do gate.
- **Parte visual incompleta:** os scripts de extração de quadros, rostos e emoção facial não estão no repositório; só o resultado `visual_emotions.csv`.

---

## Estrutura do repositório

| Caminho | Função |
|---|---|
| `src/data/` | Prepara o MUStARD++, gera o split único (`create_splits.py`) e os 10 splits (`create_splits_repetidos.py`), extrai o áudio |
| `src/features/valencia/` | Valência e emoção de cada modalidade |
| `src/features/embeddings/` | Representações (BART, wav2vec2, rosto) |
| `src/incongruence/` | Grupo A/B de cada fala |
| `src/pool_selection/` | Avaliação dos 45 candidatos, escolha dos 7 membros e sensibilidade ao peso da diversidade (`compare_diversity.py`); `config.py` lê o `SPLIT_ID` |
| `src/pool_training/` | Treino e carregamento da pool congelada |
| `src/selection/` | Seleção dinâmica com os 3 roteamentos (`ds_routing.py`) |
| `src/moe/` | Soft-MoE |
| `src/static/` | Estáticos multimodais |
| `src/statistical_comparison.py` | McNemar + Holm e bootstrap por split |
| `src/run_experiment.py` | Orquestra pool, seleção dinâmica, MoE, estáticos, sensibilidade e custos (`--split`, `--split all`) |
| `src/aggregate_splits.py` | Agrega os 10 splits: tabelas, Friedman/Nemenyi, Wilcoxon/Holm, erros e custos |
| `src/prototipo/` | Outra linha de experimentos do time (4 classificadores unimodais); não faz parte deste experimento |

### Onde ficam os resultados

| Caminho | Conteúdo |
|---|---|
| `data/results/pool/split_XX/` | CV dos candidatos, definição da pool, pool congelada e sensibilidade à diversidade |
| `data/processed/selection/split_XX/` | Predições, resumos e tempos da seleção dinâmica; `k_<K>/` para a sensibilidade ao k |
| `data/processed/moe/split_XX/`, `data/processed/static/split_XX/` | Soft-MoE e estáticos |
| `data/processed/eval/split_XX/` | Resumo consolidado e `custos.csv` do split |
| `data/processed/final/` | Resultados agregados dos 10 splits |

Sem `--split`, os mesmos arquivos ficam nas pastas sem `split_XX/` (split único).

---

## Como rodar

Pré-requisitos: Python 3.10+ e FFmpeg. Instale com `pip install -r requirements.txt` e rode a partir da raiz.

### Preparação (uma vez)

```
# dados
python src/data/prepare_mustard.py
python src/data/create_splits.py              # split unico
python src/data/create_splits_repetidos.py    # 10 splits do protocolo final
python src/data/extract_audio.py

# valencia e grupo A/B
python src/features/valencia/text_valence.py
python src/features/valencia/context_valence.py
python src/features/valencia/audio_valence.py        # retomavel: apague audio_valence.csv para recalcular
python src/features/valencia/visual_valence.py
python src/incongruence/build_incongruence.py

# representacoes
python src/features/embeddings/build_context_text.py
python src/features/embeddings/extract_text_features.py
python src/features/embeddings/extract_context_features.py
python src/features/embeddings/extract_audio_features.py
python src/features/embeddings/build_visual_features.py
```

### Experimento

```
# split unico
python -m src.run_experiment

# um split do protocolo (ex.: 00), com sensibilidade ao k e a diversidade
python -m src.run_experiment --split 00 --include-k-sweep --include-diversity

# os 10 splits (retomavel: pools ja treinadas sao reaproveitadas)
python -m src.run_experiment --split all --include-k-sweep --include-diversity

# agregacao e testes estatisticos
python src/aggregate_splits.py
```

`--only ds moe static` roda só as famílias escolhidas. A seleção da pool é a etapa demorada (cerca de 25 min por split na máquina usada).

---

## Referências

- Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS*.
- Castro et al. (2019). Towards Multimodal Sarcasm Detection (An *Obviously* Perfect Paper). *ACL*.
- Cruz, Sabourin & Cavalcanti (2018). Dynamic classifier selection: Recent advances and perspectives. *Information Fusion*, 41.
- Demšar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *JMLR*, 7.
- Jacobs, Jordan, Nowlan & Hinton (1991). Adaptive Mixtures of Local Experts. *Neural Computation*, 3(1).
- Ko, Sabourin & Britto (2008). From dynamic classifier selection to dynamic ensemble selection. *Pattern Recognition*, 41(5).
- Lewis et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *ACL*.
- Ma et al. (2024). emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation. *Findings of ACL*.
- Ray et al. (2022). A Multimodal Corpus for Emotion Recognition in Sarcasm. *LREC*.
- Woods, Kegelmeyer & Bowyer (1997). Combination of multiple classifiers using local accuracy estimates. *IEEE TPAMI*, 19(4).
