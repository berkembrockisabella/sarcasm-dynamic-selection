# Seleção Dinâmica de Classificadores para Detecção Multimodal de Sarcasmo

Projeto da disciplina Experiência Criativa: Projeto Transformador II (PUCPR).

Detecta sarcasmo em falas de séries de comédia usando **texto**, **contexto** (falas anteriores), **áudio** e **rosto**, e investiga se a **incongruência emocional entre essas modalidades** ajuda a escolher, fala a fala, qual classificador deve decidir.

- **Dataset:** [MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus) — 1202 falas (601 sarcásticas, 601 não).
- **Pergunta:** buscar competência só entre falas com o mesmo padrão de concordância entre modalidades melhora a seleção dinâmica? E a seleção dinâmica supera um classificador estático multimodal?

> 🚧 **Experimento em andamento.** Resultados parciais de uma única rodada (um split), apenas descritivos — nenhum teste estatístico aplicado ainda.

---

## O experimento

```
            MUStARD++ (1202 falas)
                     |
     TREINO 70% | DSEL 15% | TESTE 15%   (divididos por falante)
                     |
     +---------------+----------------+
     |                                |
 Representacoes                  Valencia emocional
 (BART, wav2vec2, rosto)         de cada modalidade
     |                                |
 Pool de 7 especialistas         Grupo A (modalidades concordam)
 (escolhida e treinada no TREINO)  ou B (discordam)
     |                                |
     +---------------+----------------+
                     |
     Selecao dinamica no DSEL: OLA | KNORA-E | KNORA-U
              sem roteamento | com roteamento
                     |
     Comparacao com classificadores estaticos (TESTE)
```

| Conjunto | Falas | Papel |
|---|---|---|
| Treino | 897 | Escolher e treinar os classificadores |
| DSEL | 158 | Medir em que tipo de fala cada classificador acerta |
| Teste | 147 | Medir o resultado final (nunca usado para ajustar nada) |

A divisão é **por falante**: todas as falas de um personagem ficam no mesmo conjunto.

**Para que serve cada conjunto na seleção dinâmica:**

- **Treino:** os 7 classificadores aprendem aqui. Depois ficam congelados.
- **DSEL:** serve de **base de consulta**. Para cada fala do teste, procuram-se aqui as falas parecidas e vê-se quais classificadores acertaram nelas; quem acertou mais decide. Fica separado do treino porque os classificadores já viram o treino e acertariam mais nele do que em falas novas.
- **Teste:** só mede o resultado final. Nada é ajustado com ele.

### Estrutura do repositório

| Pasta | Função |
|---|---|
| `src/data/` | Prepara a base MUStARD++, divide em treino/DSEL/teste por falante e extrai o áudio dos vídeos |
| `src/features/valencia/` | Calcula a valência e a emoção de cada modalidade (texto, contexto, áudio, rosto) |
| `src/features/embeddings/` | Gera as representações numéricas (BART, wav2vec2, emoções do rosto) usadas pelos classificadores |
| `src/incongruence/` | Define o grupo A (modalidades concordam) ou B (discordam) de cada fala. [Detalhes](src/incongruence/README.md) |
| `src/pool_selection/` | Avalia os 45 candidatos (15 combinações de modalidades × 3 classificadores) e escolhe os 7 especialistas |
| `src/pool_training/` | Treina os 7 especialistas no conjunto de treino e salva os modelos |
| `src/selection/` | Seleção dinâmica (OLA, KNORA-E, KNORA-U), sem e com roteamento. [Detalhes](src/selection/README.md) |
| `src/eval/` | Compara a seleção dinâmica com os classificadores estáticos |


### 1. Pool de 7 especialistas

Cada fala é representada por embeddings de texto e contexto (BART-base), áudio (wav2vec2-base) e estatísticas das emoções do rosto. No treino, são avaliadas 15 combinações de modalidades × 3 classificadores (SVM, regressão logística, random forest) com validação cruzada por falante; ficam os 7 que equilibram desempenho e diversidade:

| C01 | C02 | C03 | C04 | C05 | C06 | C07 |
|---|---|---|---|---|---|---|
| T+C+F · SVM | T+C · SVM | T+C+A+F · RL | T+C+A+F · SVM | T+C+A · RL | C+A+F · SVM | C+A · SVM |

### 2. Roteamento por incongruência (grupo A/B)

Para cada modalidade, um classificador de emoções pré-treinado dá a **valência** (P(positivas) − P(negativas), de −1 a +1) e a emoção predominante, que define a polaridade: 🟢 positiva (happy), 🔴 negativa (angry, disgust, fear, sad) ou ⚪ neutra (neutral, surprise, sem sinal).

- **Grupo A:** as 4 modalidades têm a mesma polaridade (concordam).
- **Grupo B:** pelo menos uma difere (discordam).

| | Treino | DSEL | Teste |
|---|---|---|---|
| Grupo A | 125 | 16 | 32 |
| Grupo B | 772 | 142 | 115 |

### 3. Seleção dinâmica

Para cada fala do teste, procuram-se os **k = 7 vizinhos mais parecidos no DSEL** (o efeito de outros valores de k está na seção de sensibilidade) e verifica-se quais especialistas acertaram esses vizinhos.

| | Onde procura os vizinhos |
|---|---|
| **Sem roteamento** | No DSEL inteiro |
| **Com roteamento** | Só no DSEL do mesmo grupo (A ou B) da fala de teste |

| Método | Quem decide |
|---|---|
| **OLA** | O especialista que mais acertou os vizinhos |
| **KNORA-E** | Votam só os que acertaram **todos** os vizinhos (se nenhum, reduz a vizinhança) |
| **KNORA-U** | Todos votam, com peso = nº de vizinhos acertados |

### 4. Classificadores estáticos (referência)

Treinados uma vez no treino e aplicados igual a todas as falas, recebendo **as 4 modalidades juntas**: SVM, regressão logística, random forest (hiperparâmetros da validação cruzada do treino) e o voto dos 3.

---

## Resultados parciais (teste, 147 falas)

| Método | Acurácia | F1-macro |
|---|---|---|
| Estático random forest | 0,673 | 0,651 |
| Estático SVM | 0,687 | 0,666 |
| Estático regressão logística | 0,680 | 0,672 |
| Estático voto dos 3 | 0,694 | 0,672 |
| OLA — sem / com roteamento | 0,701 / 0,701 | 0,690 / 0,689 |
| KNORA-E — sem roteamento | 0,701 | 0,689 |
| KNORA-U — sem / com roteamento | 0,714 / 0,707 | 0,703 / 0,696 |
| **KNORA-E — com roteamento** | **0,735** | **0,722** |
| *Oráculo (algum especialista acerta)* | *0,816* | — |

**O que se observa nesta rodada:**

- Todas as configurações de seleção dinâmica ficaram acima dos estáticos (+1,7 a +7,0 pontos de F1-macro).
- O roteamento só mudou o resultado no **KNORA-E** (+3,3 pontos), principalmente no grupo A (F1-macro 0,667 → 0,752).
- As diferenças são de poucas falas (de 2 a 10 em 147) e ainda precisam ser confirmadas.
- A pool tem **pouca diversidade**: o oráculo é 0,816 e todos os especialistas usam o contexto, então eles tendem a errar nas mesmas falas.

### Sensibilidade ao tamanho da vizinhança (k)

O mesmo experimento foi repetido com k = 3, 5, 7, 9, 11, 15, 21 e 31 (nenhum k foi escolhido olhando o teste). F1-macro:

| k | KNORA-E sem | KNORA-E com | KNORA-U sem | KNORA-U com | OLA sem | OLA com |
|---|---|---|---|---|---|---|
| 3 | 0,682 | 0,701 | 0,709 | 0,701 | 0,686 | 0,690 |
| 7 | 0,689 | 0,722 | 0,703 | 0,697 | 0,690 | 0,689 |
| 11 | 0,695 | 0,714 | 0,697 | 0,697 | 0,704 | 0,711 |
| 21 | 0,695 | 0,722 | 0,697 | 0,697 | 0,741 | 0,715 |

- **KNORA-E:** com roteamento ficou acima de sem roteamento em todos os k (+1 a +3 pontos).
- **KNORA-U:** quase não muda com o k nem com o roteamento.
- **OLA:** oscila bastante entre os k (ruído), sem tendência clara.
- Com roteamento, o grupo A tem só 16 falas no DSEL; para k > 16 a vizinhança dele é limitada a essas 16.

**Limitações:** um único split; grupo A pequeno no DSEL (16 falas) e formado em 82% por falas "tudo neutro"; scripts da parte visual (frames, rostos, emoção facial) ausentes no repositório — só o resultado `visual_emotions.csv` existe.

---

## Como rodar

Pré-requisitos: Python 3.10+ e FFmpeg. Instale com `pip install -r requirements.txt` e rode, a partir da raiz:

```
# dados
python src/data/prepare_mustard.py
python src/data/create_splits.py
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

# pool
python -m src.pool_selection.build_pool
python -m src.pool_training.train_frozen_pool

# experimento
python src/selection/ds_routing.py          # OLA / KNORA, sem e com roteamento
python src/eval/static_multimodal.py        # estaticos x selecao dinamica
python src/selection/ds_routing_k_sweep.py  # sensibilidade ao k (tabela e grafico)
```

Resultados em `data/processed/selection/ds_routing_summary.csv`, `data/processed/eval/static_multimodal_summary.csv`, `data/processed/selection/ds_routing_k_sweep.csv` (com o gráfico `.png` ao lado).

Os scripts em `src/pool_selection/prototipo/`, `src/selection/dcs_*.py`, `src/eval/static_ensemble.py` e `src/eval/compare_static_vs_dcs.py` pertencem a outra linha de experimentos do time (4 classificadores unimodais) e não fazem parte deste experimento.

---

## Referências

- Castro et al. (2019). Towards Multimodal Sarcasm Detection (An *Obviously* Perfect Paper). *ACL*.
- Cruz, Sabourin & Cavalcanti (2018). Dynamic classifier selection: Recent advances and perspectives. *Information Fusion*, 41.
- Ko, Sabourin & Britto (2008). From dynamic classifier selection to dynamic ensemble selection. *Pattern Recognition*, 41(5).
- Ray et al. (2022). A Multimodal Corpus for Emotion Recognition in Sarcasm. *LREC*.
- Woods, Kegelmeyer & Bowyer (1997). Combination of multiple classifiers using local accuracy estimates. *IEEE TPAMI*, 19(4).
