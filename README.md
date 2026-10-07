# Seleção Dinâmica de Classificadores para Detecção Multimodal de Sarcasmo

Projeto da disciplina Experiência Criativa: Projeto Transformador II (PUCPR).

O projeto detecta sarcasmo em vídeos curtos de séries de comédia combinando quatro fontes de informação de cada fala — **texto**, **contexto** (as falas anteriores), **áudio** e **rosto** — e investiga se a **incongruência emocional entre essas fontes** ajuda a escolher, fala a fala, qual classificador deve decidir.

- **Dataset:** [MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus) — 1202 falas rotuladas como sarcásticas (601) ou não (601), cada uma com vídeo, áudio, transcrição e contexto conversacional.
- **Pergunta do experimento:** restringir a busca de competência às falas com o mesmo padrão de concordância entre modalidades melhora a seleção dinâmica de classificadores? E a seleção dinâmica supera um classificador estático que vê todas as modalidades?

---

## Status

🚧 **Experimento em andamento.** Os números abaixo são **resultados parciais** de uma única rodada (um split treino/DSEL/teste). Nenhum teste estatístico foi aplicado ainda; as diferenças entre os métodos são apenas descritivas.

## Resultados parciais em uma frase

Nesta rodada, todas as configurações de seleção dinâmica tiveram F1-macro **maior** que os classificadores estáticos multimodais (de +1,7 a +7,0 pontos), e a melhor foi o **KNORA-E com roteamento por incongruência (F1-macro 0,722)**.

---

## Sumário

1. [O experimento em uma página](#1-o-experimento-em-uma-página)
2. [Etapa por etapa: o que entra, o que acontece, o que sai](#2-etapa-por-etapa)
3. [Resultados](#3-resultados)
4. [Limitações](#4-limitações)
5. [Como reproduzir](#5-como-reproduzir)
6. [Estrutura do repositório](#6-estrutura-do-repositório)
7. [Detalhes técnicos](#7-detalhes-técnicos)
8. [Referências](#8-referências)

---

## 1. O experimento em uma página

### A ideia

Sarcasmo costuma aparecer como um **desencontro** entre o que é dito e como/onde é dito: texto elogioso com tom neutro e rosto impassível, ou uma fala positiva logo depois de uma situação ruim (Castro et al., 2019; Ray et al., 2022). A hipótese do projeto é que esse desencontro indica *que tipo de fala* temos — e, portanto, *quais classificadores* tendem a acertá-la.

Em vez de usar sempre o mesmo classificador (abordagem **estática**), usamos **seleção dinâmica**: para cada fala nova, procuramos falas parecidas já rotuladas e escolhemos o(s) classificador(es) que acertaram essas falas parecidas. O experimento testa se fazer essa busca **só entre falas com o mesmo padrão de incongruência** (o "roteamento") melhora a escolha.

### Visão geral do fluxo

```
                 MUStARD++ (1202 falas: video, audio, texto, contexto)
                                     |
                 TREINO 70%  |  DSEL 15%  |  TESTE 15%   (divididos por falante)
                                     |
          +--------------------------+---------------------------+
          |                                                      |
  RAMO 1: o que os classificadores veem            RAMO 2: o que decide o roteamento
          |                                                      |
  embeddings de texto, contexto, audio            valencia emocional de cada modalidade
  + emocoes do rosto                                             |
          |                                       polaridade de cada modalidade
  pool de 7 especialistas                                        |
  (escolhida e treinada so no TREINO)             grupo A (concordam) ou B (discordam)
          |                                                      |
          +--------------------------+---------------------------+
                                     |
              SELECAO DINAMICA: para cada fala do TESTE, vizinhos no DSEL
              OLA | KNORA-E | KNORA-U   x   sem roteamento | com roteamento
                                     |
              comparacao com classificadores ESTATICOS multimodais
              (F1-macro e acuracia no TESTE)
```

### Papel de cada conjunto

| Conjunto | Falas | Para que serve | Nunca é usado para |
|---|---|---|---|
| **Treino** | 897 | Escolher e treinar os classificadores | — |
| **DSEL** | 158 | Medir *em que tipo de fala* cada classificador acerta (região de competência) | Treinar classificadores |
| **Teste** | 147 | Medir o resultado final | Ajustar qualquer coisa |

Os três conjuntos são separados **por falante**: todas as falas de um personagem ficam no mesmo conjunto, para o modelo nunca ser avaliado com uma voz ou um rosto que já viu no treino (Saeb et al., 2017). A proporção de sarcasmo fica perto de 50% em cada um.

---

## 2. Etapa por etapa

Cada etapa abaixo traz **entrada → processamento → saída** e o script responsável. O exemplo hipotético usado é a fala *"Oh, great. Another meeting."*, dita depois de *"I just scheduled a two-hour meeting at 6pm."*

### 2.1 Preparação dos dados

| | |
|---|---|
| **Entrada** | `data/raw/mustard++_text.csv` (falas-alvo + falas de contexto) e os vídeos |
| **Processamento** | Mantém as 1202 falas-alvo, liga cada uma ao seu vídeo, corrige uma linha desalinhada do CSV original (chave `"Disgust"` → `1_S11E03_067_u`), divide por falante em treino/DSEL/teste (testa 1000 sementes e escolhe a mais próxima de 50/50) e extrai o áudio (mono, 16 kHz) |
| **Saída** | `data/processed/mustard_prepared.csv`, `data/splits/{train,dsel,test}.csv`, `data/processed/audio/<KEY>.wav` |
| **Scripts** | `src/data/prepare_mustard.py`, `src/data/create_splits.py`, `src/data/extract_audio.py` |

### 2.2 Ramo 1 — Representações de cada fala

Cada fala vira quatro vetores de características:

| Modalidade | Entrada | Modelo | Saída |
|---|---|---|---|
| Texto (T) | a fala | BART-base | vetor de 768 dimensões |
| Contexto (C) | falas anteriores da cena | BART-base | vetor de 768 dimensões |
| Áudio (A) | `.wav` | wav2vec2-base | vetor de 768 dimensões |
| Rosto (F) | emoções por frame | média e desvio de 7 emoções + nº de rostos + indicador de rosto ausente | 16 valores |

Saída em `data/processed/features/` (os `.npy` não são versionados). Scripts: `build_context_text.py`, `extract_text_features.py`, `extract_context_features.py`, `extract_audio_features.py`, `build_visual_features.py` (todos em `src/features/`).

### 2.3 Ramo 1 — A pool de 7 especialistas (só no treino)

| | |
|---|---|
| **Entrada** | As representações do treino |
| **Processamento** | Avalia **15 combinações de modalidades** (T, A, T+C, T+C+A+F, …) × **3 classificadores** (SVM, regressão logística, random forest) com busca de hiperparâmetros e validação cruzada de 5 partes por falante (métrica: F1-macro). Dos 15 melhores, escolhe 7 de forma gulosa, com pontuação 0,7 × desempenho + 0,3 × diversidade. Depois, treina os 7 no treino inteiro e os congela |
| **Saída** | `data/results/pool/pool_selection/` (validação cruzada e definição da pool) e `data/results/pool/frozen_pools/base/` (modelos treinados) |
| **Scripts** | `python -m src.pool_selection.build_pool`, `python -m src.pool_training.train_frozen_pool` (detalhes nos READMEs dessas pastas) |

A pool obtida:

| Membro | Modalidades | Classificador | F1-macro na validação cruzada do treino |
|---|---|---|---|
| C01 | T + C + F | SVM | 0,738 |
| C02 | T + C | SVM | 0,735 |
| C03 | T + C + A + F | Regressão logística | 0,732 |
| C04 | T + C + A + F | SVM | 0,728 |
| C05 | T + C + A | Regressão logística | 0,725 |
| C06 | C + A + F | SVM | 0,718 |
| C07 | C + A | SVM | 0,718 |

O experimento usa a versão **BASE** da pool (os especialistas veem só as representações das modalidades, sem features de incongruência), para que a única diferença entre "sem" e "com" roteamento seja o roteamento.

### 2.4 Ramo 2 — Valência, polaridade e grupo A/B

**a) Valência de cada modalidade** (detalhes na [seção 7.1](#71-cálculo-de-valência)). Um classificador de emoções já treinado dá a probabilidade de 7 emoções, e:

```
valencia    = P(happy) - P(angry + disgust + fear + sad)       em [-1, 1]
intensidade = P(happy) + P(angry + disgust + fear + sad)       em [0, 1]
(neutral e surprise nao entram)
```

**b) Polaridade:** cada modalidade recebe a polaridade da sua emoção predominante.

| Emoção predominante | Polaridade |
|---|---|
| happy | 🟢 positiva |
| angry, disgust, fear, sad | 🔴 negativa |
| neutral, surprise, vazio (sem texto, sem emoção válida no áudio, sem rosto) | ⚪ neutra |

**c) Grupo (o roteamento):**

```
texto 🟢 | contexto 🔴 | audio ⚪ | rosto ⚪
     as 4 polaridades sao iguais?  --sim-->  grupo A (modalidades concordam)
                                   --nao-->  grupo B (modalidades discordam)
```

| | Treino | DSEL | Teste | Total |
|---|---|---|---|---|
| Grupo A | 125 | 16 | 32 | 173 |
| Grupo B | 772 | 142 | 115 | 1029 |

O mesmo script também calcula o **vetor de incongruência** (diferenças de valência entre os 6 pares de modalidades, oposições de polaridade e o indicador de "cara de paisagem"), descrito na [seção 7.2](#72-vetor-de-incongruência). Esse vetor não é usado neste experimento; aqui entra só o grupo A/B.

| | |
|---|---|
| **Saída** | `data/processed/*_valence.csv` e `data/processed/incongruence.csv` (coluna `grupo`) |
| **Scripts** | `src/features/fix_visual_anger.py`, `text_valence.py`, `context_valence.py`, `audio_valence.py`, `visual_valence.py`, `src/incongruence/build_incongruence.py` |

### 2.5 Seleção dinâmica (o experimento principal)

Para **cada fala do teste**:

**Passo 1 — achar os k = 7 vizinhos mais parecidos no DSEL.** "Parecido" é a distância euclidiana entre as quatro modalidades concatenadas (cada uma padronizada com média e desvio do treino e normalizada para ter peso parecido).

| Sem roteamento | Com roteamento |
|---|---|
| Procura no DSEL inteiro (158 falas) | Procura só no DSEL **do mesmo grupo** da fala de teste (16 se for A, 142 se for B) |

**Passo 2 — ver quem acertou esses vizinhos e decidir.** Exemplo com uma fala de teste:

```
          v1 v2 v3 v4 v5 v6 v7   acertos
   C01    ok ok ok  x ok ok ok     6
   C02    ok ok ok ok ok ok ok     7
   C03    ok  x ok ok ok ok ok     6
   C04     x ok  x ok ok  x ok     4
   ...
```

| Método | Regra | No exemplo |
|---|---|---|
| **OLA** (Woods et al., 1997) | O membro com mais acertos entre os vizinhos decide sozinho (empate: maior acurácia no DSEL) | C02 decide |
| **KNORA-E** (Ko et al., 2008) | Votam só os membros que acertaram **todos** os vizinhos; se nenhum, descarta o vizinho mais distante e tenta de novo; se ninguém acerta nem o mais próximo, vota a pool inteira | só C02 vota |
| **KNORA-U** (Ko et al., 2008) | Todos os que acertaram algum vizinho votam, com peso = nº de vizinhos acertados | C01 = 6, C02 = 7, C03 = 6, C04 = 4… |

**Passo 3 — a decisão final** (sarcástica ou não) é comparada com o rótulo verdadeiro.

São **6 configurações**: 3 métodos × 2 roteamentos. Como referência, o script também calcula o melhor membro sozinho, o voto da pool inteira e o **oráculo** (fração das falas em que pelo menos um membro acerta — o teto de qualquer seleção).

| | |
|---|---|
| **Saída** | `data/processed/selection/ds_routing_test.csv` (uma linha por fala e roteamento, com a decisão de cada método e quais membros decidiram) e `ds_routing_summary.csv` |
| **Script** | `src/selection/ds_routing.py` |

### 2.6 Linhas de base estáticas multimodais

Um classificador **estático** é treinado uma vez e aplicado igual a todas as falas, sem DSEL e sem seleção. Para a comparação ser justa, os estáticos recebem **as quatro modalidades juntas** (representações concatenadas):

| Estático | Hiperparâmetros |
|---|---|
| SVM (T+C+A+F) | os escolhidos na validação cruzada do treino (`cv_results.csv`) |
| Regressão logística (T+C+A+F) | idem |
| Random forest (T+C+A+F) | idem |
| Voto majoritário dos 3 | — |

O SVM e a regressão logística com T+C+A+F são os mesmos candidatos que aparecem na pool (C04 e C03). Ou seja, a comparação é entre **usar sempre o mesmo classificador multimodal** e **escolher dinamicamente entre ele e outros seis**.

### 2.7 Avaliação

- **F1-macro:** média do F1 das duas classes (sarcástica e não sarcástica). É a métrica principal porque pesa as duas classes igualmente.
- **Acurácia:** fração das falas classificadas corretamente.
- **F1-macro por grupo:** o mesmo F1-macro calculado separado para as falas do grupo A e do grupo B do teste, para ver onde o roteamento muda o resultado.

Nenhum teste estatístico é aplicado nesta etapa do experimento.

| | |
|---|---|
| **Saída** | `data/processed/eval/static_multimodal_summary.csv` (métricas) e `static_multimodal_test.csv` (predição de cada método em cada fala do teste) |
| **Script** | `src/eval/static_multimodal.py` |

---

## 3. Resultados

Todos os números abaixo são no **teste (147 falas)**, com um único split. As tabelas completas estão nos CSVs citados.

### 3.1 Seleção dinâmica: sem × com roteamento

| Método | Sem roteamento | Com roteamento |
|---|---|---|
| OLA | 0,690 | 0,689 |
| KNORA-E | 0,689 | **0,722** |
| KNORA-U | 0,703 | 0,696 |

F1-macro por grupo (A tem 32 falas no teste, B tem 115):

| Método | Grupo A sem | Grupo A com | Grupo B sem | Grupo B com |
|---|---|---|---|---|
| OLA | 0,667 | 0,681 | 0,696 | 0,688 |
| KNORA-E | 0,667 | **0,752** | 0,694 | 0,712 |
| KNORA-U | 0,667 | 0,639 | 0,712 | 0,712 |

Referências: melhor membro sozinho (C01) 0,693 · voto da pool inteira 0,696 · **oráculo 0,816** (acurácia).

**Leitura:** o roteamento só muda o resultado de forma visível no **KNORA-E** (+3,3 pontos de F1-macro), principalmente no grupo A (0,667 → 0,752). Fala a fala, o KNORA-E com roteamento acerta 6 falas que a versão sem roteamento erra, e erra 1 que ela acerta. No OLA e no KNORA-U, as duas versões quase não diferem.

### 3.2 Seleção dinâmica × estáticos multimodais

| Método | Acurácia | F1-macro |
|---|---|---|
| Estático random forest | 0,673 | 0,651 |
| Estático SVM | 0,687 | 0,666 |
| Estático regressão logística | 0,680 | 0,672 |
| Estático voto dos 3 | 0,694 | 0,672 |
| OLA sem / com roteamento | 0,701 / 0,701 | 0,690 / 0,689 |
| KNORA-E sem roteamento | 0,701 | 0,689 |
| KNORA-U sem / com roteamento | 0,714 / 0,707 | 0,703 / 0,696 |
| **KNORA-E com roteamento** | **0,735** | **0,722** |

Todas as configurações de seleção dinâmica ficaram acima de todos os estáticos (+1,7 a +7,0 pontos de F1-macro). Contando fala a fala, as maiores diferenças são as do KNORA-E com roteamento:

| Comparação | Falas que só a seleção dinâmica acerta | Falas que só o estático acerta |
|---|---|---|
| KNORA-E com roteamento × random forest | 13 | 4 |
| KNORA-E com roteamento × regressão logística | 11 | 3 |
| KNORA-E com roteamento × SVM | 11 | 4 |
| KNORA-E com roteamento × voto dos 3 | 10 | 4 |

### 3.3 Interpretação

- **A direção é consistente nesta rodada.** A seleção dinâmica nunca ficou abaixo dos estáticos, e o KNORA-E com roteamento foi a melhor configuração. As diferenças, porém, são de poucas falas (de 2 a 10 em 147), então ainda é preciso confirmar se elas se repetem em outros splits e com os testes estatísticos que forem definidos para o experimento final.
- **A pool tem pouca diversidade, o que limita qualquer seleção.** O oráculo chega a só 0,816: em quase 1 de cada 5 falas, *todos* os 7 membros erram. C01 e C02 discordam em apenas 1,4% das falas na validação cruzada. Quando os especialistas erram nas mesmas falas, escolher entre eles muda pouco.
- **O contexto domina.** Os 7 membros usam o contexto, e o contexto é a modalidade mais forte sozinha (F1-macro 0,70 na validação cruzada, contra 0,64 do texto, 0,60 do áudio e 0,56 do rosto). Isso deixa os membros parecidos entre si.
- **O grupo A é pequeno e pouco informativo.** Só 16 falas no DSEL, e 82% do grupo A são falas em que as quatro modalidades são neutras — "concordar" aqui é, na maioria, "nenhuma emoção detectada".

---

## 4. Limitações

1. **Um único split e teste pequeno (147 falas).** Diferenças de poucos pontos de F1-macro correspondem a poucas falas. Um próximo passo possível é repetir com vários splits por falante (por exemplo, 5 ou 10 sementes) e reportar média e desvio.
2. **Grupo A pequeno no DSEL (16 falas).** Com k = 7, cada fala do grupo A usa quase metade do DSEL-A como vizinhança.
3. **Critério de roteamento simples.** A polaridade vem só da emoção predominante, e o grupo A é dominado por falas "tudo neutro"; o grupo A tem *mais* sarcasmo (59,6%) que o B (48,5%), o contrário da intuição inicial.
4. **Classificadores de emoção externos.** As valências dependem de classificadores pré-treinados que veem só a emoção de superfície (ver [seção 7.1](#71-cálculo-de-valência)); o classificador de texto vem de um autor individual no HuggingFace, sem validação acadêmica publicada.
5. **Parte visual não reproduzível do zero.** Os scripts de extração de frames, recorte de rostos e emoção facial não estão no repositório, só o resultado (`visual_emotions.csv`). A probabilidade de raiva desse arquivo foi reconstruída (ver [seção 7.3](#73-correções-feitas-nos-dados)).
6. **Sem testes estatísticos ainda.** Os resultados são descritivos; a comparação estatística entre os métodos fica para a versão final do experimento.

---

## 5. Como reproduzir

**Pré-requisitos:** Python 3.10+ e FFmpeg instalado no sistema. Depois:

```
pip install -r requirements.txt
```

Todos os comandos rodam **a partir da raiz do repositório**, nesta ordem:

| # | Comando | O que faz |
|---|---|---|
| 1 | `python src/data/prepare_mustard.py` | Prepara as 1202 falas |
| 2 | `python src/data/create_splits.py` | Treino / DSEL / teste por falante |
| 3 | `python src/data/extract_audio.py` | Extrai o áudio (pula os `.wav` que já existem) |
| 4 | `python src/features/fix_visual_anger.py` | Corrige a raiva no `visual_emotions.csv` (não faz nada se já estiver corrigido) |
| 5 | `python src/features/text_valence.py` | Valência do texto |
| 6 | `python src/features/context_valence.py` | Valência do contexto |
| 7 | `python src/features/audio_valence.py` | Valência do áudio ⚠️ |
| 8 | `python src/features/visual_valence.py` | Valência do rosto |
| 9 | `python src/incongruence/build_incongruence.py` | Polaridade, grupo A/B e vetor de incongruência |
| 10 | `python src/features/build_context_text.py` | Texto do contexto de cada fala |
| 11 | `python src/features/extract_text_features.py` | Embeddings do texto |
| 12 | `python src/features/extract_context_features.py` | Embeddings do contexto |
| 13 | `python src/features/extract_audio_features.py` | Embeddings do áudio |
| 14 | `python src/features/build_visual_features.py` | Features do rosto |
| 15 | `python -m src.pool_selection.build_pool` | Escolhe a pool de 7 (validação cruzada no treino) |
| 16 | `python -m src.pool_training.train_frozen_pool` | Treina e congela a pool |
| 17 | `python src/selection/ds_routing.py` | **OLA / KNORA-E / KNORA-U sem e com roteamento** |
| 18 | `python src/eval/static_multimodal.py` | **Estáticos multimodais × seleção dinâmica** |

⚠️ `audio_valence.py` é retomável: se `data/processed/audio_valence.csv` já tiver todas as falas, ele não processa nada. Para recalcular do zero, apague ou mova esse CSV antes.

Os modelos de emoção e de embeddings (passos 5–7 e 11–13) são baixados automaticamente na primeira execução e ficam em cache; esses passos usam GPU se houver, e rodam (mais devagar) sem ela. O passo 15 é o mais demorado (busca de hiperparâmetros em 45 candidatos).

### Outros scripts do repositório (não fazem parte deste experimento)

A equipe também mantém uma linha de experimentos com **4 classificadores unimodais** (T, C, A, F) em vez da pool de 7: `src/pool_selection/prototipo/train_*_classifier.py` → `src/selection/dcs_local_accuracy.py`, `dcs_incongruence.py`, `dcs_combined.py` → `src/eval/static_ensemble.py` → `src/eval/compare_static_vs_dcs.py`. Os resultados de `dcs_combined` em `data/processed/selection/` podem estar desatualizados até o script ser rodado de novo com o `incongruence.csv` atual.

---

## 6. Estrutura do repositório

```
data/
  raw/                          dados originais do MUStARD++ (nao versionados os videos)
  splits/                       train.csv, dsel.csv, test.csv
  processed/
    audio/                      .wav extraidos (nao versionado)
    *_valence.csv               valencia por modalidade
    visual_emotions.csv         emocoes por rosto/frame (com a raiva reconstruida)
    incongruence.csv            polaridades, grupo A/B e vetor de incongruencia
    context_text.csv            contexto conversacional reconstruido por fala
    features/                   embeddings por modalidade (.npy nao versionados) + chaves
    selection/                  resultados da selecao dinamica (ds_routing_* e dcs_*)
    eval/                       estaticos multimodais e comparacoes do time
    pool_predictions/           predicoes dos 4 classificadores unimodais (linha do time)
    pool_analysis/              diversidade dos 4 classificadores unimodais (linha do time)
  results/
    pool/
      pool_selection/           validacao cruzada e definicao da pool de 7
      frozen_pools/             pools treinadas (base e com incongruencia; .joblib nao versionados)

src/
  data/
    prepare_mustard.py          le o CSV bruto, filtra as 1202 falas, liga cada uma ao video
    create_splits.py            divide em treino/DSEL/teste por falante
    extract_audio.py            extrai o audio (.wav, mono, 16 kHz) com ffmpeg
  features/
    fix_visual_anger.py         reconstroi a probabilidade de raiva no visual_emotions.csv
    text_valence.py             valencia do texto
    context_valence.py          valencia do contexto
    audio_valence.py            valencia do audio
    visual_valence.py           valencia do rosto
    build_context_text.py       texto do contexto de cada fala
    extract_text_features.py    embeddings do texto (BART-base)
    extract_context_features.py embeddings do contexto (BART-base)
    extract_audio_features.py   embeddings do audio (wav2vec2-base)
    build_visual_features.py    features do rosto
  incongruence/
    build_incongruence.py       polaridades, grupo A/B e vetor de incongruencia
  pool_selection/               formacao da pool de 7 (ver README proprio)
    prototipo/                  4 classificadores unimodais e analises (linha do time)
  pool_training/                treino das pools base e com incongruencia (ver README proprio)
  selection/
    ds_routing.py               OLA, KNORA-E e KNORA-U, sem e com roteamento (este experimento)
    dcs_local_accuracy.py       DCS com os 4 unimodais, vizinhos nas features (linha do time)
    dcs_incongruence.py         DCS com os 4 unimodais, vizinhos na incongruencia (linha do time)
    dcs_combined.py             DCS com os 4 unimodais, os dois espacos (linha do time)
  eval/
    static_multimodal.py        estaticos multimodais x selecao dinamica (este experimento)
    static_ensemble.py          ensemble estatico dos 4 unimodais (linha do time)
    compare_static_vs_dcs.py    compara ensemble estatico e DCS (linha do time)

requirements.txt                dependencias Python
```

---

## 7. Detalhes técnicos

### 7.1 Cálculo de valência

Valência é um número entre −1 (muito negativo) e +1 (muito positivo). Cada modalidade passa por um classificador de emoções categóricas **já treinado** (não treinado por nós), que devolve a probabilidade de cada emoção. Todos os rótulos são traduzidos para um vocabulário único de 7 emoções (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) por um dicionário `MAPA_ROTULOS` em cada script.

```
valencia    = P(emocoes positivas) - P(emocoes negativas)      em [-1, 1]
intensidade = P(emocoes positivas) + P(emocoes negativas)      em [0, 1]
```

A intensidade separa uma fala neutra (valência 0, intensidade ≈ 0) de uma ambivalente (valência 0, intensidade ≈ 1). Ela **não é arousal**: a correlação com o arousal anotado no MUStARD++ é praticamente zero.

| Modalidade | Script | Classificador | Positivas | Negativas |
|---|---|---|---|---|
| Texto | `text_valence.py` | `Tanneru/Emotion-Classification-DeBERTa-v3-Large` (HuggingFace) | happy | angry, disgust, fear, sad |
| Contexto | `context_valence.py` | o mesmo, aplicado às falas anteriores da cena concatenadas | happy | angry, disgust, fear, sad |
| Áudio | `audio_valence.py` | `emotion2vec_plus_large` (FunASR; Ma et al., 2024) | happy | angry, disgust, fear, sad |
| Rosto | `visual_valence.py` | emoções por frame de `visual_emotions.csv`, média por fala | happy | angry, disgust, fear, sad |

Observações por script:

- **Texto:** só o rótulo `anger` precisa de tradução (→ `angry`). **Atenção:** o checkpoint é de um autor individual no HuggingFace, com baixa adoção e sem validação acadêmica publicada encontrada; os números do model card não vêm de um benchmark independente.
- **Áudio:** o modelo devolve 9 classes; `other` e `unknown` são retiradas (massa guardada em `p_descartada`) e as 7 restantes são renormalizadas. Se `other + unknown ≥ 0,5`, a fala é tratada como sem emoção válida (`vazio`, valência 0) — 39 falas. Os rótulos são lidos da própria resposta do modelo, nunca fixados à mão. **Atenção:** há um [issue aberto no FunASR](https://github.com/modelscope/FunASR/issues/2728) relatando que o mesmo modelo devolve 9 classes pelo FunASR e 5 pelo ModelScope; o script imprime os rótulos da primeira fala para conferência.
- **Rosto:** a valência e a intensidade são calculadas por frame e depois têm a média tirada por fala. 2 falas não têm rosto detectado e recebem valência 0.

### 7.2 Vetor de incongruência

Calculado por `build_incongruence.py` e salvo em `incongruence.csv`. **Não é usado no experimento principal** (que usa só a coluna `grupo`), mas é usado pela pool com incongruência (`frozen_pools/incongruence/`) e pelos scripts `dcs_*` do time.

Para cada um dos 6 pares de modalidades:

```
inc_a_b = (valencia_a - valencia_b) / 2      em [-1, 1]   (com sinal: positivo = "a" mais positiva)
opp_a_b = 1 se a e b tem polaridade e elas sao contrarias (positiva x negativa), senao 0
```

E por fala:

```
n_oppositions = soma dos 6 opp_a_b
deadpan       = 1 se audio e rosto tem polaridade neutra ("cara de paisagem"), senao 0
```

- A versão absoluta das diferenças foi retirada: duas modalidades perto de 0 sempre dão diferença pequena, e o sarcasmo tem mais modalidades neutras, então ela misturava "desacordo" com "pouca emoção". A oposição de polaridade mede desacordo sem esse efeito.
- Na literatura, a incongruência do sarcasmo é principalmente o contraste entre a emoção de superfície e a emoção pretendida (Ray et al., 2022; Farabi et al., 2024). Os classificadores só veem a superfície, então o vetor usa o **contexto** como representante da situação: texto positivo com contexto negativo (Du et al., 2022).
- `deadpan` marca a entrega com tom neutro e rosto impassível, descrita por Castro et al. (2019) como pista de sarcasmo.

**Análise exploratória** (treino + DSEL, 1055 falas; só médias, sem testes de significância):

| Medida | Não sarcástico | Sarcástico |
|---|---|---|
| `deadpan` (áudio e rosto neutros) | 19,5% | 27,7% |
| `inc_text_context` (texto − contexto) | −0,028 | +0,033 |
| Intensidade do rosto | 0,649 | 0,578 |
| Intensidade do contexto | 0,498 | 0,410 |
| Proporção de sarcasmo no grupo A / B | — | 59,6% / 48,5% |

- Como referência de teto (só análise, nunca feature): a diferença entre a valência do texto e a valência *anotada pelos humanos* no MUStARD++ é ≈ 0,55 no sarcasmo contra ≈ 0,08 no não sarcasmo. O contraste superfície × intenção existe nos dados; o que falta é medi-lo sem a anotação humana, que não existe para um vídeo novo e está ligada ao rótulo de sarcasmo.
- Uma versão anterior estimava essa valência pretendida com um regressor treinado no treino; foi retirada porque a estimativa era fraca (Spearman ≈ 0,15–0,34) e colocava um modelo aprendido antes da seleção dinâmica.

### 7.3 Correções feitas nos dados

| Problema | Onde | Correção |
|---|---|---|
| A fala `1_S11E03_067_u` tem a chave `"Disgust"` no CSV original | `mustard++_text.csv` | Corrigida em `prepare_mustard.py` e em `create_splits.py`. Sem isso, a fala sumia do DSEL em todo merge por chave |
| A coluna `angry` estava zerada em todos os 11.568 frames, embora o classificador facial preveja "anger" em 403 frames | `visual_emotions.csv` | `fix_visual_anger.py` reconstrói `angry = 1 − soma das outras 6` em cada frame (supõe que o classificador tem exatamente essas 7 classes). Depois da correção, a emoção mais provável de cada frame coincide com `predicted_emotion` em 100% dos frames. O original continua no histórico do git |

---

## 8. Referências

- Britto, A. S., Sabourin, R., & Oliveira, L. E. S. (2014). Dynamic selection of classifiers — A comprehensive review. *Pattern Recognition*.
- Castro, S., Hazarika, D., Pérez-Rosas, V., Zimmermann, R., Mihalcea, R., & Poria, S. (2019). Towards Multimodal Sarcasm Detection (An *Obviously* Perfect Paper). *ACL 2019*.
- Cruz, R. M. O., Sabourin, R., & Cavalcanti, G. D. C. (2018). Dynamic classifier selection: Recent advances and perspectives. *Information Fusion*, 41, 195–216.
- Du, Y., Li, T., Pathan, M. S., Teklehaimanot, H. K., & Yang, Z. (2022). An Effective Sarcasm Detection Approach Based on Sentimental Context and Individual Expression Habits. *Cognitive Computation*, 14, 78–90.
- Farabi, S., Ranasinghe, T., Kanojia, D., Kong, Y., & Zampieri, M. (2024). A Survey of Multimodal Sarcasm Detection. *IJCAI 2024*.
- Ko, A. H. R., Sabourin, R., & Britto, A. S. (2008). From dynamic classifier selection to dynamic ensemble selection. *Pattern Recognition*, 41(5), 1718–1731.
- Ma, Z., et al. (2024). emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation. *Findings of ACL 2024*.
- Ray, A., Mishra, S., Nunna, A., & Bhattacharyya, P. (2022). A Multimodal Corpus for Emotion Recognition in Sarcasm. *LREC 2022*.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5).
- Woods, K., Kegelmeyer, W. P., & Bowyer, K. (1997). Combination of multiple classifiers using local accuracy estimates. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 19(4), 405–410.
