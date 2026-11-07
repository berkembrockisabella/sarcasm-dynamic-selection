# Classificadores Estáticos Multimodais (`src/static/`)

Esta pasta contém os scripts responsáveis pelos baselines estáticos multimodais utilizados na comparação com as estratégias de Seleção Dinâmica de Classificadores.

Os classificadores desta etapa recebem conjuntamente as quatro modalidades disponíveis:

- texto;
- contexto;
- áudio;
- visual.

Diferentemente das estratégias de seleção dinâmica, os modelos estáticos utilizam a mesma combinação multimodal para todas as instâncias.

## Arquivos

```text
src/static/
├── README.md
└── static_multimodal.py
```

### `static_multimodal.py`

Treina e avalia os classificadores estáticos multimodais utilizando a combinação completa:

```text
text + context + audio + visual
```

São considerados três classificadores:

- Support Vector Machine (SVM);
- Regressão Logística;
- Random Forest.

Os modelos são treinados exclusivamente sobre o conjunto `TRAIN` e avaliados sobre o conjunto `TEST`.

Os hiperparâmetros utilizados não são ajustados novamente nesta etapa. São reutilizadas as melhores configurações obtidas anteriormente durante a validação cruzada realizada no `TRAIN` na etapa de formação da pool.

## Features

As representações das quatro modalidades são carregadas por meio do `FeatureStore` e concatenadas para formar a entrada de cada classificador.

O fluxo geral é:

```text
texto
contexto
áudio
visual
   ↓
concatenação das features
   ↓
SVM / Regressão Logística / Random Forest
   ↓
predição de sarcasmo
```

## Baselines avaliados

Além dos três classificadores individuais, também é avaliado um baseline de voto majoritário entre eles:

```text
SVM
 +
Regressão Logística
 +
Random Forest
 ↓
voto majoritário
```

A classe prevista pelo ensemble corresponde à classe escolhida por pelo menos dois dos três classificadores.

## Métricas

São calculadas as seguintes métricas:

- acurácia;
- precisão;
- recall;
- F1-score;
- F1-macro;
- ROC-AUC, quando disponível.

O F1-macro obtido durante a validação cruzada no conjunto de treino também é mantido nos resultados detalhados para referência.

## Saídas

Os resultados são salvos em:

```text
data/processed/static/<split>/
```

São gerados três arquivos principais:

```text
static_multimodal_test.csv
static_multimodal.csv
static_multimodal_summary.csv
```

### `static_multimodal_test.csv`

Contém as predições por instância para:

- SVM;
- Regressão Logística;
- Random Forest;
- voto majoritário dos três classificadores.

### `static_multimodal.csv`

Contém os resultados detalhados dos métodos, incluindo as métricas calculadas no conjunto de teste e o F1-macro obtido anteriormente na validação cruzada do conjunto de treino.

### `static_multimodal_summary.csv`

Contém uma versão padronizada e resumida das métricas utilizadas para comparação com os demais métodos experimentais.

## Execução

O script pode ser executado a partir da raiz do projeto com:

```bash
python -m src.static.static_multimodal
```

## Papel no experimento

Esses modelos funcionam como baselines de fusão estática multimodal.

A comparação com as estratégias de seleção dinâmica permite avaliar se a escolha adaptativa de classificadores por instância oferece vantagem em relação ao uso de um único classificador treinado com todas as modalidades.

O voto majoritário também fornece uma referência de ensemble sem seleção dinâmica, permitindo distinguir ganhos decorrentes apenas da combinação de classificadores daqueles obtidos pela seleção adaptativa.
