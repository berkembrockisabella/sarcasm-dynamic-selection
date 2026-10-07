# Pool selection

Módulo responsável pela **formação da *pool* de classificadores**. A seleção é realizada uma única vez, somente sobre o conjunto de treinamento e sem incorporar as *features* de incongruência multimodal para garantir uma comparação controlada do experimento.

## Como rodar
Execute a partir da raiz do projeto:

`python -m src.pool_selection.build_pool`

Ao final da execução, a definição dos 7 especialistas estará disponível em:

`data/results/pool/pool_selection/pool_definition.csv`

## Visão geral
O processo de seleção de classificadores da *pool* segue o fluxo:

```
            TRAIN
              │
              ▼
    ┌─────────────────────┐
    │   FORMAÇÃO DA POOL  │
    |---------------------|
    │  15 combinações de  │
    │     modalidades     │
    │          ×          │
    │  3 classificadores  │
    │          ×          │
    │   hiperparâmetros   │
    ├─────────────────────┤
    │      5-fold CV      │
    │                     │
    │  Out-of-Fold (OOF)  │
    │     predictions     │
    │          │          │
    │     ┌────┴────┐     │
    │     ▼         ▼     │
    desempenho  diversidade
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ SELEÇÃO DA POOL     │
    |---------------------|
    │  desempenho +       │
    │  diversidade        │
    └──────────┬──────────┘
               │
               ▼
        7 especialistas
               │
               ▼
       pool_definition.csv
```

## Estrutura
```
pool_selection/
├── build_pool.py                   orquestra todo o processo de formação da pool
├── classifiers.py                  define os classificadores avaliados e sua construção
├── config.py                       contém caminhos, configurações e espaços de hiperparâmetros
├── cross_validation.py             executa a validação cruzada, otimização de hiperparâmetros e geração das predições OOF
├── diversity.py                    calcula as métricas de diversidade entre classificadores
├── feature_loader.py               carrega e combina as representações das diferentes modalidades
├── modality_combinations.py        gera as combinações de modalidades
├── select_pool.py                  seleciona os 7 membros finais considerando desempenho e diversidade
└── README.md
```

## Combinações de modalidades
São consideradas todas as 15 combinações não vazias das quatro modalidades disponíveis:
* text
* context
* audio
* visual

Isso inclui combinações unimodais, bimodais, trimodais e multimodal completa.

Exemplos: text, audio, text+audio, text+visual, context+audio, text+audio+visual, text+context+audio+visual

## Classificadores
Para cada combinação de modalidades são avaliadas três famílias de classificadores:

* Support Vector Machine (SVM)
* Logistic Regression (LR)
* Random Forest (RF)

Cada combinação entre modalidades e família de classificador constitui um candidato à pool.

Com 15 combinações de modalidades e 3 classificadores:

`15 × 3 = 45 candidatos`

Os diferentes hiperparâmetros não constituem membros independentes da pool. Para cada candidato, a validação cruzada determina a configuração de hiperparâmetros utilizada em sua especificação final.

## Validação cruzada

A avaliação dos candidatos utiliza validação cruzada com **5 folds**, respeitando o agrupamento por falante (SPEAKER).

Os mesmos *folds* são utilizados para todos os candidatos, permitindo uma comparação consistente entre combinações de modalidades e classificadores. As predições realizadas sobre os *folds* de validação são armazenadas como **Out-of-Fold predictions (OOF)**.

As predições OOF permitem avaliar desempenho e diversidade sem calcular essas métricas sobre as mesmas amostras utilizadas para ajustar cada modelo.

## Diversidade

Além do desempenho individual, a formação da pool considera a **diversidade** entre os candidatos.

As métricas são calculadas a partir das predições OOF.

Entre as análises disponíveis estão:

* Disagreement
* Double Fault
* Q-statistic
* Cohen's Kappa entre classificadores
* Kappa-Error Diagram

O Kappa-Error Diagram representa cada par de classificadores por:

```
X = Cohen's Kappa entre suas predições
Y = erro médio do par
```

Assim, é possível analisar conjuntamente desempenho e similaridade entre os candidatos.

## Seleção da *pool*
A seleção final da *pool* é realizada por meio de uma estratégia gulosa que considera simultaneamente o desempenho individual dos classificadores e sua diversidade em relação aos membros já selecionados. 

O objetivo é evitar que a pool seja formada apenas pelos classificadores com maior desempenho individual, pois modelos com comportamentos muito semelhantes podem produzir erros redundantes. Assim, busca-se uma *pool* composta por especialistas que apresentem **bom desempenho e padrões de erro complementares**.

A seleção é realizada a partir das predições OOF produzidas durante a validação cruzada, evitando avaliar a diversidade sobre predições obtidas diretamente nos dados utilizados para ajustar cada modelo.

O procedimento ocorre iterativamente:

1. Os candidatos são inicialmente avaliados de acordo com seu desempenho médio na validação cruzada.
2. O candidato com melhor desempenho é escolhido como primeiro membro da pool.
3. Para cada candidato ainda não selecionado, é calculada sua diversidade média em relação aos classificadores que já pertencem à pool.
4. Desempenho e diversidade são combinados em um único critério de seleção.
5. O candidato com maior valor nesse critério é adicionado à pool.
6. O processo é repetido até que sejam selecionados 7 especialistas.

Para um candidato $$(C_i)$$, esse critério pode ser representado por:

$$Score(C_i) = \alpha \cdot P(C_i) + (1-\alpha) \cdot D(C_i, S)$$

em que:

* $$(P(C_i))$$ representa o desempenho normalizado do candidato $$(C_i)$$, obtido durante a validação cruzada;
* $$(D(C_i,S))$$ representa sua diversidade média em relação ao conjunto $$(S)$$ de classificadores já selecionados;
* $$(\alpha)$$ controla a importância relativa entre desempenho e diversidade.

Na configuração utilizada, o desempenho possui peso de 0.70 e a diversidade peso de 0.30.

Essa abordagem produz uma *pool* heterogênea não apenas em termos de modalidades e famílias de classificadores, mas também em relação aos padrões de decisão observados durante a validação cruzada.

A seleção resulta em uma especificação contendo, para cada membro:
* modalidades
* classificador
* hiperparâmetros

Por exemplo:
```
C01 = text+audio | SVM | C=10, gamma=0.01
C02 = text+visual | Random Forest | ...
C03 = context+audio | Logistic Regression | ...
...
C07 = ...
```

Nesta etapa os modelos finais não são treinados. O resultado é apenas a **definição congelada da pool**.

## Saídas
```
data/
└── results/
    └── pool/
        └── pool_selection/
            ├── fold_assignments.csv
            ├── cv_results.csv
            ├── fold_metrics.csv
            ├── oof_predictions.csv
            ├── pairwise_diversity.csv
            ├── kappa_error.png
            └── pool_definition.csv       definição congelada da pool
```

A definição resultante `pool_definition.csv` será posteriormente utilizada para treinar duas versões da mesma pool em `pool_training`

Dessa forma, composição da pool, classificadores hiperparâmetros permanecem constantes entre as condições.
