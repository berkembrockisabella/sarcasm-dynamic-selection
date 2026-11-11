# Treino da pool (`src/pool_training/`)

Treina os 7 membros escolhidos em `pool_selection` com o treino inteiro do split e salva os modelos congelados.

## Como rodar

Pelo orquestrador, que treina a pool só se ela ainda não existir:

```
python -m src.run_experiment --split 00
```

Ou direto: `SPLIT_ID=00 python -m src.pool_training.train_frozen_pool`.

## Visão geral
Cada especificação de `pool_definition.csv` é instanciada de novo e treinada com todo o conjunto TRAIN, recebendo somente as features das suas modalidades:

```
   pool_definition.csv
           │
           ▼
   features das modalidades
   de cada especialista
           │
           ▼
      C01 ... C07
           │
           ▼
  frozen_pools/
```

Por exemplo:

`C01 = text + audio`

recebe:

`[text_embedding, audio_embedding]`

## Estrutura
```
pool_training/
├── load_frozen_pool.py            carrega a pool já treinada
├── train_frozen_pool.py           treina a pool
└── README.md
```

## Entrada e saída

| | Caminho |
|---|---|
| Definição da pool | `data/results/pool/split_XX/pool_selection/pool_definition.csv` |
| Pool treinada | `data/results/pool/split_XX/frozen_pools/` (`C01.joblib` a `C07.joblib` e `pool_metadata.csv`) |

`pool_metadata.csv` registra informações dos modelos treinados, incluindo:
* identificador do membro
* candidato original
* modalidades
* classificador
* hiperparâmetros
* número de features
* caminho do modelo

Uma variante da *pool* usa as mesmas pastas com uma subpasta extra (por exemplo, `pool_selection/div_0.8/` e `frozen_pools/div_0.8/`).