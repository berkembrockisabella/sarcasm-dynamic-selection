# Pool training

Módulo responsável pelo **treinamento final dos classificadores da *pool* previamente selecionada**.

Utiliza diretamente o arquivo produzido a partir de `pool selection`:

`data/results/pool/pool_selection/pool_definition.csv`

## Como rodar
Execute a partir da raiz do projeto:

`python -m src.pool_training.train_frozen_pool`

Ao final da execução, a *pool* treinada estará disponível em:

`data/results/pool/frozen_pools/base`

(a pasta se chama `base` por compatibilidade com os modelos já treinados; há uma única pool)

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
  frozen_pools/base/
```

Por exemplo:

`C01 = text + audio`

recebe:

`[text_embedding, audio_embedding]`

Os modelos utilizados durante os folds da validação cruzada não são reutilizados.

A incongruência entre modalidades **não** entra como feature dos especialistas. Ela é usada apenas no roteamento da seleção dinâmica (grupo A/B), em `src/selection/ds_routing.py`.

## Estrutura
```
pool_training/
├── load_frozen_pool.py            carrega a pool já treinada
├── train_frozen_pool.py           treina a pool
└── README.md
```

## Saídas
```
data/
└── results/
    └── pool/
        ├── pool_selection/
        │   └── pool_definition.csv
        │
        └── frozen_pools/
            └── base/
                ├── C01.joblib ... C07.joblib
                └── pool_metadata.csv
```

Os arquivos `.joblib` não são versionados (estão no `.gitignore`); são recriados rodando `train_frozen_pool`.

`pool_metadata.csv` registra informações dos modelos treinados, incluindo:
* identificador do membro
* candidato original
* modalidades
* classificador
* hiperparâmetros
* número de features
* caminho do modelo
