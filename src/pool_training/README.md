# Pool training

Módulo responsável pelo **treinamento final dos classificadores da *pool* previamente selecionada**.

Utiliza diretamente o arquivo produzido a partir de `pool selection`:

`data/results/pool_selection/pool_definition.csv`

## Como rodar
Execute a partir da raiz do projeto:

`python -m src.pool_training.train_frozen_pool`

Ao final da execução, as *pools* treinadas estarão disponíveis em:

`data/results/frozen_pools`

## Visão geral
O processo de treinamento das *pools* segue o fluxo:

```
          pool_definition.csv
                   │
                   ▼
       ┌──────────────────────┐
       │                      │
       ▼                      ▼
      BASE               INCONGRUENCE
       │                      │
   features               features
   originais              originais
       │                      +
       │                  vetor de
       │                incongruência
       │                      │
       ▼                      ▼
  C01 ... C07            C01 ... C07
       │                      │
       ▼                      ▼
frozen_pools/base/   frozen_pools/incongruence/
```

A partir da definição `pool_definition.csv` são treinadas duas versões pareadas da mesma *pool*:

* **BASE**: utiliza somente as features das modalidades
* **INCONGRUENCE**: utiliza as mesmas features + vetor de incongruência multimodal

## Estrutura
```
pool_training/
├── __init__.py
├── load_frozen_pool.py            carrega uma pool já treinada
├── train_frozen_pool.py           treina as 2 versões da pool (com e sem incongruência)
└── README.md
```

## Condição BASE

Corresponde à versão **sem** incongruência, cada especialista recebe somente as features correspondentes às suas modalidades.

Por exemplo:

`C01 = text + audio`

recebe:

`[text_embedding, audio_embedding]`

## Condição INCONGRUENCE

Corresponde à versão **com** incongruência, cada especialista recebe as features correspondentes às suas modalidades acrescidas das incongruências existentes entre elas.

Para:

`C01 = text + audio`

recebe:

`[text_embedding, audio_embedding, inc_text_audio]`

## Treinamento

Após a definição da *pool*, cada especificação é novamente instanciada e treinada utilizando todo o conjunto TRAIN.

Os modelos utilizados durante os folds da validação cruzada não são reutilizados.

## Saídas
```
data/
└── results/
    ├── pool_selection/
    │   └── pool_definition.csv
    │
    └── frozen_pools/
       ├── base/
       │   ├── C01.joblib
       │   ├── C02.joblib
       │   ├── C03.joblib
       │   ├── C04.joblib
       │   ├── C05.joblib
       │   ├── C06.joblib
       │   ├── C07.joblib
       │   └── pool_metadata.csv
       │
       └── incongruence/
              ├── C01.joblib
              ├── C02.joblib
              ├── C03.joblib
              ├── C04.joblib
              ├── C05.joblib
              ├── C06.joblib
              ├── C07.joblib
              └── pool_metadata.csv
```

`metadata.csv` registra informações dos modelos treinados, incluindo:
* identificador do membro
* candidato original
* modalidades
* classificador
* hiperparâmetros
* condição
* número de features
* caminho do modelo
