# Treino da pool (`src/pool_training/`)

Treina os 7 membros escolhidos em `pool_selection` com o treino inteiro do split e salva os modelos congelados.

## Como rodar

Pelo orquestrador, que treina a pool só se ela ainda não existir:

```
python -m src.run_experiment --split 00
```

Ou direto: `SPLIT_ID=00 python -m src.pool_training.train_frozen_pool`.

## Entrada e saída

| | Caminho |
|---|---|
| Definição da pool | `data/results/pool/split_XX/pool_selection/pool_definition.csv` |
| Pool treinada | `data/results/pool/split_XX/frozen_pools/` (`C01.joblib` a `C07.joblib` e `pool_metadata.csv`) |

Uma variante da pool usa as mesmas pastas com uma subpasta extra (por exemplo, `pool_selection/div_0.8/` e `frozen_pools/div_0.8/`).

Cada membro recebe só as features das suas modalidades. A incongruência não entra como feature: ela só define os grupos do roteamento.

Os `.joblib` não são versionados (estão no `.gitignore`); são recriados pelo `train_frozen_pool`, que leva segundos por split.
