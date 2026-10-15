# Mixture of Experts (MoE)

Este módulo implementa a abordagem *Mixture of Experts* (MoE), combinando as previsões de classificadores especializados por meio de um mecanismo de roteamento (*gate*), que aprende a determinar a contribuição de cada especialista para uma determinada instância.

## Como rodar

A execução do MoE deve seguir os comandos e parâmetros definidos pelos scripts deste módulo e pelo orquestrador principal do projeto, `src/run_experiment.py`.

A partir da raiz do repositório, executar:

```
python -m src.moe.run_moe
```

Deve ser executado após a definição e treinamento da pool.

## Mecanismo de roteamento (*Gate*)

O gate é responsável por determinar a contribuição dos especialistas de acordo com as informações disponíveis para cada instância.

## Comparação com Seleção Dinâmica

O DES e o MoE são estratégias de combinação adaptativa, mas diferem no mecanismo utilizado para determinar a contribuição dos especialistas.

| Aspecto               | DES                                                    | MoE                                          |
| --------------------- | ------------------------------------------------------ | -------------------------------------------- |
| Mecanismo de decisão  | Estimação de competência dos classificadores.          | Roteamento aprendido por um gate.            |
| Adaptação             | Seleção ou combinação baseada na competência estimada. | Ponderação das previsões conforme a entrada. |
| Treinamento adicional | Depende do algoritmo de seleção utilizado.             | Treinamento do mecanismo de roteamento.      |

## Saídas
```
data/
└── processed/
    └── moe/
        ├── moe_config.json
        ├── moe_summary.csv
        ├── moe_test_predictions.csv
        └── moe_training_history.csv
```