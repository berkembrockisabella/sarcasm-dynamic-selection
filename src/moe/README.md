# Mixture of Experts (MoE)

Este módulo implementa a abordagem *Mixture of Experts* (MoE), que combina as previsões de classificadores especializados por meio de um mecanismo de roteamento (*gate*), que aprende a determinar a contribuição de cada especialista para uma determinada instância.

## Como rodar

A execução do MoE deve seguir os comandos e parâmetros definidos pelos scripts deste módulo e pelo orquestrador principal do projeto, `src/run_experiment.py`.

A partir da raiz do repositório, executar:

```
python -m src.moe.run_moe
```

Deve ser executado após a definição e treinamento da *pool* de especialistas.

## Estrutura

Os principais arquivos do módulo são:

| Arquivo | Responsabilidade |
|---|---|
| `moe_expert_predictions.py` | Calcula as probabilidades de sarcarmo estimadas de cada fala por cada especialista. |
| `moe_gate.py` | Define a rede neural MLP responsável por produzir os pesos de cada especialista. |
| `moe.py` | Implementa o mecanismo MoE, combinando as probabilidades dos especialistas de acordo com os pesos produzidos pelo gate. |
| `run_moe.py` | Orquestra a execução do MoE, carrega os especialistas, treina o gate no DSEL e avalia o modelo no TEST. |

## Mecanismo de roteamento (*Gate*)

O gate é uma rede neural do tipo *Multi-Layer Perceptron* (MLP), responsável por aprender e determinar a contribuição de cada especialista para uma determinada instância.

Sua arquitetura é formada por:
* Uma camada de entrada que recebe as características utilizadas pelo mecanismo de roteamento;
* Uma camada totalmente conectada com 32 neurônios;
* Uma função de ativação ReLU;
* Uma camada de Dropout com taxa de 0,2;
* Uma camada de saída com um neurônio para cada especialista da pool.

A camada de saída produz **logits**, isto é, pontuações brutas que representam a preferência do gate por cada especialista. A função *Softmax* transforma essas pontuações em pesos não negativos cuja soma é igual a 1.

Esses pesos são utilizados para calcular uma soma ponderada das probabilidades produzidas pelos especialistas, resultando na previsão final do MoE.

## Treinamento

O treinamento do MoE utiliza o conjunto DSEL para ajustar os parâmetros do gate.

O treinamento conta com:
* **Função de perda:** *Binary Cross-Entropy* (BCE);
* **Otimizador:** Adam;
* **Taxa de aprendizado:** 0,001;
* **Weight decay:** 0,0001;
* **Tamanho do lote:** 32;
* **Número máximo de épocas:** 200;
* **Critério de parada:** paciência de 20 épocas.

## Comparação com Seleção Dinâmica

O DES e o MoE são estratégias de combinação adaptativa, mas diferem no mecanismo utilizado para determinar a contribuição dos especialistas.

| Aspecto               | DES                                                    | MoE                                          |
| --------------------- | ------------------------------------------------------ | -------------------------------------------- |
| Mecanismo de contribuição  | Estima a competência dos especialistas        | Aprende os pesos por meio de um gate             |
| Decisão | Seleciona ou combina especialistas conforme a estratégia DES utilizada | Calcula uma soma ponderada das probabilidades |
| Aprendizado do mecanismo de decisão | Depende da estratégia DES utilizada | Treina uma rede neural no DSEL |
| Adaptação             | Baseada na competência estimada para a instância  | Baseada nos pesos produzidos pelo gate a partir de sua entrada |

## Saídas
Os resultados são armazenados no diretório `data/processed/moe/`:

```
data/
└── processed/
    └── moe/
        ├── moe_config.json             configurações utilizadas
        ├── moe_results.csv             métricas de avaliação final
        ├── moe_summary.csv             métricas para comparação
        ├── moe_test.csv                previsões conjunto TEST
        └── moe_training_history.csv    histórico do treinamento do gate,
                                        incluindo perdas registradas épocas
```