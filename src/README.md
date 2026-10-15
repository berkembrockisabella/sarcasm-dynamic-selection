# Código-fonte (src)
Esta pasta contém toda a implementação dos experimentos de detecção multimodal de sarcasmo.


| Módulo | Descrição |
| :--- | :--- |
| `data/` | Preparação, carregamento e organização dos dados utilizados nos experimentos |
| `features/` | Disponibilização e organização das representações multimodais utilizadas pelos classificadores |
| `incongruence/` | Cálculo e preparação das características relacionadas à incongruência multimodal |
| `pool_selection/` | Avaliação de candidatos e seleção de especialistas para compor a *pool* |
| `pool_training/` | Treinamento, persistência e carregamento dos classificadores selecionados |
| `moe/` | Implementação de Mixture of Experts (MoE) |
| `selection/` | Implementação dos métodos de Seleção Dinâmica |
| `static/` | Implementação dos modelos estáticos |
| **`run_experiment.py`** | Ponto de entrada para orquestrar a execução dos experimentos |

## Fluxo geral
1. **Preparação dos dados:** carregamento das representações multimodais e dos rótulos.
2. **Formação da pool:** avaliação de combinações de modalidades, classificadores e hiperparâmetros, utilizando exclusivamente o conjunto TRAIN.
3. **Seleção dos especialistas:** escolha de uma pool que considere desempenho preditivo e diversidade entre os classificadores.
4. **Treinamento final dos especialistas:** treinamento final dos modelos selecionados e armazenamento dos artefatos necessários para reutilização.
5. **Estimação de competência ou roteamento:** utilização do conjunto DSEL para configurar os métodos de combinação adaptativa.
6. **Avaliação final:** aplicação dos métodos ao conjunto TEST e cálculo das métricas de desempenho.

## Métodos experimentais
O projeto contempla diferentes estratégias de classificação e combinação de especialistas:

* **Classificadores estáticos:** utilizam um classificador individual como referência.
* **Seleção dinâmica de classificadores (DES):** seleciona ou combina especialistas de acordo com a competência estimada para cada exemplo.
* **DES com incongruência multimodal:** incorpora explicitamente características de incongruência ao processo de seleção dinâmica.
* **Mixture of Experts (MoE):** utiliza um mecanismo de roteamento aprendido para ponderar as previsões dos especialistas.

## `run_experiment.py`
Orquestra os experimentos principais de Seleção Dinâmica (DS), Mixture of Experts (MoE) e baselines estáticos

Para executar:
`python -m src.run_experiment`


