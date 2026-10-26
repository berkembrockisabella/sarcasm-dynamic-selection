# Seleção Dinâmica de Classificadores (DS)

Este módulo implementa a seleção dinâmica de classificadores (*Dynamic Selection* - DS), utilizando uma pool congelada de 7 especialistas já treinados. Para cada fala, estima-se a competência dos especialistas com base em exemplos semelhantes do conjunto DSEL e determina quais serão utilizados na predição.

## Como rodar

A partir da raiz do repositório, executar:

```
python -m src.selection.ds_routing
```
Deve ser executado após a definição e treinamento da *pool* de especialistas.

Para executar a análise de sensibilidade ao tamanho da vizinhança:

```
python -m src.selection.ds_routing_k_sweep
```

## Estrutura

| Arquivo | Faz parte do experimento? | O que faz |
|---|---|---|
| `ds_routing.py` | **Sim (principal)** | OLA, KNORA-E e KNORA-U, cada um **sem** e **com** roteamento por incongruência (k = 7) |
| `ds_routing_k_sweep.py` | Sim (análise extra) | Repete o `ds_routing.py` variando o k (3 a 31) para ver se o resultado depende do k |

## Como funciona o `ds_routing.py`

```
 TREINO (897)            DSEL (158)                    TESTE (147)
     |                       |                              |
 1. pool já             2. mede quem acerta            3. para cada fala:
    treinada em            em cada fala                   - acha os 7 vizinhos no DSEL
    `pool_training`        do DSEL                        - vê quem acertou os vizinhos
                                                          - OLA / KNORA-E / KNORA-U
                                                            escolhem quem decide
```

Passo a passo do script:

1. **Carrega os dados:** treino, DSEL e teste, mais o grupo de cada fala (`grupo` A ou B, de `data/processed/incongruence.csv`).
2. **Carrega a pool congelada** (os 7 modelos já treinados) e calcula as predições deles no DSEL e no teste.
3. **Monta o espaço de vizinhança:** concatena as 4 modalidades (texto, contexto, áudio, rosto), cada uma padronizada e normalizada, para ter peso parecido na distância.
4. **Acha os 7 vizinhos** de cada fala de teste dentro do DSEL:
   - **sem roteamento:** procura no DSEL inteiro;
   - **com roteamento:** procura só entre as falas do DSEL do mesmo grupo da fala de teste (A: todas as modalidades com a mesma polaridade; B: pelo menos uma difere).
5. **Escolhe quem decide** olhando quem acertou esses vizinhos:
   - **OLA:** o especialista que mais acertou os vizinhos decide sozinho;
   - **KNORA-E:** votam só os que acertaram **todos** os vizinhos (se nenhum, reduz a vizinhança);
   - **KNORA-U:** todos votam, com peso igual ao número de vizinhos que acertaram.
6. **Salva** as predições e o resumo em `data/processed/selection/ds_routing_test.csv` e `ds_routing_summary.csv`.

**Recapitulando:**

O DSEL é utilizado para **selecionar** quem decide cada fala: procura as falas parecidas e vê quem acertou nelas.

Enquanto o TESTE tem como função **avaliar**: só recebe as predições; nada é ajustado com ele.

Nesta pasta, o treino só é lido para uma coisa: ajustar o `StandardScaler` que padroniza as features antes de calcular a distância entre falas. Ele é ajustado só no treino para que DSEL e teste sejam transformados com a mesma régua, sem influência do teste. Nenhum classificador é treinado ou alterado aqui.

## Métodos de seleção

O módulo implementa três estratégias, com e sem roteamento por incongruência:

* **OLA (*Overall Local Accuracy*):** seleciona o especialista com maior acurácia na região de competência.
* **KNORA-E (*K-Nearest Oracles Eliminate*):** seleciona os especialistas que acertam todos os vizinhos, reduzindo a vizinhança quando necessário.
* **KNORA-U (*K-Nearest Oracles Union*):** combina os especialistas de acordo com a quantidade de vizinhos classificados corretamente.

## Roteamento por incongruência

O roteamento restringe a busca por vizinhos no DSEL ao grupo de incongruência da fala avaliada:

* **Sem roteamento:** busca os vizinhos em todo o conjunto DSEL.
* **Com roteamento:** busca os vizinhos apenas no mesmo grupo, A ou B, definido pela concordância entre as polaridades emocionais das modalidades.

A região de competência utiliza representações de texto, contexto, áudio e rosto, padronizadas e normalizadas para o cálculo das distâncias. O tamanho padrão da vizinhança é (k = 7).


## Saídas

Os resultados são armazenados no diretório `data/processed/selection/`:

```
data/
└── processed/
    └── selection/
        ├── ds_routing_group.csv       métricas de avaliação dos experimentos com grupos
        ├── ds_routing_k_sweep.csv     resultados da análise de sensibilidade ao k
        ├── ds_routing_k_sweep.png     imagem com os resultados ao k
        ├── ds_routing_summary.csv     métricas de avaliação dos experimentos
        └── ds_routing_test.csv        predições dos métodos, com e sem roteamento
```