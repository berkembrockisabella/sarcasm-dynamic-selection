# Seleção dinâmica (`src/selection/`)

Esta pasta contém o passo final do experimento: dado um pool de 7 classificadores já treinado, escolher **para cada fala do teste** qual(is) deles vai(ão) decidir.

## Arquivos

| Arquivo | Faz parte do experimento? | O que faz |
|---|---|---|
| `ds_routing.py` | **Sim (principal)** | OLA, KNORA-E e KNORA-U, cada um **sem** e **com** roteamento por incongruência (k = 7) |
| `ds_routing_k_sweep.py` | Sim (análise extra) | Repete o `ds_routing.py` variando o k (3 a 31) para ver se o resultado depende do k |
| `dcs_local_accuracy.py`, `dcs_incongruence.py`, `dcs_combined.py` | **Não** | Outra linha de experimentos do time (4 classificadores unimodais: T, C, A, F); não usam a pool de 7 |

## Como funciona o `ds_routing.py`

```
 TREINO (897)            DSEL (158)                    TESTE (147)
     |                       |                              |
 1. treina a pool        2. mede quem acerta            3. para cada fala:
    (já feito em            em cada fala                   - acha os 7 vizinhos no DSEL
    pool_training)          do DSEL                        - vê quem acertou os vizinhos
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

## O que acontece em cada etapa (e onde)

Os classificadores **não são treinados** nesta pasta. A divisão de trabalho é esta:

| Etapa | Onde | Conjunto usado | O que acontece |
|---|---|---|---|
| Escolher os 7 especialistas | `src/pool_selection/` | Treino | Validação cruzada por falante entre os 45 candidatos |
| **Treinar** os 7 especialistas | `src/pool_training/` | Treino | Cada um aprende a detectar sarcasmo; depois ficam congelados |
| **Selecionar** quem decide cada fala | `src/selection/` | DSEL | Procura as falas parecidas e vê quem acertou nelas |
| **Avaliar** | `src/selection/` | Teste | Só recebe as predições; nada é ajustado com ele |

Nesta pasta, o treino só é lido para uma coisa: ajustar o `StandardScaler` que padroniza as features antes de calcular a distância entre falas. Ele é ajustado só no treino para que DSEL e teste sejam transformados com a mesma régua, sem influência do teste. Nenhum classificador é treinado ou alterado aqui.

## O que mudou em `pool_training`

Antes havia duas pools treinadas (uma "base" e uma com a incongruência como feature de entrada). Isso não fazia parte do desenho do experimento, então foi removido:

- agora existe **uma única pool de 7 especialistas**, treinada só com as features das modalidades;
- a incongruência **não entra como feature** dos classificadores: ela só define o grupo A/B usado no roteamento;
- a pasta com os modelos continua se chamando `data/results/pool/frozen_pools/base` (nome mantido para não retreinar), e `load_frozen_pool()` não recebe mais argumento.

## Ordem para rodar

```
python -m src.pool_selection.build_pool          # escolhe os 7 especialistas
python -m src.pool_training.train_frozen_pool    # treina a pool
python src/selection/ds_routing.py               # seleção dinâmica (principal)
python src/selection/ds_routing_k_sweep.py       # opcional: sensibilidade ao k
```
