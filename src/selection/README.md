# Seleção Dinâmica de Classificadores (`src/selection/`)

Este módulo implementa a seleção dinâmica de classificadores (*Dynamic Selection* - DS), utilizando uma pool congelada de 7 especialistas já treinados. Para cada fala de teste, `ds_routing.py` escolhe quais dos especialistas serão utilizados na predição, com base em falas parecidas do DSEL.

## Como rodar

Pelo orquestrador (recomendado):

```
python -m src.run_experiment --split 00 --only ds
```

Ou direto, com o split no ambiente: `SPLIT_ID=00 python -m src.selection.ds_routing`. `DS_K` muda o tamanho da vizinhança (padrão 7) e `DS_N_SORTEIOS` o número de sorteios dos controles aleatórios (padrão 10).

## Como funciona

1. **Carrega os dados:** treino, DSEL e teste do split e os grupos de incongruência de cada fala (`grupo` A ou B e grupo de conflito, de `data/processed/incongruence.csv` ou, com `--incongruencia calibrada`, de `data/processed/incongruence_calibrada/split_XX.csv`).
2. **Carrega a pool congelada** (os 7 modelos já treinados) e calcula as predições deles no DSEL e no teste.
3. **Monta o espaço de vizinhança:** concatena as 4 modalidades (texto, contexto, áudio, rosto), cada uma padronizada (ajustada só no treino) e normalizada (L2), para ter peso parecido na distância.
4. **Acha os 7 vizinhos** de cada fala de teste dentro do DSEL:
   - **sem roteamento:** procura no DSEL inteiro;
   - **com roteamento:** procura só entre as falas do DSEL do mesmo grupo da fala de teste (A: todas as modalidades com a mesma polaridade; B: pelo menos uma difere).
5. **Escolhe quem decide** olhando quem acertou esses vizinhos:
   - **OLA (*Overall Local Accuracy*):** o especialista que mais acertou os vizinhos decide sozinho;
   - **KNORA-E (*K-Nearest Oracles Eliminate*):** votam só os que acertaram **todos** os vizinhos (se nenhum, reduz a vizinhança);
   - **KNORA-U (*K-Nearest Oracles Union*):** todos votam, com peso igual ao número de vizinhos que acertaram.

## Roteamentos

| Roteamento | Onde procura os vizinhos |
|---|---|
| `sem_roteamento` | No conjunto DSEL inteiro |
| `com_roteamento` | Só no grupo A/B da fala (A: as 4 modalidades com a mesma polaridade) |
| `conflito` | Só no grupo de conflito da fala (alguma modalidade positiva e outra negativa) |
| `aleatorio` | Grupos sorteados com o mesmo tamanho de A e B (controle do A/B) |
| `conflito_aleatorio` | Grupos sorteados com o mesmo tamanho dos de conflito (controle do conflito) |

Os controles aleatórios usam 10 sorteios; o resumo traz cada sorteio e a média.

## Saídas
Os resultados são armazenados em `data/processed/selection/split_XX/`

| Arquivo | Conteúdo |
|---|---|
| `ds_routing_dsel.csv` | Voto e oráculo no DSEL (critério de escolha entre variantes) |
| `ds_routing_group.csv` | Métricas por grupo A/B |
| `ds_routing_test.csv` | Predição de cada método e roteamento, por fala |
| `ds_routing_summary.csv` | Métricas no teste, com pior e melhor membro, voto da pool e oráculo |
| `ds_routing_tempo.csv` | Tempo da seleção por fala |
| `ds_routing_membros_test.csv` | Predição de cada membro no teste (análise de erros) |
| `k_<K>/` | Mesmas saídas para outros valores de k (sensibilidade) |
