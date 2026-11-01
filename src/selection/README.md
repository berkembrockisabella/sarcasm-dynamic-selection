# Seleção dinâmica (`src/selection/`)

`ds_routing.py` escolhe, para cada fala de teste, quais dos 7 membros da pool congelada decidem, com base em falas parecidas do DSEL.

## Como rodar

Pelo orquestrador (recomendado):

```
python -m src.run_experiment --split 00 --only ds
```

Ou direto, com o split no ambiente: `SPLIT_ID=00 python -m src.selection.ds_routing`. `DS_K` muda o tamanho da vizinhança (padrão 7) e `DS_N_SORTEIOS` o número de sorteios dos controles aleatórios (padrão 10).

## Como funciona

1. Carrega treino, DSEL e teste do split e os grupos de incongruência de cada fala.
2. Calcula as predições dos 7 membros no DSEL e no teste.
3. Monta o espaço de vizinhança: as 4 modalidades concatenadas, cada uma padronizada (ajustada só no treino) e normalizada (L2).
4. Acha os k = 7 vizinhos de cada fala de teste no DSEL, conforme o roteamento.
5. Decide com cada método:
   - **OLA:** o membro que mais acertou os vizinhos;
   - **KNORA-E:** votam os que acertaram todos os vizinhos (se nenhum, a vizinhança diminui);
   - **KNORA-U:** todos votam, com peso igual ao número de vizinhos acertados.

## Roteamentos

| Roteamento | Onde procura os vizinhos |
|---|---|
| `sem_roteamento` | No DSEL inteiro |
| `com_roteamento` | Só no grupo A/B da fala (A: as 4 modalidades com a mesma polaridade) |
| `conflito` | Só no grupo de conflito da fala (alguma modalidade positiva e outra negativa) |
| `aleatorio` | Grupos sorteados com o mesmo tamanho de A e B (controle do A/B) |
| `conflito_aleatorio` | Grupos sorteados com o mesmo tamanho dos de conflito (controle do conflito) |

Os controles aleatórios usam 10 sorteios; o resumo traz cada sorteio e a média.

## Saídas (`data/processed/selection/split_XX/`)

| Arquivo | Conteúdo |
|---|---|
| `ds_routing_test.csv` | Predição de cada método e roteamento, por fala |
| `ds_routing_summary.csv` | Métricas no teste, com pior e melhor membro, voto da pool e oráculo |
| `ds_routing_group.csv` | Métricas por grupo A/B |
| `ds_routing_membros_test.csv` | Predição de cada membro no teste (análise de erros) |
| `ds_routing_dsel.csv` | Voto e oráculo no DSEL (critério de escolha entre variantes) |
| `ds_routing_tempo.csv` | Tempo da seleção por fala |
| `k_<K>/` | Mesmas saídas para outros valores de k (sensibilidade) |
