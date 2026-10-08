# Incongruência (`src/incongruence/`)

Esta pasta define, para cada fala, o **grupo A ou B** que é usado no roteamento da seleção dinâmica.

## Arquivo

| Arquivo | O que faz |
|---|---|
| `build_incongruence.py` | Junta a valência e a emoção das 4 modalidades (texto, contexto, áudio, rosto), calcula a polaridade de cada uma e define o grupo da fala |

## Conceitos

- **Valência:** número entre −1 e +1 que diz o quão positiva ou negativa é a emoção de uma modalidade (P(positivas) − P(negativas)). É calculada antes, pelos scripts de `src/features/valencia/`.
- **Polaridade:** classe da emoção predominante da modalidade: **positiva** (happy), **negativa** (angry, disgust, fear, sad) ou **neutra** (neutral, surprise, sem sinal).
- **Grupo A (concordância):** as 4 modalidades têm a mesma polaridade.
- **Grupo B (incongruência):** pelo menos uma modalidade tem polaridade diferente das outras. Neutra conta como uma polaridade própria, então "texto positivo e áudio neutro" já é grupo B.

```
 texto: positiva     contexto: positiva     áudio: positiva     rosto: positiva   ->  grupo A
 texto: positiva     contexto: negativa     áudio: neutra       rosto: positiva   ->  grupo B
```

## Entrada e saída

- **Entrada:** `data/processed/{text,context,audio,visual}_valence.csv` (gerados em `src/features/valencia/`).
- **Saída:** `data/processed/incongruence.csv`, uma linha por fala, com:
  - as 4 valências e as 4 polaridades;
  - `grupo` (A ou B), que é o único dado usado pelo roteamento em `src/selection/ds_routing.py`;
  - `inc_*` (diferença de valência entre cada par de modalidades), `n_oppositions` (pares com polaridades opostas) e `deadpan` (áudio e rosto neutros), usados só pela outra linha de experimentos (`src/selection/dcs_*.py`).

## Como rodar

```
python src/incongruence/build_incongruence.py
```

Rode depois dos quatro scripts de valência.
