# Código-fonte (`src`)

Implementação do experimento de seleção dinâmica de classificadores com roteamento por incongruência para detecção multimodal de sarcasmo. A visão geral, o protocolo e os resultados estão no [README principal](../README.md).

## Módulos

| Módulo | O que faz |
| :--- | :--- |
| `analysis/` | Diagnóstico dos extratores de emoção contra a anotação |
| `data/` | Prepara o MUStARD++, gera os 10 splits por falante e extrai o áudio |
| `features/embeddings/` | Representações usadas pelos classificadores (BART, wav2vec2, rosto) |
| `features/valencia/` | Emoção e valência de cada modalidade, incluindo os extratores alternativos e a reextração do rosto do falante |
| `incongruence/` | Grupos de incongruência (A/B e conflito), originais e calibrados |
| `pool_selection/` | Validação cruzada dos 45 candidatos, escolha dos 7 membros e variantes da pool |
| `pool_training/` | Treino e carregamento da pool congelada |
| `moe/` | Soft-MoE |
| `selection/` | Seleção dinâmica (OLA, KNORA-E, KNORA-U) com os roteamentos |
| `static/` | Classificadores estáticos multimodais |

## Scripts na raiz de `src`

| Script | O que faz |
| :--- | :--- |
| `run_experiment.py` | Orquestra pool, seleção dinâmica, MoE, estáticos, sensibilidade e custos de um split (`--split 00`) ou de todos (`--split all`) |
| `aggregate_splits.py` | Agrega os 10 splits: tabelas, Friedman/Nemenyi, Wilcoxon/Holm, erros e custos; aceita o nome de uma variante |
| `compare_variants.py` | Compara as variantes da pool e da incongruência com a configuração padrão |
| `statistical_comparison.py` | McNemar + Holm e bootstrap dentro de cada split (complemento; pulado com `--sem-estatistica`) |

## Como os caminhos dependem do split

`pool_selection/config.py` lê `SPLIT_ID` (definido pelo `run_experiment.py`) e as opções de variante. Os resultados de cada split ficam em subpastas `split_XX/`, e os de cada variante numa subpasta com o nome dela (por exemplo, `split_03/div_0.8/`).
