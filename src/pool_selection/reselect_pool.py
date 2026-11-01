'''
Escolhe os 7 membros de uma variante da pool reaproveitando a validacao
cruzada ja feita (cv_results.csv e oof_predictions.csv do split).

Execute da raiz do projeto, com a variante no ambiente:
    POOL_DIVERSITY=0.8 SPLIT_ID=00 python -m src.pool_selection.reselect_pool
(o run_experiment faz isso sozinho com --diversity / --max-contexto)
'''

import pandas as pd

from .config import POOL_DEFINITION_PATH, POOL_DIVERSITY, POOL_MAX_CONTEXTO, RESULTS_DIR, VARIANTE
from .select_pool import select_diverse_pool

pasta_cv = RESULTS_DIR / "pool_selection"
resultados = pd.read_csv(pasta_cv / "cv_results.csv")
oof = pd.read_csv(pasta_cv / "oof_predictions.csv")

# Mesmas entradas que o build_pool passa para a escolha gulosa: predicoes
# fora da amostra de cada candidato e o rotulo verdadeiro do treino
y = oof["y_true"].to_numpy()
predicoes = {
    cid: oof[f"{cid}__pred"].to_numpy()
    for cid in resultados["candidate_id"]
}

pool = select_diverse_pool(resultados, predicoes, y)
POOL_DEFINITION_PATH.parent.mkdir(parents=True, exist_ok=True)
pool.to_csv(POOL_DEFINITION_PATH, index=False)

print(f"Variante: {VARIANTE or 'padrao'} (diversidade = {POOL_DIVERSITY}, max. com contexto = {POOL_MAX_CONTEXTO or 'sem limite'})")
print(pool[["selection_order", "candidate_id", "mean_f1", "mean_disagreement_with_selected"]].to_string(index=False))
print("Membros com contexto:", int(pool["candidate_id"].str.split("__").str[0].str.split("+").apply(lambda m: "context" in m).sum()))
print("Salvo em:", POOL_DEFINITION_PATH)
