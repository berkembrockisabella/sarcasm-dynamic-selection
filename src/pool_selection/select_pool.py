'''
Seleção final da pool de classificadores, balanceando desempenho e diversidade
'''

import pandas as pd
from .config import POOL_SIZE, TOP_K_FOR_DIVERSITY
from .diversity import mean_disagreement

def select_diverse_pool(results_df, predictions, y):
    """
    Estratégia gulosa:
    - restringe aos TOP_K candidatos por F1;
    - começa pelo melhor F1;
    - adiciona candidatos equilibrando F1 normalizado e disagreement médio.

    score = 0.70 * performance + 0.30 * diversity
    """
    candidates = (
        results_df
        .sort_values(["mean_f1", "std_f1"], ascending=[False, True])
        .head(TOP_K_FOR_DIVERSITY)
        .copy()
    )

    min_f1 = candidates["mean_f1"].min()
    max_f1 = candidates["mean_f1"].max()

    def norm_f1(value):
        if max_f1 == min_f1:
            return 1.0
        return (value - min_f1) / (max_f1 - min_f1)

    selected = [candidates.iloc[0]["candidate_id"]]
    selection_rows = [{
        "selection_order": 1,
        "candidate_id": selected[0],
        "selection_score": 1.0,
        "mean_disagreement_with_selected": 0.0,
    }]

    while len(selected) < min(POOL_SIZE, len(candidates)):
        best = None

        for _, row in candidates.iterrows():
            cid = row["candidate_id"]
            if cid in selected:
                continue

            div = mean_disagreement(cid, selected, predictions, y)
            score = 0.70 * norm_f1(row["mean_f1"]) + 0.30 * div

            item = (score, div, row["mean_f1"], cid)
            if best is None or item > best:
                best = item

        score, div, _, cid = best
        selected.append(cid)
        selection_rows.append({
            "selection_order": len(selected),
            "candidate_id": cid,
            "selection_score": score,
            "mean_disagreement_with_selected": div,
        })

    selection = pd.DataFrame(selection_rows)
    return selection.merge(results_df, on="candidate_id", how="left")
