'''
Construção da pool de classificadores
'''

import numpy as np
import pandas as pd

from .config import PARAM_GRIDS, RESULTS_DIR, TRAIN_PATH
from .cross_validation import make_folds, tune_and_oof
from .diversity import pairwise_table, plot_kappa_error
from .feature_loader import FeatureStore
from .modality_combinations import (
    all_modality_combinations,
    combo_name,
)
from .select_pool import select_diverse_pool

def run():
    output_dir = RESULTS_DIR / "pool_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH).copy()
    required_columns = {"KEY", "SPEAKER", "Sarcasm"}
    missing_columns = required_columns - set(train.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes em TRAIN: {sorted(missing_columns)}")
    
    train["KEY"] = train["KEY"].astype(str)
    train["Sarcasm"] = train["Sarcasm"].astype(int)
    train["SPEAKER"] = train["SPEAKER"].astype(str)
    
    keys = train["KEY"].tolist()
    y = train["Sarcasm"].to_numpy()
    groups = train["SPEAKER"].astype(str).to_numpy()
    
    print(f"\nInstâncias de treino: {len(train)}")
    print(f"Speakers: {train['SPEAKER'].nunique()}")
    print(f"Distribuição das classes: {train['Sarcasm'].value_counts().to_dict()}")

    # Cria 5 folds agrupados por SPEAKER
    # Os folds são criados uma única vez e reutilizados, garantindo uma comparação justa
    folds = make_folds(y, groups)

    fold_assignment = np.full(len(train), -1, dtype=int)
    for fold_id, (_, val_idx) in enumerate(folds, start=1):
        fold_assignment[val_idx] = fold_id
    if np.any(fold_assignment == -1):
        raise RuntimeError("Existem instâncias que não foram atribuídas a nenhum fold de validação.")
    
    pd.DataFrame({
        "KEY": keys,
        "SPEAKER": groups,
        "Sarcasm": y,
        "fold": fold_assignment,
    }).to_csv(output_dir / 'fold_assignments.csv', index=False)
    print(f"\nFolds salvos em: {output_dir / 'fold_assignments.csv'}")

    
    # candidatos formados apenas com as features das modalidades
    store = FeatureStore()
    result_rows = []
    fold_rows = []
    predictions = {}
    oof_table = pd.DataFrame({"KEY": keys, "y_true": y})

    combos = all_modality_combinations()
    total = len(combos) * len(PARAM_GRIDS)
    current = 0

    for combo in combos:
        # Avalia todas as combinações de modalidades.
        modalities = combo_name(combo)
        X = store.matrix(keys, combo)

        # Avalia todos os classificadores e hiperparâmetros.
        for classifier in PARAM_GRIDS:
            current += 1
            candidate_id = f"{modalities}__{classifier}"
            print(f"[{current}/{total}] : {candidate_id}")

            # Grid search + cv + OOF predictions
            summary, fold_df, pred, prob = tune_and_oof(X, y, groups, classifier, folds)
            result_rows.append({
                "candidate_id": candidate_id,
                "modalities": modalities,
                "classifier": classifier,
                "n_features": X.shape[1],
                **summary,
            })

            fold_df = fold_df.copy()
            fold_df.insert(0, "candidate_id", candidate_id)
            fold_rows.append(fold_df)

            predictions[candidate_id] = pred
            oof_table[f"{candidate_id}__pred"] = pred
            oof_table[f"{candidate_id}__prob"] = prob

    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(
        ["mean_f1", "std_f1"],
        ascending=[False, True]
    ).reset_index(drop=True)
    results_df.to_csv(output_dir / "cv_results.csv", index=False)
    
    pd.concat(fold_rows, ignore_index=True).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    
    oof_table.to_csv(output_dir / "oof_predictions.csv", index=False)

    # Calcular diversidade a partir de OOF
    diversity_df = pairwise_table(y, predictions)
    print("build_poool")
    print("Linhas:", len(diversity_df))
    print("Colunas:", diversity_df.columns.tolist())
    print(diversity_df.head())
    diversity_df.to_csv(output_dir / "pairwise_diversity.csv", index=False)
    plot_kappa_error(diversity_df, output_path=(output_dir / "kappa_error.png"))

    # Seleção dos 7 especialistas da pool
    selected = select_diverse_pool(results_df, predictions, y)
    selected.to_csv(output_dir / "pool_definition.csv", index=False) # Salva definição congelada da pool

    print("\nPool selecionada:")
    print(selected[
        ["selection_order", "candidate_id", "mean_f1",
        "mean_disagreement_with_selected"]
    ].to_string(index=False))
    print(f"\nResultados: {output_dir}")

def main():
    run()

if __name__ == "__main__":
    main()
