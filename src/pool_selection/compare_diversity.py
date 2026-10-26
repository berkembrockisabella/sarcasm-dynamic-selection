"""Compara pesos desempenho/diversidade usando TRAIN para selecionar e DSEL para validar.

Execute da raiz do projeto:
    python -m src.pool_selection.compare_diversity_configs

Pressupõe que o módulo esteja em src/pool_selection e que já existam
cv_results.csv e oof_predictions.csv gerados por build_pool.py.
"""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .classifiers import build_classifier
from .config import RESULTS_DIR, SPLIT_DIR, POOL_SIZE, TOP_K_FOR_DIVERSITY, RANDOM_STATE
from .diversity import mean_disagreement
from .feature_loader import FeatureStore

DSEL_PATH = SPLIT_DIR.parent / "dsel.csv"
# Peso do desempenho; o peso da diversidade será 1 - alpha.
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _select(results, predictions, y, alpha):
    candidates = (results.sort_values(["mean_f1", "std_f1"], ascending=[False, True]).head(TOP_K_FOR_DIVERSITY).copy())
    lo, hi = candidates["mean_f1"].min(), candidates["mean_f1"].max()
    candidates["performance_norm"] = 1.0 if hi == lo else (candidates["mean_f1"] - lo) / (hi - lo)
    selected = [candidates.iloc[0]["candidate_id"]]
    rows = [{"selection_order": 1, "candidate_id": selected[0],
            "performance_norm": float(candidates.iloc[0]["performance_norm"]),
            "mean_disagreement": 0.0, "selection_score": 1.0}]

    while len(selected) < min(POOL_SIZE, len(candidates)):
        best = None
        for _, row in candidates.iterrows():
            cid = row["candidate_id"]
            if cid in selected:
                continue
            div = mean_disagreement(cid, selected, predictions, y)
            score = alpha * float(row["performance_norm"]) + (1 - alpha) * div
            # Desempate determinístico: score, diversidade, F1 e ID.
            item = (score, div, float(row["mean_f1"]), cid)
            if best is None or item > best[0]:
                best = (item, row, div, score)
        _, row, div, score = best
        selected.append(row["candidate_id"])
        rows.append({"selection_order": len(selected), "candidate_id": row["candidate_id"],
                    "performance_norm": float(row["performance_norm"]),
                    "mean_disagreement": float(div), "selection_score": float(score)})
    selected_df = pd.DataFrame(rows).merge(results, on="candidate_id", how="left")
    selected_df["alpha_performance"] = alpha
    selected_df["alpha_diversity"] = 1 - alpha
    return selected_df


def _dsel_predictions(train, dsel, selected, store):
    y_train = train["Sarcasm"].astype(int).to_numpy()
    y_dsel = dsel["Sarcasm"].astype(int).to_numpy()
    train_keys = train["KEY"].astype(str).tolist()
    dsel_keys = dsel["KEY"].astype(str).tolist()
    preds = {}
    for _, row in selected.iterrows():
        cid = row["candidate_id"]
        modalities = row["modalities"].split("+")
        X_train = store.matrix(train_keys, modalities)
        X_dsel = store.matrix(dsel_keys, modalities)
        params = json.loads(row["best_params"])
        model = build_classifier(row["classifier"])
        model.set_params(**params)
        model.fit(X_train, y_train)
        preds[cid] = model.predict(X_dsel)
    return y_dsel, preds


def main():
    output_dir = RESULTS_DIR / "diversity_config_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_path = RESULTS_DIR / "pool_selection" / "cv_results.csv"
    oof_path = RESULTS_DIR / "pool_selection" / "oof_predictions.csv"
    for path in (SPLIT_DIR, DSEL_PATH, cv_path, oof_path):
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    train, dsel = pd.read_csv(SPLIT_DIR / "train.csv"), pd.read_csv(DSEL_PATH)
    for name, df in [("TRAIN", train), ("DSEL", dsel)]:
        missing = {"KEY", "Sarcasm"} - set(df.columns)
        if missing:
            raise ValueError(f"Colunas ausentes em {name}: {sorted(missing)}")
    train["KEY"] = train["KEY"].astype(str)
    dsel["KEY"] = dsel["KEY"].astype(str)
    train["Sarcasm"] = train["Sarcasm"].astype(int)
    dsel["Sarcasm"] = dsel["Sarcasm"].astype(int)

    results = pd.read_csv(cv_path)
    oof = pd.read_csv(oof_path)
    y_train_oof = oof["y_true"].astype(int).to_numpy()
    predictions = {cid: oof[f"{cid}__pred"].astype(int).to_numpy() for cid in results["candidate_id"] if f"{cid}__pred" in oof.columns}
    results = results[results["candidate_id"].isin(predictions)].copy()
    if len(predictions) < POOL_SIZE:
        raise ValueError("Não há predições OOF suficientes para selecionar a pool.")

    store = FeatureStore()
    summary_rows, selection_frames = [], []
    for alpha in ALPHAS:
        selected = _select(results, predictions, y_train_oof, alpha)
        selected.insert(0, "configuration", f"perf_{alpha:.1f}_div_{1-alpha:.1f}")
        selection_frames.append(selected)
        y_true, dsel_preds = _dsel_predictions(train, dsel, selected, store)
        # Predição por votação majoritária da pool; empate resolvido pela classe 0.
        pred_matrix = np.column_stack(list(dsel_preds.values()))
        ensemble_pred = (pred_matrix.sum(axis=1) > (pred_matrix.shape[1] / 2)).astype(int)
        summary_rows.append({
            "configuration": f"perf_{alpha:.1f}_div_{1-alpha:.1f}",
            "alpha_performance": alpha, "alpha_diversity": 1-alpha,
            "pool_size": len(selected),
            "dsel_accuracy_majority_vote": accuracy_score(y_true, ensemble_pred),
            "dsel_f1_macro_majority_vote": f1_score(y_true, ensemble_pred, average="macro", zero_division=0),
            "dsel_precision_macro_majority_vote": precision_score(y_true, ensemble_pred, average="macro", zero_division=0),
            "dsel_recall_macro_majority_vote": recall_score(y_true, ensemble_pred, average="macro", zero_division=0),
            "dsel_mean_member_f1_macro": np.mean([f1_score(y_true, p, average="macro", zero_division=0) for p in dsel_preds.values()]),
            "dsel_mean_pairwise_disagreement": np.mean([
                np.mean(dsel_preds[a] != dsel_preds[b])
                for i, a in enumerate(dsel_preds) for b in list(dsel_preds)[i+1:]
            ]) if len(dsel_preds) > 1 else 0.0,
            "selected_candidates": ";".join(selected["candidate_id"].tolist()),
        })
        print(f"{alpha:.1f}/{1-alpha:.1f}: F1-macro DSEL={summary_rows[-1]['dsel_f1_macro_majority_vote']:.4f}; "
              f"Acc={summary_rows[-1]['dsel_accuracy_majority_vote']:.4f}")

    summary = pd.DataFrame(summary_rows).sort_values(
        ["dsel_f1_macro_majority_vote", "dsel_mean_member_f1_macro"], ascending=False
    )
    summary.to_csv(output_dir / "configuration_summary.csv", index=False)
    pd.concat(selection_frames, ignore_index=True).to_csv(output_dir / "selected_pools_by_configuration.csv", index=False)
    print("\nResumo ordenado por F1-macro da votação majoritária em DSEL:")
    print(summary[["configuration", "dsel_f1_macro_majority_vote", "dsel_accuracy_majority_vote",
                   "dsel_mean_member_f1_macro", "dsel_mean_pairwise_disagreement"]].to_string(index=False))
    print(f"\nArquivos salvos em: {output_dir}")
    print("Use DSEL para escolher a configuração; mantenha TEST intocado para a avaliação final única.")

if __name__ == "__main__":
    main()
