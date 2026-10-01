'''
Validação cruzada estratificada 5-fold com grupos (StratifiedGroupKFold)
para seleção de hiperparâmetros e geração de predições fora da amostra (OOF).
'''

import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .classifiers import build_classifier
from .config import N_SPLITS, PARAM_GRIDS, RANDOM_STATE, SCORING

def make_folds(y, groups):
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return list(cv.split(np.zeros(len(y)), y, groups))

def tune_and_oof(X, y, groups, classifier_name, folds):
    """
    GridSearchCV nos mesmos 5 folds para a escolha de hiperparâmetros
    Com os melhores parâmetros congelados, refaz os 5 folds e gera OOF.
    """
    base = build_classifier(classifier_name)

    search = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRIDS[classifier_name],
        scoring=SCORING,
        cv=folds,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y, groups=groups)

    best_params = search.best_params_
    oof_pred = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros(len(y), dtype=float)
    fold_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        model = clone(base).set_params(**best_params)
        model.fit(X[train_idx], y[train_idx])

        pred = model.predict(X[val_idx])
        prob = model.predict_proba(X[val_idx])[:, 1]

        oof_pred[val_idx] = pred
        oof_prob[val_idx] = prob

        fold_rows.append({
            "fold": fold_id,
            "f1_macro": f1_score(y[val_idx], pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y[val_idx], pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y[val_idx], pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y[val_idx], pred),
            "pr_auc": average_precision_score(y[val_idx], prob),
        })

    fold_df = pd.DataFrame(fold_rows)

    summary = {
        "best_params": json.dumps(best_params, sort_keys=True),
        "grid_best_f1": float(search.best_score_),
        "mean_f1": float(fold_df["f1_macro"].mean()),
        "std_f1": float(fold_df["f1_macro"].std(ddof=1)),
        "mean_precision": float(fold_df["precision_macro"].mean()),
        "mean_recall": float(fold_df["recall_macro"].mean()),
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "mean_pr_auc": float(fold_df["pr_auc"].mean()),
    }
    return summary, fold_df, oof_pred, oof_prob
