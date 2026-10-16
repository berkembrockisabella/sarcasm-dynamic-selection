from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_PATH = Path(
    "data/processed/features/visual_features.csv"
)

TRAIN_PATH = Path("data/splits/train.csv")
DSEL_PATH = Path("data/splits/dsel.csv")
TEST_PATH = Path("data/splits/test.csv")

MODEL_DIR = Path("models/pool")
PREDICTIONS_DIR = Path(
    "data/processed/pool_predictions"
)

MODEL_PATH = (
    MODEL_DIR / "visual_logistic_regression.joblib"
)

DSEL_PRED_PATH = (
    PREDICTIONS_DIR / "visual_dsel_predictions.csv"
)

TEST_PRED_PATH = (
    PREDICTIONS_DIR / "visual_test_predictions.csv"
)


def prepare_split(
    split_df,
    features_df
):

    merged = split_df.merge(
        features_df,
        on="KEY",
        how="left",
        validate="one_to_one"
    )

    if merged.isna().any().any():
        raise ValueError(
            "Existem valores ausentes após alinhar "
            "o split com as features visuais."
        )

    feature_columns = [
        col
        for col in features_df.columns
        if col != "KEY"
    ]

    X = (
        merged[feature_columns]
        .to_numpy()
    )

    y = (
        merged["Sarcasm"]
        .astype(int)
        .to_numpy()
    )

    return X, y


def print_metrics(
    name,
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"F1 Macro:  {f1_macro:.4f}")


def main():

    print(
        "Carregando features visuais..."
    )

    features_df = pd.read_csv(
        FEATURES_PATH
    )

    print(
        f"Features: {features_df.shape}"
    )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    dsel_df = pd.read_csv(
        DSEL_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    print("\nTamanhos dos conjuntos:")
    print(f"Train: {len(train_df)}")
    print(f"DSEL:  {len(dsel_df)}")
    print(f"Test:  {len(test_df)}")

    X_train, y_train = prepare_split(
        train_df,
        features_df
    )

    X_dsel, y_dsel = prepare_split(
        dsel_df,
        features_df
    )

    X_test, y_test = prepare_split(
        test_df,
        features_df
    )

    print("\nDimensões:")
    print(f"X_train: {X_train.shape}")
    print(f"X_dsel:  {X_dsel.shape}")
    print(f"X_test:  {X_test.shape}")

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )

    print(
        "\nExecutando validação cruzada "
        "por falante no Train..."
    )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision": "precision",
        "recall": "recall",
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        groups=train_df["SPEAKER"],
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print("\nResultados CV - Train")

    for metric in scoring:

        values = cv_results[
            f"test_{metric}"
        ]

        print(
            f"{metric}: "
            f"{values.mean():.4f} "
            f"+/- {values.std():.4f}"
        )

    print(
        "\nTreinando classificador visual "
        "no Train completo..."
    )

    model.fit(
        X_train,
        y_train
    )

    dsel_pred = model.predict(
        X_dsel
    )

    dsel_prob = model.predict_proba(
        X_dsel
    )[:, 1]

    print_metrics(
        "Desempenho no DSEL",
        y_dsel,
        dsel_pred
    )

    test_pred = model.predict(
        X_test
    )

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    dsel_output = pd.DataFrame(
        {
            "KEY": dsel_df["KEY"].values,
            "label": y_dsel,
            "prediction": dsel_pred,
            "prob_sarcasm": dsel_prob,
        }
    )

    test_output = pd.DataFrame(
        {
            "KEY": test_df["KEY"].values,
            "label": y_test,
            "prediction": test_pred,
            "prob_sarcasm": test_prob,
        }
    )

    dsel_output.to_csv(
        DSEL_PRED_PATH,
        index=False
    )

    test_output.to_csv(
        TEST_PRED_PATH,
        index=False
    )

    print("\nArquivos salvos:")
    print(MODEL_PATH)
    print(DSEL_PRED_PATH)
    print(TEST_PRED_PATH)

    print(
        "\nClassificador visual F concluído."
    )


if __name__ == "__main__":
    main()