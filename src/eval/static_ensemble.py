from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# --------------------------------------------------
# Caminhos
# --------------------------------------------------

PREDICTIONS_DIR = Path(
    "data/processed/pool_predictions"
)

OUTPUT_DIR = Path(
    "data/processed/eval"
)

DSEL_OUTPUT_PATH = (
    OUTPUT_DIR / "static_ensemble_dsel.csv"
)

TEST_OUTPUT_PATH = (
    OUTPUT_DIR / "static_ensemble_test.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "static_ensemble_summary.csv"
)


# --------------------------------------------------
# Arquivos dos especialistas
# --------------------------------------------------

DSEL_FILES = {
    "T": PREDICTIONS_DIR / "text_dsel_predictions.csv",
    "C": PREDICTIONS_DIR / "context_dsel_predictions.csv",
    "A": PREDICTIONS_DIR / "audio_dsel_predictions.csv",
    "F": PREDICTIONS_DIR / "visual_dsel_predictions.csv",
}

TEST_FILES = {
    "T": PREDICTIONS_DIR / "text_test_predictions.csv",
    "C": PREDICTIONS_DIR / "context_test_predictions.csv",
    "A": PREDICTIONS_DIR / "audio_test_predictions.csv",
    "F": PREDICTIONS_DIR / "visual_test_predictions.csv",
}


# --------------------------------------------------
# Métricas
# --------------------------------------------------

def calculate_metrics(
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
    }


def print_metrics(
    name,
    metrics
):
    print(f"\n{name}")

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"F1 Macro:  "
        f"{metrics['f1_macro']:.4f}"
    )


# --------------------------------------------------
# Carregamento e alinhamento
# --------------------------------------------------

def load_predictions(files):
    merged = None

    for model_name, path in files.items():

        df = pd.read_csv(
            path
        )

        required_columns = {
            "KEY",
            "label",
            "prob_sarcasm",
        }

        missing = (
            required_columns
            -
            set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{model_name}: "
                f"colunas ausentes: {missing}"
            )

        current = df[
            [
                "KEY",
                "label",
                "prob_sarcasm",
            ]
        ].copy()

        current = current.rename(
            columns={
                "label":
                    f"label_{model_name}",

                "prob_sarcasm":
                    f"prob_{model_name}",
            }
        )

        if merged is None:

            merged = current

        else:

            merged = merged.merge(
                current,
                on="KEY",
                how="inner",
                validate="one_to_one"
            )

    # --------------------------------------------------
    # Confere labels
    # --------------------------------------------------

    reference_label = (
        f"label_{list(files.keys())[0]}"
    )

    for model_name in files.keys():

        label_column = (
            f"label_{model_name}"
        )

        inconsistent = (
            merged[label_column]
            !=
            merged[reference_label]
        )

        if inconsistent.any():
            raise ValueError(
                f"Labels inconsistentes "
                f"para o modelo {model_name}."
            )

    merged["label"] = (
        merged[reference_label]
        .astype(int)
    )

    label_columns = [
        column
        for column in merged.columns
        if column.startswith("label_")
    ]

    merged = merged.drop(
        columns=label_columns
    )

    return merged


# --------------------------------------------------
# Ensemble estático
# --------------------------------------------------

def run_static_ensemble(
    files,
    output_path,
    dataset_name
):

    print(
        f"\nProcessando {dataset_name}..."
    )

    df = load_predictions(
        files
    )

    model_names = list(
        files.keys()
    )

    probability_columns = [
        f"prob_{model}"
        for model in model_names
    ]

    # --------------------------------------------------
    # Média simples das probabilidades
    # --------------------------------------------------

    df["ensemble_probability"] = (
        df[
            probability_columns
        ]
        .mean(axis=1)
    )

    # Threshold padrão
    df["ensemble_prediction"] = (
        df["ensemble_probability"]
        >= 0.5
    ).astype(int)

    # --------------------------------------------------
    # Métricas
    # --------------------------------------------------

    y_true = (
        df["label"]
        .to_numpy()
    )

    y_pred = (
        df["ensemble_prediction"]
        .to_numpy()
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    print_metrics(
        f"Ensemble estático - {dataset_name}",
        metrics
    )

    # --------------------------------------------------
    # Salva
    # --------------------------------------------------

    df.to_csv(
        output_path,
        index=False
    )

    return metrics


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Static Ensemble - média das "
        "probabilidades T, C, A e F"
    )

    # --------------------------------------------------
    # DSEL
    # --------------------------------------------------

    dsel_metrics = (
        run_static_ensemble(
            DSEL_FILES,
            DSEL_OUTPUT_PATH,
            "DSEL"
        )
    )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    test_metrics = (
        run_static_ensemble(
            TEST_FILES,
            TEST_OUTPUT_PATH,
            "Test"
        )
    )

    # --------------------------------------------------
    # Resumo
    # --------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "dataset":
                    "DSEL",

                "method":
                    "Static_Average",

                "accuracy":
                    dsel_metrics[
                        "accuracy"
                    ],

                "precision":
                    dsel_metrics[
                        "precision"
                    ],

                "recall":
                    dsel_metrics[
                        "recall"
                    ],

                "f1":
                    dsel_metrics[
                        "f1"
                    ],

                "f1_macro":
                    dsel_metrics[
                        "f1_macro"
                    ],
            },
            {
                "dataset":
                    "Test",

                "method":
                    "Static_Average",

                "accuracy":
                    test_metrics[
                        "accuracy"
                    ],

                "precision":
                    test_metrics[
                        "precision"
                    ],

                "recall":
                    test_metrics[
                        "recall"
                    ],

                "f1":
                    test_metrics[
                        "f1"
                    ],

                "f1_macro":
                    test_metrics[
                        "f1_macro"
                    ],
            },
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print(
        "\nArquivos salvos:"
    )

    print(
        DSEL_OUTPUT_PATH
    )

    print(
        TEST_OUTPUT_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nConcluído."
    )


if __name__ == "__main__":
    main()