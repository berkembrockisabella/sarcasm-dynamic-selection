from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Configuração
# --------------------------------------------------

K = 7

MODELS = ["T", "C", "A", "F"]

# Features de incongruencia (src/incongruence/build_incongruence.py):
# diferenca de valencia COM SINAL entre os 6 pares de modalidades, numero de
# pares com polaridades opostas e o indicador de "cara de paisagem" (audio e
# rosto neutros). As versoes absolutas (abs_inc_*) foram retiradas do csv
# porque misturavam desacordo com quantidade de emocao
INCONGRUENCE_COLUMNS = [
    "inc_text_context",
    "inc_text_audio",
    "inc_text_visual",
    "inc_context_audio",
    "inc_context_visual",
    "inc_audio_visual",
    "n_oppositions",
    "deadpan",
]


# --------------------------------------------------
# Caminhos
# --------------------------------------------------

TRAIN_PATH = Path(
    "data/splits/train.csv"
)

DSEL_PATH = Path(
    "data/splits/dsel.csv"
)

TEST_PATH = Path(
    "data/splits/test.csv"
)

INCONGRUENCE_PATH = Path(
    "data/processed/incongruence.csv"
)

PREDICTIONS_DIR = Path(
    "data/processed/pool_predictions"
)

DSEL_PREDICTIONS = {
    "T": PREDICTIONS_DIR / "text_dsel_predictions.csv",
    "C": PREDICTIONS_DIR / "context_dsel_predictions.csv",
    "A": PREDICTIONS_DIR / "audio_dsel_predictions.csv",
    "F": PREDICTIONS_DIR / "visual_dsel_predictions.csv",
}

TEST_PREDICTIONS = {
    "T": PREDICTIONS_DIR / "text_test_predictions.csv",
    "C": PREDICTIONS_DIR / "context_test_predictions.csv",
    "A": PREDICTIONS_DIR / "audio_test_predictions.csv",
    "F": PREDICTIONS_DIR / "visual_test_predictions.csv",
}

OUTPUT_DIR = Path(
    "data/processed/selection"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "dcs_incongruence_test.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "dcs_incongruence_summary.csv"
)


# --------------------------------------------------
# Alinha features de incongruência
# --------------------------------------------------

def align_incongruence(
    split_df,
    incongruence_df
):

    merged = (
        split_df[["KEY"]]
        .merge(
            incongruence_df[
                ["KEY"] + INCONGRUENCE_COLUMNS
            ],
            on="KEY",
            how="left",
            validate="one_to_one"
        )
    )

    if merged[
        INCONGRUENCE_COLUMNS
    ].isna().any().any():

        missing = merged[
            merged[
                INCONGRUENCE_COLUMNS
            ].isna().any(axis=1)
        ]

        raise ValueError(
            "Existem instâncias sem "
            "incongruência.\n"
            f"{missing['KEY'].tolist()[:5]}"
        )

    return (
        merged[
            INCONGRUENCE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


# --------------------------------------------------
# Carrega previsões
# --------------------------------------------------

def load_pool_predictions(
    paths,
    reference_df
):

    predictions = {}
    probabilities = {}

    labels = None

    for model_name, path in paths.items():

        df = pd.read_csv(
            path
        )

        aligned = (
            reference_df[["KEY"]]
            .merge(
                df,
                on="KEY",
                how="left",
                validate="one_to_one"
            )
        )

        if aligned.isna().any().any():

            raise ValueError(
                f"Dados ausentes nas previsões "
                f"do modelo {model_name}."
            )

        predictions[model_name] = (
            aligned["prediction"]
            .astype(int)
            .to_numpy()
        )

        probabilities[model_name] = (
            aligned["prob_sarcasm"]
            .astype(float)
            .to_numpy()
        )

        current_labels = (
            aligned["label"]
            .astype(int)
            .to_numpy()
        )

        if labels is None:

            labels = current_labels

        elif not np.array_equal(
            labels,
            current_labels
        ):

            raise ValueError(
                f"Labels inconsistentes "
                f"no modelo {model_name}."
            )

    return (
        labels,
        predictions,
        probabilities
    )


# --------------------------------------------------
# Métricas
# --------------------------------------------------

def print_metrics(
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

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"F1 Macro:  {f1_macro:.4f}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(
        f"DCS por Incongruência - k = {K}\n"
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

    incongruence_df = pd.read_csv(
        INCONGRUENCE_PATH
    )

    print(
        "Features de incongruência:"
    )

    for column in INCONGRUENCE_COLUMNS:
        print(
            f" - {column}"
        )

    # --------------------------------------------------
    # Features
    # --------------------------------------------------

    X_train = align_incongruence(
        train_df,
        incongruence_df
    )

    X_dsel = align_incongruence(
        dsel_df,
        incongruence_df
    )

    X_test = align_incongruence(
        test_df,
        incongruence_df
    )

    print(
        "\nDimensões:"
    )

    print(
        f"Train: {X_train.shape}"
    )

    print(
        f"DSEL:  {X_dsel.shape}"
    )

    print(
        f"Test:  {X_test.shape}"
    )

    # --------------------------------------------------
    # Padronização
    #
    # O scaler é ajustado SOMENTE no Train.
    # --------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        X_train
    )

    X_dsel = scaler.transform(
        X_dsel
    )

    X_test = scaler.transform(
        X_test
    )

    # --------------------------------------------------
    # Predições dos especialistas
    # --------------------------------------------------

    (
        y_dsel,
        dsel_predictions,
        _
    ) = load_pool_predictions(
        DSEL_PREDICTIONS,
        dsel_df
    )

    (
        y_test,
        test_predictions,
        test_probabilities
    ) = load_pool_predictions(
        TEST_PREDICTIONS,
        test_df
    )

    # --------------------------------------------------
    # Competência global no DSEL
    # para desempate
    # --------------------------------------------------

    global_accuracy = {}

    for model in MODELS:

        global_accuracy[model] = np.mean(
            dsel_predictions[model]
            ==
            y_dsel
        )

    print(
        "\nAcurácia global no DSEL:"
    )

    for model in MODELS:

        print(
            f"{model}: "
            f"{global_accuracy[model]:.4f}"
        )

    # --------------------------------------------------
    # kNN no espaço de incongruência
    # --------------------------------------------------

    print(
        f"\nConstruindo região de competência "
        f"com k={K}..."
    )

    knn = NearestNeighbors(
        n_neighbors=K,
        metric="euclidean"
    )

    knn.fit(
        X_dsel
    )

    distances, neighbor_indices = (
        knn.kneighbors(
            X_test
        )
    )

    # --------------------------------------------------
    # Seleção dinâmica
    # --------------------------------------------------

    final_predictions = []
    final_probabilities = []
    selected_models = []

    result_rows = []

    for test_index in range(
        len(test_df)
    ):

        neighbors = (
            neighbor_indices[
                test_index
            ]
        )

        local_competence = {}

        # Competência de cada especialista
        for model in MODELS:

            neighbor_predictions = (
                dsel_predictions[
                    model
                ][neighbors]
            )

            neighbor_labels = (
                y_dsel[
                    neighbors
                ]
            )

            competence = np.mean(
                neighbor_predictions
                ==
                neighbor_labels
            )

            local_competence[
                model
            ] = competence

        # Melhor competência local
        best_competence = max(
            local_competence.values()
        )

        tied_models = [
            model
            for model in MODELS
            if local_competence[model]
            == best_competence
        ]

        # Desempate pela acurácia global
        # no DSEL
        selected_model = max(
            tied_models,
            key=lambda model:
            global_accuracy[model]
        )

        selected_prediction = (
            test_predictions[
                selected_model
            ][test_index]
        )

        selected_probability = (
            test_probabilities[
                selected_model
            ][test_index]
        )

        final_predictions.append(
            selected_prediction
        )

        final_probabilities.append(
            selected_probability
        )

        selected_models.append(
            selected_model
        )

        result_rows.append(
            {
                "KEY":
                    test_df.iloc[
                        test_index
                    ]["KEY"],

                "label":
                    y_test[
                        test_index
                    ],

                "selected_model":
                    selected_model,

                "prediction":
                    selected_prediction,

                "prob_sarcasm":
                    selected_probability,

                "competence_T":
                    local_competence["T"],

                "competence_C":
                    local_competence["C"],

                "competence_A":
                    local_competence["A"],

                "competence_F":
                    local_competence["F"],

                "mean_neighbor_distance":
                    distances[
                        test_index
                    ].mean(),
            }
        )

    final_predictions = np.array(
        final_predictions
    )

    # --------------------------------------------------
    # Resultados
    # --------------------------------------------------

    print(
        "\nDesempenho DCS por Incongruência:"
    )

    metrics = print_metrics(
        y_test,
        final_predictions
    )

    print(
        "\nQuantidade de vezes que cada "
        "especialista foi selecionado:"
    )

    selection_counts = (
        pd.Series(
            selected_models
        )
        .value_counts()
        .reindex(
            MODELS,
            fill_value=0
        )
    )

    for model in MODELS:

        count = (
            selection_counts[
                model
            ]
        )

        percentage = (
            count
            /
            len(test_df)
            *
            100
        )

        print(
            f"{model}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------
    # Salva
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df = pd.DataFrame(
        result_rows
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    summary = pd.DataFrame(
        [
            {
                "method":
                    "DCS_Incongruence",

                "k":
                    K,

                "num_incongruence_features":
                    len(
                        INCONGRUENCE_COLUMNS
                    ),

                "accuracy":
                    metrics["accuracy"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1":
                    metrics["f1"],

                "f1_macro":
                    metrics["f1_macro"],

                "selected_T":
                    selection_counts["T"],

                "selected_C":
                    selection_counts["C"],

                "selected_A":
                    selection_counts["A"],

                "selected_F":
                    selection_counts["F"],
            }
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
        OUTPUT_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nConcluído."
    )


if __name__ == "__main__":
    main()