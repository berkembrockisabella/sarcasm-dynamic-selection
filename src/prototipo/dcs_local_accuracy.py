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
from sklearn.preprocessing import StandardScaler, normalize


# --------------------------------------------------
# Configuração
# --------------------------------------------------

K = 7

MODELS = ["T", "C", "A", "F"]


# --------------------------------------------------
# Caminhos
# --------------------------------------------------

TRAIN_PATH = Path("data/splits/train.csv")
DSEL_PATH = Path("data/splits/dsel.csv")
TEST_PATH = Path("data/splits/test.csv")

FEATURES_DIR = Path("data/processed/features")

TEXT_EMBEDDINGS_PATH = FEATURES_DIR / "text_embeddings.npy"
TEXT_KEYS_PATH = FEATURES_DIR / "text_keys.csv"

CONTEXT_EMBEDDINGS_PATH = FEATURES_DIR / "context_embeddings.npy"
CONTEXT_KEYS_PATH = FEATURES_DIR / "context_keys.csv"

AUDIO_EMBEDDINGS_PATH = FEATURES_DIR / "audio_embeddings.npy"
AUDIO_KEYS_PATH = FEATURES_DIR / "audio_keys.csv"

VISUAL_FEATURES_PATH = FEATURES_DIR / "visual_features.csv"

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
    OUTPUT_DIR / "dcs_local_accuracy_test.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "dcs_local_accuracy_summary.csv"
)


# --------------------------------------------------
# Carregamento de embeddings
# --------------------------------------------------

def load_embedding_file(
    embeddings_path,
    keys_path
):

    embeddings = np.load(
        embeddings_path
    )

    keys_df = pd.read_csv(
        keys_path
    )

    if len(embeddings) != len(keys_df):

        raise ValueError(
            f"Inconsistência entre "
            f"{embeddings_path} e {keys_path}"
        )

    return embeddings, keys_df["KEY"].astype(str).tolist()


def align_embeddings(
    split_df,
    embeddings,
    keys
):

    key_to_index = {
        key: index
        for index, key in enumerate(keys)
    }

    missing = [
        key
        for key in split_df["KEY"]
        if key not in key_to_index
    ]

    if missing:

        raise ValueError(
            f"{len(missing)} KEYs sem embedding. "
            f"Exemplos: {missing[:5]}"
        )

    indices = [
        key_to_index[key]
        for key in split_df["KEY"]
    ]

    return embeddings[indices]


# --------------------------------------------------
# Visual
# --------------------------------------------------

def align_visual_features(
    split_df,
    visual_df
):

    merged = split_df[["KEY"]].merge(
        visual_df,
        on="KEY",
        how="left",
        validate="one_to_one"
    )

    if merged.isna().any().any():

        raise ValueError(
            "Existem valores ausentes nas "
            "features visuais."
        )

    feature_columns = [
        column
        for column in visual_df.columns
        if column != "KEY"
    ]

    return (
        merged[feature_columns]
        .to_numpy(dtype=np.float32)
    )


# --------------------------------------------------
# Prepara representação multimodal
# --------------------------------------------------

def scale_and_normalize(
    X_train,
    X_dsel,
    X_test
):

    # Scaler ajustado somente no Train
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_dsel_scaled = scaler.transform(
        X_dsel
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Cada modalidade recebe norma aproximadamente igual
    X_train_scaled = normalize(
        X_train_scaled,
        norm="l2"
    )

    X_dsel_scaled = normalize(
        X_dsel_scaled,
        norm="l2"
    )

    X_test_scaled = normalize(
        X_test_scaled,
        norm="l2"
    )

    return (
        X_train_scaled,
        X_dsel_scaled,
        X_test_scaled,
    )


def build_multimodal_features(
    train_df,
    dsel_df,
    test_df
):

    print(
        "Carregando representações T, C, A e F..."
    )

    # -------------------------
    # Texto
    # -------------------------

    text_embeddings, text_keys = (
        load_embedding_file(
            TEXT_EMBEDDINGS_PATH,
            TEXT_KEYS_PATH
        )
    )

    T_train = align_embeddings(
        train_df,
        text_embeddings,
        text_keys
    )

    T_dsel = align_embeddings(
        dsel_df,
        text_embeddings,
        text_keys
    )

    T_test = align_embeddings(
        test_df,
        text_embeddings,
        text_keys
    )

    T_train, T_dsel, T_test = (
        scale_and_normalize(
            T_train,
            T_dsel,
            T_test
        )
    )

    # -------------------------
    # Contexto
    # -------------------------

    context_embeddings, context_keys = (
        load_embedding_file(
            CONTEXT_EMBEDDINGS_PATH,
            CONTEXT_KEYS_PATH
        )
    )

    C_train = align_embeddings(
        train_df,
        context_embeddings,
        context_keys
    )

    C_dsel = align_embeddings(
        dsel_df,
        context_embeddings,
        context_keys
    )

    C_test = align_embeddings(
        test_df,
        context_embeddings,
        context_keys
    )

    C_train, C_dsel, C_test = (
        scale_and_normalize(
            C_train,
            C_dsel,
            C_test
        )
    )

    # -------------------------
    # Áudio
    # -------------------------

    audio_embeddings, audio_keys = (
        load_embedding_file(
            AUDIO_EMBEDDINGS_PATH,
            AUDIO_KEYS_PATH
        )
    )

    A_train = align_embeddings(
        train_df,
        audio_embeddings,
        audio_keys
    )

    A_dsel = align_embeddings(
        dsel_df,
        audio_embeddings,
        audio_keys
    )

    A_test = align_embeddings(
        test_df,
        audio_embeddings,
        audio_keys
    )

    A_train, A_dsel, A_test = (
        scale_and_normalize(
            A_train,
            A_dsel,
            A_test
        )
    )

    # -------------------------
    # Visual
    # -------------------------

    visual_df = pd.read_csv(
        VISUAL_FEATURES_PATH
    )

    F_train = align_visual_features(
        train_df,
        visual_df
    )

    F_dsel = align_visual_features(
        dsel_df,
        visual_df
    )

    F_test = align_visual_features(
        test_df,
        visual_df
    )

    F_train, F_dsel, F_test = (
        scale_and_normalize(
            F_train,
            F_dsel,
            F_test
        )
    )

    # -------------------------
    # Concatenação
    # -------------------------

    X_train = np.concatenate(
        [
            T_train,
            C_train,
            A_train,
            F_train,
        ],
        axis=1
    )

    X_dsel = np.concatenate(
        [
            T_dsel,
            C_dsel,
            A_dsel,
            F_dsel,
        ],
        axis=1
    )

    X_test = np.concatenate(
        [
            T_test,
            C_test,
            A_test,
            F_test,
        ],
        axis=1
    )

    print(
        f"Train multimodal: {X_train.shape}"
    )

    print(
        f"DSEL multimodal:  {X_dsel.shape}"
    )

    print(
        f"Test multimodal:  {X_test.shape}"
    )

    return X_train, X_dsel, X_test


# --------------------------------------------------
# Carregamento das previsões do pool
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
        f"DCS Local Accuracy - k = {K}\n"
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

    # --------------------------------------------------
    # Espaço para busca dos vizinhos
    # --------------------------------------------------

    _, X_dsel, X_test = (
        build_multimodal_features(
            train_df,
            dsel_df,
            test_df
        )
    )

    # --------------------------------------------------
    # Predições dos classificadores
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
    # usada apenas para desempate
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
    # kNN sobre o DSEL
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

        # -------------------------
        # Competência local
        # -------------------------

        for model in MODELS:

            neighbor_predictions = (
                dsel_predictions[model][
                    neighbors
                ]
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

        # -------------------------
        # Melhor competência
        # -------------------------

        best_competence = max(
            local_competence.values()
        )

        tied_models = [
            model
            for model in MODELS
            if local_competence[model]
            == best_competence
        ]

        # -------------------------
        # Desempate:
        # maior desempenho global no DSEL
        # -------------------------

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
        "\nDesempenho DCS Local Accuracy:"
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
    # Salva resultados
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
                    "DCS_Local_Accuracy",

                "k":
                    K,

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