from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Caminhos
# --------------------------------------------------

EMOTIONS_PATH = Path(
    "data/processed/visual_emotions.csv"
)

PREPARED_PATH = Path(
    "data/processed/mustard_prepared.csv"
)

OUTPUT_PATH = Path(
    "data/processed/features/visual_features.csv"
)


# --------------------------------------------------
# Configuração
# --------------------------------------------------

EMOTION_COLUMNS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def main():

    print("Carregando emoções visuais...")

    emotions_df = pd.read_csv(
        EMOTIONS_PATH
    )

    prepared_df = pd.read_csv(
        PREPARED_PATH
    )

    print(
        f"Faces disponíveis: {len(emotions_df)}"
    )

    print(
        f"Instâncias esperadas: {len(prepared_df)}"
    )

    # --------------------------------------------------
    # Agregação por KEY
    # --------------------------------------------------

    print(
        "\nAgregando probabilidades por instância..."
    )

    grouped_mean = (
        emotions_df
        .groupby("KEY")[EMOTION_COLUMNS]
        .mean()
        .add_suffix("_mean")
    )

    grouped_std = (
        emotions_df
        .groupby("KEY")[EMOTION_COLUMNS]
        .std()
        .fillna(0)
        .add_suffix("_std")
    )

    face_count = (
        emotions_df
        .groupby("KEY")
        .size()
        .rename("face_count")
    )

    # Junta tudo
    visual_features = pd.concat(
        [
            grouped_mean,
            grouped_std,
            face_count,
        ],
        axis=1
    )

    visual_features = (
        visual_features
        .reset_index()
    )

    # --------------------------------------------------
    # Garante todas as 1202 instâncias
    # --------------------------------------------------

    all_keys = (
        prepared_df[["KEY"]]
        .drop_duplicates()
        .copy()
    )

    result = all_keys.merge(
        visual_features,
        on="KEY",
        how="left"
    )

    # Marca instâncias sem face
    result["visual_missing"] = (
        result["face_count"]
        .isna()
        .astype(int)
    )

    # Para instâncias sem faces:
    # probabilidades agregadas = 0
    # face_count = 0
    feature_columns = [
        column
        for column in result.columns
        if column != "KEY"
    ]

    result[feature_columns] = (
        result[feature_columns]
        .fillna(0)
    )

    result["face_count"] = (
        result["face_count"]
        .astype(int)
    )

    # --------------------------------------------------
    # Validação
    # --------------------------------------------------

    print(
        f"\nInstâncias finais: {len(result)}"
    )

    print(
        "Instâncias sem informação visual:",
        result["visual_missing"].sum()
    )

    print(
        "Total de features visuais:",
        len(result.columns) - 1
    )

    print(
        "Valores ausentes:",
        result.isna().sum().sum()
    )

    # --------------------------------------------------
    # Salva
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSalvo em: {OUTPUT_PATH}"
    )

    print(
        "\nPrimeiras linhas:"
    )

    print(
        result.head().to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()