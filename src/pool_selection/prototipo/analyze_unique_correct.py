from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Caminhos
# --------------------------------------------------

INPUT_PATH = Path(
    "data/processed/pool_analysis/aligned_dsel_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/processed/pool_analysis"
)

DETAILS_PATH = (
    OUTPUT_DIR / "unique_correct_details.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "unique_correct_summary.csv"
)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Carregando previsões alinhadas...")

    df = pd.read_csv(
        INPUT_PATH
    )

    model_names = [
        "T",
        "C",
        "A",
        "F",
    ]

    # --------------------------------------------------
    # Calcula quais classificadores acertaram
    # --------------------------------------------------

    for model in model_names:

        pred_column = (
            f"prediction_{model}"
        )

        correct_column = (
            f"correct_{model}"
        )

        df[correct_column] = (
            df[pred_column]
            ==
            df["label"]
        ).astype(int)

    correct_columns = [
        f"correct_{model}"
        for model in model_names
    ]

    # Quantos classificadores acertaram
    df["num_correct"] = (
        df[correct_columns]
        .sum(axis=1)
    )

    # --------------------------------------------------
    # Identifica quando apenas um acertou
    # --------------------------------------------------

    def get_unique_model(row):

        if row["num_correct"] != 1:
            return None

        for model in model_names:

            if row[
                f"correct_{model}"
            ] == 1:
                return model

        return None

    df["unique_correct_model"] = (
        df.apply(
            get_unique_model,
            axis=1
        )
    )

    # --------------------------------------------------
    # Filtra casos únicos
    # --------------------------------------------------

    unique_df = df[
        df["num_correct"] == 1
    ].copy()

    print(
        f"\nTotal de instâncias no DSEL: "
        f"{len(df)}"
    )

    print(
        "Instâncias com apenas "
        f"1 classificador correto: "
        f"{len(unique_df)}"
    )

    print(
        f"Percentual: "
        f"{len(unique_df) / len(df) * 100:.2f}%"
    )

    # --------------------------------------------------
    # Resumo por especialista
    # --------------------------------------------------

    summary = (
        unique_df[
            "unique_correct_model"
        ]
        .value_counts()
        .reindex(
            model_names,
            fill_value=0
        )
        .rename_axis(
            "model"
        )
        .reset_index(
            name="unique_correct_count"
        )
    )

    summary[
        "percentage_of_unique_cases"
    ] = (
        summary[
            "unique_correct_count"
        ]
        /
        len(unique_df)
        *
        100
    )

    summary[
        "percentage_of_dsel"
    ] = (
        summary[
            "unique_correct_count"
        ]
        /
        len(df)
        *
        100
    )

    print(
        "\nCasos em que somente "
        "um especialista acertou:\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Exibe os casos
    # --------------------------------------------------

    columns_to_show = [
        "KEY",
        "label",
        "prediction_T",
        "prediction_C",
        "prediction_A",
        "prediction_F",
        "unique_correct_model",
    ]

    print(
        "\nDetalhes dos casos únicos:\n"
    )

    print(
        unique_df[
            columns_to_show
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Junta informações originais do DSEL
    # --------------------------------------------------

    dsel_path = Path(
        "data/splits/dsel.csv"
    )

    dsel_df = pd.read_csv(
        dsel_path
    )

    extra_columns = [
        "KEY",
        "SENTENCE",
        "SPEAKER",
        "SCENE",
    ]

    available_columns = [
        col
        for col in extra_columns
        if col in dsel_df.columns
    ]

    unique_df = unique_df.merge(
        dsel_df[
            available_columns
        ],
        on="KEY",
        how="left"
    )

    # --------------------------------------------------
    # Salva
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    unique_df.to_csv(
        DETAILS_PATH,
        index=False
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print(
        "\nArquivos salvos:"
    )

    print(
        DETAILS_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nAnálise concluída."
    )


if __name__ == "__main__":
    main()