from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Caminhos
# --------------------------------------------------

STATIC_PATH = Path(
    "data/processed/eval/static_ensemble_test.csv"
)

DCS_PATH = Path(
    "data/processed/selection/dcs_combined_test.csv"
)

OUTPUT_DIR = Path(
    "data/processed/eval"
)

DETAILS_PATH = (
    OUTPUT_DIR / "compare_static_vs_dcs_details.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "compare_static_vs_dcs_summary.csv"
)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(
        "Comparando Ensemble Estático "
        "vs DCS Combinado...\n"
    )

    static_df = pd.read_csv(
        STATIC_PATH
    )

    dcs_df = pd.read_csv(
        DCS_PATH
    )

    # --------------------------------------------------
    # Seleciona e renomeia colunas
    # --------------------------------------------------

    static = static_df[
        [
            "KEY",
            "label",
            "ensemble_prediction",
            "ensemble_probability",
        ]
    ].copy()

    static = static.rename(
        columns={
            "ensemble_prediction":
                "static_prediction",

            "ensemble_probability":
                "static_probability",
        }
    )

    dcs = dcs_df[
        [
            "KEY",
            "label",
            "prediction",
            "prob_sarcasm",
            "selected_model",
            "competence_T",
            "competence_C",
            "competence_A",
            "competence_F",
        ]
    ].copy()

    dcs = dcs.rename(
        columns={
            "label":
                "label_dcs",

            "prediction":
                "dcs_prediction",

            "prob_sarcasm":
                "dcs_probability",
        }
    )

    # --------------------------------------------------
    # Alinha por KEY
    # --------------------------------------------------

    merged = static.merge(
        dcs,
        on="KEY",
        how="inner",
        validate="one_to_one"
    )

    if len(merged) != len(static_df):
        raise ValueError(
            "Nem todas as instâncias foram "
            "alinhadas entre Static e DCS."
        )

    # --------------------------------------------------
    # Confere labels
    # --------------------------------------------------

    if not (
        merged["label"]
        ==
        merged["label_dcs"]
    ).all():

        raise ValueError(
            "Existem labels diferentes "
            "entre os arquivos."
        )

    merged = merged.drop(
        columns=["label_dcs"]
    )

    # --------------------------------------------------
    # Acertos e erros
    # --------------------------------------------------

    merged["static_correct"] = (
        merged["static_prediction"]
        ==
        merged["label"]
    ).astype(int)

    merged["dcs_correct"] = (
        merged["dcs_prediction"]
        ==
        merged["label"]
    ).astype(int)

    # --------------------------------------------------
    # Categoria de comparação
    # --------------------------------------------------

    def classify_case(row):

        static_correct = (
            row["static_correct"] == 1
        )

        dcs_correct = (
            row["dcs_correct"] == 1
        )

        if static_correct and dcs_correct:
            return "both_correct"

        if static_correct and not dcs_correct:
            return "static_only"

        if not static_correct and dcs_correct:
            return "dcs_only"

        return "both_wrong"

    merged["comparison"] = (
        merged.apply(
            classify_case,
            axis=1
        )
    )

    # --------------------------------------------------
    # Resumo
    # --------------------------------------------------

    summary = (
        merged["comparison"]
        .value_counts()
        .reindex(
            [
                "both_correct",
                "static_only",
                "dcs_only",
                "both_wrong",
            ],
            fill_value=0
        )
        .rename_axis(
            "category"
        )
        .reset_index(
            name="count"
        )
    )

    summary["percentage"] = (
        summary["count"]
        /
        len(merged)
        *
        100
    )

    print(
        "Resumo da comparação:\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Estatísticas adicionais
    # --------------------------------------------------

    both_correct = (
        merged["comparison"]
        ==
        "both_correct"
    ).sum()

    static_only = (
        merged["comparison"]
        ==
        "static_only"
    ).sum()

    dcs_only = (
        merged["comparison"]
        ==
        "dcs_only"
    ).sum()

    both_wrong = (
        merged["comparison"]
        ==
        "both_wrong"
    ).sum()

    print(
        "\nInterpretação:"
    )

    print(
        f"Ambos acertaram: {both_correct}"
    )

    print(
        f"Só o ensemble estático acertou: "
        f"{static_only}"
    )

    print(
        f"Só o DCS acertou: "
        f"{dcs_only}"
    )

    print(
        f"Ambos erraram: "
        f"{both_wrong}"
    )

    # --------------------------------------------------
    # Analisa quais especialistas salvaram casos
    # em que o ensemble errou
    # --------------------------------------------------

    dcs_only_df = merged[
        merged["comparison"]
        ==
        "dcs_only"
    ].copy()

    if not dcs_only_df.empty:

        selected_summary = (
            dcs_only_df[
                "selected_model"
            ]
            .value_counts()
            .reindex(
                [
                    "T",
                    "C",
                    "A",
                    "F",
                ],
                fill_value=0
            )
            .rename_axis(
                "selected_model"
            )
            .reset_index(
                name="count"
            )
        )

        selected_summary[
            "percentage"
        ] = (
            selected_summary[
                "count"
            ]
            /
            len(dcs_only_df)
            *
            100
        )

        print(
            "\nNos casos em que somente "
            "o DCS acertou:"
        )

        print(
            selected_summary.to_string(
                index=False
            )
        )

    else:

        selected_summary = pd.DataFrame(
            columns=[
                "selected_model",
                "count",
                "percentage",
            ]
        )

        print(
            "\nO DCS não teve nenhum caso "
            "exclusivo de acerto."
        )

    # --------------------------------------------------
    # Diferença das probabilidades
    # --------------------------------------------------

    merged[
        "probability_difference"
    ] = (
        merged["dcs_probability"]
        -
        merged["static_probability"]
    )

    # --------------------------------------------------
    # Junta texto original para inspeção
    # --------------------------------------------------

    test_path = Path(
        "data/splits/test.csv"
    )

    test_df = pd.read_csv(
        test_path
    )

    optional_columns = [
        "KEY",
        "SENTENCE",
        "SPEAKER",
        "SCENE",
    ]

    available_columns = [
        col
        for col in optional_columns
        if col in test_df.columns
    ]

    merged = merged.merge(
        test_df[
            available_columns
        ],
        on="KEY",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------
    # Salva resultados
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        DETAILS_PATH,
        index=False
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # Salva resumo dos modelos selecionados
    if not selected_summary.empty:

        selected_summary.to_csv(
            OUTPUT_DIR
            / "dcs_only_selected_models.csv",
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

    if not selected_summary.empty:

        print(
            OUTPUT_DIR
            / "dcs_only_selected_models.csv"
        )

    print(
        "\nComparação concluída."
    )


if __name__ == "__main__":
    main()