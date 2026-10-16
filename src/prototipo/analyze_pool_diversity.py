from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Caminhos
# --------------------------------------------------

PREDICTIONS_DIR = Path(
    "data/processed/pool_predictions"
)

OUTPUT_DIR = Path(
    "data/processed/pool_analysis"
)

MODELS = {
    "T": PREDICTIONS_DIR / "text_dsel_predictions.csv",
    "C": PREDICTIONS_DIR / "context_dsel_predictions.csv",
    "A": PREDICTIONS_DIR / "audio_dsel_predictions.csv",
    "F": PREDICTIONS_DIR / "visual_dsel_predictions.csv",
}


# --------------------------------------------------
# Métricas de diversidade
# --------------------------------------------------

def disagreement(pred1, pred2):
    """
    Proporção de instâncias em que
    os dois classificadores discordam.
    """
    return np.mean(pred1 != pred2)


def double_fault(y_true, pred1, pred2):
    """
    Proporção de instâncias em que
    os dois classificadores erram juntos.
    """
    error1 = pred1 != y_true
    error2 = pred2 != y_true

    return np.mean(error1 & error2)


def q_statistic(y_true, pred1, pred2):
    """
    Q-statistic baseado em acertos e erros
    dos dois classificadores.
    """

    correct1 = pred1 == y_true
    correct2 = pred2 == y_true

    n11 = np.sum(correct1 & correct2)
    n00 = np.sum(~correct1 & ~correct2)
    n10 = np.sum(correct1 & ~correct2)
    n01 = np.sum(~correct1 & correct2)

    denominator = (
        n11 * n00
        +
        n10 * n01
    )

    if denominator == 0:
        return np.nan

    q = (
        (n11 * n00)
        -
        (n10 * n01)
    ) / denominator

    return q


# --------------------------------------------------
# Oracle
# --------------------------------------------------

def oracle_accuracy(
    y_true,
    predictions
):
    """
    Considera a instância correta se
    pelo menos um classificador acertou.
    """

    correct_matrix = (
        predictions
        == y_true[:, None]
    )

    oracle_correct = (
        correct_matrix.any(axis=1)
    )

    return oracle_correct.mean()


def oracle_coverage_table(
    y_true,
    predictions,
    model_names
):
    """
    Conta quantos classificadores acertaram
    cada instância.
    """

    correct_matrix = (
        predictions
        == y_true[:, None]
    )

    num_correct = (
        correct_matrix.sum(axis=1)
    )

    result = pd.DataFrame(
        {
            "num_classifiers_correct": num_correct
        }
    )

    summary = (
        result["num_classifiers_correct"]
        .value_counts()
        .sort_index()
        .rename_axis(
            "num_classifiers_correct"
        )
        .reset_index(
            name="count"
        )
    )

    summary["percentage"] = (
        summary["count"]
        /
        len(result)
        *
        100
    )

    return summary


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Carregando previsões do DSEL...\n"
    )

    dataframes = {}

    for model_name, path in MODELS.items():

        df = pd.read_csv(path)

        required_columns = {
            "KEY",
            "label",
            "prediction",
        }

        missing = (
            required_columns
            -
            set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{model_name}: colunas ausentes: "
                f"{missing}"
            )

        dataframes[model_name] = df[
            [
                "KEY",
                "label",
                "prediction",
            ]
        ].copy()

        print(
            f"{model_name}: "
            f"{len(df)} instâncias"
        )

    # --------------------------------------------------
    # Usa T como base para alinhamento
    # --------------------------------------------------

    base = dataframes["T"].rename(
        columns={
            "prediction": "prediction_T"
        }
    )

    for model_name in [
        "C",
        "A",
        "F",
    ]:

        current = (
            dataframes[model_name]
            .rename(
                columns={
                    "prediction":
                    f"prediction_{model_name}",
                    "label":
                    f"label_{model_name}",
                }
            )
        )

        base = base.merge(
            current,
            on="KEY",
            how="inner",
            validate="one_to_one"
        )

    # --------------------------------------------------
    # Confere se labels são iguais
    # --------------------------------------------------

    for model_name in [
        "C",
        "A",
        "F",
    ]:

        label_column = (
            f"label_{model_name}"
        )

        inconsistent = (
            base["label"]
            !=
            base[label_column]
        )

        if inconsistent.any():
            raise ValueError(
                f"Labels inconsistentes "
                f"entre T e {model_name}."
            )

        base = base.drop(
            columns=[label_column]
        )

    print(
        f"\nInstâncias alinhadas: "
        f"{len(base)}"
    )

    # --------------------------------------------------
    # Matrizes de predição
    # --------------------------------------------------

    model_names = [
        "T",
        "C",
        "A",
        "F",
    ]

    y_true = (
        base["label"]
        .astype(int)
        .to_numpy()
    )

    predictions = np.column_stack(
        [
            base[
                f"prediction_{model}"
            ]
            .astype(int)
            .to_numpy()
            for model in model_names
        ]
    )

    # --------------------------------------------------
    # Desempenho individual
    # --------------------------------------------------

    print(
        "\nAcurácia individual no DSEL:"
    )

    individual_results = []

    for index, model_name in enumerate(
        model_names
    ):

        accuracy = np.mean(
            predictions[:, index]
            ==
            y_true
        )

        individual_results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
            }
        )

        print(
            f"{model_name}: "
            f"{accuracy:.4f}"
        )

    individual_df = pd.DataFrame(
        individual_results
    )

    individual_df.to_csv(
        OUTPUT_DIR
        / "individual_accuracy.csv",
        index=False
    )

    # --------------------------------------------------
    # Métricas par-a-par
    # --------------------------------------------------

    pair_results = []

    print(
        "\nDiversidade entre pares:\n"
    )

    for i in range(
        len(model_names)
    ):

        for j in range(
            i + 1,
            len(model_names)
        ):

            model1 = model_names[i]
            model2 = model_names[j]

            pred1 = predictions[:, i]
            pred2 = predictions[:, j]

            dis = disagreement(
                pred1,
                pred2
            )

            dfault = double_fault(
                y_true,
                pred1,
                pred2
            )

            q = q_statistic(
                y_true,
                pred1,
                pred2
            )

            pair_results.append(
                {
                    "model_1": model1,
                    "model_2": model2,
                    "disagreement": dis,
                    "double_fault": dfault,
                    "q_statistic": q,
                }
            )

            print(
                f"{model1} x {model2}"
            )

            print(
                f"  Disagreement: "
                f"{dis:.4f}"
            )

            print(
                f"  Double-fault: "
                f"{dfault:.4f}"
            )

            print(
                f"  Q-statistic: "
                f"{q:.4f}"
            )

            print()

    pair_df = pd.DataFrame(
        pair_results
    )

    pair_df.to_csv(
        OUTPUT_DIR
        / "pairwise_diversity.csv",
        index=False
    )

    # --------------------------------------------------
    # Matrizes
    # --------------------------------------------------

    disagreement_matrix = (
        pd.DataFrame(
            index=model_names,
            columns=model_names,
            dtype=float
        )
    )

    double_fault_matrix = (
        pd.DataFrame(
            index=model_names,
            columns=model_names,
            dtype=float
        )
    )

    q_matrix = (
        pd.DataFrame(
            index=model_names,
            columns=model_names,
            dtype=float
        )
    )

    for model1 in model_names:

        for model2 in model_names:

            i = model_names.index(
                model1
            )

            j = model_names.index(
                model2
            )

            if model1 == model2:

                disagreement_matrix.loc[
                    model1,
                    model2
                ] = 0.0

                double_fault_matrix.loc[
                    model1,
                    model2
                ] = np.mean(
                    predictions[:, i]
                    !=
                    y_true
                )

                q_matrix.loc[
                    model1,
                    model2
                ] = 1.0

            else:

                disagreement_matrix.loc[
                    model1,
                    model2
                ] = disagreement(
                    predictions[:, i],
                    predictions[:, j]
                )

                double_fault_matrix.loc[
                    model1,
                    model2
                ] = double_fault(
                    y_true,
                    predictions[:, i],
                    predictions[:, j]
                )

                q_matrix.loc[
                    model1,
                    model2
                ] = q_statistic(
                    y_true,
                    predictions[:, i],
                    predictions[:, j]
                )

    disagreement_matrix.to_csv(
        OUTPUT_DIR
        / "disagreement_matrix.csv"
    )

    double_fault_matrix.to_csv(
        OUTPUT_DIR
        / "double_fault_matrix.csv"
    )

    q_matrix.to_csv(
        OUTPUT_DIR
        / "q_statistic_matrix.csv"
    )

    # --------------------------------------------------
    # Oracle
    # --------------------------------------------------

    oracle = oracle_accuracy(
        y_true,
        predictions
    )

    print(
        "\nOracle accuracy:"
    )

    print(
        f"{oracle:.4f}"
    )

    print(
        "\nIsso significa que, em "
        f"{oracle * 100:.2f}% "
        "das instâncias do DSEL, "
        "pelo menos um dos quatro "
        "classificadores acertou."
    )

    oracle_summary = (
        oracle_coverage_table(
            y_true,
            predictions,
            model_names
        )
    )

    print(
        "\nQuantidade de classificadores "
        "corretos por instância:"
    )

    print(
        oracle_summary.to_string(
            index=False
        )
    )

    oracle_summary.to_csv(
        OUTPUT_DIR
        / "oracle_coverage.csv",
        index=False
    )

    # --------------------------------------------------
    # Salva previsões alinhadas
    # --------------------------------------------------

    base.to_csv(
        OUTPUT_DIR
        / "aligned_dsel_predictions.csv",
        index=False
    )

    print(
        "\nArquivos salvos em:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nAnálise concluída."
    )


if __name__ == "__main__":
    main()