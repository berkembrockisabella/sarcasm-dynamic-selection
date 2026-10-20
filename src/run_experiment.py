"""
Orquestra os experimentos principais de:
- Seleção Dinâmica (DS)
- Mixture of Experts (MoE)
- Baselines estáticos
- Comparação estatística

Execução recomendada, a partir da raiz do projeto:

    python -m src.run_experiment

Exemplos:

    python -m src.run_experiment --only moe
    python -m src.run_experiment --only ds
    python -m src.run_experiment --only static
    python -m src.run_experiment --only ds static
    python -m src.run_experiment --include-k-sweep
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"

POOL_RESULTS = (
    DATA
    / "results"
    / "pool"
    / "pool_selection"
)

FROZEN_POOLS = (
    DATA
    / "results"
    / "pool"
    / "frozen_pools"
)


# ============================================================
# EXECUÇÃO DE UMA ETAPA
# ============================================================

def run_step(label, command):
    """
    Executa uma etapa do experimento.

    Todas as etapas são executadas a partir da raiz
    do projeto.
    """

    print(
        "\n" + "=" * 76,
        flush=True
    )

    print(
        f"ETAPA: {label}",
        flush=True
    )

    print(
        "Comando:",
        " ".join(
            str(item)
            for item in command
        ),
        flush=True
    )

    print(
        "=" * 76,
        flush=True
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True
    )


# ============================================================
# VERIFICAÇÃO DA POOL
# ============================================================

def frozen_pool_is_complete(pool_dir):
    """
    Verifica se a frozen pool contém:
    - pool_metadata.csv
    - C01.joblib ... C07.joblib
    """

    metadata = (
        pool_dir
        / "pool_metadata.csv"
    )

    expected_models = [
        pool_dir
        / f"C{i:02d}.joblib"
        for i in range(1, 8)
    ]

    return (
        metadata.exists()
        and all(
            model.exists()
            for model in expected_models
        )
    )


def ensure_pool():
    """
    Garante que:

    1. A seleção da pool já foi executada.
    2. A pool já foi treinada.

    Se os arquivos já existirem,
    as etapas não são executadas novamente.
    """

    definition = (
        POOL_RESULTS
        / "pool_definition.csv"
    )

    cv_results = (
        POOL_RESULTS
        / "cv_results.csv"
    )

    # --------------------------------------------------------
    # Pool selection
    # --------------------------------------------------------

    if (
        not definition.exists()
        or not cv_results.exists()
    ):

        run_step(
            "Seleção da pool (CV no TRAIN)",
            [
                sys.executable,
                "-m",
                "src.pool_selection.build_pool",
            ]
        )

    else:

        print(
            "\nPool selection já encontrada."
        )

        print(
            f"Definição: {definition}"
        )

    # --------------------------------------------------------
    # Frozen pool
    # --------------------------------------------------------

    pool_ok = frozen_pool_is_complete(
        FROZEN_POOLS
    )

    if not pool_ok:

        run_step(
            "Treinamento da pool congelada",
            [
                sys.executable,
                "-m",
                "src.pool_training.train_frozen_pool",
            ]
        )

    else:

        print(
            "\nFrozen pool já encontrada:"
        )

        print(
            FROZEN_POOLS
        )


# ============================================================
# CONSOLIDAÇÃO DOS RESULTADOS
# ============================================================

def consolidate():
    """
    Junta os arquivos de resumo dos experimentos
    em um único CSV.
    """

    sources = [
        (
            DATA
            / "processed"
            / "selection"
            / "ds_routing_summary.csv",
            "DS",
        ),
        (
            DATA
            / "processed"
            / "moe"
            / "moe_summary.csv",
            "MoE",
        ),
        (
            DATA
            / "processed"
            / "static"
            / "static_multimodal_summary.csv",
            "Estático",
        ),
    ]

    frames = []

    for path, family in sources:

        if not path.exists():

            print(
                f"Resumo não encontrado: {path}"
            )

            continue

        df = pd.read_csv(
            path
        )

        # Padroniza nome da coluna do método
        if "metodo" in df.columns:

            df["familia"] = family
            frames.append(df)

        elif "method" in df.columns:

            df = df.rename(
                columns={
                    "method": "metodo"
                }
            )

            df["familia"] = family
            frames.append(df)

        else:

            print(
                f"Aviso: {path.name} não possui "
                "'metodo' nem 'method'."
            )

    if not frames:

        print(
            "\nNenhum resumo encontrado "
            "para consolidar."
        )

        return

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    output = (
        DATA
        / "processed"
        / "eval"
        / "all_experiments_summary.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        output,
        index=False
    )

    print(
        "\n" + "=" * 76
    )

    print(
        "RESUMO CONSOLIDADO"
    )

    print(
        "=" * 76
    )

    print(
        f"\nArquivo salvo em:\n{output}"
    )

    columns_to_show = [
        column
        for column in (
            "familia",
            "metodo",
            "roteamento",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f1_macro",
        )
        if column in combined.columns
    ]

    if columns_to_show:

        print(
            combined[
                columns_to_show
            ].to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Executa os experimentos de DS, MoE, "
            "baselines estáticos e comparação estatística."
        )
    )

    parser.add_argument(
        "--only",
        nargs="+",
        choices=[
            "ds",
            "moe",
            "static",
        ],
        help=(
            "Executa apenas as famílias "
            "de experimentos informadas."
        ),
    )

    parser.add_argument(
        "--include-k-sweep",
        action="store_true",
        help=(
            "Executa também a análise de "
            "sensibilidade do k para DS."
        ),
    )

    args = parser.parse_args()

    # Se --only não for informado:
    # executa DS + MoE + Static
    selected = set(
        args.only
        or [
            "ds",
            "moe",
            "static",
        ]
    )

    print(
        "\n" + "=" * 76
    )

    print(
        "EXPERIMENTOS - SARCASM DYNAMIC SELECTION"
    )

    print(
        "=" * 76
    )

    print(
        f"\nRaiz do projeto:\n{ROOT}"
    )

    print(
        "\nExperimentos selecionados:"
    )

    print(
        ", ".join(
            sorted(selected)
        )
    )

    try:

        # ====================================================
        # POOL
        # ====================================================

        if selected:
            ensure_pool()

        # ====================================================
        # DYNAMIC SELECTION
        # ====================================================

        if "ds" in selected:

            run_step(
                (
                    "Seleção dinâmica "
                    "(OLA, KNORA-E, KNORA-U; "
                    "com/sem roteamento)"
                ),
                [
                    sys.executable,
                    "-m",
                    "src.selection.ds_routing",
                ]
            )

            if args.include_k_sweep:

                run_step(
                    "Sensibilidade do k",
                    [
                        sys.executable,
                        "-m",
                        "src.selection.ds_routing_k_sweep",
                    ]
                )

        # ====================================================
        # MIXTURE OF EXPERTS
        # ====================================================

        if "moe" in selected:

            run_step(
                (
                    "Soft-MoE "
                    "(gate treinado em DSEL, "
                    "avaliação em TEST)"
                ),
                [
                    sys.executable,
                    "-m",
                    "src.moe.run_moe",
                ]
            )

        # ====================================================
        # BASELINES ESTÁTICOS
        # ====================================================

        if "static" in selected:

            ds_result = (
                DATA
                / "processed"
                / "selection"
                / "ds_routing_test.csv"
            )

            # O baseline estático usa resultados do DS
            # para algumas comparações.
            if not ds_result.exists():

                run_step(
                    (
                        "Seleção dinâmica necessária "
                        "para comparação estática"
                    ),
                    [
                        sys.executable,
                        "-m",
                        "src.selection.ds_routing",
                    ]
                )

            run_step(
                "Baselines estáticos multimodais",
                [
                    sys.executable,
                    "-m",
                    "src.static.static_multimodal",
                ]
            )

        # ====================================================
        # CONSOLIDAÇÃO
        # ====================================================

        consolidate()

        # ====================================================
        # COMPARAÇÃO ESTATÍSTICA
        # ====================================================

        run_step(
            "Comparação estatística dos métodos",
            [
                sys.executable,
                "-m",
                "src.statistical_comparison",
            ]
        )

        # ====================================================
        # FINAL
        # ====================================================

        print(
            "\n" + "=" * 76
        )

        print(
            "EXPERIMENTOS CONCLUÍDOS"
        )

        print(
            "=" * 76
        )

    except subprocess.CalledProcessError as exc:

        print(
            (
                f"\nERRO: etapa falhou com código "
                f"{exc.returncode}. "
                "Execução interrompida para não gerar "
                "comparação incompleta."
            ),
            file=sys.stderr
        )

        raise SystemExit(
            exc.returncode
        )


if __name__ == "__main__":
    main()