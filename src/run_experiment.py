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
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"

# Repeticoes do experimento (protocolo E0): --split 03 (ou a variavel de
# ambiente SPLIT_ID=03) roda tudo em data/splits/split_03/ e grava em
# subpastas split_03/ -- os scripts filhos herdam SPLIT_ID pelo ambiente.
# Sem split, tudo continua nos caminhos originais (split unico).
SPLIT_SUFFIX = ""
POOL_RESULTS = None
FROZEN_POOLS = None


def set_split(split_id):
    global SPLIT_SUFFIX, POOL_RESULTS, FROZEN_POOLS

    split_id = (split_id or "").strip()

    if split_id:
        os.environ["SPLIT_ID"] = split_id

    SPLIT_SUFFIX = f"split_{split_id}" if split_id else ""

    POOL_RESULTS = (
        DATA
        / "results"
        / "pool"
        / SPLIT_SUFFIX
        / "pool_selection"
    )

    FROZEN_POOLS = (
        DATA
        / "results"
        / "pool"
        / SPLIT_SUFFIX
        / "frozen_pools"
    )


set_split(os.environ.get("SPLIT_ID"))


# ============================================================
# EXECUÇÃO DE UMA ETAPA
# ============================================================

# Custo de cada etapa executada (protocolo E5): tempo de parede e, se o
# psutil estiver instalado, tempo de CPU e pico de memoria do processo e
# dos filhos (amostrados a cada 0,5 s -- valores aproximados)
CUSTOS = []

# Valores de k da sensibilidade (E3); o k = 7 e o experimento principal
KS_SENSIBILIDADE = [3, 5, 11, 15]


def run_step(label, command, env=None):
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

    ambiente = os.environ.copy()
    if env:
        ambiente.update(env)

    try:
        import psutil
    except ImportError:
        psutil = None

    inicio = time.perf_counter()
    pico_mb = None
    cpu_s = None

    if psutil is None:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            env=ambiente,
        )
    else:
        processo = subprocess.Popen(command, cwd=ROOT, env=ambiente)
        pico = 0
        cpu_por_pid = {}
        while processo.poll() is None:
            try:
                raiz = psutil.Process(processo.pid)
                processos = [raiz] + raiz.children(recursive=True)
            except psutil.Error:
                processos = []
            memoria = 0
            for proc in processos:
                try:
                    memoria += proc.memory_info().rss
                    tempos_cpu = proc.cpu_times()
                    cpu_por_pid[proc.pid] = tempos_cpu.user + tempos_cpu.system
                except psutil.Error:
                    pass
            pico = max(pico, memoria)
            time.sleep(0.5)
        if processo.returncode != 0:
            raise subprocess.CalledProcessError(processo.returncode, command)
        pico_mb = pico / 1024 ** 2
        cpu_s = sum(cpu_por_pid.values())

    CUSTOS.append({
        "split": SPLIT_SUFFIX or "unico",
        "etapa": label,
        "segundos_parede": time.perf_counter() - inicio,
        "segundos_cpu_aprox": cpu_s,
        "pico_memoria_mb_aprox": pico_mb,
    })


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
            / SPLIT_SUFFIX
            / "ds_routing_summary.csv",
            "DS",
        ),
        (
            DATA
            / "processed"
            / "moe"
            / SPLIT_SUFFIX
            / "moe_summary.csv",
            "MoE",
        ),
        (
            DATA
            / "processed"
            / "static"
            / SPLIT_SUFFIX
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
        / SPLIT_SUFFIX
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

    parser.add_argument(
        "--split",
        help=(
            "Roda em data/splits/split_<SPLIT>/ "
            "(ex.: --split 03). Sem ele, usa o split unico."
        ),
    )

    parser.add_argument(
        "--include-diversity",
        action="store_true",
        help=(
            "Executa também a sensibilidade ao peso "
            "desempenho x diversidade (avaliada no DSEL)."
        ),
    )

    args = parser.parse_args()

    # --split all: roda o experimento completo em cada um dos splits de
    # data/splits/split_XX/, um processo por split
    if args.split == "all":
        pastas = sorted((DATA / "splits").glob("split_*"))
        if not pastas:
            raise SystemExit(
                "Nenhum split encontrado. Rode antes: "
                "python src/data/create_splits_repetidos.py"
            )
        for pasta in pastas:
            split_id = pasta.name.replace("split_", "")
            comando = [
                sys.executable,
                "-m",
                "src.run_experiment",
                "--split",
                split_id,
            ]
            if args.only:
                comando += ["--only", *args.only]
            if args.include_k_sweep:
                comando.append("--include-k-sweep")
            if args.include_diversity:
                comando.append("--include-diversity")
            run_step(f"Split {split_id}", comando)
        print("\nTodos os splits concluidos.")
        return

    if args.split:
        set_split(args.split)

    print(
        f"\nSplit: {SPLIT_SUFFIX or 'split unico (data/splits/)'}"
    )

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

                # Mesmo script do experimento principal, so com outro k --
                # sem os sorteios aleatorios (E3 compara so sem x A/B).
                # Saida em data/processed/selection/<split>/k_<K>/
                for k in KS_SENSIBILIDADE:
                    run_step(
                        f"Sensibilidade do k (k = {k})",
                        [
                            sys.executable,
                            "-m",
                            "src.selection.ds_routing",
                        ],
                        env={
                            "DS_K": str(k),
                            "DS_N_SORTEIOS": "0",
                        },
                    )

        # ====================================================
        # PESO DESEMPENHO x DIVERSIDADE NA POOL (E3)
        # ====================================================

        if args.include_diversity:

            # Avalia no DSEL (nunca no teste) as pools escolhidas com
            # pesos de desempenho de 0,0 a 1,0
            run_step(
                "Sensibilidade ao peso da diversidade",
                [
                    sys.executable,
                    "-m",
                    "src.pool_selection.compare_diversity",
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
                / SPLIT_SUFFIX
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
        # CUSTOS (E5)
        # ====================================================

        pasta_custos = DATA / "processed" / "eval" / SPLIT_SUFFIX
        pasta_custos.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(CUSTOS).to_csv(
            pasta_custos / "custos.csv",
            index=False,
        )
        print(f"\nCustos salvos em: {pasta_custos / 'custos.csv'}")

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