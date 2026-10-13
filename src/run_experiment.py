"""
Orquestra os experimentos principais de DS, MoE e estáticos
Para executar: `python run_experiment.py`
Exemplos:
- python run_experiment.py --only moe
- python run_experiment.py --only ds static
- python run_experiment.py --include-k-sweep
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POOL_RESULTS = DATA / "results" / "pool" / "pool_selection"
FROZEN_POOL = DATA / "results" / "pool" / "frozen_pools" / "base"

def run_step(label, command):
    print("\n" + "=" * 76, flush=True)
    print(f"ETAPA: {label}", flush=True)
    print("Comando:", " ".join(command), flush=True)
    print("=" * 76, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

def ensure_pool():
    definition = POOL_RESULTS / "pool_definition.csv"
    cv_results = POOL_RESULTS / "cv_results.csv"
    if not definition.exists() or not cv_results.exists():
        run_step("Seleção da pool (CV no TRAIN)", [sys.executable, "-m", "src.pool_selection.build_pool"])
    metadata = FROZEN_POOL / "pool_metadata.csv"
    expected_models = [FROZEN_POOL / f"C{i:02d}.joblib" for i in range(1, 8)]
    if not metadata.exists() or not all(p.exists() for p in expected_models):
        run_step("Treinamento da pool congelada", [sys.executable, "src/pool_training/train_frozen_pool.py"])

def consolidate():
    import pandas as pd
    sources = [
        (DATA / "processed" / "selection" / "ds_routing_summary.csv", "DS"),
        (DATA / "processed" / "moe" / "moe_summary.csv", "MoE"),
        (DATA / "processed" / "eval" / "static_multimodal_summary.csv", "Estático"),
    ]
    frames = []
    for path, family in sources:
        if path.exists():
            df = pd.read_csv(path)
            if "metodo" in df.columns:
                df["familia"] = family
                frames.append(df)
            elif "method" in df.columns:
                df = df.rename(columns={"method": "metodo"})
                df["familia"] = family
                frames.append(df)
    if not frames:
        print("Nenhum resumo encontrado para consolidar.")
        return
    combined = pd.concat(frames, ignore_index=True, sort=False)
    output = DATA / "processed" / "eval" / "all_experiments_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"\nResumo consolidado: {output}")
    cols = [c for c in ("familia", "metodo", "roteamento", "accuracy", "f1_macro") if c in combined.columns]
    print(combined[cols].to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Executa DS, MoE e baselines estáticos.")
    parser.add_argument("--only", nargs="+", choices=["ds", "moe", "static"], help="Executa apenas as famílias listadas.")
    parser.add_argument("--include-k-sweep", action="store_true", help="Inclui a análise de sensibilidade do k para DS.")
    args = parser.parse_args()
    selected = set(args.only or ["ds", "moe", "static"])
    try:
        if selected:
            ensure_pool()
        if "ds" in selected:
            run_step("Seleção dinâmica (OLA, KNORA-E, KNORA-U; com/sem roteamento)", [sys.executable, "src/selection/ds_routing.py"])
            if args.include_k_sweep:
                run_step("Sensibilidade do k", [sys.executable, "src/selection/ds_routing_k_sweep.py"])
        if "moe" in selected:
            run_step("Soft-MoE (gate treinado em DSEL, avaliação em TEST)", [sys.executable, "src/selection/run_moe.py"])
        if "static" in selected:
            if not (DATA / "processed" / "selection" / "ds_routing_test.csv").exists():
                run_step("Seleção dinâmica necessária para comparação", [sys.executable, "src/selection/ds_routing.py"])
            run_step("Baselines estáticos multimodais", [sys.executable, "src/eval/static_multimodal.py"])
        consolidate()
    except subprocess.CalledProcessError as exc:
        print(f"\nERRO: etapa falhou com código {exc.returncode}. Execução interrompida para não gerar comparação incompleta.", file=sys.stderr)
        raise SystemExit(exc.returncode)

if __name__ == "__main__":
    main()
