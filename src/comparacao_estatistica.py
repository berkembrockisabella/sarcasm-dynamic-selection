"""
Comparação estatística pareada:
McNemar pareado + correção de Holm + IC bootstrap pareado.
- Métricas por método
- Teste de McNemar
- Correção de Holm
- Intervalos de confiança de 95% por bootstrap pareado

Para rodar: `python src/statistical_comparison.py`

Saídas:
    metricas_por_metodo.csv
    mcnemar_holm.csv
    bootstrap_diferencas_ic95.csv
"""
import itertools
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_PATH = SRC_DIR / "data" / "processed"
ds_test = RESULTS_PATH / "selection" / "ds_routing_test.csv"
moe_test = RESULTS_PATH / "moe" / "moe_test.csv"
static_test = RESULTS_PATH / "static" / "static_multimodal_test.csv"

OUT_DIR = RESULTS_PATH / "statistical_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOTSTRAP = 1000
ALPHA = 0.05

METRICS = ["accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"]

# Métricas
def metrics(y, pred, score=None):
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": np.nan,
    }
    if score is not None and len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, score)
    return out


# Correção de Holm-Bonferroni
def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


# Teste exato de McNemar
def mcnemar_exact(y, pred_a, pred_b):
    """
    Compara os erros de dois métodos sobre as mesmas amostras.

    n10: A correto e B incorreto.
    n01: A incorreto e B correto.
    """
    a_correct = pred_a == y
    b_correct = pred_b == y
    n10 = int(np.sum(a_correct & ~b_correct))
    n01 = int(np.sum(~a_correct & b_correct))
    discordant = n10 + n01
    
    if discordant == 0:
        p = 1.0
    else:
        p =float(binomtest(min(n10, n01), n=discordant, p=0.5, alternative="two-sided").pvalue)
    return n10, n01, discordant, p

# Carregar .csv com os testes
def load_data():
    """
    Carrega os arquivos .csv de teste e retorna um DataFrame concatenado com as colunas necessárias para a análise estatística.
    """
    for path in [ds_test, moe_test, static_test]:
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    # Mixture of Experts (MoE)
    df_moe = pd.read_csv(moe_test)
    df_moe = df_moe.rename(columns={"KEY": "id", "Sarcasm": "y_true", "prediction": "y_pred", "prob_sarcasm": "y_score"})
    if not {"id", "y_true", "y_pred", "y_score"}.issubset(df_moe.columns):
        raise ValueError("O .CSV do MoE deve conter as colunas KEY, Sarcasm, prediction e prob_sarcasm.")
    df_moe["metodo"] = "MoE"
    
    # Seleção dinâmica (DS)
    df_ds = pd.read_csv(ds_test)
    df_ds = df_ds.rename(columns={"KEY": "id", "Sarcasm": "y_true"})
    ds_methods = ["OLA", "KNORA-E", "KNORA-U"]
    required_ds = {"id", "y_true", "roteamento", *ds_methods}
    if not required_ds.issubset(df_ds.columns):
        raise ValueError("O .CSV de DS deve conter as colunas KEY, Sarcasm, roteamento, OLA, KNORA-E e KNORA-U.")
    df_ds = df_ds.melt(id_vars=["id", "y_true", "roteamento"], value_vars=ds_methods, var_name="metodo", value_name="y_pred")
    df_ds["metodo"] = (df_ds["metodo"] + " (" + df_ds["roteamento"].astype(str) + ")")
    df_ds = df_ds.drop(columns=["roteamento"])
    
    # Classificadores estáticos
    df_static = pd.read_csv(static_test)
    df_static = df_static.rename(columns={"KEY": "id", "Sarcasm": "y_true"})
    static_methods = ["Estatico SVM", "Estatico Random forest", "Estatico voto dos 3"]
    required_static = {"id", "y_true", *static_methods}
    if not required_static.issubset(df_static.columns):
        raise ValueError("O .CSV de classificadores estáticos deve conter as colunas KEY, Sarcasm, Estatico SVM, Estatico Random forest e Estatico voto dos 3.")
    df_static = df_static.melt(id_vars=["id", "y_true"], value_vars=static_methods, var_name="metodo", value_name="y_pred")
    
    # Concatenar todos os DataFrames em um único DataFrame
    df = pd.concat([df_moe, df_ds, df_static], ignore_index=True, sort=False)
    
    # O score é opcional.
    if "y_score" not in df.columns:
        df["y_score"] = np.nan
    
    df["id"] = df["id"].astype(str).str.strip() # Remove espaços acidentais nos IDs textuais.
    if (df["id"] == "").any():
        raise ValueError("Existem IDs vazios nos arquivos.")

    # Valida dados ausentes.
    if df[["id", "y_true", "y_pred"]].isna().any().any():
        raise ValueError("Existem IDs, rótulos reais ou predições ausentes. Verifique os arquivos CSV.")

    # Valida rótulos binários.
    for col in ["y_true", "y_pred"]:
        values = set(pd.to_numeric(df[col], errors="raise").unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"A coluna {col} deve conter somente 0 e 1. Valores encontrados: {values}")
        df[col] = df[col].astype(int)

    # Um ID deve aparecer uma única vez por método.
    duplicated = df.duplicated(["metodo", "id"], keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, ["metodo", "id"]].head(10)
        raise ValueError("Existem IDs duplicados dentro de um método:\n" + examples.to_string(index=False))

    # Todos os métodos precisam avaliar os mesmos IDs.
    methods = df["metodo"].unique().tolist()
    reference_ids = set(df.loc[df["metodo"] == methods[0], "id"])
    for method in methods[1:]:
        method_ids = set(df.loc[df["metodo"] == method, "id"])

        if method_ids != reference_ids:
            missing = reference_ids - method_ids
            extra = method_ids - reference_ids

            raise ValueError(
                f"O método '{method}' não possui os mesmos IDs que os demais.\n"
                f"IDs ausentes: {len(missing)}\n"
                f"IDs adicionais: {len(extra)}"
            )

    # O rótulo real deve ser igual em todos os arquivos.
    if df.groupby("id")["y_true"].nunique().max() > 1:
        raise ValueError("Um mesmo ID possui valores diferentes de y_true.")

    return df

# Execução principal
def main():
    df = load_data()

    methods = df["metodo"].unique().tolist()
    ids = sorted(df["id"].unique(), key=str)

    # Ordena os rótulos reais pela mesma sequência de IDs
    y = df.drop_duplicates("id").set_index("id").loc[ids, "y_true"].to_numpy()
    
    preds, scores = {}, {}
    
    for method in methods:
        part = df[df["metodo"] == method].set_index("id").loc[ids]
        preds[method] = part["y_pred"].to_numpy(dtype=int)
        score = pd.to_numeric(part["y_score"], errors="coerce").to_numpy(dtype=float)
        # AUC requer um score válido para cada amostra.
        scores[method] = score if np.isfinite(score).all() else None

    # Métricas individuais
    metric_rows = []
    for method in methods:
        metric_rows.append({
            "metodo": method,
            "n_amostras": len(y),
            **metrics(y, preds[method], scores[method])
        })
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "metricas_por_metodo.csv", index=False)

    # McNemar + correção de Holm
    pair_rows = []
    for a, b in itertools.combinations(methods, 2):
        n10, n01, discordant, p = mcnemar_exact(y, preds[a], preds[b])
        pair_rows.append({"metodo_a": a,
                        "metodo_b": b,
                        "a_correto_b_errado": n10,
                        "a_errado_b_correto": n01,
                        "pares_discordantes": discordant,
                        "p_valor": p
                    })
    adjusted = holm([row["p_valor"] for row in pair_rows])
    for row, p_adj in zip(pair_rows, adjusted):
        row["p_holm"] = p_adj
        row["significativo_05_holm"] = bool(p_adj < ALPHA)
    pd.DataFrame(pair_rows).to_csv(OUT_DIR / "mcnemar_holm.csv", index=False)

    # Bootstrap pareado e IC de 95%
    rng = np.random.default_rng(SEED)
    boot_rows = []
    n = len(y)
    for a, b in itertools.combinations(methods, 2):
        observed_a, observed_b = metrics(y, preds[a], scores[a]), metrics(y, preds[b], scores[b])
        samples = {metric: [] for metric in METRICS}
        for _ in range(N_BOOTSTRAP):
            ix = rng.integers(0, n, size=n)  # mesma reamostragem para A e B
            y_boot = y[ix]
            score_a = None if scores[a] is None else scores[a][ix]
            score_b = None if scores[b] is None else scores[b][ix]
            ma = metrics(y_boot, preds[a][ix], score_a)
            mb = metrics(y_boot, preds[b][ix], score_b)
            for metric in METRICS:
                va = ma[metric]
                vb = mb[metric]
                if np.isfinite(va) and np.isfinite(vb):
                    samples[metric].append(va - vb)
        # Intervalos percentis de 95%
        for metric in METRICS:
            vals = np.asarray(samples[metric], dtype=float)
            va = observed_a[metric]
            vb = observed_b[metric]
            observed = (va - vb if np.isfinite(va) and np.isfinite(vb) else np.nan)
            low, high = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
            boot_rows.append({"metodo_a": a,
                            "metodo_b": b,
                            "metrica": metric,
                            "diferenca_a_menos_b": observed,
                            "ic95_inferior": low,
                            "ic95_superior": high,
                            "replicas_validas": len(vals),
                            "replicas_solicitadas": N_BOOTSTRAP})
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "bootstrap_diferencas_ic95.csv", index=False)

    # Resumo
    print(f"Resultados salvos em: {OUT_DIR.resolve()}")
    print(f"Número de amostras: {len(y)}")
    print(f"Número de métodos: {len(methods)}")
    print(f"Número de comparações: {len(pair_rows)}")
    print(f"Réplicas bootstrap: {N_BOOTSTRAP}")
    print()
    print("Arquivos gerados:")
    print(" - metricas_por_metodo.csv")
    print(" - mcnemar_holm.csv")
    print(" - bootstrap_diferencas_ic95.csv")


if __name__ == "__main__":
    main()
