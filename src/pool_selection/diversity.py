'''
Analisar diversidade entre classificadores, com base em suas predições
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

def contingency(y_true, pred_a, pred_b):
    """
    Calcula a tabela de contingência de acertos/erros para dois classificadores.
    """

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    n11 = np.sum(correct_a & correct_b)
    n10 = np.sum(correct_a & ~correct_b)
    n01 = np.sum(~correct_a & correct_b)
    n00 = np.sum(~correct_a & ~correct_b)

    return n11, n10, n01, n00


def disagreement_measure(y_true, pred_a, pred_b):
    """
    Proporção de exemplos em que apenas um dos classificadores está correto.
    """
    n11, n10, n01, n00 = contingency(y_true, pred_a, pred_b)
    n = n11 + n10 + n01 + n00
    
    return (n10 + n01) / n

def double_fault_measure(y_true, pred_a, pred_b):
    """
    Proporção de exemplos em que ambos os classificadores erram.
    """
    n11, n10, n01, n00 = contingency(y_true, pred_a, pred_b)
    n = n11 + n10 + n01 + n00

    return n00 / n

def q_statistic(y_true, pred_a, pred_b):
    """
    Q-statistic de Yule.
    """
    n11, n10, n01, n00 = contingency(y_true, pred_a, pred_b)
    den = n11 * n00 + n10 * n01
    return 0.0 if den == 0 else (n11 * n00 - n10 * n01) / den

def pairwise_table(y_true, predictions):
    """
    Calcula métricas de diversidade para todos os pares de classificadores.
    """
    names = list(predictions.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pred_a = np.asarray(predictions[a])
            pred_b = np.asarray(predictions[b])
            n11, n10, n01, n00 = contingency(y_true, pred_a, pred_b)
            disagreement = disagreement_measure(y_true, pred_a, pred_b)
            double_fault = double_fault_measure(y_true, pred_a, pred_b)
            q_stat = q_statistic(y_true, pred_a, pred_b)
            
            kappa = cohen_kappa_score(pred_a, pred_b)
            error_a = np.mean(pred_a != y_true)
            error_b = np.mean(pred_b != y_true)
            mean_error = (error_a + error_b) / 2
            
            rows.append({
                "candidate_a": a,
                "candidate_b": b,
                
                "n11": n11,
                "n10": n10,
                "n01": n01,
                "n00": n00,
                
                "disagreement": disagreement,
                "double_fault": double_fault,
                "q_statistic": q_stat,
                "kappa": kappa,
                
                "error_a": error_a,
                "error_b": error_b,
                "mean_error": mean_error,
            })
    diversity_df = pd.DataFrame(rows)
    print("pairwise_table")
    print("Linhas:", len(diversity_df))
    print("Colunas:", diversity_df.columns.tolist())
    print(diversity_df.head())
    return diversity_df

def mean_disagreement(candidate, selected, predictions, y):
    if not selected:
        return 0.0
    return float(np.mean([
        disagreement_measure(y, predictions[candidate], predictions[s])
        for s in selected
    ]))
    
def plot_kappa_error(diversity_df, output_path=None, title="Kappa-Error Diagram",):
    """
    Plota o diagrama Kappa-Error.

    Eixo X: Cohen's Kappa entre as predições dos dois classificadores.
    Eixo Y: erro médio do par.

    Cada ponto representa um par de classificadores.

    Interpretação geral:
        menor erro -> melhor desempenho;
        menor kappa -> maior diversidade.
    """

    if diversity_df.empty:
        raise ValueError("diversity_df está vazio")

    required_columns = {"kappa", "mean_error"}
    missing = (required_columns - set(diversity_df.columns))

    if missing:
        raise ValueError(f"Colunas necessárias ausentes: {sorted(missing)}")

    # Remover eventuais NaN
    plot_df = diversity_df.dropna(
        subset=[
            "kappa",
            "mean_error",
        ]
    )

    if plot_df.empty:
        raise ValueError("Não existem pares válidos para plotar o Kappa-Error.")

    # Gráfico
    plt.figure(figsize=(10, 7))
    plt.scatter(
        plot_df["kappa"],
        plot_df["mean_error"],
        alpha=0.7,
        s=45,
    )

    plt.xlabel("Cohen's Kappa")
    plt.ylabel("Mean Error")
    plt.title(title)
    plt.grid(alpha=0.3)

    # Kappa normalmente varia de -1 a 1
    plt.xlim(-1.05, 1.05,)
    plt.tight_layout()

    if output_path is not None:
        output_path = str(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches="tight",)
        print(f"Kappa-Error salvo em: {output_path}")

    plt.close()