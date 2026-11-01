import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Compara as variantes da pool (ablacao do peso da diversidade e do limite de
# membros com contexto) com a pool padrao (diversidade 0.3, sem limite).
# Criterio de escolha: F1-macro do voto da pool no DSEL -- o teste so e
# reportado, nunca usado para escolher.
# Rodar depois de aggregate_splits.py para a padrao e para cada variante:
#   python src/aggregate_splits.py
#   python src/aggregate_splits.py div_0.8
#   python src/compare_variants.py

PASTA_FINAL = "data/processed/final"
METODOS = [
    "OLA sem_roteamento",
    "KNORA-E sem_roteamento",
    "KNORA-U sem_roteamento",
    "KNORA-E com_roteamento",
    "KNORA-E conflito",
    "Voto da pool inteira",
    "Melhor membro",
    "Soft-MoE",
]
ALFA = 0.05

variantes = [""] + sorted(
    os.path.basename(os.path.dirname(p))
    for p in glob.glob(f"{PASTA_FINAL}/*/resumo_metodos.csv")
)
splits = sorted(os.path.basename(p) for p in glob.glob("data/splits/split_*"))

linhas = []
por_split = {}
for variante in variantes:
    pasta = os.path.join(PASTA_FINAL, variante)
    nome = variante or "padrao"
    resumo = pd.read_csv(os.path.join(pasta, "resumo_metodos.csv")).set_index("metodo")
    por_split[nome] = pd.read_csv(os.path.join(pasta, "resultados_por_split.csv"))

    # Criterio no DSEL, media dos splits
    dsel = []
    for split in splits:
        arquivo = os.path.join("data/processed/selection", split, variante, "ds_routing_dsel.csv")
        if os.path.exists(arquivo):
            dsel.append(pd.read_csv(arquivo))
    dsel = pd.concat(dsel, ignore_index=True) if dsel else pd.DataFrame()

    composicao = pd.read_csv(os.path.join(pasta, "pool_composicao.csv"))
    com_contexto = composicao["modalities"].str.split("+").apply(lambda m: "context" in m).mean()

    linha = {
        "variante": nome,
        "f1_voto_dsel": round(100 * dsel["f1_macro_voto_dsel"].mean(), 1) if len(dsel) else np.nan,
        "oraculo_dsel": round(100 * dsel["oraculo_dsel"].mean(), 1) if len(dsel) else np.nan,
        "membros_com_contexto_pct": round(100 * com_contexto, 1),
        "oraculo_teste": resumo.loc["Oraculo (algum membro acerta)", "accuracy_media"],
    }
    for metodo in METODOS:
        if metodo in resumo.index:
            linha[f"{metodo}_f1"] = resumo.loc[metodo, "f1_macro_media"]
    linhas.append(linha)

comparacao = pd.DataFrame(linhas)
comparacao.to_csv(f"{PASTA_FINAL}/comparacao_variantes.csv", index=False)

# Wilcoxon pareado (10 splits) de cada variante contra a padrao, por metodo,
# com correcao de Holm sobre todos os testes
linhas = []
base = por_split["padrao"]
for nome, tabela in por_split.items():
    if nome == "padrao":
        continue
    for metodo in METODOS:
        a = tabela[tabela["metodo"] == metodo].set_index("split")["f1_macro"]
        b = base[base["metodo"] == metodo].set_index("split")["f1_macro"]
        pares = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        if len(pares) == 0:
            continue
        diferencas = pares["a"] - pares["b"]
        if (diferencas == 0).all():
            p_valor = 1.0
        else:
            p_valor = wilcoxon(pares["a"], pares["b"], zero_method="wilcox").pvalue
        linhas.append({
            "variante": nome,
            "metodo": metodo,
            "diferenca_media_f1_pp": round(100 * diferencas.mean(), 1),
            "vitorias": int((diferencas > 0).sum()),
            "derrotas": int((diferencas < 0).sum()),
            "p_valor": p_valor,
        })

testes = pd.DataFrame(linhas)
if len(testes):
    ordem = np.argsort(testes["p_valor"].to_numpy())
    p = testes["p_valor"].to_numpy()[ordem]
    m = len(p)
    ajustado = np.empty(m)
    ajustado[ordem] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p), 1.0)
    testes["p_holm"] = ajustado
    testes["significativo_holm"] = testes["p_holm"] < ALFA
testes.to_csv(f"{PASTA_FINAL}/wilcoxon_variantes.csv", index=False)

print(f"Variantes comparadas: {', '.join(comparacao['variante'])}")
print("\nCriterio de escolha (DSEL) e resultados no teste (F1-macro %, media dos splits):")
print(comparacao.to_string(index=False))
melhor = comparacao.loc[comparacao["f1_voto_dsel"].idxmax(), "variante"] if comparacao["f1_voto_dsel"].notna().any() else "-"
print(f"\nVariante com maior F1 do voto no DSEL: {melhor}")
if len(testes):
    print("\nWilcoxon contra a pool padrao:")
    print(testes.round(4).to_string(index=False))
print("\nSalvo em:", f"{PASTA_FINAL}/comparacao_variantes.csv", "e", f"{PASTA_FINAL}/wilcoxon_variantes.csv")
