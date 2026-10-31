import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon
from sklearn.metrics import cohen_kappa_score

# Agrega os resultados dos 10 splits (protocolo final do TCC):
#   E1 -- comparacao principal: media, desvio, pior e melhor split por
#         metodo; Friedman + Nemenyi sobre o F1-macro (Demsar, 2006)
#   E2 -- ablacao do roteamento: Wilcoxon pareado + correcao de Holm
#         (A/B x sem e A/B x aleatorio, para OLA, KNORA-E e KNORA-U)
#   E3 -- sensibilidade ao k e ao peso desempenho x diversidade (descritivo)
#   E4 -- analise de erros: erro do seletor x fala que nenhum membro acerta
#   E5 -- custo computacional
# Rodar da raiz do repositorio, depois de: python -m src.run_experiment --split all
# (com --include-k-sweep e --include-diversity para o E3)

sys.path.insert(0, "src")
from pool_selection.config import N_SPLITS, PARAM_GRIDS
from pool_selection.feature_loader import FeatureStore

PASTA_SAIDA = "data/processed/final"
METRICAS = ["f1_macro", "accuracy", "precision", "recall"]
METODOS_DS = ["OLA", "KNORA-E", "KNORA-U"]
KS_SENSIBILIDADE = [3, 5, 7, 11, 15]
ALFA = 0.05

# Metodos do E1 -- sem roteamento, para nao misturar com a ablacao (E2).
# O oraculo nao entra no Friedman porque so tem acuracia
METODOS_E1 = [
    "Estatico SVM",
    "Estatico Regressao logistica",
    "Estatico Random forest",
    "Estatico voto dos 3",
    "Pior membro",
    "Melhor membro",
    "Voto da pool inteira",
    "OLA sem_roteamento",
    "KNORA-E sem_roteamento",
    "KNORA-U sem_roteamento",
    "Soft-MoE",
]

SPLITS = sorted(os.path.basename(p) for p in glob.glob("data/splits/split_*"))
if not SPLITS:
    raise SystemExit("Nenhum split em data/splits/split_*")

os.makedirs(PASTA_SAIDA, exist_ok=True)


# ============================================================
# 1. Resumos por split
# ============================================================

linhas = []
splits_incompletos = []

for split in SPLITS:
    arquivos = [
        f"data/processed/selection/{split}/ds_routing_summary.csv",
        f"data/processed/moe/{split}/moe_summary.csv",
        f"data/processed/static/{split}/static_multimodal_summary.csv",
    ]
    if not all(os.path.exists(a) for a in arquivos):
        splits_incompletos.append(split)
        continue
    for arquivo in arquivos:
        tabela = pd.read_csv(arquivo)
        for _, row in tabela.iterrows():
            metodo = str(row["metodo"])
            # Cada sorteio aleatorio isolado fica fora -- entra so a media
            if re.search(r"aleatorio_\d+$", metodo):
                continue
            # Remove o detalhe entre parenteses para o nome ser o mesmo em
            # todos os splits -- ex.: "Melhor membro (C03)" -> "Melhor membro"
            if metodo.startswith(("Estatico", "Melhor membro", "Pior membro")):
                metodo = re.sub(r" \(.*\)$", "", metodo)
            linha = {"split": split, "metodo": metodo}
            for metrica in METRICAS:
                linha[metrica] = row.get(metrica, np.nan)
            linhas.append(linha)

if splits_incompletos:
    print("AVISO -- splits sem resultados completos (ignorados):", splits_incompletos)

resultados = pd.DataFrame(linhas)
SPLITS_OK = sorted(resultados["split"].unique())
resultados.to_csv(f"{PASTA_SAIDA}/resultados_por_split.csv", index=False)


# ============================================================
# 2. E1 -- media, desvio, pior e melhor split (em %, 1 casa decimal)
# ============================================================

linhas = []
for metodo, parte in resultados.groupby("metodo"):
    linha = {"metodo": metodo, "n_splits": len(parte)}
    for metrica in METRICAS:
        valores = parte[metrica] * 100
        linha[f"{metrica}_media"] = round(valores.mean(), 1)
        linha[f"{metrica}_dp"] = round(valores.std(ddof=1), 1)
        linha[f"{metrica}_min"] = round(valores.min(), 1)
        linha[f"{metrica}_max"] = round(valores.max(), 1)
    # Pior e melhor split pelo F1-macro (oraculo: pela acuracia)
    referencia = "f1_macro" if parte["f1_macro"].notna().any() else "accuracy"
    linha["pior_split"] = parte.loc[parte[referencia].idxmin(), "split"]
    linha["melhor_split"] = parte.loc[parte[referencia].idxmax(), "split"]
    linhas.append(linha)

resumo_metodos = pd.DataFrame(linhas).sort_values("f1_macro_media", ascending=False)
resumo_metodos.to_csv(f"{PASTA_SAIDA}/resumo_metodos.csv", index=False)


# ============================================================
# 3. E1 -- Friedman + Nemenyi sobre o F1-macro
# ============================================================

tabela_e1 = resultados[resultados["metodo"].isin(METODOS_E1)].pivot(
    index="split", columns="metodo", values="f1_macro"
)[METODOS_E1].dropna()
n_blocos = len(tabela_e1)
n_metodos = len(METODOS_E1)

# Rank 1 = maior F1-macro no split; empates recebem o rank medio
ranks = tabela_e1.rank(axis=1, ascending=False, method="average")
rank_medio = ranks.mean().sort_values()

chi2_friedman, p_friedman = friedmanchisquare(*[tabela_e1[m] for m in METODOS_E1])

# Distancia critica de Nemenyi -- q_alfa e a amplitude studentizada
# dividida por raiz de 2 (Demsar, 2006)
q_alfa = studentized_range.ppf(1 - ALFA, n_metodos, np.inf) / np.sqrt(2)
distancia_critica = q_alfa * np.sqrt(n_metodos * (n_metodos + 1) / (6 * n_blocos))

pd.DataFrame([{
    "n_splits": n_blocos,
    "n_metodos": n_metodos,
    "chi2_friedman": chi2_friedman,
    "p_friedman": p_friedman,
    "distancia_critica_nemenyi": distancia_critica,
}]).to_csv(f"{PASTA_SAIDA}/friedman.csv", index=False)

pd.DataFrame({"metodo": rank_medio.index, "rank_medio": rank_medio.values}).to_csv(
    f"{PASTA_SAIDA}/ranks_medios.csv", index=False
)

linhas = []
for i, metodo_a in enumerate(METODOS_E1):
    for metodo_b in METODOS_E1[i + 1:]:
        diferenca = rank_medio[metodo_a] - rank_medio[metodo_b]
        linhas.append({
            "metodo_a": metodo_a,
            "metodo_b": metodo_b,
            "diferenca_rank": diferenca,
            # So vale como pos-teste se o Friedman rejeitar H0
            "significativo_nemenyi": bool(abs(diferenca) > distancia_critica and p_friedman < ALFA),
        })
pd.DataFrame(linhas).to_csv(f"{PASTA_SAIDA}/nemenyi_pares.csv", index=False)


# ============================================================
# 4. E2 -- Wilcoxon pareado + Holm (ablacao do roteamento)
# ============================================================

linhas = []
for metodo in METODOS_DS:
    for comparado in ["sem_roteamento", "aleatorio"]:
        a = resultados[resultados["metodo"] == f"{metodo} com_roteamento"].set_index("split")["f1_macro"]
        b = resultados[resultados["metodo"] == f"{metodo} {comparado}"].set_index("split")["f1_macro"]
        pares = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        diferencas = pares["a"] - pares["b"]
        # Todas as diferencas zero -- o roteamento nao mudou nenhuma predicao;
        # o Wilcoxon nao se aplica e o p fica 1.0
        if (diferencas == 0).all():
            p_valor = 1.0
            estatistica = np.nan
        else:
            estatistica, p_valor = wilcoxon(pares["a"], pares["b"], zero_method="wilcox")
        linhas.append({
            "metodo": metodo,
            "comparacao": f"com_roteamento x {comparado}",
            "n_splits": len(pares),
            "diferenca_media_f1_pp": round(diferencas.mean() * 100, 1),
            "vitorias": int((diferencas > 0).sum()),
            "empates": int((diferencas == 0).sum()),
            "derrotas": int((diferencas < 0).sum()),
            "estatistica_w": estatistica,
            "p_valor": p_valor,
        })

wilcoxon_e2 = pd.DataFrame(linhas)

# Correcao de Holm sobre os 6 testes
ordem = np.argsort(wilcoxon_e2["p_valor"].to_numpy())
p_ordenado = wilcoxon_e2["p_valor"].to_numpy()[ordem]
m = len(p_ordenado)
p_holm_ordenado = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p_ordenado), 1.0)
p_holm = np.empty(m)
p_holm[ordem] = p_holm_ordenado
wilcoxon_e2["p_holm"] = p_holm
wilcoxon_e2["significativo_holm"] = wilcoxon_e2["p_holm"] < ALFA
wilcoxon_e2.to_csv(f"{PASTA_SAIDA}/wilcoxon_roteamento.csv", index=False)

# F1-macro por grupo A/B no teste -- ajuda a ler o E2. Os sorteios aleatorios
# viram uma linha so (media dos sorteios) por split e grupo
linhas = []
for split in SPLITS_OK:
    arquivo = f"data/processed/selection/{split}/ds_routing_group.csv"
    if not os.path.exists(arquivo):
        continue
    tabela = pd.read_csv(arquivo)
    tabela["metodo"] = tabela["metodo"].str.replace(r"aleatorio_\d+$", "aleatorio", regex=True)
    tabela["metodo"] = tabela["metodo"].str.replace(r" \(C\d+\)$", "", regex=True)
    tabela = tabela.groupby(["metodo", "grupo"], as_index=False)["f1_macro"].mean()
    tabela["split"] = split
    linhas.append(tabela)

if linhas:
    por_grupo = pd.concat(linhas, ignore_index=True)
    por_grupo = por_grupo.groupby(["metodo", "grupo"])["f1_macro"].agg(["mean", "std"]).mul(100).round(1).reset_index()
    por_grupo.columns = ["metodo", "grupo", "f1_macro_media", "f1_macro_dp"]
    por_grupo.to_csv(f"{PASTA_SAIDA}/resumo_por_grupo.csv", index=False)


# ============================================================
# 5. E3 -- sensibilidade ao k e ao peso desempenho x diversidade
# ============================================================

linhas = []
for split in SPLITS_OK:
    for k in KS_SENSIBILIDADE:
        pasta = f"data/processed/selection/{split}" + ("" if k == 7 else f"/k_{k}")
        arquivo = f"{pasta}/ds_routing_summary.csv"
        if not os.path.exists(arquivo):
            continue
        tabela = pd.read_csv(arquivo)
        for metodo in METODOS_DS:
            for roteamento in ["sem_roteamento", "com_roteamento"]:
                valor = tabela.loc[tabela["metodo"] == f"{metodo} {roteamento}", "f1_macro"]
                if len(valor):
                    linhas.append({"split": split, "k": k, "metodo": metodo,
                                   "roteamento": roteamento, "f1_macro": valor.iloc[0]})

if linhas:
    sens_k = pd.DataFrame(linhas)
    sens_k = sens_k.groupby(["metodo", "roteamento", "k"])["f1_macro"].agg(["mean", "std", "count"]).reset_index()
    sens_k["mean"] = (sens_k["mean"] * 100).round(1)
    sens_k["std"] = (sens_k["std"] * 100).round(1)
    sens_k.columns = ["metodo", "roteamento", "k", "f1_macro_media", "f1_macro_dp", "n_splits"]
    sens_k.to_csv(f"{PASTA_SAIDA}/sensibilidade_k.csv", index=False)
else:
    sens_k = None
    print("AVISO -- sem resultados de k (rode com --include-k-sweep)")

linhas = []
pools_distintas = []
for split in SPLITS_OK:
    arquivo = f"data/results/pool/{split}/diversity_config_validation/configuration_summary.csv"
    if not os.path.exists(arquivo):
        continue
    tabela = pd.read_csv(arquivo)
    tabela["split"] = split
    linhas.append(tabela)
    # Quantas pools diferentes os 11 pesos geram -- se for 1, o peso nao importa
    conjuntos = tabela["selected_candidates"].apply(lambda texto: frozenset(texto.split(";")))
    pools_distintas.append({"split": split, "n_pools_distintas": conjuntos.nunique()})

if linhas:
    diversidade = pd.concat(linhas, ignore_index=True)
    sens_div = diversidade.groupby("alpha_performance").agg(
        f1_macro_dsel_media=("dsel_f1_macro_majority_vote", "mean"),
        f1_macro_dsel_dp=("dsel_f1_macro_majority_vote", "std"),
        discordancia_media=("dsel_mean_pairwise_disagreement", "mean"),
    ).reset_index()
    sens_div[["f1_macro_dsel_media", "f1_macro_dsel_dp"]] = (sens_div[["f1_macro_dsel_media", "f1_macro_dsel_dp"]] * 100).round(1)
    sens_div["discordancia_media"] = (sens_div["discordancia_media"] * 100).round(1)
    sens_div.to_csv(f"{PASTA_SAIDA}/sensibilidade_diversidade.csv", index=False)
    pd.DataFrame(pools_distintas).to_csv(f"{PASTA_SAIDA}/sensibilidade_diversidade_pools.csv", index=False)
else:
    sens_div = None
    print("AVISO -- sem resultados de diversidade (rode com --include-diversity)")


# ============================================================
# 6. E4 -- analise de erros
# ============================================================

# Categorias por fala de teste:
#   acerto
#   erro_seletor -- o metodo errou, mas algum membro da pool acertaria
#   erro_pool    -- nenhum membro da pool acerta (limite da pool/features)
linhas = []
diversidade_pool = []
composicao = []

for split in SPLITS_OK:
    membros = pd.read_csv(f"data/processed/selection/{split}/ds_routing_membros_test.csv")
    colunas_membros = [c for c in membros.columns if re.fullmatch(r"C\d+", c)]
    acertos_membros = membros[colunas_membros].eq(membros["Sarcasm"], axis=0)
    membros["oraculo"] = acertos_membros.any(axis=1)

    # Diversidade da pool no teste: discordancia e kappa medios entre pares
    discordancias = []
    kappas = []
    for i, a in enumerate(colunas_membros):
        for b in colunas_membros[i + 1:]:
            discordancias.append((membros[a] != membros[b]).mean())
            kappas.append(cohen_kappa_score(membros[a], membros[b]))
    diversidade_pool.append({
        "split": split,
        "discordancia_media": np.mean(discordancias),
        "kappa_medio": np.nanmean(kappas),
        "oraculo": membros["oraculo"].mean(),
    })

    # Composicao da pool -- modalidades e classificadores dos 7 membros
    metadados = pd.read_csv(f"data/results/pool/{split}/frozen_pools/pool_metadata.csv")
    for _, row in metadados.iterrows():
        composicao.append({"split": split, "member_id": row["member_id"],
                           "modalities": row["modalities"], "classifier": row["classifier"]})

    # Predicoes de cada metodo no teste
    predicoes = {}
    ds = pd.read_csv(f"data/processed/selection/{split}/ds_routing_test.csv")
    for roteamento in ["sem_roteamento", "com_roteamento"]:
        parte = ds[ds["roteamento"] == roteamento].set_index("KEY")
        for metodo in METODOS_DS:
            predicoes[f"{metodo} {roteamento}"] = parte[metodo]
    moe = pd.read_csv(f"data/processed/moe/{split}/moe_test.csv").set_index("KEY")
    predicoes["Soft-MoE"] = moe["prediction"]
    estatico = pd.read_csv(f"data/processed/static/{split}/static_multimodal_test.csv").set_index("KEY")
    for coluna in ["Estatico SVM", "Estatico Regressao logistica", "Estatico Random forest", "Estatico voto dos 3"]:
        if coluna in estatico.columns:
            predicoes[coluna] = estatico[coluna]

    base = membros.set_index("KEY")
    for metodo, predicao in predicoes.items():
        predicao = predicao.reindex(base.index)
        acertou = predicao.to_numpy() == base["Sarcasm"].to_numpy()
        for (grupo, classe, oraculo, acerto) in zip(base["grupo"], base["Sarcasm"], base["oraculo"], acertou):
            if acerto:
                categoria = "acerto"
            elif oraculo:
                categoria = "erro_seletor"
            else:
                categoria = "erro_pool"
            linhas.append({"split": split, "metodo": metodo, "grupo": grupo,
                           "classe": "sarcasmo" if classe == 1 else "nao_sarcasmo",
                           "categoria": categoria})

erros = pd.DataFrame(linhas)

# Proporcao de cada categoria (todas as falas de teste dos 10 splits juntas)
for recorte, nome_arquivo in [([], "erros_geral.csv"), (["grupo"], "erros_por_grupo.csv"), (["classe"], "erros_por_classe.csv")]:
    contagem = erros.groupby(["metodo", *recorte, "categoria"]).size().unstack("categoria", fill_value=0)
    proporcao = contagem.div(contagem.sum(axis=1), axis=0).mul(100).round(1)
    proporcao["n_falas"] = contagem.sum(axis=1)
    proporcao.reset_index().to_csv(f"{PASTA_SAIDA}/{nome_arquivo}", index=False)

diversidade_pool = pd.DataFrame(diversidade_pool)
diversidade_pool.to_csv(f"{PASTA_SAIDA}/pool_diversidade_teste.csv", index=False)

composicao = pd.DataFrame(composicao)
composicao.to_csv(f"{PASTA_SAIDA}/pool_composicao.csv", index=False)

# Em quantos membros (de todos os splits) cada modalidade aparece
uso_modalidades = {
    modalidade: float(round(100 * composicao["modalities"].str.split("+").apply(lambda m: modalidade in m).mean(), 1))
    for modalidade in ["text", "context", "audio", "visual"]
}


# ============================================================
# 7. E5 -- custo computacional
# ============================================================

linhas = []
for split in SPLITS_OK:
    arquivo = f"data/processed/eval/{split}/custos.csv"
    if os.path.exists(arquivo):
        linhas.append(pd.read_csv(arquivo))

if linhas:
    custos = pd.concat(linhas, ignore_index=True)
    # A etapa do k aparece com o valor no nome; agrupa por etapa exata
    custos_resumo = custos.groupby("etapa").agg(
        n=("segundos_parede", "size"),
        segundos_parede_media=("segundos_parede", "mean"),
        segundos_parede_dp=("segundos_parede", "std"),
        segundos_cpu_media=("segundos_cpu_aprox", "mean"),
        pico_memoria_mb_media=("pico_memoria_mb_aprox", "mean"),
    ).round(1).reset_index()
    custos_resumo.to_csv(f"{PASTA_SAIDA}/custos_etapas.csv", index=False)
else:
    custos_resumo = None
    print("AVISO -- sem custos.csv (a pool pode ter sido reaproveitada de uma execucao anterior)")

linhas = []
for split in SPLITS_OK:
    arquivo = f"data/processed/selection/{split}/ds_routing_tempo.csv"
    if os.path.exists(arquivo):
        tabela = pd.read_csv(arquivo)
        tabela["roteamento"] = tabela["roteamento"].str.replace(r"aleatorio_\d+$", "aleatorio", regex=True)
        linhas.append(tabela)
if linhas:
    tempo_ds = pd.concat(linhas).groupby("roteamento")["ms_por_fala_3_metodos"].agg(["mean", "std"]).round(2).reset_index()
    tempo_ds.to_csv(f"{PASTA_SAIDA}/custos_selecao_por_fala.csv", index=False)

# Contagens analiticas -- por candidato, o GridSearchCV treina
# grid x N_SPLITS modelos + 1 refit, e o OOF refaz os N_SPLITS folds
n_combinacoes = 15
treinos_por_classificador = {
    nome: int(np.prod([len(v) for v in grade.values()])) * N_SPLITS + 1 + N_SPLITS
    for nome, grade in PARAM_GRIDS.items()
}
treinos_candidatos = sum(treinos_por_classificador.values())

store = FeatureStore()
chave = pd.read_csv(f"data/splits/{SPLITS_OK[0]}/test.csv")["KEY"].iloc[0]
dim_gate = store.matrix([chave], ("text", "context", "audio", "visual")).shape[1]
parametros_gate = dim_gate * 32 + 32 + 32 * 7 + 7

pd.DataFrame([
    {"abordagem": "Estaticos (3 classificadores, 4 modalidades)",
     "treinos_busca_hiperparametros": treinos_candidatos,
     "treinos_finais": 3, "parametros_treinaveis_extras": 0},
    {"abordagem": "Pool + selecao dinamica",
     "treinos_busca_hiperparametros": n_combinacoes * treinos_candidatos,
     "treinos_finais": 7, "parametros_treinaveis_extras": 0},
    {"abordagem": "Pool + Soft-MoE",
     "treinos_busca_hiperparametros": n_combinacoes * treinos_candidatos,
     "treinos_finais": 7, "parametros_treinaveis_extras": parametros_gate},
]).to_csv(f"{PASTA_SAIDA}/custos_analiticos.csv", index=False)


# ============================================================
# 8. Resumo no terminal
# ============================================================

print(f"Splits agregados: {len(SPLITS_OK)} ({', '.join(SPLITS_OK)})")

print("\nE1 -- F1-macro e acuracia (%): media +- dp [pior, melhor]")
for _, row in resumo_metodos.iterrows():
    print(f"  {row['metodo']:<32} F1 {row['f1_macro_media']:>5} +- {row['f1_macro_dp']:<4} "
          f"[{row['f1_macro_min']}, {row['f1_macro_max']}]   acc {row['accuracy_media']}")

print(f"\nFriedman: chi2 = {chi2_friedman:.2f}, p = {p_friedman:.4f} "
      f"({n_metodos} metodos, {n_blocos} splits); CD Nemenyi = {distancia_critica:.2f}")
print("Ranks medios:")
print(rank_medio.round(2).to_string())

print("\nE2 -- Wilcoxon + Holm (F1-macro, com_roteamento contra):")
print(wilcoxon_e2[["metodo", "comparacao", "diferenca_media_f1_pp", "vitorias", "empates",
                   "derrotas", "p_valor", "p_holm"]].round(4).to_string(index=False))

if sens_k is not None:
    print("\nE3 -- F1-macro medio por k:")
    print(sens_k.pivot_table(index=["metodo", "roteamento"], columns="k", values="f1_macro_media").to_string())
if sens_div is not None:
    print("\nE3 -- peso do desempenho (F1-macro do voto no DSEL):")
    print(sens_div.to_string(index=False))
    print("Pools distintas entre os 11 pesos, por split:",
          pd.DataFrame(pools_distintas)["n_pools_distintas"].tolist())

print(f"\nE4 -- categorias de erro (% das falas de teste, {len(SPLITS_OK)} splits juntos):")
print(pd.read_csv(f"{PASTA_SAIDA}/erros_geral.csv").to_string(index=False))
print("\nDiversidade da pool no teste (media dos splits):",
      diversidade_pool[["discordancia_media", "kappa_medio", "oraculo"]].mean().round(3).to_dict())
print("Membros que usam cada modalidade (%):", uso_modalidades)

if custos_resumo is not None:
    print("\nE5 -- tempo medio por etapa (s):")
    print(custos_resumo[["etapa", "n", "segundos_parede_media", "segundos_cpu_media",
                         "pico_memoria_mb_media"]].to_string(index=False))
print("\nE5 -- contagens:")
print(pd.read_csv(f"{PASTA_SAIDA}/custos_analiticos.csv").to_string(index=False))

print("\nSalvo em:", PASTA_SAIDA)
