import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Calibracao da polaridade de cada modalidade, feita so com o TREINO de cada
# split. A mesma regra vale para as 4 modalidades:
#   valencia V = P(happy) - P(angry + disgust + fear + sad)
#   positiva se V > tau, negativa se V < -tau, neutra no meio (zona morta)
# Para cada modalidade e cada split, escolhe a fonte (extrator) e o tau que
# mais concordam (kappa de Cohen) com a emocao explicita anotada no
# MUStARD++ -- so nas falas do treino; DSEL e teste ficam so para conferir.
# Depois refaz os grupos com a mesma regra de build_incongruence.py:
#   A/B -- as 4 modalidades com a mesma polaridade (A) ou nao (B)
#   n_oppositions -- pares com polaridades opostas (positiva x negativa)
# Saida: data/processed/incongruence_calibrada/<split>.csv, lida pelo
# ds_routing.py no lugar de incongruence.csv

PASTA_SAIDA = "data/processed/incongruence_calibrada"
MODALIDADES = ["text", "context", "audio", "visual"]
NEGATIVAS = ["angry", "disgust", "fear", "sad"]
TAUS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# Candidatos por modalidade -- os que nao existirem no disco sao ignorados
FONTES = {
    "text": {
        "deberta": "data/processed/text_valence.csv",
        "jhartmann": "data/processed/extratores/text_jhartmann.csv",
    },
    "context": {
        "deberta": "data/processed/context_valence.csv",
        "jhartmann": "data/processed/extratores/context_jhartmann.csv",
        "deberta_anterior": "data/processed/extratores/context_anterior_deberta.csv",
        "jhartmann_anterior": "data/processed/extratores/context_anterior_jhartmann.csv",
    },
    "audio": {
        "emotion2vec": "data/processed/audio_valence.csv",
    },
    "visual": {
        "rosto_media": "data/processed/visual_valence.csv",
        "rosto_falante_media": "data/processed/extratores/visual_falante_media.csv",
        "rosto_falante_pico": "data/processed/extratores/visual_falante_pico.csv",
    },
}

# Polaridade da anotacao -- mesmo criterio de diagnostico_extratores.py
POLARIDADE_ANOTACAO = {
    "Happiness": "positiva", "Excitement": "positiva",
    "Sadness": "negativa", "Anger": "negativa", "Frustration": "negativa",
    "Disgust": "negativa", "Fear": "negativa",
    "Neutral": "neutra", "Surprise": "neutra",
}

anotacao = pd.read_csv("data/raw/mustard++_text.csv")
anotacao = anotacao[anotacao["Sarcasm"].notna()][["KEY", "Explicit_Emotion"]].copy()
anotacao.loc[anotacao["KEY"] == "Disgust", "KEY"] = "1_S11E03_067_u"
anotacao["polaridade_anotada"] = anotacao["Explicit_Emotion"].map(POLARIDADE_ANOTACAO)
anotacao = anotacao.set_index("KEY")["polaridade_anotada"]

# Valencia de cada fonte, alinhada pelas KEYs da anotacao. Emocao ausente
# (sem rosto, audio sem emocao valida) vira valencia 0.0 -- neutra
valencias = {}
for modalidade, fontes in FONTES.items():
    for fonte, caminho in fontes.items():
        if not os.path.exists(caminho):
            print("Fonte ausente, ignorada:", caminho)
            continue
        tabela = pd.read_csv(caminho).set_index("KEY")
        v = tabela["happy"].fillna(0.0) - tabela[NEGATIVAS].fillna(0.0).sum(axis=1)
        valencias[(modalidade, fonte)] = v.reindex(anotacao.index).fillna(0.0)

# Polaridade original (argmax, de incongruence.csv) -- so para comparar
original = pd.read_csv("data/processed/incongruence.csv").set_index("KEY").reindex(anotacao.index)


def polaridade(v, tau):
    return pd.Series(np.where(v > tau, "positiva", np.where(v < -tau, "negativa", "neutra")), index=v.index)


splits = sorted(os.path.basename(p) for p in glob.glob("data/splits/split_*"))
conjuntos_por_split = {s: f"data/splits/{s}" for s in splits}

os.makedirs(PASTA_SAIDA, exist_ok=True)
escolhas = []
grupos_resumo = []

for split, pasta in conjuntos_por_split.items():
    chaves = {nome: pd.read_csv(f"{pasta}/{nome}.csv")["KEY"].tolist() for nome in ["train", "dsel", "test"]}
    resultado = pd.DataFrame(index=anotacao.index)

    for modalidade in MODALIDADES:
        # Escolha da fonte e do tau so no treino
        melhor = None
        for (mod, fonte), v in valencias.items():
            if mod != modalidade:
                continue
            for tau in TAUS:
                pol = polaridade(v, tau)
                kappa = cohen_kappa_score(anotacao[chaves["train"]], pol[chaves["train"]])
                if melhor is None or kappa > melhor[0]:
                    melhor = (kappa, fonte, tau, pol)

        kappa_treino, fonte, tau, pol = melhor
        resultado[f"{modalidade}_polarity"] = pol
        linha = {"split": split, "modalidade": modalidade, "fonte": fonte, "tau": tau}
        for nome in ["train", "dsel", "test"]:
            k = chaves[nome]
            linha[f"kappa_original_{nome}"] = cohen_kappa_score(anotacao[k], original.loc[k, f"{modalidade}_polarity"])
            linha[f"kappa_calibrado_{nome}"] = cohen_kappa_score(anotacao[k], pol[k])
        linha["neutro_original_pct"] = 100 * (original[f"{modalidade}_polarity"] == "neutra").mean()
        linha["neutro_calibrado_pct"] = 100 * (pol == "neutra").mean()
        escolhas.append(linha)

    # Grupos com a mesma regra de build_incongruence.py
    colunas = [f"{m}_polarity" for m in MODALIDADES]
    resultado["grupo"] = np.where(resultado[colunas].nunique(axis=1) == 1, "A", "B")
    n_oposicoes = 0
    for i, a in enumerate(MODALIDADES):
        for b in MODALIDADES[i + 1:]:
            pa, pb = resultado[f"{a}_polarity"], resultado[f"{b}_polarity"]
            n_oposicoes = n_oposicoes + ((pa != "neutra") & (pb != "neutra") & (pa != pb)).astype(int)
    resultado["n_oppositions"] = n_oposicoes
    resultado["deadpan"] = ((resultado["audio_polarity"] == "neutra") & (resultado["visual_polarity"] == "neutra")).astype(int)
    resultado = resultado.reset_index()
    resultado.to_csv(f"{PASTA_SAIDA}/{split}.csv", index=False)

    tudo_neutro = (resultado[colunas] == "neutra").all(axis=1)
    for nome in ["train", "dsel", "test"]:
        parte = resultado[resultado["KEY"].isin(chaves[nome])]
        grupos_resumo.append({
            "split": split,
            "conjunto": nome,
            "n": len(parte),
            "grupo_A": int((parte["grupo"] == "A").sum()),
            "grupo_A_tudo_neutro": int((tudo_neutro[parte.index] & (parte["grupo"] == "A")).sum()),
            "conflito": int((parte["n_oppositions"] > 0).sum()),
        })

escolhas = pd.DataFrame(escolhas)
grupos_resumo = pd.DataFrame(grupos_resumo)
escolhas.to_csv(f"{PASTA_SAIDA}/calibracao_escolhas.csv", index=False)
grupos_resumo.to_csv(f"{PASTA_SAIDA}/calibracao_grupos.csv", index=False)

print(f"Polaridade calibrada para {len(conjuntos_por_split)} splits.")
print("\nFonte escolhida por modalidade (nº de splits):")
print(escolhas.groupby(["modalidade", "fonte"]).size().to_string())
print("\nKappa com a anotacao, media dos splits (original -> calibrado):")
media = escolhas.groupby("modalidade")[[c for c in escolhas.columns if c.startswith("kappa") or c.startswith("neutro")]].mean().round(3)
print(media.to_string())
print("\nGrupos (media dos splits):")
print(grupos_resumo.groupby("conjunto")[["n", "grupo_A", "grupo_A_tudo_neutro", "conflito"]].mean().round(1).to_string())
print("\nSalvo em:", PASTA_SAIDA)
