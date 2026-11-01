import os

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

# Diagnostico dos extratores de emocao: compara a polaridade e a valencia
# que cada modalidade recebeu (texto, contexto, audio, rosto) com a
# anotacao humana do MUStARD++ para a fala -- Explicit_Emotion (emocao
# expressa, anotada vendo a fala completa) e Valence (escala 1 a 8).
# Pergunta: o excesso de "neutro" vem da fala ou do extrator?
# A emocao implicita nao entra -- fica fora do escopo do trabalho

PASTA_SAIDA = "data/processed/diagnostico"
MODALIDADES = ["text", "context", "audio", "visual"]

# Polaridade da anotacao -- mesmo criterio do projeto: happy positiva,
# angry/disgust/fear/sad negativas, neutral/surprise neutras. Excitement
# conta como positiva e Frustration como negativa
POLARIDADE_ANOTACAO = {
    "Happiness": "positiva",
    "Excitement": "positiva",
    "Sadness": "negativa",
    "Anger": "negativa",
    "Frustration": "negativa",
    "Disgust": "negativa",
    "Fear": "negativa",
    "Neutral": "neutra",
    "Surprise": "neutra",
}

anotacao = pd.read_csv("data/raw/mustard++_text.csv")
anotacao = anotacao[anotacao["Sarcasm"].notna()][["KEY", "Sarcasm", "Explicit_Emotion", "Valence"]].copy()
# Mesma correcao de prepare_mustard.py -- linha desalinhada do CSV original
anotacao.loc[anotacao["KEY"] == "Disgust", "KEY"] = "1_S11E03_067_u"
anotacao["polaridade_anotada"] = anotacao["Explicit_Emotion"].map(POLARIDADE_ANOTACAO)

incongruencia = pd.read_csv("data/processed/incongruence.csv")
base = incongruencia.merge(anotacao, on="KEY", how="inner")

linhas = []
for modalidade in MODALIDADES:
    prevista = base[f"{modalidade}_polarity"]
    anotada = base["polaridade_anotada"]
    com_emocao = anotada != "neutra"
    rho, p_rho = spearmanr(base[f"{modalidade}_valence"], base["Valence"])
    linhas.append({
        "modalidade": modalidade,
        "n": len(base),
        "concordancia_polaridade": (prevista == anotada).mean(),
        "kappa_polaridade": cohen_kappa_score(prevista, anotada),
        # Falas que os anotadores viram com emocao, mas o extrator deu neutro
        "neutro_quando_anotada_com_emocao": (prevista[com_emocao] == "neutra").mean(),
        # Falas com emocao nos dois lados, mas de sinal oposto
        "polaridade_invertida": ((prevista != "neutra") & com_emocao & (prevista != anotada)).mean(),
        "spearman_valencia": rho,
        "p_spearman": p_rho,
    })

resumo = pd.DataFrame(linhas)

# Polaridade anotada por grupo de roteamento -- o grupo A ("tudo neutro" em
# 81% dos casos) tambem e neutro para os anotadores?
grupos = []
for coluna in ["grupo"]:
    tabela = pd.crosstab(base[coluna], base["polaridade_anotada"], normalize="index").mul(100).round(1)
    tabela["n"] = base[coluna].value_counts()
    grupos.append(tabela.reset_index().rename(columns={coluna: "grupo"}))
base["grupo_conflito"] = (base["n_oppositions"] > 0).map({True: "conflito", False: "sem_conflito"})
tabela = pd.crosstab(base["grupo_conflito"], base["polaridade_anotada"], normalize="index").mul(100).round(1)
tabela["n"] = base["grupo_conflito"].value_counts()
grupos.append(tabela.reset_index().rename(columns={"grupo_conflito": "grupo"}))
por_grupo = pd.concat(grupos, ignore_index=True)

# Matriz de confusao da polaridade (linhas = extrator, colunas = anotacao)
confusao = []
for modalidade in MODALIDADES:
    matriz = pd.crosstab(base[f"{modalidade}_polarity"], base["polaridade_anotada"])
    matriz.insert(0, "modalidade", modalidade)
    confusao.append(matriz.reset_index().rename(columns={f"{modalidade}_polarity": "polaridade_extrator"}))
confusao = pd.concat(confusao, ignore_index=True)

os.makedirs(PASTA_SAIDA, exist_ok=True)
resumo.to_csv(f"{PASTA_SAIDA}/extratores_vs_anotacao.csv", index=False)
por_grupo.to_csv(f"{PASTA_SAIDA}/grupos_vs_anotacao.csv", index=False)
confusao.to_csv(f"{PASTA_SAIDA}/confusao_polaridade.csv", index=False)

print(f"Diagnostico calculado para {len(base)} instancias.")
print("\nPolaridade anotada (Explicit_Emotion):")
print(base["polaridade_anotada"].value_counts(normalize=True).mul(100).round(1).to_string())
print("\nExtrator x anotacao:")
print(resumo.round(3).to_string(index=False))
print("\nPolaridade anotada (%) por grupo de roteamento:")
print(por_grupo.to_string(index=False))
print("\nSalvo em:", PASTA_SAIDA)
