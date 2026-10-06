import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

# Valencia pretendida (implicita) de cada fala. Na literatura, a
# incongruencia do sarcasmo e o contraste entre a emocao de superficie e a
# emocao pretendida (Ray et al., 2022; Farabi et al., 2024) -- os 4
# classificadores de emocao so veem a superficie, entao a valencia
# pretendida precisa ser estimada. O alvo e a valencia percebida anotada no
# MUStARD++ (escala 1-9, coluna Valence), usada SO no split de treino --
# DSEL e teste recebem a predicao do modelo, nunca a anotacao
df = pd.read_csv("data/processed/mustard_prepared.csv")
treino = pd.read_csv("data/splits/train.csv")

MODALIDADES = ["text", "context", "audio", "visual"]
EMOCOES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
N_FOLDS = 5

# Features: as 7 probabilidades de emocao de cada modalidade (28 colunas).
# Instancia sem rosto detectado fica com probabilidade 0 em tudo, o mesmo
# valor neutro usado nos scripts de valencia
dados = df[["KEY", "SPEAKER", "Valence", "Sarcasm"]].copy()
for modalidade in MODALIDADES:
    tabela = pd.read_csv(f"data/processed/{modalidade}_valence.csv")
    tabela = tabela[["KEY"] + EMOCOES].rename(
        columns={emocao: f"{modalidade}_{emocao}" for emocao in EMOCOES}
    )
    dados = dados.merge(tabela, on="KEY", how="left")

FEATURES = [f"{modalidade}_{emocao}" for modalidade in MODALIDADES for emocao in EMOCOES]
dados[FEATURES] = dados[FEATURES].fillna(0.0)

# Valence 1-9 (5 = neutro) vai para [-1, 1], a mesma escala das valencias
# dos classificadores: (v - 5) / 4
dados["valence_anotada"] = (dados["Valence"] - 5) / 4
eh_treino = dados["KEY"].isin(treino["KEY"])

# Random forest em vez de regressao linear porque o que interessa sao as
# combinacoes entre modalidades (ex.: texto positivo com rosto neutro), que
# um modelo linear nao representa
regressor = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42)

# Instancias de treino recebem predicao out-of-fold (o modelo que preve uma
# instancia nunca viu ela) -- sem isso a feature do treino seria quase a
# propria anotacao e o classificador final aprenderia com um valor que
# nao existe no DSEL/teste. Folds separados por falante, igual aos splits
dados["implied_valence"] = 0.0
dados.loc[eh_treino, "implied_valence"] = cross_val_predict(
    regressor,
    dados.loc[eh_treino, FEATURES],
    dados.loc[eh_treino, "valence_anotada"],
    groups=dados.loc[eh_treino, "SPEAKER"],
    cv=GroupKFold(n_splits=N_FOLDS),
)

# DSEL e teste: modelo treinado no treino inteiro
regressor.fit(dados.loc[eh_treino, FEATURES], dados.loc[eh_treino, "valence_anotada"])
dados.loc[~eh_treino, "implied_valence"] = regressor.predict(dados.loc[~eh_treino, FEATURES])
dados["implied_source"] = eh_treino.map({True: "oof_treino", False: "modelo_treino"})

# A anotacao e o rotulo nao vao para o csv -- so a predicao
resultado = dados[["KEY", "implied_valence", "implied_source"]]
resultado.to_csv("data/processed/implied_valence.csv", index=False)

# Qualidade da estimativa: correlacao de Spearman com a anotacao no treino
# (out-of-fold) e no DSEL. O teste nao e avaliado aqui
dsel = pd.read_csv("data/splits/dsel.csv")
eh_dsel = dados["KEY"].isin(dsel["KEY"])
rho_treino = dados.loc[eh_treino, ["implied_valence", "valence_anotada"]].corr("spearman").iloc[0, 1]
rho_dsel = dados.loc[eh_dsel, ["implied_valence", "valence_anotada"]].corr("spearman").iloc[0, 1]

print("Valencia pretendida estimada para", len(resultado), "instancias.")
print("Instancias de treino (out-of-fold):", eh_treino.sum())
print("Instancias de DSEL/teste (modelo do treino):", (~eh_treino).sum())
print(f"\nSpearman com a valencia anotada -- treino (oof): {rho_treino:.3f} | DSEL: {rho_dsel:.3f}")
print("\nValencia pretendida media por classe (treino + DSEL):")
print(dados[eh_treino | eh_dsel].groupby("Sarcasm")["implied_valence"].mean().round(3))
print("\nEstatisticas de implied_valence:")
print(resultado["implied_valence"].describe())
print("Salvo em: data/processed/implied_valence.csv")
