import pandas as pd

emocoes_df = pd.read_csv("data/processed/visual_emotions.csv")

EMOCOES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
POSITIVAS = ["happy"]
NEGATIVAS = ["angry", "disgust", "fear", "sad"]

emocoes_df["face_valence"] = (
    emocoes_df[POSITIVAS].sum(axis=1) - emocoes_df[NEGATIVAS].sum(axis=1)
)

# Media de cada emocao entre todos os rostos/frames de uma instancia
medias_por_emocao = {emocao: (emocao, "mean") for emocao in EMOCOES}

resultado = emocoes_df.groupby("KEY").agg(
    n_faces=("face_valence", "size"),
    visual_valence=("face_valence", "mean"),
    **medias_por_emocao,
).reset_index()

resultado["emocao_predominante"] = resultado[EMOCOES].idxmax(axis=1)

resultado.to_csv("data/processed/visual_valence.csv", index=False)

print("Valencia visual calculada para", len(resultado), "instancias.")
print("\nDistribuicao da emocao predominante:")
print(resultado["emocao_predominante"].value_counts())
print("\nEstatisticas de visual_valence:")
print(resultado["visual_valence"].describe())
print("Salvo em: data/processed/visual_valence.csv")