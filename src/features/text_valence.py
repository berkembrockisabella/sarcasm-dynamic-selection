import pandas as pd
from transformers import pipeline

# Le o dataset preparado
df = pd.read_csv("data/processed/mustard_prepared.csv")

# Classificador de emocao categorica: 6 emocoes basicas de Ekman + neutro
classificador = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)

# Traduz os rotulos nativos do classificador para o vocabulario padrao do
# projeto (o mesmo ja usado na expressao facial)
MAPA_ROTULOS = {
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
}

EMOCOES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
POSITIVAS = {"happy"}
NEGATIVAS = {"angry", "disgust", "fear", "sad"}

linhas = []

for _, row in df.iterrows():
    sentence = row["SENTENCE"]

    if not isinstance(sentence, str) or sentence.strip() == "":
        probs = {emocao: 0.0 for emocao in EMOCOES}
        emocao_predominante = "vazio"
        valencia = 0.0
    else:
        saida = classificador(sentence, truncation=True)[0]
        probs = {
            MAPA_ROTULOS.get(item["label"], item["label"]): item["score"]
            for item in saida
        }
        emocao_predominante = max(probs, key=probs.get)
        p_positiva = sum(probs.get(c, 0.0) for c in POSITIVAS)
        p_negativa = sum(probs.get(c, 0.0) for c in NEGATIVAS)
        valencia = p_positiva - p_negativa

    linha = {"KEY": row["KEY"], "SENTENCE": sentence}
    linha.update({emocao: probs.get(emocao, 0.0) for emocao in EMOCOES})
    linha["emocao_predominante"] = emocao_predominante
    linha["text_valence"] = valencia
    linhas.append(linha)

resultado = pd.DataFrame(linhas)
resultado.to_csv("data/processed/text_valence.csv", index=False)

print("Valencia textual calculada para", len(resultado), "instancias.")
print("\nDistribuicao da emocao predominante:")
print(resultado["emocao_predominante"].value_counts())
print("\nEstatisticas de text_valence:")
print(resultado["text_valence"].describe())
print("Salvo em: data/processed/text_valence.csv")