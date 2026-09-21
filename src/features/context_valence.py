import pandas as pd
from transformers import pipeline

df = pd.read_csv("data/processed/mustard_prepared.csv")
bruto = pd.read_csv("data/raw/mustard++_text.csv")

contexto_por_scene = {}
for scene, grupo in bruto.groupby("SCENE"):
    linhas_contexto = grupo[grupo["KEY"].str.contains("_c_")].copy()
    linhas_contexto = linhas_contexto.sort_values("KEY")
    contexto_por_scene[scene] = " ".join(linhas_contexto["SENTENCE"].astype(str))

# DeBERTa-v3-large fine-tuned em GoEmotions + ISEAR + DAIR-AI -- mesmo
# classificador usado em text_valence.py, para manter o vocabulario de
# emocoes consistente entre os dois
classificador = pipeline(
    "text-classification",
    model="Tanneru/Emotion-Classification-DeBERTa-v3-Large",
    top_k=None,
)

# Mesma traducao de rotulos usada em text_valence.py, para manter o
# vocabulario de emocoes consistente entre os dois -- so "anger" difere
MAPA_ROTULOS = {
    "anger": "angry",
}

EMOCOES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
POSITIVAS = {"happy"}
NEGATIVAS = {"angry", "disgust", "fear", "sad"}

linhas = []

for _, row in df.iterrows():
    texto_contexto = contexto_por_scene.get(row["SCENE"], "")

    if texto_contexto.strip() == "":
        probs = {emocao: 0.0 for emocao in EMOCOES}
        emocao_predominante = "vazio"
        valencia = 0.0
    else:
        saida = classificador(texto_contexto, truncation=True)[0]
        probs = {
            MAPA_ROTULOS.get(item["label"], item["label"]): item["score"]
            for item in saida
        }
        emocao_predominante = max(probs, key=probs.get)
        p_positiva = sum(probs.get(c, 0.0) for c in POSITIVAS)
        p_negativa = sum(probs.get(c, 0.0) for c in NEGATIVAS)
        valencia = p_positiva - p_negativa

    linha = {"KEY": row["KEY"], "context": texto_contexto}
    linha.update({emocao: probs.get(emocao, 0.0) for emocao in EMOCOES})
    linha["emocao_predominante"] = emocao_predominante
    linha["context_valence"] = valencia
    linhas.append(linha)

resultado = pd.DataFrame(linhas)
resultado.to_csv("data/processed/context_valence.csv", index=False)

print("Valencia de contexto calculada para", len(resultado), "instancias.")
print("\nDistribuicao da emocao predominante:")
print(resultado["emocao_predominante"].value_counts())
print("\nEstatisticas de context_valence:")
print(resultado["context_valence"].describe())
print("Salvo em: data/processed/context_valence.csv")
