import os

import pandas as pd
from transformers import pipeline

# Extratores alternativos de emocao para texto e contexto -- candidatos que a
# calibracao (src/incongruence/calibrar_polaridade.py) compara com os atuais
# usando so o treino de cada split. Gera, no mesmo formato dos *_valence.csv:
#   text_jhartmann.csv             -- fala, distilroberta j-hartmann
#   context_jhartmann.csv          -- contexto inteiro, distilroberta j-hartmann
#   context_anterior_jhartmann.csv -- so a fala anterior, distilroberta j-hartmann
#   context_anterior_deberta.csv   -- so a fala anterior, DeBERTa atual
# O j-hartmann foi treinado com varios conjuntos, entre eles o MELD (dialogos
# de Friends), mesmo dominio de sitcom do MUStARD++. A fala anterior entra
# porque o contexto inteiro (varias falas juntas) tende a virar "neutro"

PASTA_SAIDA = "data/processed/extratores"
EMOCOES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
POSITIVAS = {"happy"}
NEGATIVAS = {"angry", "disgust", "fear", "sad"}

MODELOS = {
    "jhartmann": ("j-hartmann/emotion-english-distilroberta-base", {"anger": "angry", "joy": "happy", "sadness": "sad"}),
    "deberta": ("Tanneru/Emotion-Classification-DeBERTa-v3-Large", {"anger": "angry"}),
}

df = pd.read_csv("data/processed/mustard_prepared.csv")
bruto = pd.read_csv("data/raw/mustard++_text.csv")

# Contexto inteiro e fala anterior de cada cena -- mesma ordenacao de
# context_valence.py (c_00, c_01, ...)
contexto_inteiro = {}
fala_anterior = {}
for scene, grupo in bruto.groupby("SCENE"):
    linhas_contexto = grupo[grupo["KEY"].str.contains("_c_")].sort_values("KEY")
    frases = linhas_contexto["SENTENCE"].astype(str).tolist()
    contexto_inteiro[scene] = " ".join(frases)
    fala_anterior[scene] = frases[-1] if frases else ""

ENTRADAS = {
    "text": lambda row: row["SENTENCE"],
    "context": lambda row: contexto_inteiro.get(row["SCENE"], ""),
    "context_anterior": lambda row: fala_anterior.get(row["SCENE"], ""),
}

# (entrada, modelo) a gerar -- texto e contexto inteiro com DeBERTa ja
# existem em text_valence.csv e context_valence.csv
COMBINACOES = [
    ("text", "jhartmann"),
    ("context", "jhartmann"),
    ("context_anterior", "jhartmann"),
    ("context_anterior", "deberta"),
]

os.makedirs(PASTA_SAIDA, exist_ok=True)
classificadores = {}

for entrada, nome_modelo in COMBINACOES:
    caminho = f"{PASTA_SAIDA}/{entrada}_{nome_modelo}.csv"
    if os.path.exists(caminho):
        print("Ja existe, pulando:", caminho)
        continue

    modelo, mapa_rotulos = MODELOS[nome_modelo]
    if nome_modelo not in classificadores:
        classificadores[nome_modelo] = pipeline("text-classification", model=modelo, top_k=None)
    classificador = classificadores[nome_modelo]

    linhas = []
    for _, row in df.iterrows():
        texto = ENTRADAS[entrada](row)
        # Entrada vazia vira neutro (0.0), como nos outros scripts
        if not isinstance(texto, str) or texto.strip() == "":
            probs = {emocao: 0.0 for emocao in EMOCOES}
            emocao_predominante = "vazio"
        else:
            saida = classificador(texto, truncation=True)[0]
            probs = {mapa_rotulos.get(item["label"], item["label"]): item["score"] for item in saida}
            emocao_predominante = max(probs, key=probs.get)
        linha = {"KEY": row["KEY"]}
        linha.update({emocao: probs.get(emocao, 0.0) for emocao in EMOCOES})
        linha["emocao_predominante"] = emocao_predominante
        linha["valence"] = sum(probs.get(c, 0.0) for c in POSITIVAS) - sum(probs.get(c, 0.0) for c in NEGATIVAS)
        linhas.append(linha)

    resultado = pd.DataFrame(linhas)
    resultado.to_csv(caminho, index=False)
    print(f"\n{entrada} / {nome_modelo}: calculado para {len(resultado)} instancias.")
    print(resultado["emocao_predominante"].value_counts().to_string())
    print("Salvo em:", caminho)
