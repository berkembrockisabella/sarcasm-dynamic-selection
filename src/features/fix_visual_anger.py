import pandas as pd

# Corrige a emocao "angry" no visual_emotions.csv. O classificador facial
# chama a classe de "anger" (ver predicted_emotion), mas a coluna de
# probabilidade foi gravada como "angry" e ficou zerada em todos os frames --
# a probabilidade da raiva se perdeu na troca de nome (o mesmo problema que o
# MAPA_ROTULOS resolve no texto). Os scripts que geraram o csv (frames,
# recorte de rostos, emocao facial) nao estao no repositorio, entao a
# correcao e feita aqui, antes de visual_valence.py e build_visual_features.py.
#
# Suposicao: o classificador facial tem exatamente estas 7 classes e as
# probabilidades somam 1 -- entao a massa que falta em cada frame e a
# probabilidade de raiva. Evidencia: frames previstos como "anger" tem ~55% de
# massa faltando, os outros ~3%. O original continua no historico do git.
CAMINHO = "data/processed/visual_emotions.csv"
OUTRAS = ["disgust", "fear", "happy", "neutral", "sad", "surprise"]

df = pd.read_csv(CAMINHO)

if df["angry"].abs().sum() > 0:
    # Ja corrigido (ou o csv foi gerado de novo com a raiva certa) -- nao
    # mexe, para o script poder rodar mais de uma vez sem estragar nada
    print("Coluna angry ja tem valores -- nada a corrigir.")
else:
    massa_faltando = 1 - df[OUTRAS].sum(axis=1)
    # clip em 0: arredondamento pode deixar a soma das outras um pouco acima de 1
    df["angry"] = massa_faltando.clip(lower=0.0)

    frames_raiva = df["predicted_emotion"] == "anger"
    # Conferencia: nos frames previstos como raiva, a raiva reconstruida
    # deve ser a maior probabilidade
    emocoes = ["angry"] + OUTRAS
    maior = df.loc[frames_raiva, emocoes].idxmax(axis=1)
    confere = (maior == "angry").mean()

    df.to_csv(CAMINHO, index=False)

    print("Raiva reconstruida para", len(df), "frames.")
    print("Frames previstos como anger:", frames_raiva.sum())
    print(f"Desses, angry e a maior probabilidade em: {confere:.1%}")
    print("\nangry reconstruida -- previstos como anger x demais:")
    print(df.groupby(frames_raiva)["angry"].describe().round(3))
    print("Salvo em:", CAMINHO)
