import os

import cv2
import numpy as np
import pandas as pd

# Fase 2 da reextracao facial (depois de extract_face_tracks.py):
#   1. escolhe, em cada fala, a trilha de rosto do personagem que fala, pela
#      identidade: para cada personagem (SHOW + SPEAKER) monta um "rosto
#      medio" (prototipo) a partir das suas falas e escolhe a trilha mais
#      parecida com ele. O prototipo comeca pelo maior rosto de cada fala e e
#      refinado algumas vezes -- como o maior rosto ja acerta a maioria, o
#      prototipo converge para o rosto do personagem
#   2. se nenhuma trilha se parece com o personagem, o falante esta fora da
#      tela -- a fala fica sem rosto (emocao vazia, valencia 0.0), em vez de
#      usar o rosto de quem escuta
#   3. personagens genericos (PERSON, OTHER...) ou com poucas falas nao tem
#      prototipo -- usa o maior rosto da fala
#   4. classifica a emocao de cada recorte do falante com o ViT
# Para conferir so a escolha do falante (sem torch): SO_ESCOLHA=1
# Para refazer so a agregacao a partir de visual_falante_quadros.csv: SO_AGREGAR=1

PASTA_VIDEOS = "data/raw/all_videos-MUStarARD/all_videos/final_utterance_videos"
PASTA_TRILHAS = "data/processed/rostos_trilhas"
PASTA_SAIDA = "data/processed/extratores"
PASTA_INSPECAO = "data/processed/inspecao_rostos"
MODELO_EMOCAO = "trpakov/vit-face-expression"

GENERICOS = {"PERSON", "PERSON1", "PERSON3", "OTHER", "MODERATOR", "MEMBER-GIRL", "MEMBER-BOY"}
MIN_FALAS_PROTOTIPO = 5     # personagens com menos falas usam o maior rosto
MIN_DETECCOES = 2           # trilhas mais curtas nao concorrem
LIMIAR_IDENTIDADE = 0.363   # cosseno do SFace para "mesma pessoa" (limiar do OpenCV)
ITERACOES = 4
MARGEM = 0.15               # folga em volta do rosto no recorte
N_INSPECAO = 40             # falas com recortes salvos para conferir a mao

EMOCOES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
POSITIVAS = {"happy"}
NEGATIVAS = {"angry", "disgust", "fear", "sad"}

SO_ESCOLHA = os.environ.get("SO_ESCOLHA", "0") == "1"
SO_AGREGAR = os.environ.get("SO_AGREGAR", "0") == "1"

falas = pd.read_csv("data/processed/mustard_prepared.csv")[["KEY", "SHOW", "SPEAKER"]]

if SO_AGREGAR:
    falas = falas.iloc[0:0]

# Trilhas de cada fala: assinatura media (normalizada), area total, quadros
trilhas_por_fala = {}
for _, row in falas.iterrows():
    dados = np.load(os.path.join(PASTA_TRILHAS, f"{row['KEY']}.npz"))
    trilhas = []
    for trilha in np.unique(dados["trilha"]):
        mascara = dados["trilha"] == trilha
        if mascara.sum() < MIN_DETECCOES:
            continue
        assinaturas = dados["assinatura"][mascara]
        assinaturas = assinaturas / np.linalg.norm(assinaturas, axis=1, keepdims=True)
        media = assinaturas.mean(axis=0)
        trilhas.append({
            "trilha": int(trilha),
            "assinatura": media / np.linalg.norm(media),
            "area": float((dados["deteccao"][mascara, 2] * dados["deteccao"][mascara, 3]).sum()),
            "quadros": dados["quadro"][mascara],
            "deteccoes": dados["deteccao"][mascara],
        })
    trilhas_por_fala[row["KEY"]] = trilhas

# Prototipo de cada personagem, refinado iterativamente
falas["personagem"] = falas["SHOW"].astype(str) + "/" + falas["SPEAKER"].astype(str)
contagem = falas["personagem"].value_counts()
prototipos = {}
for personagem, n in contagem.items():
    if n < MIN_FALAS_PROTOTIPO or personagem.split("/")[1] in GENERICOS:
        continue
    chaves = falas.loc[falas["personagem"] == personagem, "KEY"].tolist()
    # Comeco: o maior rosto de cada fala
    escolhidas = [max(trilhas_por_fala[k], key=lambda t: t["area"])["assinatura"] for k in chaves if trilhas_por_fala[k]]
    prototipo = np.mean(escolhidas, axis=0)
    prototipo = prototipo / np.linalg.norm(prototipo)
    for _ in range(ITERACOES):
        escolhidas = []
        for k in chaves:
            if not trilhas_por_fala[k]:
                continue
            melhor = max(trilhas_por_fala[k], key=lambda t: float(t["assinatura"] @ prototipo))
            if float(melhor["assinatura"] @ prototipo) >= LIMIAR_IDENTIDADE:
                escolhidas.append(melhor["assinatura"])
        if not escolhidas:
            break
        prototipo = np.mean(escolhidas, axis=0)
        prototipo = prototipo / np.linalg.norm(prototipo)
    prototipos[personagem] = prototipo

# Escolha do falante em cada fala
linhas = []
escolha_por_fala = {}
for _, row in falas.iterrows():
    trilhas = trilhas_por_fala[row["KEY"]]
    linha = {"KEY": row["KEY"], "personagem": row["personagem"], "n_trilhas": len(trilhas)}
    escolhida = None
    if not trilhas:
        linha["metodo"] = "sem_rosto"
    elif row["personagem"] in prototipos:
        prototipo = prototipos[row["personagem"]]
        melhor = max(trilhas, key=lambda t: float(t["assinatura"] @ prototipo))
        linha["similaridade"] = float(melhor["assinatura"] @ prototipo)
        if linha["similaridade"] >= LIMIAR_IDENTIDADE:
            escolhida = melhor
            linha["metodo"] = "identidade"
        else:
            linha["metodo"] = "falante_fora_da_tela"
    else:
        escolhida = max(trilhas, key=lambda t: t["area"])
        linha["metodo"] = "maior_rosto"
    linha["quadros_falante"] = 0 if escolhida is None else len(escolhida["quadros"])
    escolha_por_fala[row["KEY"]] = escolhida
    linhas.append(linha)

escolhas = pd.DataFrame(linhas, columns=["KEY", "personagem", "n_trilhas", "metodo", "similaridade", "quadros_falante"])
os.makedirs(PASTA_SAIDA, exist_ok=True)
if not SO_AGREGAR:
    escolhas.to_csv(f"{PASTA_SAIDA}/visual_falante_escolha.csv", index=False)

print(f"Falante escolhido para {len(escolhas)} falas ({len(prototipos)} personagens com prototipo).")
print(escolhas["metodo"].value_counts().to_string())
print("\nSimilaridade com o prototipo (falas com prototipo):")
if len(escolhas):
    print(escolhas["similaridade"].astype(float).describe().round(3).to_string())

# Recortes do falante: os quadros sao lidos de novo do video, so os da trilha
# escolhida -- para a inspecao (primeiras falas) e para o ViT
classificador = None
if not SO_ESCOLHA and not SO_AGREGAR:
    from PIL import Image
    from transformers import pipeline
    # top_k = 7 -- com top_k=None algumas versoes do transformers devolvem so
    # as 5 classes mais provaveis e as outras viram ausentes
    classificador = pipeline("image-classification", model=MODELO_EMOCAO, top_k=len(EMOCOES))

linhas_quadros = []
for numero, key in enumerate(falas["KEY"]):
    escolhida = escolha_por_fala[key]
    inspecionar = numero < N_INSPECAO
    if escolhida is None or (SO_ESCOLHA and not inspecionar):
        if not SO_ESCOLHA:
            linha = {"KEY": key, "quadro": -1}
            linha.update({emocao: np.nan for emocao in EMOCOES})
            linhas_quadros.append(linha)
        continue

    caixas = {int(q): d for q, d in zip(escolhida["quadros"], escolhida["deteccoes"])}
    video = cv2.VideoCapture(os.path.join(PASTA_VIDEOS, f"{key}.mp4"))
    recortes = []
    n_quadro = 0
    while True:
        ok, quadro = video.read()
        if not ok:
            break
        if n_quadro in caixas:
            altura, largura = quadro.shape[:2]
            x, y, w, h = [float(v) for v in caixas[n_quadro][:4]]
            x0, y0 = int(max(0, x - MARGEM * w)), int(max(0, y - MARGEM * h))
            x1, y1 = int(min(largura, x + w * (1 + MARGEM))), int(min(altura, y + h * (1 + MARGEM)))
            if x1 > x0 and y1 > y0:
                recortes.append((n_quadro, quadro[y0:y1, x0:x1].copy()))
        n_quadro += 1
    video.release()

    if inspecionar:
        pasta = os.path.join(PASTA_INSPECAO, key)
        os.makedirs(pasta, exist_ok=True)
        for indice, (n, recorte) in enumerate(recortes[:6]):
            cv2.imwrite(os.path.join(pasta, f"falante_{indice:02d}_q{n}.jpg"), recorte)

    if SO_ESCOLHA:
        continue

    imagens = [Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)) for _, recorte in recortes]
    for (n, _), saida in zip(recortes, classificador(imagens)):
        linha = {"KEY": key, "quadro": n}
        linha.update({item["label"].lower(): item["score"] for item in saida})
        linhas_quadros.append(linha)
    if (numero + 1) % 100 == 0:
        print(f"{numero + 1} falas classificadas...")

print("Recortes para inspecao em:", PASTA_INSPECAO)
if SO_ESCOLHA:
    raise SystemExit

if SO_AGREGAR:
    quadros = pd.read_csv(f"{PASTA_SAIDA}/visual_falante_quadros.csv")
else:
    quadros = pd.DataFrame(linhas_quadros)
    quadros.to_csv(f"{PASTA_SAIDA}/visual_falante_quadros.csv", index=False)

# Quadro classificado (quadro >= 0) com classe ausente = probabilidade ~0
# (fora das mais provaveis); fala sem rosto tem quadro = -1
classificados = quadros["quadro"] >= 0
quadros.loc[classificados, EMOCOES] = quadros.loc[classificados, EMOCOES].fillna(0.0)
quadros["valence"] = quadros["happy"] - quadros[list(NEGATIVAS)].sum(axis=1)

# Agregacao por fala: media dos quadros do falante e, como alternativa, o
# quadro de emocao mais intensa -- a calibracao escolhe pelo treino
for nome in ["media", "pico"]:
    linhas = []
    for key, grupo in quadros.groupby("KEY", sort=False):
        validos = grupo[grupo["quadro"] >= 0]
        if len(validos) == 0:
            probs = {emocao: 0.0 for emocao in EMOCOES}
            emocao_predominante = "vazio"
        elif nome == "media":
            probs = validos[EMOCOES].mean().to_dict()
            emocao_predominante = max(probs, key=probs.get)
        else:
            probs = validos.loc[validos["valence"].abs().idxmax(), EMOCOES].to_dict()
            emocao_predominante = max(probs, key=probs.get)
        linha = {"KEY": key, "n_faces": len(validos)}
        linha.update(probs)
        linha["emocao_predominante"] = emocao_predominante
        linha["valence"] = sum(probs[c] for c in POSITIVAS) - sum(probs[c] for c in NEGATIVAS)
        linhas.append(linha)
    resultado = pd.DataFrame(linhas)
    caminho = f"{PASTA_SAIDA}/visual_falante_{nome}.csv"
    resultado.to_csv(caminho, index=False)
    print(f"\nEmocao facial do falante ({nome}) calculada para {len(resultado)} instancias.")
    print(resultado["emocao_predominante"].value_counts().to_string())
    print("Salvo em:", caminho)
