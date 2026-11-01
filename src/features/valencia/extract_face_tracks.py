import os
import urllib.request

import cv2
import numpy as np
import pandas as pd

# Fase 1 da reextracao facial: encontra e acompanha os rostos de cada fala.
#   - amostra QUADROS_POR_SEGUNDO quadros do video da fala
#   - detecta rostos com o YuNet e calcula a "assinatura" de identidade de
#     cada rosto com o SFace (os dois do OpenCV, leves, rodam em CPU)
#   - acompanha cada rosto entre quadros pela sobreposicao das caixas, sem
#     atravessar cortes de cena
# Salva, por fala, as trilhas com quadros, caixas/pontos e assinaturas em
# data/processed/rostos_trilhas/<KEY>.npz. A fase 2 (extract_face_emotions.py)
# usa as assinaturas para achar o rosto do personagem que fala.
# Por que identidade: na conferencia a mao, escolher o maior rosto ou o com
# mais movimento de boca acertou o falante em so ~60% das falas -- quem
# escuta aparece em primeiro plano com frequencia.
# Retomavel: falas com .npz ja salvo sao puladas.

PASTA_VIDEOS = "data/raw/all_videos-MUStarARD/all_videos/final_utterance_videos"
PASTA_SAIDA = "data/processed/rostos_trilhas"

MODELOS = {
    "models/face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "models/face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

QUADROS_POR_SEGUNDO = 3
CONFIANCA_MINIMA = 0.8      # deteccoes abaixo disso sao descartadas
SOBREPOSICAO_MINIMA = 0.3   # IoU para considerar o mesmo rosto entre quadros
CORTE_DE_CENA = 35.0        # diferenca media entre quadros acima disso = corte
N_VIDEOS = int(os.environ.get("N_VIDEOS", "0"))  # 0 = todos (teste: N_VIDEOS=20)

os.makedirs(PASTA_SAIDA, exist_ok=True)
for caminho, url in MODELOS.items():
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    if not os.path.exists(caminho):
        print("Baixando", caminho)
        urllib.request.urlretrieve(url, caminho)

detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320), CONFIANCA_MINIMA)
reconhecedor = cv2.FaceRecognizerSF.create("models/face_recognition_sface_2021dec.onnx", "")

falas = pd.read_csv("data/processed/mustard_prepared.csv")["KEY"].tolist()
if N_VIDEOS:
    falas = falas[:N_VIDEOS]

processadas = 0
for key in falas:
    saida = os.path.join(PASTA_SAIDA, f"{key}.npz")
    if os.path.exists(saida):
        continue

    video = cv2.VideoCapture(os.path.join(PASTA_VIDEOS, f"{key}.mp4"))
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    passo = max(1, int(round(fps / QUADROS_POR_SEGUNDO)))

    # Uma linha por deteccao: trilha, quadro, 15 valores do YuNet, assinatura
    trilha_ids, quadros, deteccoes_salvas, assinaturas = [], [], [], []
    ativas = {}           # trilha -> ultima caixa, so dentro da cena atual
    proxima_trilha = 0
    miniatura_anterior = None
    n_quadro = 0
    while True:
        ok, quadro = video.read()
        if not ok:
            break
        if n_quadro % passo == 0:
            altura, largura = quadro.shape[:2]

            # Corte de cena: nenhuma trilha antiga continua depois dele
            miniatura = cv2.resize(cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY), (64, 36)).astype(float)
            if miniatura_anterior is not None and np.abs(miniatura - miniatura_anterior).mean() > CORTE_DE_CENA:
                ativas = {}
            miniatura_anterior = miniatura

            detector.setInputSize((largura, altura))
            _, deteccoes = detector.detect(quadro)
            deteccoes = [] if deteccoes is None else deteccoes
            usadas = set()
            novas_ativas = dict(ativas)
            for deteccao in deteccoes:
                x, y, w, h = [float(v) for v in deteccao[:4]]
                melhor, melhor_iou = None, SOBREPOSICAO_MINIMA
                for trilha, (tx, ty, tw, th) in ativas.items():
                    if trilha in usadas:
                        continue
                    inter_w = max(0.0, min(x + w, tx + tw) - max(x, tx))
                    inter_h = max(0.0, min(y + h, ty + th) - max(y, ty))
                    inter = inter_w * inter_h
                    iou = inter / (w * h + tw * th - inter) if inter > 0 else 0.0
                    if iou > melhor_iou:
                        melhor, melhor_iou = trilha, iou
                if melhor is None:
                    melhor = proxima_trilha
                    proxima_trilha += 1
                usadas.add(melhor)
                novas_ativas[melhor] = (x, y, w, h)

                # Assinatura de identidade do rosto alinhado pelos 5 pontos
                alinhado = reconhecedor.alignCrop(quadro, deteccao)
                assinatura = reconhecedor.feature(alinhado).flatten()
                trilha_ids.append(melhor)
                quadros.append(n_quadro)
                deteccoes_salvas.append(np.asarray(deteccao[:15], dtype=np.float32))
                assinaturas.append(assinatura.astype(np.float32))
            ativas = novas_ativas
        n_quadro += 1
    video.release()

    np.savez_compressed(
        saida,
        trilha=np.asarray(trilha_ids, dtype=np.int32),
        quadro=np.asarray(quadros, dtype=np.int32),
        deteccao=np.asarray(deteccoes_salvas, dtype=np.float32).reshape(-1, 15),
        assinatura=np.asarray(assinaturas, dtype=np.float32).reshape(-1, 128),
    )
    processadas += 1
    if processadas % 50 == 0:
        print(f"{processadas} falas processadas...")

feitas = len([f for f in os.listdir(PASTA_SAIDA) if f.endswith(".npz")])
print(f"\nTrilhas de rostos calculadas para {processadas} falas nesta execucao ({feitas} no total).")
print("Salvo em:", PASTA_SAIDA)
