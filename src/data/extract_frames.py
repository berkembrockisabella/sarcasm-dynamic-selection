import os
import cv2
import pandas as pd

# Lê o dataset preparado
df = pd.read_csv("data/processed/mustard_prepared.csv")

# Pasta onde os frames serão salvos
frames_dir = "data/processed/frames"

# Cria a pasta principal caso não exista
os.makedirs(frames_dir, exist_ok=True)

# Quantidade de frames que queremos extrair por vídeo
num_frames = 10

# Percorre todas as instâncias do dataset
for _, row in df.iterrows():

    key = row["KEY"]
    video_path = row["video_path"]

    # Cria uma pasta específica para cada vídeo
    output_dir = os.path.join(frames_dir, key)
    os.makedirs(output_dir, exist_ok=True)

    # Evita extrair novamente se já existirem os 10 frames
    frames_existentes = [
        arquivo
        for arquivo in os.listdir(output_dir)
        if arquivo.endswith(".jpg")
    ]

    if len(frames_existentes) >= num_frames:
        continue

    # Abre o vídeo
    video = cv2.VideoCapture(video_path)

    # Quantidade total de frames do vídeo
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Não foi possível ler o vídeo: {key}")
        video.release()
        continue

    # Calcula o intervalo para distribuir os frames ao longo do vídeo
    intervalo = max(total_frames // num_frames, 1)

    frame_atual = 0
    frames_salvos = 0

    while True:
        sucesso, frame = video.read()

        if not sucesso:
            break

        # Salva somente os frames selecionados
        if frame_atual % intervalo == 0 and frames_salvos < num_frames:

            caminho_frame = os.path.join(
                output_dir,
                f"frame_{frames_salvos:02d}.jpg"
            )

            cv2.imwrite(caminho_frame, frame)

            frames_salvos += 1

        frame_atual += 1

    video.release()

print("Extração de frames finalizada.")

# Conta quantas pastas de instâncias foram criadas
pastas = [
    pasta
    for pasta in os.listdir(frames_dir)
    if os.path.isdir(os.path.join(frames_dir, pasta))
]

print("Quantidade de vídeos processados:", len(pastas))