import os
import subprocess
import pandas as pd

# Lê o dataset já preparado
df = pd.read_csv("data/processed/mustard_prepared.csv")

# Pasta onde os áudios serão salvos
audio_dir = "data/processed/audio"

# Cria a pasta caso ela ainda não exista
os.makedirs(audio_dir, exist_ok=True)

for _, row in df.iterrows():

    # Caminho do vídeo da instância
    video_path = row["video_path"]

    # Usa o KEY para nomear o arquivo de áudio
    key = row["KEY"]
    audio_path = os.path.join(audio_dir, f"{key}.wav")

    # Evita extrair novamente se o arquivo já existir
    if os.path.exists(audio_path):
        continue

    # Comando do FFmpeg
    comando = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-y",
        audio_path
    ]

    # Executa o FFmpeg sem mostrar toda a saída no terminal
    subprocess.run(
        comando,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

print("Extração de áudio finalizada.")

# Lista apenas arquivos .wav da pasta de áudio
arquivos_audio = [
    arquivo
    for arquivo in os.listdir(audio_dir)
    if arquivo.endswith(".wav")
]

print("Quantidade de áudios extraídos:", len(arquivos_audio))