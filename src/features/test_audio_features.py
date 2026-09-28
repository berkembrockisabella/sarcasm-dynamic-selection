from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from transformers import AutoProcessor, AutoModel


AUDIO_PATH = Path(
    "data/processed/audio/1_10004_u.wav"
)

MODEL_NAME = "facebook/wav2vec2-base-960h"


def main():

    print("Arquivo:", AUDIO_PATH)

    # Carrega áudio
    audio, sample_rate = sf.read(
        AUDIO_PATH
    )

    print("Sample rate:", sample_rate)
    print("Formato:", audio.shape)
    print(
        "Duração:",
        round(len(audio) / sample_rate, 2),
        "segundos"
    )

    # Wav2Vec2 espera mono 16 kHz
    if sample_rate != 16000:
        raise ValueError(
            f"Áudio está em {sample_rate} Hz. "
            "Esperado: 16000 Hz."
        )

    # Converte para float32
    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    print("\nCarregando processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    print("Carregando Wav2Vec2...")

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    # Prepara entrada
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    print("\nGerando embedding...")

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # [batch, tempo, dimensão]
    hidden_states = (
        outputs.last_hidden_state
    )

    print(
        "Hidden states:",
        hidden_states.shape
    )

    # Média ao longo do tempo
    embedding = (
        hidden_states
        .mean(dim=1)
        .squeeze(0)
    )

    print(
        "Embedding:",
        embedding.shape
    )

    print(
        "Primeiros valores:"
    )

    print(
        embedding[:10]
    )


if __name__ == "__main__":
    main()