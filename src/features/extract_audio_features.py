from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from transformers import AutoModel, AutoProcessor


# --------------------------------------------------
# Caminhos
# --------------------------------------------------

AUDIO_DIR = Path("data/processed/audio")

PREPARED_PATH = Path(
    "data/processed/mustard_prepared.csv"
)

OUTPUT_DIR = Path(
    "data/processed/features"
)

EMBEDDINGS_PATH = (
    OUTPUT_DIR / "audio_embeddings.npy"
)

KEYS_PATH = (
    OUTPUT_DIR / "audio_keys.csv"
)

PARTIAL_EMBEDDINGS_PATH = (
    OUTPUT_DIR / "audio_embeddings_partial.npy"
)

PARTIAL_KEYS_PATH = (
    OUTPUT_DIR / "audio_keys_partial.csv"
)


# --------------------------------------------------
# Modelo
# --------------------------------------------------

MODEL_NAME = "facebook/wav2vec2-base-960h"

EXPECTED_SAMPLE_RATE = 16000


# --------------------------------------------------
# Funções auxiliares
# --------------------------------------------------

def load_audio(audio_path):
    audio, sample_rate = sf.read(audio_path)

    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"{audio_path.name}: "
            f"sample rate = {sample_rate}, "
            f"esperado = {EXPECTED_SAMPLE_RATE}"
        )

    # Caso apareça áudio estéreo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    return audio


def extract_embedding(
    audio,
    processor,
    model,
    device
):

    inputs = processor(
        audio,
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # Média ao longo do eixo temporal
    embedding = (
        outputs
        .last_hidden_state
        .mean(dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return embedding


def save_partial(
    embeddings,
    keys
):

    if not embeddings:
        return

    np.save(
        PARTIAL_EMBEDDINGS_PATH,
        np.vstack(embeddings)
    )

    pd.DataFrame(
        {"KEY": keys}
    ).to_csv(
        PARTIAL_KEYS_PATH,
        index=False
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Carregando dados...")

    df = pd.read_csv(
        PREPARED_PATH
    )

    keys = (
        df["KEY"]
        .astype(str)
        .tolist()
    )

    print(
        f"Total de instâncias: {len(keys)}"
    )

    # --------------------------------------------------
    # Dispositivo
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Dispositivo: {device}"
    )

    # --------------------------------------------------
    # Modelo
    # --------------------------------------------------

    print(
        "Carregando processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )

    print(
        "Carregando Wav2Vec2..."
    )

    model = (
        AutoModel
        .from_pretrained(
            MODEL_NAME
        )
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Retomada automática
    # --------------------------------------------------

    embeddings = []
    processed_keys = []

    if (
        PARTIAL_EMBEDDINGS_PATH.exists()
        and PARTIAL_KEYS_PATH.exists()
    ):

        print(
            "\nArquivos parciais encontrados."
        )

        partial_embeddings = np.load(
            PARTIAL_EMBEDDINGS_PATH
        )

        partial_keys_df = pd.read_csv(
            PARTIAL_KEYS_PATH
        )

        processed_keys = (
            partial_keys_df["KEY"]
            .astype(str)
            .tolist()
        )

        embeddings = [
            row
            for row
            in partial_embeddings
        ]

        print(
            f"Retomando a partir de "
            f"{len(processed_keys)} "
            f"instâncias processadas."
        )

    processed_set = set(
        processed_keys
    )

    # --------------------------------------------------
    # Extração
    # --------------------------------------------------

    print(
        "\nIniciando extração "
        "dos embeddings de áudio...\n"
    )

    total = len(keys)

    for key in keys:

        if key in processed_set:
            continue

        audio_path = (
            AUDIO_DIR
            / f"{key}.wav"
        )

        if not audio_path.exists():

            print(
                f"[ERRO] Áudio não encontrado: "
                f"{audio_path}"
            )

            continue

        try:

            audio = load_audio(
                audio_path
            )

            embedding = (
                extract_embedding(
                    audio,
                    processor,
                    model,
                    device
                )
            )

            embeddings.append(
                embedding
            )

            processed_keys.append(
                key
            )

            processed_set.add(
                key
            )

            print(
                f"Processadas: "
                f"{len(processed_keys)}/{total} "
                f"- {key}"
            )

            # Salva a cada 25 áudios
            if (
                len(processed_keys)
                % 25
                == 0
            ):

                save_partial(
                    embeddings,
                    processed_keys
                )

                print(
                    "Checkpoint salvo."
                )

        except Exception as error:

            print(
                f"[ERRO] {key}: {error}"
            )

    # --------------------------------------------------
    # Validação
    # --------------------------------------------------

    if len(processed_keys) != total:

        print(
            "\nATENÇÃO:"
        )

        print(
            f"Esperado: {total}"
        )

        print(
            f"Processado: "
            f"{len(processed_keys)}"
        )

        save_partial(
            embeddings,
            processed_keys
        )

        print(
            "\nArquivos parciais salvos."
        )

        return

    # --------------------------------------------------
    # Salva arquivos finais
    # --------------------------------------------------

    final_embeddings = np.vstack(
        embeddings
    )

    np.save(
        EMBEDDINGS_PATH,
        final_embeddings
    )

    pd.DataFrame(
        {
            "KEY": processed_keys
        }
    ).to_csv(
        KEYS_PATH,
        index=False
    )

    print(
        "\nExtração concluída."
    )

    print(
        f"Formato final: "
        f"{final_embeddings.shape}"
    )

    print(
        "\nArquivos salvos:"
    )

    print(
        EMBEDDINGS_PATH
    )

    print(
        KEYS_PATH
    )

    # --------------------------------------------------
    # Remove checkpoints
    # --------------------------------------------------

    if PARTIAL_EMBEDDINGS_PATH.exists():
        PARTIAL_EMBEDDINGS_PATH.unlink()

    if PARTIAL_KEYS_PATH.exists():
        PARTIAL_KEYS_PATH.unlink()

    print(
        "\nArquivos parciais removidos."
    )

    print(
        "\nConcluído."
    )


if __name__ == "__main__":
    main()