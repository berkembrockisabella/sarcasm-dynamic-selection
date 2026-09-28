from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# Caminhos
INPUT_PATH = Path("data/processed/context_text.csv")
OUTPUT_DIR = Path("data/processed/features")

EMBEDDINGS_PATH = OUTPUT_DIR / "context_embeddings.npy"
KEYS_PATH = OUTPUT_DIR / "context_keys.csv"


# Modelo
MODEL_NAME = "facebook/bart-base"

# Processamento
BATCH_SIZE = 16
MAX_LENGTH = 512


def mean_pooling(last_hidden_state, attention_mask):
    """
    Calcula a média apenas dos tokens válidos,
    ignorando padding.
    """
    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed = torch.sum(
        last_hidden_state * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


def main():

    print("Carregando contextos...")

    df = pd.read_csv(INPUT_PATH)

    df = df[["KEY", "CONTEXT"]].copy()

    df["CONTEXT"] = (
        df["CONTEXT"]
        .fillna("")
        .astype(str)
    )

    print(f"Total de instâncias: {len(df)}")

    # Cria pasta de saída
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Define dispositivo
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Dispositivo: {device}")

    # Carrega tokenizer
    print("Carregando tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # Carrega modelo
    print("Carregando BART...")

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    all_embeddings = []

    total = len(df)

    print("\nIniciando extração dos embeddings de contexto...\n")

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_texts = (
            df.iloc[start:end]["CONTEXT"]
            .tolist()
        )

        # Tokenização
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        # Move para CPU/GPU
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Inferência
        with torch.no_grad():

            outputs = model(**inputs)

            embeddings = mean_pooling(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
        )

        all_embeddings.append(
            embeddings
        )

        print(
            f"Processadas: {end}/{total}"
        )

    # Junta todos os batches
    all_embeddings = np.vstack(
        all_embeddings
    )

    print("\nExtração concluída.")

    print(
        f"Formato final dos embeddings: "
        f"{all_embeddings.shape}"
    )

    # Salva embeddings
    np.save(
        EMBEDDINGS_PATH,
        all_embeddings
    )

    # Salva KEYs na mesma ordem
    df[["KEY"]].to_csv(
        KEYS_PATH,
        index=False
    )

    print("\nArquivos salvos:")

    print(EMBEDDINGS_PATH)
    print(KEYS_PATH)

    print("\nConcluído.")


if __name__ == "__main__":
    main()