from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/mustard++_text.csv")
OUTPUT_PATH = Path("data/processed/context_text.csv")


def main():

    print("Carregando MUStARD++...")

    df = pd.read_csv(INPUT_PATH)

    contexts = []

    for scene, group in df.groupby("SCENE", sort=False):

        # Linhas de contexto
        context_rows = group[
            group["KEY"].astype(str).str.contains("_c_")
        ].copy()

        # Fala principal
        utterance_rows = group[
            group["Sarcasm"].notna()
        ].copy()

        if utterance_rows.empty:
            continue

        # Em princípio há uma fala principal por cena
        utterance = utterance_rows.iloc[0]

        key = utterance["KEY"]

        # Correção conhecida do dataset
        if key == "Disgust" and scene == "1_S11E03_067":
            key = "1_S11E03_067_u"

        # Mantém ordem c_00, c_01, c_02...
        context_rows = context_rows.sort_values("KEY")

        context_parts = []

        for _, row in context_rows.iterrows():

            speaker = str(row["SPEAKER"]).strip()
            sentence = str(row["SENTENCE"]).strip()

            # Remove quebras de linha
            sentence = " ".join(sentence.split())

            context_parts.append(
                f"{speaker}: {sentence}"
            )

        context_text = " ".join(context_parts)

        contexts.append(
            {
                "KEY": key,
                "SCENE": scene,
                "CONTEXT": context_text,
                "NUM_CONTEXT_UTTERANCES": len(context_rows),
            }
        )

    result = pd.DataFrame(contexts)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nConcluído.")

    print(
        f"Instâncias geradas: {len(result)}"
    )

    print(
        f"Sem contexto: "
        f"{(result['NUM_CONTEXT_UTTERANCES'] == 0).sum()}"
    )

    print(
        f"\nSalvo em: {OUTPUT_PATH}"
    )

    print("\nExemplo:\n")

    print(
        result.iloc[0].to_string()
    )


if __name__ == "__main__":
    main()