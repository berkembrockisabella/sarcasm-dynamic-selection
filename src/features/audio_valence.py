import os
import pandas as pd
from funasr import AutoModel

df = pd.read_csv("data/processed/mustard_prepared.csv")
audio_dir = "data/processed/audio"
output_path = "data/processed/audio_valence.csv"

# emotion2vec+ (Ma et al., ACL 2024 Findings) -- reconhece 9 categorias,
classificador = AutoModel(model="iic/emotion2vec_plus_large", hub="hf")

POSITIVAS = {"happy"}
NEGATIVAS = {"angry", "disgusted", "fearful", "sad"}

# Traducao dos rotulos nativos para o vocabulario padrao do projeto --
# usada so na hora de montar a linha salva no csv
MAPA_ROTULOS = {
    "disgusted": "disgust",
    "fearful": "fear",
    "surprised": "surprise",
    "<unk>": "unknown",
}

if os.path.exists(output_path):
    keys_processadas = set(pd.read_csv(output_path)["KEY"].astype(str))
    print("Ja processadas:", len(keys_processadas))
else:
    keys_processadas = set()

novos_resultados = []
total_processadas = len(keys_processadas)
total_sem_audio = 0
rotulos_impressos = False

for _, row in df.iterrows():
    key = str(row["KEY"])
    if key in keys_processadas:
        continue
    audio_path = os.path.join(audio_dir, f"{key}.wav")
    if not os.path.exists(audio_path):
        print(f"Audio nao encontrado: {key}")
        total_sem_audio += 1
        continue
    try:
        saida = classificador.generate(
            audio_path, granularity="utterance", extract_embedding=False
        )[0]
        # o modelo devolve rotulos bilingues (ex.: "开心/happy"); ficamos
        # so com a parte em ingles -- "<unk>" nao tem "/" e fica como esta
        rotulos_nativos = [
            rotulo.split("/")[-1] if "/" in rotulo else rotulo
            for rotulo in saida["labels"]
        ]
        probs_por_rotulo = dict(zip(rotulos_nativos, saida["scores"]))

        if not rotulos_impressos:
            print("Rotulos do classificador:", list(probs_por_rotulo.keys()))
            rotulos_impressos = True

        emocao_predominante = max(probs_por_rotulo, key=probs_por_rotulo.get)
        p_positiva = sum(probs_por_rotulo.get(c, 0.0) for c in POSITIVAS)
        p_negativa = sum(probs_por_rotulo.get(c, 0.0) for c in NEGATIVAS)

        linha = {"KEY": key}
        linha.update({
            MAPA_ROTULOS.get(rotulo, rotulo): valor
            for rotulo, valor in probs_por_rotulo.items()
        })
        linha["emocao_predominante"] = MAPA_ROTULOS.get(emocao_predominante, emocao_predominante)
        linha["audio_valence"] = p_positiva - p_negativa
        novos_resultados.append(linha)
        total_processadas += 1
    except Exception as erro:
        print(f"Erro ao processar {key}: {erro}")
    if len(novos_resultados) >= 100:
        pd.DataFrame(novos_resultados).to_csv(
            output_path,
            mode="a" if os.path.exists(output_path) else "w",
            header=not os.path.exists(output_path),
            index=False,
        )
        novos_resultados = []
        print("Progresso salvo. Total processado:", total_processadas)

if len(novos_resultados) > 0:
    pd.DataFrame(novos_resultados).to_csv(
        output_path,
        mode="a" if os.path.exists(output_path) else "w",
        header=not os.path.exists(output_path),
        index=False,
    )

resultado_final = pd.read_csv(output_path)
print("\nValencia acustica calculada para", len(resultado_final), "instancias.")
print("Instancias sem audio:", total_sem_audio)
print("\nDistribuicao da emocao predominante:")
print(resultado_final["emocao_predominante"].value_counts())
print("\nEstatisticas de audio_valence:")
print(resultado_final["audio_valence"].describe())
print("Salvo em:", output_path)