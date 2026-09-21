import os
import pandas as pd
from speechbrain.inference.interfaces import foreign_class

df = pd.read_csv("data/processed/mustard_prepared.csv")
audio_dir = "data/processed/audio"
output_path = "data/processed/audio_valence.csv"

classificador = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
)
indice_para_rotulo = classificador.hparams.label_encoder.ind2lab
print("Rotulos do classificador:", indice_para_rotulo)

POSITIVAS = {"hap"}
NEGATIVAS = {"ang", "sad"}

# Traducao dos rotulos nativos para o vocabulario padrao do projeto --
# usada so na hora de montar a linha salva no csv
MAPA_ROTULOS = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}

if os.path.exists(output_path):
    keys_processadas = set(pd.read_csv(output_path)["KEY"].astype(str))
    print("Ja processadas:", len(keys_processadas))
else:
    keys_processadas = set()

novos_resultados = []
total_processadas = len(keys_processadas)
total_sem_audio = 0

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
        out_prob, score, index, text_lab = classificador.classify_file(audio_path)
        probs = out_prob.squeeze().tolist()
        probs_por_rotulo = {indice_para_rotulo[i]: probs[i] for i in range(len(probs))}
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