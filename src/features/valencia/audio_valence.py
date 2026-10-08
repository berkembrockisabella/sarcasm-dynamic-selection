import os
import pandas as pd
from funasr import AutoModel

df = pd.read_csv("data/processed/mustard_prepared.csv")
audio_dir = "data/processed/audio"
output_path = "data/processed/audio_valence.csv"

# emotion2vec+ (Ma et al., ACL 2024 Findings) -- reconhece 9 categorias
classificador = AutoModel(model="iic/emotion2vec_plus_large", hub="hf")

# Traducao dos rotulos nativos para o vocabulario padrao do projeto --
# aplicada logo apos a chamada do classificador, igual aos outros scripts
MAPA_ROTULOS = {
    "disgusted": "disgust",
    "fearful": "fear",
    "surprised": "surprise",
    "<unk>": "unknown",
}

EMOCOES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
POSITIVAS = {"happy"}
NEGATIVAS = {"angry", "disgust", "fear", "sad"}

# "other" e "unknown" nao sao emocoes do vocabulario padrao -- a massa
# delas e retirada e as 7 emocoes restantes sao renormalizadas para somar 1,
# deixando o audio na mesma escala das outras modalidades
DESCARTADAS = {"other", "unknown"}

# Se a maior parte da massa foi para other/unknown, renormalizar amplifica
# ruido (ex.: 0.07 de valencia sobre 8% de massa valida vira 0.83) -- nesse
# caso a instancia e tratada como sem emocao valida, igual entrada vazia
MINIMO_MASSA_VALIDA = 0.5

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

        probs = {
            MAPA_ROTULOS.get(rotulo, rotulo): valor
            for rotulo, valor in probs_por_rotulo.items()
        }
        p_descartada = sum(probs.get(c, 0.0) for c in DESCARTADAS)
        p_valida = 1.0 - p_descartada

        if p_valida < MINIMO_MASSA_VALIDA:
            probs_validas = {emocao: 0.0 for emocao in EMOCOES}
            emocao_predominante = "vazio"
            valencia = 0.0
        else:
            probs_validas = {
                emocao: probs.get(emocao, 0.0) / p_valida for emocao in EMOCOES
            }
            emocao_predominante = max(probs_validas, key=probs_validas.get)
            p_positiva = sum(probs_validas[c] for c in POSITIVAS)
            p_negativa = sum(probs_validas[c] for c in NEGATIVAS)
            valencia = p_positiva - p_negativa

        linha = {"KEY": key}
        linha.update(probs_validas)
        linha["p_descartada"] = p_descartada
        linha["emocao_predominante"] = emocao_predominante
        linha["audio_valence"] = valencia
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
print("Instancias sem emocao valida (other/unknown >= 50%):",
      (resultado_final["emocao_predominante"] == "vazio").sum())
print("\nDistribuicao da emocao predominante:")
print(resultado_final["emocao_predominante"].value_counts())
print("\nEstatisticas de audio_valence:")
print(resultado_final["audio_valence"].describe())
print("Salvo em:", output_path)