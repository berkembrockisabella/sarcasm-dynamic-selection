import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Gera os 10 splits por falante usados nas repeticoes do experimento final
# (protocolo E0). Mesma divisao de create_splits.py (70% treino, 15% DSEL,
# 15% teste, por falante). Os seeds sao percorridos em ordem (0, 1, 2, ...) e
# ficam os 10 primeiros que passam no criterio de aceitacao abaixo.
# Por que o criterio: o MUStARD++ tem so 27 falantes, muito desiguais
# (CHANDLER sozinho tem 157 falas, 76% sarcasticas), entao sortear sem
# controle gerava DSEL de 49 a 365 falas e teste com 27% a 60% de sarcasmo --
# a dispersao entre splits mediria quais falantes cairam onde, nao os metodos.
# O split unico original (data/splits/train.csv etc.) continua intocado.
N_SPLITS = 10
MIN_FALAS = 120
MAX_FALAS = 240
MIN_PROP = 0.40
MAX_PROP = 0.60
PASTA_SAIDA = "data/splits"

df = pd.read_csv("data/raw/mustard++_text.csv")

# Mantém apenas as 1202 instâncias principais
df = df[df["Sarcasm"].notna()].copy()

# Mesma correcao de create_splits.py -- linha desalinhada do CSV original
df.loc[df["SCENE"] == "1_S11E03_067", "KEY"] = "1_S11E03_067_u"

linhas = []
seed = -1
seeds_testados = 0

while len(linhas) < N_SPLITS:

    seed += 1
    seeds_testados += 1

    # 70% treino e 30% restante, agrupando por falante
    split_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(split_1.split(df, groups=df["SPEAKER"]))
    treino = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    # Divide os 30% restantes entre DSEL e teste, tambem por falante
    split_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    dsel_idx, test_idx = next(split_2.split(temp, groups=temp["SPEAKER"]))
    dsel = temp.iloc[dsel_idx].copy()
    teste = temp.iloc[test_idx].copy()

    # Criterio de aceitacao -- DSEL e teste com tamanho e proporcao de
    # sarcasmo razoaveis; o treino fica com o restante (~70%)
    aceito = True
    for conjunto in [dsel, teste]:
        if not (MIN_FALAS <= len(conjunto) <= MAX_FALAS):
            aceito = False
        if not (MIN_PROP <= conjunto["Sarcasm"].mean() <= MAX_PROP):
            aceito = False
    if not aceito:
        continue

    # Nenhum falante pode aparecer em dois conjuntos
    falantes_treino = set(treino["SPEAKER"])
    falantes_dsel = set(dsel["SPEAKER"])
    falantes_teste = set(teste["SPEAKER"])
    if falantes_treino & falantes_dsel or falantes_treino & falantes_teste or falantes_dsel & falantes_teste:
        raise ValueError(f"Seed {seed}: falante repetido entre conjuntos")

    nome_split = f"split_{len(linhas):02d}"
    pasta = os.path.join(PASTA_SAIDA, nome_split)
    os.makedirs(pasta, exist_ok=True)
    treino.to_csv(os.path.join(pasta, "train.csv"), index=False)
    dsel.to_csv(os.path.join(pasta, "dsel.csv"), index=False)
    teste.to_csv(os.path.join(pasta, "test.csv"), index=False)

    linhas.append({
        "split": nome_split,
        "seed": seed,
        "n_train": len(treino),
        "n_dsel": len(dsel),
        "n_test": len(teste),
        "prop_sarcasm_train": round(treino["Sarcasm"].mean(), 3),
        "prop_sarcasm_dsel": round(dsel["Sarcasm"].mean(), 3),
        "prop_sarcasm_test": round(teste["Sarcasm"].mean(), 3),
        "n_speakers_train": len(falantes_treino),
        "n_speakers_dsel": len(falantes_dsel),
        "n_speakers_test": len(falantes_teste),
    })

resumo = pd.DataFrame(linhas)
caminho_resumo = os.path.join(PASTA_SAIDA, "splits_resumo.csv")
resumo.to_csv(caminho_resumo, index=False)

print(f"{N_SPLITS} splits gerados ({seeds_testados} seeds testados).")
print(resumo.to_string(index=False))
print("\nTamanhos (min / max):")
print(resumo[["n_train", "n_dsel", "n_test"]].describe().loc[["min", "max"]])
print("\nSalvo em:", caminho_resumo)
