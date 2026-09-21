import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Lê o dataset
df = pd.read_csv("data/raw/mustard++_text.csv")

# Mantém apenas as 1202 instâncias principais
df = df[df["Sarcasm"].notna()].copy()

melhor_split = None
melhor_diferenca = float("inf")

# Testa diferentes divisões
for seed in range(1000):

    # 70% treino e 30% restante
    split_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed
    )

    train_idx, temp_idx = next(
        split_1.split(df, groups=df["SPEAKER"])
    )

    train_temp = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    # Divide os 30% restantes entre DSEL e Test
    split_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed
    )

    dsel_idx, test_idx = next(
        split_2.split(temp, groups=temp["SPEAKER"])
    )

    dsel_temp = temp.iloc[dsel_idx].copy()
    test_temp = temp.iloc[test_idx].copy()

    # Calcula a proporção de sarcasmo de cada conjunto
    prop_train = train_temp["Sarcasm"].mean()
    prop_dsel = dsel_temp["Sarcasm"].mean()
    prop_test = test_temp["Sarcasm"].mean()

    # Mede o quanto cada conjunto se afasta de 50/50
    diferenca = (
        abs(prop_train - 0.5)
        + abs(prop_dsel - 0.5)
        + abs(prop_test - 0.5)
    )

    # Guarda a melhor divisão encontrada
    if diferenca < melhor_diferenca:
        melhor_diferenca = diferenca
        melhor_split = (
            seed,
            train_temp,
            dsel_temp,
            test_temp
        )

# Recupera a melhor divisão
seed, train, dsel, test = melhor_split

print("Seed escolhida:", seed)

print("\nTrain:", len(train))
print(train["Sarcasm"].value_counts())

print("\nDSEL:", len(dsel))
print(dsel["Sarcasm"].value_counts())

print("\nTest:", len(test))
print(test["Sarcasm"].value_counts())

# Salva os conjuntos em arquivos CSV
train.to_csv("data/splits/train.csv", index=False)
dsel.to_csv("data/splits/dsel.csv", index=False)
test.to_csv("data/splits/test.csv", index=False)

print("\nSplits salvos em data/splits/")