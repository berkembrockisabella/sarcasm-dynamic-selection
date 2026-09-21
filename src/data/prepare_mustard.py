import os
import pandas as pd

# Caminho do CSV do MUStARD++
csv_path = "data/raw/mustard++_text.csv"

# Caminho dos vídeos das falas principais
video_dir = "data/raw/all_videos-MUStarARD/all_videos/final_utterance_videos"

# Lê o CSV
df = pd.read_csv(csv_path)

# Mantém apenas as 1202 instâncias principais
df_principal = df[df["Sarcasm"].notna()].copy()

# Corrige uma linha desalinhada no CSV original
df_principal.loc[
    df_principal["SCENE"] == "1_S11E03_067",
    "KEY"
] = "1_S11E03_067_u"

# Cria o caminho do vídeo correspondente a cada instância
df_principal["video_path"] = df_principal["KEY"].apply(
    lambda key: os.path.join(video_dir, f"{key}.mp4")
)

# Verifica se todos os vídeos existem
df_principal["video_exists"] = df_principal["video_path"].apply(os.path.exists)

# Informações básicas do dataset
print("Total de linhas no CSV:", len(df))
print("Instâncias principais:", len(df_principal))

print("\nQuantidade por classe:")
print(df_principal["Sarcasm"].value_counts())

print("\nQuantidade de falantes:")
print(df_principal["SPEAKER"].nunique())

print("\nQuantidade de vídeos encontrados:")
print(df_principal["video_exists"].sum(), "/", len(df_principal))

# Verifica se existe alguma instância sem vídeo
faltando_video = df_principal[~df_principal["video_exists"]]

print("\nQuantidade sem vídeo:", len(faltando_video))

if len(faltando_video) > 0:
    print(faltando_video[["SCENE", "KEY", "SENTENCE"]])

# Salva o dataset preparado
os.makedirs("data/processed", exist_ok=True)

df_principal.to_csv(
    "data/processed/mustard_prepared.csv",
    index=False
)

print("\nDataset preparado salvo em:")
print("data/processed/mustard_prepared.csv")