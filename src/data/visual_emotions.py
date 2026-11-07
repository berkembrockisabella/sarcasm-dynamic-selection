import os
import numpy as np
import pandas as pd
import torch

from PIL import Image

# Compatibilidade com bibliotecas que ainda usam aliases antigos do NumPy
if "long" not in np.__dict__:
    np.long = np.int64

if "ulong" not in np.__dict__:
    np.ulong = np.uint64

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)


# Modelo ViT de reconhecimento de expressão facial
MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

# Pasta com as faces recortadas pelo MTCNN
faces_dir = "data/processed/faces_mtcnn"

# Dataset preparado
dataset_path = "data/processed/mustard_prepared.csv"

# Arquivo onde os resultados serão salvos
output_path = "data/processed/visual_emotions.csv"


# Define CPU ou GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo:", device)


# Carrega o processador de imagens
processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

# Carrega o modelo ViT de reconhecimento de emoção
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()


# Lê o dataset preparado
df = pd.read_csv(dataset_path)


# Verifica se já existe um arquivo de resultados
if os.path.exists(output_path):

    resultados_df = pd.read_csv(output_path)

    # Cria conjunto com faces já processadas
    faces_processadas = set(
        zip(
            resultados_df["KEY"].astype(str),
            resultados_df["face"].astype(str)
        )
    )

    print(
        "Faces já processadas:",
        len(faces_processadas)
    )

else:

    resultados_df = pd.DataFrame()

    faces_processadas = set()


# Lista temporária para novos resultados
novos_resultados = []

# Contadores
total_processadas = len(faces_processadas)
total_sem_face = 0


# Percorre todas as instâncias do dataset
for indice, row in df.iterrows():

    key = str(row["KEY"])

    pasta_faces = os.path.join(
        faces_dir,
        key
    )

    # Caso a pasta da instância não exista
    if not os.path.exists(pasta_faces):

        print(f"Pasta não encontrada: {key}")

        total_sem_face += 1

        continue


    # Lista as faces da instância
    faces = sorted([
        arquivo
        for arquivo in os.listdir(pasta_faces)
        if arquivo.lower().endswith(".jpg")
    ])


    # Caso nenhuma face tenha sido encontrada
    if len(faces) == 0:

        print(f"Nenhuma face disponível para: {key}")

        total_sem_face += 1

        continue


    # Processa cada face da instância
    for face_name in faces:

        identificador = (
            key,
            face_name
        )

        # Se já foi processada, pula
        if identificador in faces_processadas:
            continue


        face_path = os.path.join(
            pasta_faces,
            face_name
        )


        try:

            # Abre a imagem em RGB
            image = Image.open(
                face_path
            ).convert("RGB")


            # Pré-processamento necessário para o ViT
            inputs = processor(
                images=image,
                return_tensors="pt"
            )


            # Move os dados para CPU ou GPU
            inputs = {
                chave: valor.to(device)
                for chave, valor in inputs.items()
            }


            # Inferência
            with torch.no_grad():

                outputs = model(
                    **inputs
                )


            # Converte logits em probabilidades
            probabilidades = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]


            # Dicionário das emoções
            emocoes = {}

            for i, prob in enumerate(probabilidades):

                label = model.config.id2label[i].lower()

                emocoes[label] = prob.item()


            # Emoção com maior probabilidade
            emocao_predita = max(
                emocoes,
                key=emocoes.get
            )


            # Guarda o resultado
            resultado = {

                "KEY": key,

                "face": face_name,

                "angry": emocoes.get(
                    "angry",
                    0.0
                ),

                "disgust": emocoes.get(
                    "disgust",
                    0.0
                ),

                "fear": emocoes.get(
                    "fear",
                    0.0
                ),

                "happy": emocoes.get(
                    "happy",
                    0.0
                ),

                "neutral": emocoes.get(
                    "neutral",
                    0.0
                ),

                "sad": emocoes.get(
                    "sad",
                    0.0
                ),

                "surprise": emocoes.get(
                    "surprise",
                    0.0
                ),

                "predicted_emotion": emocao_predita
            }


            novos_resultados.append(
                resultado
            )


            faces_processadas.add(
                identificador
            )


            total_processadas += 1


        except Exception as erro:

            print(
                f"Erro ao processar {key}/{face_name}: {erro}"
            )


        # Salva a cada 100 novas faces processadas
        if len(novos_resultados) >= 100:

            novos_df = pd.DataFrame(
                novos_resultados
            )

            # Se o CSV ainda não existe, cria
            if not os.path.exists(output_path):

                novos_df.to_csv(
                    output_path,
                    index=False
                )

            else:

                # Adiciona novas linhas ao CSV existente
                novos_df.to_csv(
                    output_path,
                    mode="a",
                    header=False,
                    index=False
                )

            novos_resultados = []

            print(
                f"Faces processadas até agora: {total_processadas}"
            )


# Salva os resultados restantes
if len(novos_resultados) > 0:

    novos_df = pd.DataFrame(
        novos_resultados
    )

    if not os.path.exists(output_path):

        novos_df.to_csv(
            output_path,
            index=False
        )

    else:

        novos_df.to_csv(
            output_path,
            mode="a",
            header=False,
            index=False
        )


print("\nProcessamento finalizado.")

print(
    "Total de faces processadas:",
    total_processadas
)

print(
    "Instâncias sem faces disponíveis:",
    total_sem_face
)

print(
    "Resultado salvo em:",
    output_path
)