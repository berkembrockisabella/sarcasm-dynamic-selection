import os
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN

# Lê o dataset preparado
df = pd.read_csv("data/processed/mustard_prepared.csv")

# Pasta onde estão os frames
frames_dir = "data/processed/frames"

# Pasta onde as faces recortadas serão salvas
faces_dir = "data/processed/faces_mtcnn"

# Cria a pasta de saída caso ela não exista
os.makedirs(faces_dir, exist_ok=True)

# Cria o detector de faces MTCNN
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=True,
    post_process=False
)

# Contadores
total_instancias = 0
total_frames = 0
total_faces = 0
frames_sem_face = 0

# Percorre todas as instâncias do dataset
for _, row in df.iterrows():

    key = row["KEY"]

    # Pasta com os frames da instância
    input_dir = os.path.join(frames_dir, key)

    # Pasta onde as faces da instância serão salvas
    output_dir = os.path.join(faces_dir, key)

    os.makedirs(output_dir, exist_ok=True)

    # Verifica se a pasta de frames existe
    if not os.path.exists(input_dir):
        print(f"Pasta de frames não encontrada: {key}")
        continue

    # Lista os frames da instância
    frames = sorted([
        arquivo
        for arquivo in os.listdir(input_dir)
        if arquivo.lower().endswith(".jpg")
    ])

    for frame_name in frames:

        frame_path = os.path.join(
            input_dir,
            frame_name
        )

        # Nome da face correspondente ao frame
        output_path = os.path.join(
            output_dir,
            frame_name.replace("frame", "face")
        )

        # Se a face já foi processada, não processa novamente
        if os.path.exists(output_path):
            continue

        # Abre o frame
        image = Image.open(frame_path).convert("RGB")

        # Detecta as faces
        boxes, probs = mtcnn.detect(image)

        total_frames += 1

        # Caso nenhuma face seja encontrada
        if boxes is None or probs is None:
            frames_sem_face += 1
            continue

        # Remove detecções sem probabilidade válida
        deteccoes_validas = [
            (box, prob)
            for box, prob in zip(boxes, probs)
            if prob is not None
        ]

        if len(deteccoes_validas) == 0:
            frames_sem_face += 1
            continue

        # Escolhe a face com maior confiança
        box, prob = max(
            deteccoes_validas,
            key=lambda item: item[1]
        )

        x1, y1, x2, y2 = box.astype(int)

        # Garante que o recorte fique dentro dos limites da imagem
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, image.width)
        y2 = min(y2, image.height)

        # Evita recortes inválidos
        if x2 <= x1 or y2 <= y1:
            frames_sem_face += 1
            continue

        # Recorta a face
        face = image.crop(
            (x1, y1, x2, y2)
        )

        # Redimensiona para 224x224
        face = face.resize(
            (224, 224)
        )

        # Salva a face
        face.save(output_path)

        total_faces += 1

    total_instancias += 1


print("\nProcessamento finalizado.")

print("Instâncias processadas:", total_instancias)
print("Frames analisados:", total_frames)
print("Faces detectadas:", total_faces)
print("Frames sem face:", frames_sem_face)


# Verifica a quantidade de faces por instância
instancias_sem_face = []
instancias_poucas_faces = []

for key in df["KEY"]:

    pasta = os.path.join(
        faces_dir,
        key
    )

    if not os.path.exists(pasta):
        instancias_sem_face.append(key)
        continue

    faces = [
        arquivo
        for arquivo in os.listdir(pasta)
        if arquivo.lower().endswith(".jpg")
    ]

    if len(faces) == 0:
        instancias_sem_face.append(key)

    elif len(faces) < 3:
        instancias_poucas_faces.append(
            (key, len(faces))
        )


print("\nInstâncias sem nenhuma face:", len(instancias_sem_face))
print(
    "Instâncias com menos de 3 faces:",
    len(instancias_poucas_faces)
)

print("\nExemplos sem face:")
print(instancias_sem_face[:20])

print("\nExemplos com poucas faces:")
print(instancias_poucas_faces[:20])