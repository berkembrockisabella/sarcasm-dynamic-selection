# Preparação dos Dados (`src/data/`)

Esta pasta contém os scripts responsáveis pela preparação dos dados do MUStARD++ utilizados nos experimentos.

O processamento inclui a preparação do dataset, extração de áudio, extração de frames, detecção de faces, reconhecimento de expressões faciais e geração dos splits utilizados nos experimentos.

## Fluxo de processamento

```text
prepare_mustard.py
├── extract_audio.py
└── extract_frames.py
    └── crop_faces_mtcnn.py
        └── visual_emotions.py

create_splits_repetidos.py
```

Para a modalidade visual, o fluxo é:

```text
vídeo
↓
frames
↓
detecção e recorte das faces
↓
reconhecimento de expressões faciais
↓
visual_emotions.csv
```

## Scripts

### `prepare_mustard.py`

Prepara os dados originais do MUStARD++ para as etapas seguintes.

O script:

- carrega `data/raw/mustard++_text.csv`;
- mantém as 1202 instâncias principais do dataset;
- corrige a chave da cena `1_S11E03_067`;
- associa cada instância ao vídeo correspondente;
- verifica a existência dos arquivos de vídeo;
- gera:

```text
data/processed/mustard_prepared.csv
```

---

### `extract_audio.py`

Extrai o áudio correspondente ao enunciado-alvo de cada instância.

O áudio é extraído dos vídeos utilizando FFmpeg, convertido para mono e reamostrado para 16 kHz.

Saída:

```text
data/processed/audio/<KEY>.wav
```

Os arquivos gerados são posteriormente utilizados na extração da representação acústica e no cálculo da valência de áudio.

---

### `extract_frames.py`

Extrai 10 frames distribuídos ao longo do vídeo correspondente a cada instância.

Saída:

```text
data/processed/frames/<KEY>/
```

Os frames são utilizados posteriormente para a detecção das faces.

---

### `crop_faces_mtcnn.py`

Realiza a detecção e o recorte das faces presentes nos frames utilizando MTCNN.

Quando mais de uma face é encontrada, é selecionada a detecção com maior confiança. A face selecionada é redimensionada para `224x224`.

Saída:

```text
data/processed/faces_mtcnn/<KEY>/
```

Frames em que nenhuma face válida é detectada são ignorados.

---

### `visual_emotions.py`

Realiza o reconhecimento de expressões faciais a partir das faces extraídas pelo MTCNN.

É utilizado o modelo pré-treinado:

```text
mo-thecreator/vit-Facial-Expression-Recognition
```

O modelo é baseado em Vision Transformer (ViT) e retorna probabilidades para sete categorias:

- `angry`
- `disgust`
- `fear`
- `happy`
- `neutral`
- `sad`
- `surprise`

Também é armazenada a emoção com maior probabilidade para cada face.

Saída:

```text
data/processed/visual_emotions.csv
```

Esse arquivo é utilizado posteriormente para a construção da representação visual e para o cálculo da valência visual.

---

### `create_splits_repetidos.py`

Gera os splits por falante utilizados nos experimentos.

As divisões mantêm os falantes separados entre os conjuntos, permitindo avaliar a generalização dos modelos para falantes não observados durante o treinamento.

Os mesmos splits são utilizados pelos diferentes classificadores e estratégias avaliadas, garantindo comparabilidade entre os experimentos.

## Arquivos gerados

Os principais arquivos intermediários produzidos por esta etapa são:

```text
data/processed/
├── mustard_prepared.csv
├── audio/
├── frames/
├── faces_mtcnn/
└── visual_emotions.csv
```

Os arquivos brutos e os arquivos multimídia intermediários de grande volume não são versionados no repositório.
