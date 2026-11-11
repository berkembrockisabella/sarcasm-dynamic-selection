# Extração de Features e Valência (`src/features/`)

Esta pasta contém os scripts responsáveis pela construção das representações das modalidades utilizadas pelos classificadores e pela extração dos valores de valência empregados no cálculo da incongruência multimodal.

A estrutura está dividida em duas etapas principais:

```text
src/features/
├── embeddings/
└── valencia/
```

- `embeddings/`: geração das representações de texto, contexto, áudio e visual utilizadas como entrada dos classificadores.
- `valencia/`: cálculo dos valores de valência associados às diferentes modalidades.

## Embeddings

A pasta `embeddings/` contém os scripts responsáveis por transformar cada modalidade em uma representação numérica adequada para os experimentos de classificação.

```text
embeddings/
├── build_context_text.py
├── build_visual_features.py
├── extract_audio_features.py
├── extract_context_features.py
└── extract_text_features.py
```

### `extract_text_features.py`

Extrai a representação da modalidade textual a partir do enunciado-alvo de cada instância.

O texto é processado utilizando o modelo pré-treinado BART e os estados ocultos são agregados para gerar um vetor semântico por instância.

---

### `build_context_text.py`

Constrói o contexto textual associado a cada instância a partir dos enunciados anteriores disponíveis na cena.

O resultado é utilizado posteriormente na extração das features de contexto.

---

### `extract_context_features.py`

Extrai a representação semântica do contexto conversacional.

O contexto é processado separadamente do enunciado-alvo, permitindo que texto e contexto sejam avaliados como modalidades independentes e também em diferentes combinações multimodais.

---

### `extract_audio_features.py`

Extrai a representação acústica dos arquivos de áudio preparados anteriormente em `src/data/`.

O áudio é processado pelo modelo pré-treinado Wav2Vec2 e suas representações temporais são agregadas para gerar um vetor acústico por instância.

---

### `build_visual_features.py`

Constrói a representação visual a partir das probabilidades de expressões faciais armazenadas em:

```text
data/processed/visual_emotions.csv
```

Para cada instância, são agregadas as probabilidades das sete categorias emocionais:

- `angry`
- `disgust`
- `fear`
- `happy`
- `neutral`
- `sad`
- `surprise`

A representação considera a média e o desvio-padrão temporal dessas probabilidades, além de informações relacionadas à disponibilidade das faces, resultando nas features utilizadas pelos classificadores.

## Valência

A pasta `valencia/` contém os scripts responsáveis pela obtenção dos valores de valência utilizados posteriormente na construção do vetor de incongruência multimodal.

```text
valencia/
├── audio_valence.py
├── context_valence.py
├── extract_face_emotions.py
├── extract_face_tracks.py
├── extratores_alternativos.py
├── text_valence.py
└── visual_valence.py
```

### `text_valence.py`

Calcula a valência associada ao enunciado textual de cada instância.

---

### `context_valence.py`

Calcula a valência correspondente ao contexto conversacional anterior ao enunciado-alvo.

---

### `audio_valence.py`

Calcula a valência da modalidade acústica a partir do áudio correspondente a cada instância.

---

### `visual_valence.py`

Calcula a valência da modalidade visual a partir das informações emocionais extraídas das faces presentes nos vídeos.

---

### `extract_face_tracks.py`

Detecta e acompanha os rostos (YuNet) e escolhe a trilha do personagem que fala (SFace). Usado só na ablação E6 (rosto do falante); a valência visual principal vem de `visual_valence.py`.

---

### `extract_face_emotions.py`

Estima a emoção do rosto do falante (`trpakov/vit-face-expression`), com média ou pico dos quadros. Saída em `data/processed/extratores/visual_falante_*.csv`, usada só na calibração (E6).

---

### `extratores_alternativos.py`

Gera as emoções de texto e contexto com `j-hartmann/emotion-english-distilroberta-base` e o contexto só com a fala anterior (DeBERTa e j-hartmann). Saída em `data/processed/extratores/`, usada só na calibração (E6).

## Fluxo geral

As features utilizadas pelos classificadores seguem o fluxo:

```text
dados preparados
↓
src/features/embeddings/
↓
representações de texto, contexto, áudio e visual
↓
classificadores
```

O cálculo da incongruência segue um fluxo independente:

```text
dados preparados
↓
src/features/valencia/
↓
valência por modalidade
↓
vetor de incongruência multimodal
↓
seleção dinâmica
```

Dessa forma, as representações utilizadas pelos classificadores permanecem separadas dos valores de valência utilizados para caracterizar a incongruência entre as modalidades.
