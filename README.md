# Seleção Dinâmica de Classificadores para Detecção Multimodal de Sarcasmo

Projeto da disciplina Experiência Criativa: Projeto Transformador II (PUCPR). O objetivo é detectar sarcasmo em enunciados multimodais (texto, áudio, expressão facial e contexto conversacional) usando um vetor de incongruência de valência entre as modalidades como base para seleção dinâmica de classificadores.

Dataset: [MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus), 1202 instâncias rotuladas para sarcasmo, com vídeo, áudio, transcrição e contexto conversacional anterior a cada fala.

## Status atual

O que já está implementado e rodando:

- Preparação do dataset e criação dos splits (treino/DSEL/teste).
- Extração de áudio dos vídeos.
- Extração de emoções faciais por frame (resultado presente em `data/processed/visual_emotions.csv`).
- Cálculo de valência para as quatro modalidades: texto, contexto, áudio e expressão facial.

O que ainda falta:

- `src/incongruence/build_incongruence.py` — construir o vetor de incongruência (|vT−vA|, |vT−vV|, |vA−vV|) a partir das valências calculadas. Arquivo ainda vazio.
- Pool de classificadores, estratégia de seleção dinâmica (DS) e comparação com Mixture of Experts (MoE) — etapas futuras do projeto.

**Atenção:** os scripts que geram `extract_frames.py`, `crop_faces.py`/`crop_faces_mtcnn.py` e `visual_emotions.py` (extração de frames, recorte de rostos e classificação de emoção facial) não estão presentes nesta cópia do repositório, apenas o resultado final deles (`data/processed/visual_emotions.csv`). Se for necessário reprocessar a parte visual do zero, esses scripts precisam ser recuperados ou reescritos.

Também existe um arquivo `data/processed/audio_egemaps.csv` que é resíduo de uma abordagem anterior (extração de features acústicas eGeMAPS + regressão Ridge) que foi abandonada em favor da classificação categórica de emoções com SpeechBrain. Pode ser removido.

## Estrutura do repositório

```
data/
  raw/                          dados originais do MUStARD++ (CSV, vídeos, anotações)
  processed/                    dados gerados pelo pipeline (áudio extraído, valências, etc.)
  splits/                       divisão em train.csv, dsel.csv, test.csv

src/
  data/
    prepare_mustard.py          le o CSV bruto, filtra as 1202 instancias principais, liga cada uma ao seu video
    create_splits.py            divide os dados em treino/DSEL/teste por falante, balanceando a proporcao de sarcasmo
    extract_audio.py            extrai o audio (.wav, mono, 16kHz) de cada video com ffmpeg
  features/
    text_valence.py             calcula a valencia do texto da fala
    context_valence.py          calcula a valencia do contexto conversacional anterior a fala
    audio_valence.py            calcula a valencia do audio da fala
    visual_valence.py           calcula a valencia da expressao facial a partir das emocoes ja extraidas
  incongruence/
    build_incongruence.py       (a implementar) vetor de incongruencia entre as valencias

notebooks/                      notebooks de exploracao (se houver)
requirements.txt                dependencias Python do projeto
```

## Pipeline, na ordem

1. `python src/data/prepare_mustard.py` — le `data/raw/mustard++_text.csv`, mantém as 1202 instâncias principais (as que têm rótulo de sarcasmo) e associa cada uma ao caminho do seu vídeo. Gera `data/processed/mustard_prepared.csv`, usado por todos os scripts seguintes.
2. `python src/data/create_splits.py` — divide as instâncias em treino (70%), DSEL (15%) e teste (15%) por falante (`GroupShuffleSplit`), testando 1000 sementes e escolhendo a que deixa a proporção de sarcasmo mais próxima de 50/50 em cada conjunto. Gera os arquivos em `data/splits/`.
3. `python src/data/extract_audio.py` — extrai o áudio de cada vídeo com FFmpeg (mono, 16kHz) para `data/processed/audio/<KEY>.wav`.
4. Extração de frames, recorte de rostos e classificação de emoção facial por frame — **scripts ausentes nesta cópia**, mas o resultado (`data/processed/visual_emotions.csv`) já existe.
5. Cálculo de valência por modalidade (`src/features/*.py`, detalhado abaixo).
6. `src/incongruence/build_incongruence.py` — a implementar.

## Cálculo de valência

Valência é um número entre -1 (muito negativo) e +1 (muito positivo) que representa o tom emocional de um sinal. O projeto calcula uma valência separada para cada uma das quatro modalidades, e todas seguem a mesma lógica:

1. Um classificador de emoções categóricas já treinado (não treinado por nós) recebe o sinal (texto, contexto, áudio ou rosto) e devolve a probabilidade de cada emoção básica.
2. A valência é a soma das probabilidades das emoções positivas menos a soma das probabilidades das emoções negativas:

```
valencia = P(emocoes positivas) - P(emocoes negativas)
```

Emoções neutras/surpresa não entram na conta (contribuem 0). Como as probabilidades de cada lado somam no máximo 1, o resultado fica sempre entre -1 e 1, sem precisar de nenhuma normalização adicional ou peso arbitrário.

Cada classificador usa sua própria nomenclatura nativa de rótulos (o de texto fala em "joy"/"sadness", o de áudio em "hap"/"sad", etc.). Para que os quatro CSVs e os scripts falem a mesma língua, todos os rótulos são traduzidos para um vocabulário único de sete emoções — o mesmo já usado em `visual_emotions.csv`: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`. Essa tradução é feita por um dicionário (`MAPA_ROTULOS`) logo após a chamada do classificador em cada script; o cálculo de valência (positivas − negativas) já usa os nomes padronizados. O classificador de áudio só reconhece 4 das 7 emoções (não distingue disgust/fear/surprise), então `audio_valence.csv` não tem essas três colunas — isso é uma limitação do classificador, não é preenchido com zero para não sugerir uma certeza que o modelo não tem.

| Modalidade | Script | Classificador usado | Positivas | Negativas |
|---|---|---|---|---|
| Texto | `src/features/text_valence.py` | `j-hartmann/emotion-english-distilroberta-base` (HuggingFace) | happy | angry, disgust, fear, sad |
| Contexto | `src/features/context_valence.py` | `j-hartmann/emotion-english-distilroberta-base` (HuggingFace) | happy | angry, disgust, fear, sad |
| Áudio | `src/features/audio_valence.py` | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` (SpeechBrain) | happy | angry, sad |
| Expressão facial | `src/features/visual_valence.py` | emoções já extraídas em `data/processed/visual_emotions.csv` | happy | angry, disgust, fear, sad |

Detalhes por script:

- **`text_valence.py`**: aplica o classificador em `SENTENCE` (a fala de cada instância) e traduz os rótulos (`anger`→`angry`, `joy`→`happy`, `sadness`→`sad`) para o vocabulário padrão. Gera `data/processed/text_valence.csv` com `KEY`, `SENTENCE`, uma coluna por emoção (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`), `emocao_predominante` e `text_valence`.
- **`context_valence.py`**: reconstrói o contexto conversacional de cada cena a partir de `data/raw/mustard++_text.csv` (as linhas cujo `KEY` contém `_c_` são falas de contexto; são agrupadas por `SCENE` e concatenadas em ordem). Aplica o mesmo classificador de texto (com a mesma tradução de rótulos) nesse contexto. Gera `data/processed/context_valence.csv` com `KEY`, `context`, as sete colunas de emoção, `emocao_predominante` e `context_valence`.
- **`audio_valence.py`**: aplica o classificador de emoção em fala do SpeechBrain (treinado no IEMOCAP) em cada arquivo `.wav`. O mapeamento índice→rótulo nativo das classes é obtido em tempo de execução (`classificador.hparams.label_encoder.ind2lab`), nunca fixado manualmente, para evitar inverter os rótulos por engano. O cálculo de valência (positiva − negativa) usa os rótulos nativos do classificador (`ang`, `hap`, `neu`, `sad`); a tradução para o vocabulário padrão (`angry`, `happy`, `neutral`, `sad`) acontece só na hora de montar a linha salva no CSV. O script é retomável: salva o progresso a cada 100 áudios processados em `data/processed/audio_valence.csv`, então pode ser interrompido e reexecutado sem reprocessar o que já foi feito.
- **`visual_valence.py`**: usa as emoções por rosto/frame já calculadas em `data/processed/visual_emotions.csv` (já no vocabulário padrão, nenhuma tradução necessária), aplica a fórmula positiva−negativa por linha e depois tira a média de cada emoção e da valência por `KEY` (uma instância pode ter vários rostos/frames). Gera `data/processed/visual_valence.csv` com `KEY`, `n_faces`, as sete colunas de emoção (médias), `emocao_predominante` e `visual_valence`.

## Como rodar

Pré-requisitos: Python 3.10+, FFmpeg instalado no sistema (não é um pacote Python).

```
pip install -r requirements.txt
```

Depois, rodar os scripts na ordem descrita em "Pipeline, na ordem". Os scripts de `src/features/` dependem apenas de `data/processed/mustard_prepared.csv` (e, no caso do áudio, de `data/processed/audio/`; no caso da expressão facial, de `data/processed/visual_emotions.csv`), então podem ser rodados em qualquer ordem entre si, uma vez que esses dados existam.

Os modelos de texto/contexto (`transformers`) e áudio (`speechbrain`) são baixados automaticamente do HuggingFace Hub na primeira execução e ficam em cache localmente.
