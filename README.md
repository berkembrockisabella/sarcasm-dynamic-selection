# Seleção Dinâmica de Classificadores para Detecção Multimodal de Sarcasmo

Projeto da disciplina Experiência Criativa: Projeto Transformador II (PUCPR). O objetivo é detectar sarcasmo em enunciados multimodais (texto, áudio, expressão facial e contexto conversacional) usando um vetor de incongruência de valência entre as modalidades como base para seleção dinâmica de classificadores.

Dataset: [MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus), 1202 instâncias rotuladas para sarcasmo, com vídeo, áudio, transcrição e contexto conversacional anterior a cada fala.

## Status atual

O que já está implementado e rodando:

- Preparação do dataset e criação dos splits (treino/DSEL/teste).
- Extração de áudio dos vídeos.
- Extração de emoções faciais por frame (resultado presente em `data/processed/visual_emotions.csv`).
- Cálculo de valência e intensidade emocional para as quatro modalidades: texto, contexto, áudio e expressão facial.
- Vetor de incongruência de valência entre os 6 pares de modalidades (`src/incongruence/build_incongruence.py`).

O que ainda falta:

- Pool de classificadores, estratégia de seleção dinâmica (DS) e comparação com Mixture of Experts (MoE) — etapas futuras do projeto.

**Atenção:** os scripts que geram `extract_frames.py`, `crop_faces.py`/`crop_faces_mtcnn.py` e `visual_emotions.py` (extração de frames, recorte de rostos e classificação de emoção facial) não estão presentes nesta cópia do repositório, apenas o resultado final deles (`data/processed/visual_emotions.csv`). Se for necessário reprocessar a parte visual do zero, esses scripts precisam ser recuperados ou reescritos.

Também existe uma pasta `wav2vec2_checkpoints/` (cache de modelo baixado, ~700 MB) que era usada pela abordagem anterior de áudio (SpeechBrain/IEMOCAP, 4 classes, substituída pelo `emotion2vec_plus_large`). Pode ser apagada; o `emotion2vec_plus_large` baixa seu próprio cache separadamente (via `funasr`/ModelScope), fora dessa pasta.

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
    build_incongruence.py       vetor de incongruencia entre as valencias (6 pares)

notebooks/                      notebooks de exploracao (se houver)
requirements.txt                dependencias Python do projeto
```

## Pipeline, na ordem

1. `python src/data/prepare_mustard.py` — le `data/raw/mustard++_text.csv`, mantém as 1202 instâncias principais (as que têm rótulo de sarcasmo) e associa cada uma ao caminho do seu vídeo. Gera `data/processed/mustard_prepared.csv`, usado por todos os scripts seguintes.
2. `python src/data/create_splits.py` — divide as instâncias em treino (70%), DSEL (15%) e teste (15%) por falante (`GroupShuffleSplit`), testando 1000 sementes e escolhendo a que deixa a proporção de sarcasmo mais próxima de 50/50 em cada conjunto. Gera os arquivos em `data/splits/`.
3. `python src/data/extract_audio.py` — extrai o áudio de cada vídeo com FFmpeg (mono, 16kHz) para `data/processed/audio/<KEY>.wav`.
4. Extração de frames, recorte de rostos e classificação de emoção facial por frame — **scripts ausentes nesta cópia**, mas o resultado (`data/processed/visual_emotions.csv`) já existe.
5. Cálculo de valência por modalidade (`src/features/*.py`, detalhado abaixo).
6. `python src/incongruence/build_incongruence.py` — vetor de incongruência (detalhado abaixo).

## Cálculo de valência

Valência é um número entre -1 (muito negativo) e +1 (muito positivo) que representa o tom emocional de um sinal. O projeto calcula uma valência separada para cada uma das quatro modalidades, e todas seguem a mesma lógica:

1. Um classificador de emoções categóricas já treinado (não treinado por nós) recebe o sinal (texto, contexto, áudio ou rosto) e devolve a probabilidade de cada emoção básica.
2. A valência é a soma das probabilidades das emoções positivas menos a soma das probabilidades das emoções negativas:

```
valencia = P(emocoes positivas) - P(emocoes negativas)
```

Emoções neutras/surpresa não entram na conta (contribuem 0). Como as probabilidades de cada lado somam no máximo 1, o resultado fica sempre entre -1 e 1, sem precisar de nenhuma normalização adicional ou peso arbitrário.

Cada script também salva a **intensidade** da modalidade (`<modalidade>_intensity`), que mede quanta emoção com polaridade o classificador viu:

```
intensidade = P(emocoes positivas) + P(emocoes negativas)        em [0, 1]
```

Ela distingue uma fala neutra (valência 0, intensidade perto de 0) de uma ambivalente (valência 0, intensidade perto de 1). Nos testes exploratórios (treino + DSEL), foi a feature que mais diferenciou as classes: as falas sarcásticas têm menos intensidade, principalmente no rosto e no contexto.

Cada classificador usa sua própria nomenclatura nativa de rótulos (o de texto fala em "joy"/"sadness", o de áudio em "disgusted"/"fearful", etc.). Para que os quatro CSVs e os scripts falem a mesma língua, todos os rótulos são traduzidos para um vocabulário único de sete emoções — o mesmo já usado em `visual_emotions.csv`: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`. Essa tradução é feita por um dicionário (`MAPA_ROTULOS`) logo após a chamada do classificador em cada script; o cálculo de valência (positivas − negativas) já usa os nomes padronizados.

| Modalidade | Script | Classificador usado | Positivas | Negativas |
|---|---|---|---|---|
| Texto | `src/features/text_valence.py` | `Tanneru/Emotion-Classification-DeBERTa-v3-Large` (HuggingFace) | happy | angry, disgust, fear, sad |
| Contexto | `src/features/context_valence.py` | `Tanneru/Emotion-Classification-DeBERTa-v3-Large` (HuggingFace) | happy | angry, disgust, fear, sad |
| Áudio | `src/features/audio_valence.py` | `emotion2vec_plus_large` (FunASR / Ma et al., ACL 2024) | happy | angry, disgust, fear, sad |
| Expressão facial | `src/features/visual_valence.py` | emoções já extraídas em `data/processed/visual_emotions.csv` | happy | angry, disgust, fear, sad |

Detalhes por script:

- **`text_valence.py`**: aplica o classificador (`Tanneru/Emotion-Classification-DeBERTa-v3-Large`, fine-tuned em DeBERTa-v3-large sobre GoEmotions + ISEAR + DAIR-AI mesclados/aumentados — substitui o `j-hartmann/emotion-english-distilroberta-base` anterior) em `SENTENCE` (a fala de cada instância) e traduz o único rótulo divergente (`anger`→`angry`) para o vocabulário padrão; os outros seis rótulos nativos do modelo (`disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) já batem exatamente. Gera `data/processed/text_valence.csv` com `KEY`, `SENTENCE`, uma coluna por emoção (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`), `emocao_predominante`, `text_valence` e `text_intensity`.

  **Atenção:** esse checkpoint é de um autor individual no HuggingFace (não uma instituição/grupo de pesquisa), com baixa adoção (~299 downloads/mês, 1 like no momento da troca) e sem validação acadêmica publicada que eu tenha encontrado — os 95,6% de acurácia/0,94 F1 reportados no model card não são de um benchmark independente. Vale considerar isso ao reportar a metodologia.
- **`context_valence.py`**: reconstrói o contexto conversacional de cada cena a partir de `data/raw/mustard++_text.csv` (as linhas cujo `KEY` contém `_c_` são falas de contexto; são agrupadas por `SCENE` e concatenadas em ordem). Aplica o mesmo classificador de texto (`Tanneru/Emotion-Classification-DeBERTa-v3-Large`, com a mesma tradução de rótulos) nesse contexto. Gera `data/processed/context_valence.csv` com `KEY`, `context`, as sete colunas de emoção, `emocao_predominante`, `context_valence` e `context_intensity`.
- **`audio_valence.py`**: aplica o `emotion2vec_plus_large` (modelo de representação de emoção em fala pré-treinado de forma auto-supervisionada, artigo ACL 2024 Findings, via biblioteca `funasr`) em cada arquivo `.wav`. Diferente do classificador anterior (SpeechBrain/IEMOCAP, só 4 classes), esse reconhece 9 categorias (`angry`, `disgusted`, `fearful`, `happy`, `neutral`, `other`, `sad`, `surprised`, `unknown`), então o lado negativo passa a incluir `disgust`/`fear` também para áudio — antes essas duas ficavam de fora só nessa modalidade por limitação do modelo. `other` e `unknown` não fazem parte do vocabulário padrão: a massa delas é retirada (guardada na coluna `p_descartada`) e as 7 emoções restantes são renormalizadas para somar 1, deixando o áudio na mesma escala das outras modalidades. Quando `other`+`unknown` ≥ 0,5, renormalizar amplificaria ruído (uma valência de 0,07 sobre 8% de massa válida viraria 0,83), então a instância é tratada como sem emoção válida (`emocao_predominante = vazio`, valência 0) — hoje são 39 instâncias. Os rótulos nativos do modelo são lidos dinamicamente da própria resposta (`saida["labels"]`/`saida["scores"]`), nunca fixados manualmente, pelo mesmo motivo de antes: evitar inverter rótulo por engano. O script é retomável: salva o progresso a cada 100 áudios processados em `data/processed/audio_valence.csv`, então pode ser interrompido e reexecutado sem reprocessar o que já foi feito. Gera `data/processed/audio_valence.csv` com `KEY`, as sete colunas de emoção (renormalizadas), `p_descartada`, `emocao_predominante`, `audio_valence` e `audio_intensity`. A versão anterior do CSV, com as 9 classes brutas antes da renormalização, está em `data/processed/audio_valence_9classes_backup.csv` (pode ser apagada).

  **Atenção:** há um [issue aberto no repositório do FunASR](https://github.com/modelscope/FunASR/issues/2728) relatando que, pro mesmo modelo `emotion2vec_plus_large`, o framework FunASR e o framework ModelScope retornam quantidades diferentes de classes (9 vs. 5) pro mesmo áudio, sem explicação da equipe mantenedora. O script imprime os rótulos recebidos na primeira instância processada (`print("Rotulos do classificador:", ...)`) — vale conferir essa saída antes de deixar o script rodar as 1202 instâncias, para confirmar que as 9 classes esperadas realmente vieram.
- **`visual_valence.py`**: usa as emoções por rosto/frame já calculadas em `data/processed/visual_emotions.csv` (já no vocabulário padrão, nenhuma tradução necessária), aplica a fórmula positiva−negativa (e a de intensidade) por linha e depois tira a média de cada emoção, da valência e da intensidade por `KEY` (uma instância pode ter vários rostos/frames). Gera `data/processed/visual_valence.csv` com `KEY`, `n_faces`, as sete colunas de emoção (médias), `emocao_predominante`, `visual_valence` e `visual_intensity`.

## Vetor de incongruência

`src/incongruence/build_incongruence.py` junta as quatro valências por `KEY` e calcula, para cada um dos 6 pares de modalidades (texto–contexto, texto–áudio, texto–visual, contexto–áudio, contexto–visual, áudio–visual):

```
inc_a_b     = (valencia_a - valencia_b) / 2      em [-1, 1]
abs_inc_a_b = |inc_a_b|                          em [0, 1]
```

A divisão por 2 só devolve a diferença (que vai de -2 a 2) para a mesma escala das valências. A versão com sinal guarda a direção do desacordo (ex.: texto positivo com contexto negativo, o padrão clássico de sarcasmo); a versão absoluta guarda só o tamanho do desacordo. A coluna `incongruencia_media` é a média das 6 incongruências absolutas, só para análise exploratória. As quatro intensidades são copiadas para o mesmo CSV, para poderem ser usadas junto com o vetor. Diferenças de intensidade entre modalidades, marcação de valores ausentes e desvio-padrão da valência facial foram testados e não diferenciaram melhor as classes, por isso não foram implementados. Instâncias sem valência em alguma modalidade (hoje, 2 sem rosto detectado) recebem valência 0, o mesmo valor usado para entrada vazia nos scripts de valência. Gera `data/processed/incongruence.csv` (sem a coluna de rótulo). O script também imprime a intensidade e a incongruência médias por classe e 5 instâncias sarcásticas com o cálculo aberto (`N_EXEMPLOS`), para conferência.

**Resultados exploratórios** (treino + DSEL, sem usar o teste): as falas sarcásticas têm valência um pouco mais positiva no texto, no áudio e no rosto, **menos** intensidade emocional (principalmente no rosto e no contexto) e **menor** incongruência absoluta entre modalidades, o contrário do que a hipótese inicial previa. Uma leitura possível é que os quatro classificadores captam a emoção expressa na superfície, que no sarcasmo tende a ser coerente entre os canais. Os efeitos são pequenos, e as valências sozinhas separam as classes um pouco melhor do que o vetor de incongruência.

## Como rodar

Pré-requisitos: Python 3.10+, FFmpeg instalado no sistema (não é um pacote Python).

```
pip install -r requirements.txt
```

Depois, rodar os scripts na ordem descrita em "Pipeline, na ordem". Os scripts de `src/features/` dependem apenas de `data/processed/mustard_prepared.csv` (e, no caso do áudio, de `data/processed/audio/`; no caso da expressão facial, de `data/processed/visual_emotions.csv`), então podem ser rodados em qualquer ordem entre si, uma vez que esses dados existam.

Os modelos de texto/contexto (`transformers`, via HuggingFace Hub) e áudio (`emotion2vec_plus_large`, via `funasr`/ModelScope) são baixados automaticamente na primeira execução e ficam em cache localmente.
