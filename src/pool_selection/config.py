'''
Configuração para a seleção da pool
'''

import os
from pathlib import Path

RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "f1_macro"
POOL_SIZE = 7
TOP_K_FOR_DIVERSITY = 15

DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "processed" / "features"

# Repeticoes do experimento (protocolo E0): com SPLIT_ID=03 os scripts leem
# data/splits/split_03/ e gravam os resultados em subpastas split_03/.
# O run_experiment.py exige o split (--split XX ou --split all)
SPLIT_ID = os.environ.get("SPLIT_ID", "").strip()
SPLIT_SUFFIX = f"split_{SPLIT_ID}" if SPLIT_ID else ""

SPLIT_DIR = DATA_DIR / "splits" / SPLIT_SUFFIX
RESULTS_DIR = DATA_DIR / "results" / "pool" / SPLIT_SUFFIX

# Variantes da pool (ablacao): peso da diversidade na escolha gulosa e limite
# de membros que usam o contexto. Os valores padrao reproduzem a pool atual.
# Uma variante reaproveita a validacao cruzada (cv_results/oof) do split --
# so muda a escolha dos 7 membros -- e grava a pool e os resultados numa
# subpasta com o nome da variante (ex.: div_0.8, div_0.3_ctx_max4)
POOL_DIVERSITY = float(os.environ.get("POOL_DIVERSITY", "0.3"))
POOL_MAX_CONTEXTO = int(os.environ.get("POOL_MAX_CONTEXTO", "0"))
DIVERSITY_PADRAO = 0.3

# Grupos de incongruencia: "original" (incongruence.csv, argmax) ou
# "calibrada" (polaridade com zona morta ajustada no treino de cada split,
# ver src/incongruence/calibrar_polaridade.py)
INCONGRUENCIA = os.environ.get("INCONGRUENCIA", "original").strip()
if INCONGRUENCIA == "calibrada":
    INCONGRUENCIA_PATH = DATA_DIR / "processed" / "incongruence_calibrada" / f"{SPLIT_SUFFIX or 'unico'}.csv"
else:
    INCONGRUENCIA_PATH = DATA_DIR / "processed" / "incongruence.csv"

partes_variante = []
if POOL_DIVERSITY != DIVERSITY_PADRAO or POOL_MAX_CONTEXTO:
    partes_variante.append(f"div_{POOL_DIVERSITY:.1f}" + (f"_ctx_max{POOL_MAX_CONTEXTO}" if POOL_MAX_CONTEXTO else ""))
if INCONGRUENCIA == "calibrada":
    partes_variante.append("inc_calibrada")
VARIANTE = "_".join(partes_variante)

POOL_DEFINITION_PATH = RESULTS_DIR / "pool_selection" / VARIANTE / "pool_definition.csv"
FROZEN_POOL_DIR = RESULTS_DIR / "frozen_pools" / VARIANTE

TRAIN_PATH = SPLIT_DIR / "train.csv"

MODALITIES = ("text", "context", "audio", "visual")

FEATURE_SOURCES = {
    "text": {
        "type": "npy",
        "features": FEATURE_DIR / "text_embeddings.npy",
        "keys": FEATURE_DIR / "text_keys.csv",
    },
    "context": {
        "type": "npy",
        "features": FEATURE_DIR / "context_embeddings.npy",
        "keys": FEATURE_DIR / "context_keys.csv",
    },
    "audio": {
        "type": "npy",
        "features": FEATURE_DIR / "audio_embeddings.npy",
        "keys": FEATURE_DIR / "audio_keys.csv",
    },
    "visual": {
        "type": "csv",
        "features": FEATURE_DIR / "visual_features.csv",
    },
}

PARAM_GRIDS = {
    "svm": {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", 0.01, 0.001],
        "model__kernel": ["rbf"],
    },
    "logistic_regression": {
        "model__C": [0.1, 1, 10],
        "model__penalty": ["l1", "l2"],
        "model__solver": ["liblinear"],
    },
    "random_forest": {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
}