'''
Configuração para a seleção da pool
'''

from pathlib import Path

RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "f1_macro"
POOL_SIZE = 7
TOP_K_FOR_DIVERSITY = 15

DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "processed" / "features"
SPLIT_DIR = DATA_DIR / "splits"
RESULTS_DIR = DATA_DIR / "results" / "pool"

TRAIN_PATH = SPLIT_DIR / "train.csv"
INCONGRUENCE_PATH = DATA_DIR / "processed" / "incongruence.csv"

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