'''
Carrega as features de cada modalidade
Monta a matriz de features para uma combinação de modalidades
'''

import numpy as np
import pandas as pd

from .config import FEATURE_SOURCES

def _load_npy_source(source):
    X = np.load(source["features"])
    keys = pd.read_csv(source["keys"])["KEY"].astype(str)
    if len(X) != len(keys):
        raise ValueError(f"{source['features']}: número de embeddings != número de KEYs")
    return pd.DataFrame({"KEY": keys, "_row": np.arange(len(keys))}), X

def load_modality(modality):
    source = FEATURE_SOURCES[modality]
    if source["type"] == "npy":
        keys_df, X = _load_npy_source(source)
    else:
        df = pd.read_csv(source["features"])
        df["KEY"] = df["KEY"].astype(str)
        cols = [c for c in df.columns if c != "KEY"]
        keys_df = df[["KEY"]].copy()
        X = df[cols].to_numpy(dtype=np.float32)

    if keys_df["KEY"].duplicated().any():
        raise ValueError(f"KEY duplicada nas features de {modality}")
    return dict(zip(keys_df["KEY"], X))

class FeatureStore:
    def __init__(self):
        self.modalities = {
            name: load_modality(name)
            for name in FEATURE_SOURCES
        }

    def matrix(self, keys, combo):
        keys = [str(k) for k in keys]
        blocks = []

        for modality in combo:
            mapping = self.modalities[modality]
            missing = [k for k in keys if k not in mapping]
            if missing:
                raise ValueError(
                    f"{len(missing)} KEYs sem features de {modality}. "
                    f"Exemplos: {missing[:5]}"
                )
            blocks.append(np.vstack([mapping[k] for k in keys]).astype(np.float32))

        return np.hstack(blocks)
