'''
Carrega as features de cada modalidade e as incongruências
Monta a matriz de features para uma combinação de modalidades
'''

from itertools import combinations
import numpy as np
import pandas as pd

from .config import FEATURE_SOURCES, INCONGRUENCE_PATH

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

def incongruence_columns_for(combo):
    # Para combinação de modalidades -> inc_<mod1>_<mod2>
    cols = []
    for a, b in combinations(combo, 2):
        cols.append(f"inc_{a}_{b}")
    return cols

class FeatureStore:
    def __init__(self):
        self.modalities = {
            name: load_modality(name)
            for name in FEATURE_SOURCES
        }

        inc = pd.read_csv(INCONGRUENCE_PATH)
        inc["KEY"] = inc["KEY"].astype(str)
        self.incongruence = inc.set_index("KEY")

    def matrix(self, keys, combo, use_incongruence=False):
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

        if use_incongruence:
            cols = incongruence_columns_for(combo)
            missing_cols = [c for c in cols if c not in self.incongruence.columns]
            if missing_cols:
                raise ValueError(f"Colunas de incongruência ausentes: {missing_cols}")

            inc_block = (
                self.incongruence
                .reindex(keys)[cols]
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            blocks.append(inc_block)

        return np.hstack(blocks)
