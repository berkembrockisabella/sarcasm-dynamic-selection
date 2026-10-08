'''
Carrega uma pool congelada
'''

import joblib
import pandas as pd

from pool_selection.config import RESULTS_DIR

# A pasta se chama base por compatibilidade com os modelos ja treinados -- ha uma unica pool
POOL_DIR = (RESULTS_DIR / "frozen_pools" / "base")

def load_frozen_pool():
    pool_dir = POOL_DIR

    if not pool_dir.exists():
        raise FileNotFoundError(f"Pool não encontrada: {pool_dir}")

    # Carregar metadados
    metadata_path = (pool_dir / "pool_metadata.csv")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadados da pool não encontrados: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Carregar classificadores
    pool = []

    for _, row in metadata.iterrows():

        member_id = row["member_id"]
        model_path = (pool_dir / f"{member_id}.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        model = joblib.load(model_path)

        pool.append({
            "member_id": member_id,
            "candidate_id": row[
                "candidate_id"
            ],
            "modalities": row[
                "modalities"
            ],
            "classifier": row[
                "classifier"
            ],
            "n_features": int(
                row["n_features"]
            ),
            "model": model,
        })

    # Verificação final
    if len(pool) != 7:
        raise RuntimeError(f"Esperados 7 membros na pool, mas foram carregados {len(pool)}.")

    return pool