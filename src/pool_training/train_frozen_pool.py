import ast
import json
import sys
from pathlib import Path

# Permite executar este arquivo diretamente, adicionando a pasta `src` ao
# caminho de importação para localizar o pacote irmão `pool_selection`.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd

from pool_selection.config import RESULTS_DIR, TRAIN_PATH
from pool_selection.classifiers import build_classifier
from pool_selection.feature_loader import FeatureStore

POOL_DEFINITION_PATH = (RESULTS_DIR / "pool_selection" / "pool_definition.csv")
FROZEN_POOLS_DIR = (RESULTS_DIR / "frozen_pools")

# Funções auxiliares
def parse_params(value):
    """
    Converte os hiperparâmetros armazenados no CSV -> dict
    """

    if isinstance(value, dict):
        return value

    if pd.isna(value):
        return {}

    value = str(value)

    try:
        return json.loads(value)

    except json.JSONDecodeError:
        return ast.literal_eval(value)


def get_best_params(row):
    """
    Obtém os hiperparâmetros do candidato
    """

    possible_columns = [
        "best_params",
        "params",
        "best_parameters",
    ]

    for column in possible_columns:
        if column in row.index:
            return parse_params(row[column])

    raise ValueError("Não foi encontrada uma coluna contendo os hiperparâmetros no pool_definition.csv. \nEsperado: best_params, params ou best_parameters.")


# Treinamento de um membro
def train_member(member_id, row, train, store, condition, output_dir):
    """
    Treina um membro da pool congelada

    condition =
    BASE: usa somente as features das modalidades
    INCONGRUENCE: usa as mesmas features + vetor de incongruência correspondente

    O classificador e os hiperparâmetros são exatamente os mesmos nas duas condições
    """

    modalities = row["modalities"]
    classifier_name = row["classifier"]

    combo = tuple(part.strip() for part in str(modalities).split("+") if part.strip())

    best_params = get_best_params(row)

    use_incongruence = (condition == "incongruence")

    keys = train["KEY"].tolist()
    y = train["Sarcasm"].to_numpy()

    # Montar matriz de features
    X = store.matrix(keys=keys, combo=combo, use_incongruence=use_incongruence)

    model = build_classifier(classifier_name)
    model.set_params(**best_params) # Aplica exatamente os mesmos hiperparâmetros selecionados durante a formação da pool

    # Treinar com todo o conjunto TRAIN
    model.fit(X, y)
    model_path = (output_dir / f"{member_id}.joblib") # Salva o modelo

    joblib.dump(model, model_path)

    print(f"  {member_id}: {modalities} | {classifier_name} | {X.shape[1]} features")

    return {
        "member_id": member_id,
        "candidate_id": row["candidate_id"],
        "modalities": modalities,
        "classifier": classifier_name,
        "best_params": json.dumps(
            best_params
        ),
        "condition": condition,
        "n_features": X.shape[1],
        "model_path": str(model_path),
    }

# Treinar uma condição
def train_condition(pool_definition, train, store, condition,):
    """
    Treina todos os membros da pool para uma condição
    """

    if condition not in {"base", "incongruence"}:
        raise ValueError("condition deve ser 'base' ou 'incongruence'")

    output_dir = (FROZEN_POOLS_DIR / condition)
    output_dir.mkdir(parents=True, exist_ok=True,)

    print(f"\nTREINANDO POOL: {condition.upper()}")

    metadata = []

    # Mesmos membros utilizados para BASE e INC
    for i, (_, row) in enumerate(pool_definition.iterrows(), start=1,):

        member_id = f"C{i:02d}"

        info = train_member(
            member_id=member_id,
            row=row,
            train=train,
            store=store,
            condition=condition,
            output_dir=output_dir,
        )

        metadata.append(info)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / "pool_metadata.csv", index=False,)
    print(f"Pool salva em: {output_dir}")

    return metadata_df


# Execução principal
def run():
    print("TRAIN FROZEN POOLS")

    # Ler definição congelada da pool
    if not POOL_DEFINITION_PATH.exists():
        raise FileNotFoundError(f"pool_definition.csv não encontrado: {POOL_DEFINITION_PATH}")

    pool_definition = pd.read_csv(POOL_DEFINITION_PATH)

    if len(pool_definition) != 7:
        raise ValueError(f"A pool congelada deveria possuir 7 membros, mas possui {len(pool_definition)}.")

    print(f"\nPool definition: {POOL_DEFINITION_PATH}")
    print(f"Membros: {len(pool_definition)}")

    train = pd.read_csv(TRAIN_PATH).copy()
    train["KEY"] = (train["KEY"].astype(str))
    train["Sarcasm"] = (train["Sarcasm"].astype(int))

    print(f"Instâncias de treino: {len(train)}")

    store = FeatureStore()

    # Pool base
    base_metadata = train_condition(
        pool_definition=pool_definition,
        train=train,
        store=store,
        condition="base",
    )

    # Pool de incongruência
    inc_metadata = train_condition(
        pool_definition=pool_definition,
        train=train,
        store=store,
        condition="incongruence",
    )

    # Verificar pareamento
    columns_to_compare = [
        "member_id",
        "candidate_id",
        "modalities",
        "classifier",
        "best_params",
    ]

    base_compare = (
        base_metadata[
            columns_to_compare
        ].reset_index(drop=True)
    )

    inc_compare = (
        inc_metadata[
            columns_to_compare
        ].reset_index(drop=True)
    )

    if not base_compare.equals(inc_compare):
        raise RuntimeError("As pools BASE e INCONGRUENCE não possuem a mesma definição.")

    print("\nTREINAMENTO CONCLUÍDO")

def main():
    run()

if __name__ == "__main__":
    main()