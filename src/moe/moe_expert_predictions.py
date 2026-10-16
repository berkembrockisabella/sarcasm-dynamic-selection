"""
Comunicação entre pool congelada e MoE.
"""

import numpy as np

def parse_combo_name(combo):
    """
    Converte a representação das modalidades para uma lista.
    """

    if isinstance(combo, str):
        return [
            modality.strip()
            for modality in combo.split("+")
            if modality.strip()
        ]

    if isinstance(combo, (list, tuple)):
        return list(combo)

    raise TypeError(f"Formato inválido para modalities. Recebido: {type(combo)}")


def get_expert_probabilities(pool, keys, store):
    """
    Obtém probabilidades da classe positiva produzidas pelos especialistas da pool.

    pool: Pool retornada por load_frozen_pool
    keys: Identificadores das amostras
    store: Repositório das features multimodais

    Retorno:
    probabilities: Matriz (n_samples, n_experts).
    member_ids: IDs dos especialistas na mesma ordem das colunas da matriz.
    """

    probabilities = []
    member_ids = []

    for member in pool:

        model = member["model"]
        combo = parse_combo_name(member["modalities"])

        # Pool: utiliza somente as features das modalidades.
        X = store.matrix(keys=keys, combo=combo)

        if not hasattr(model, "predict_proba"):
            raise TypeError(f"O especialista {member['member_id']} não possui predict_proba().")

        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        positive_index = np.where(classes == 1)[0]

        if len(positive_index) != 1:
            raise ValueError(f"Classe positiva 1 não encontrada no especialista {member['member_id']}.")

        probabilities.append(proba[:, positive_index[0]])
        member_ids.append(member["member_id"])

    if not probabilities:
        raise ValueError("A pool está vazia.")

    return (np.column_stack(probabilities), member_ids)