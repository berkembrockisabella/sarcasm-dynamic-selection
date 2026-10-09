'''
Comunicação entre pool congelada e MoE
'''

import numpy as np
from src.pool_selection.modality_combinations import parse_combo_name


def get_expert_probabilities(pool, keys, store):
    """
    Obtém probabilidades da classe positiva produzidas pelos especialistas da pool BASE.

    pool -> pool retornada por load_frozen_pool("base").
    keys -> identificadores das amostras.
    store -> repositório das features multimodais.

    Saída
    probabilities -> matriz (n_samples, n_experts).
    Cada coluna corresponde a um especialista.
    member_ids -> identificadores dos especialistas na mesma ordem das colunas da matriz.
    """

    probabilities = []
    member_ids = []

    for member in pool:
        model = member["model"]
        combo = parse_combo_name(member["modalities"])

        X = store.matrix(keys=keys, combo=combo, use_incongruence=False)

        if not hasattr(model, "predict_proba"):
            raise TypeError(f"O especialista {member['member_id']} não possui predict_proba().")

        proba = model.predict_proba(X)

        classes = np.asarray(model.classes_)
        positive_index = np.where(classes == 1)[0]

        if len(positive_index) != 1:
            raise ValueError(f"Classe positiva 1 não encontrada no especialista {member['member_id']}.")

        probabilities.append(proba[:, positive_index[0]])
        member_ids.append(member["member_id"])

    return (np.column_stack(probabilities), member_ids)