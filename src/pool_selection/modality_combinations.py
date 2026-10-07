'''
Gera todas as combinações possíveis de modalidades (áudio, contexto, texto e visual)
'''

from itertools import combinations
from .config import MODALITIES

def all_modality_combinations():
    return [
        combo
        for r in range(1, len(MODALITIES) + 1)
        for combo in combinations(MODALITIES, r)
    ]

def combo_name(combo):
    return "+".join(combo)