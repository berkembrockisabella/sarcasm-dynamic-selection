import sys
from pathlib import Path

# Permite importar os pacotes irmaos (pool_selection, pool_training) rodando
# este arquivo direto da raiz do repositorio -- mesmo esquema de
# src/pool_training/train_frozen_pool.py
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from pool_selection.feature_loader import FeatureStore
from pool_training.load_frozen_pool import load_frozen_pool

# Experimento: OLA, KNORA-E e KNORA-U, cada um sem e com roteamento por
# incongruencia. A pool e sempre a mesma (7 especialistas, versao BASE) --
# o unico fator que muda entre "sem" e "com" e o roteamento:
#   - sem roteamento: os k vizinhos da instancia de teste sao procurados no
#     DSEL inteiro
#   - com roteamento: os k vizinhos sao procurados so entre as instancias do
#     DSEL do mesmo grupo (A = as 4 modalidades tem a mesma polaridade,
#     B = pelo menos uma difere; coluna "grupo" de incongruence.csv)
# Referencias: OLA (Woods et al., 1997) e KNORA-E/U (Ko, Sabourin & Britto,
# 2008), como descritos em Cruz et al. (2018) e Britto et al. (2014)
K = 7
MODALIDADES = ["text", "context", "audio", "visual"]
METODOS = ["OLA", "KNORA-E", "KNORA-U"]
ROTEAMENTOS = ["sem_roteamento", "com_roteamento"]

treino = pd.read_csv("data/splits/train.csv")
dsel = pd.read_csv("data/splits/dsel.csv")
teste = pd.read_csv("data/splits/test.csv")
grupos = pd.read_csv("data/processed/incongruence.csv")[["KEY", "grupo"]]
dsel = dsel.merge(grupos, on="KEY", how="left")
teste = teste.merge(grupos, on="KEY", how="left")

# Sem grupo nao da para rotear -- melhor parar do que rotear errado
for nome, tabela in [("DSEL", dsel), ("teste", teste)]:
    if tabela["grupo"].isna().any():
        raise ValueError(f"{nome}: instancias sem grupo em incongruence.csv")

y_dsel = dsel["Sarcasm"].astype(int).to_numpy()
y_teste = teste["Sarcasm"].astype(int).to_numpy()

# --------------------------------------------------------------------------
# Predicoes da pool no DSEL e no teste
# --------------------------------------------------------------------------
store = FeatureStore()
pool = load_frozen_pool("base")
MEMBROS = [membro["member_id"] for membro in pool]

pred_dsel = np.zeros((len(dsel), len(pool)), dtype=int)
pred_teste = np.zeros((len(teste), len(pool)), dtype=int)
proba_teste = np.zeros((len(teste), len(pool)))

for j, membro in enumerate(pool):
    combo = tuple(parte.strip() for parte in str(membro["modalities"]).split("+"))
    X_d = store.matrix(dsel["KEY"].tolist(), combo, use_incongruence=False)
    X_t = store.matrix(teste["KEY"].tolist(), combo, use_incongruence=False)
    pred_dsel[:, j] = membro["model"].predict(X_d)
    pred_teste[:, j] = membro["model"].predict(X_t)
    proba_teste[:, j] = membro["model"].predict_proba(X_t)[:, 1]

# --------------------------------------------------------------------------
# Espaco da regiao de competencia: as 4 modalidades concatenadas. Cada
# modalidade e padronizada (scaler ajustado so no treino) e normalizada (L2)
# para ter peso parecido na distancia -- mesmo esquema de dcs_local_accuracy.py
# --------------------------------------------------------------------------
blocos_dsel = []
blocos_teste = []
for modalidade in MODALIDADES:
    escalador = StandardScaler().fit(store.matrix(treino["KEY"].tolist(), (modalidade,)))
    bloco_d = escalador.transform(store.matrix(dsel["KEY"].tolist(), (modalidade,)))
    bloco_t = escalador.transform(store.matrix(teste["KEY"].tolist(), (modalidade,)))
    blocos_dsel.append(normalize(bloco_d))
    blocos_teste.append(normalize(bloco_t))
X_dsel = np.hstack(blocos_dsel)
X_teste = np.hstack(blocos_teste)
grupo_dsel = dsel["grupo"].to_numpy()
grupo_teste = teste["grupo"].to_numpy()

# ===== SELECAO DINAMICA =====
acerto_dsel = (pred_dsel == y_dsel[:, None]).astype(int)
acuracia_global = acerto_dsel.mean(axis=0)

# Vizinhos de cada instancia de teste, nos dois roteamentos. Os indices
# guardados sao sempre do DSEL inteiro, para usar acerto_dsel direto
vizinhos = {"com_roteamento": []}
busca = NearestNeighbors(n_neighbors=K).fit(X_dsel)
vizinhos["sem_roteamento"] = list(busca.kneighbors(X_teste, return_distance=False))

for grupo in ["A", "B"]:
    indices_grupo = np.where(grupo_dsel == grupo)[0]
    k_grupo = min(K, len(indices_grupo))
    busca_grupo = NearestNeighbors(n_neighbors=k_grupo).fit(X_dsel[indices_grupo])
    vizinhos[f"grupo_{grupo}"] = (indices_grupo, busca_grupo)

for i in range(len(teste)):
    indices_grupo, busca_grupo = vizinhos[f"grupo_{grupo_teste[i]}"]
    locais = busca_grupo.kneighbors(X_teste[i : i + 1], return_distance=False)[0]
    vizinhos["com_roteamento"].append(indices_grupo[locais])

# Voto majoritario entre os membros escolhidos. Empate (numero par de votos)
# e decidido pela media das probabilidades de sarcasmo desses membros
def voto(i, escolhidos, pesos=None):
    if pesos is None:
        pesos = np.ones(len(escolhidos))
    votos_sarc = (pesos * pred_teste[i, escolhidos]).sum()
    votos_nao = (pesos * (1 - pred_teste[i, escolhidos])).sum()
    if votos_sarc != votos_nao:
        return int(votos_sarc > votos_nao)
    return int(proba_teste[i, escolhidos].mean() >= 0.5)

TODOS = np.arange(len(pool))
linhas = []

for roteamento in ROTEAMENTOS:
    for i in range(len(teste)):
        viz = vizinhos[roteamento][i]
        acertos_locais = acerto_dsel[viz]          # k x membros

        # OLA: o membro com mais acertos entre os k vizinhos decide sozinho.
        # Empate -> maior acuracia global no DSEL -> ordem da pool
        competencia = acertos_locais.mean(axis=0)
        melhor = max(TODOS, key=lambda j: (competencia[j], acuracia_global[j], -j))
        linha = {
            "KEY": teste.loc[i, "KEY"], "grupo": grupo_teste[i], "Sarcasm": y_teste[i],
            "roteamento": roteamento, "k_usado": len(viz),
        }
        linha["OLA"] = int(pred_teste[i, melhor])
        linha["OLA_membros"] = MEMBROS[melhor]

        # KNORA-E: membros que acertam TODOS os vizinhos; se nenhum, tira o
        # vizinho mais distante e tenta de novo. Se nenhum acerta nem o mais
        # proximo, usa a pool inteira (Ko et al., 2008)
        escolhidos = TODOS
        for k_atual in range(len(viz), 0, -1):
            acertam_todos = np.where(acertos_locais[:k_atual].all(axis=0))[0]
            if len(acertam_todos) > 0:
                escolhidos = acertam_todos
                break
        linha["KNORA-E"] = voto(i, escolhidos)
        linha["KNORA-E_membros"] = "+".join(MEMBROS[j] for j in escolhidos)

        # KNORA-U: cada membro vota com peso = numero de vizinhos que ele
        # acertou. Se ninguem acertou nenhum, voto simples da pool inteira
        pesos = acertos_locais.sum(axis=0)
        if pesos.sum() > 0:
            usados = np.where(pesos > 0)[0]
            linha["KNORA-U"] = voto(i, usados, pesos[usados])
            linha["KNORA-U_membros"] = "+".join(MEMBROS[j] for j in usados)
        else:
            linha["KNORA-U"] = voto(i, TODOS)
            linha["KNORA-U_membros"] = "+".join(MEMBROS)
        linhas.append(linha)

resultado = pd.DataFrame(linhas)

# Referencias, iguais nos dois roteamentos: melhor membro no DSEL sozinho,
# voto majoritario da pool inteira e oraculo (algum membro acerta)
melhor_global = int(np.argmax(acuracia_global))
referencias = {
    f"Melhor membro ({MEMBROS[melhor_global]})": pred_teste[:, melhor_global],
    "Voto da pool inteira": np.array([voto(i, TODOS) for i in range(len(teste))]),
}
oraculo = (pred_teste == y_teste[:, None]).any(axis=1).mean()

resumo = []
for roteamento in ROTEAMENTOS:
    parte = resultado[resultado["roteamento"] == roteamento]
    for metodo in METODOS:
        linha = {"metodo": metodo, "roteamento": roteamento}
        linha["accuracy"] = accuracy_score(parte["Sarcasm"], parte[metodo])
        linha["f1_macro"] = f1_score(parte["Sarcasm"], parte[metodo], average="macro")
        for grupo in ["A", "B"]:
            g = parte[parte["grupo"] == grupo]
            linha[f"f1_macro_grupo_{grupo}"] = f1_score(g["Sarcasm"], g[metodo], average="macro")
        resumo.append(linha)
for nome, predicao in referencias.items():
    resumo.append({
        "metodo": nome, "roteamento": "-",
        "accuracy": accuracy_score(y_teste, predicao),
        "f1_macro": f1_score(y_teste, predicao, average="macro"),
    })
# Oraculo: fracao das falas em que pelo menos um membro acerta -- teto de
# acuracia de qualquer selecao feita com esta pool (nao tem F1)
resumo.append({"metodo": "Oraculo (algum membro acerta)", "roteamento": "-", "accuracy": oraculo})
resumo = pd.DataFrame(resumo)

Path("data/processed/selection").mkdir(parents=True, exist_ok=True)
resultado.to_csv("data/processed/selection/ds_routing_test.csv", index=False)
resumo.to_csv("data/processed/selection/ds_routing_summary.csv", index=False)

print("Pool (BASE):", ", ".join(f"{m['member_id']}={m['modalities']}/{m['classifier']}" for m in pool))
print("Acuracia global de cada membro no DSEL:", dict(zip(MEMBROS, acuracia_global.round(3))))
print("DSEL por grupo:", dsel["grupo"].value_counts().to_dict(), "| teste por grupo:", teste["grupo"].value_counts().to_dict())
print(f"k = {K} | oraculo no teste: {oraculo:.3f}")
print("\nResultados no teste:")
print(resumo.round(4).to_string(index=False))
print("\nSalvo em: data/processed/selection/ds_routing_test.csv")
print("Salvo em: data/processed/selection/ds_routing_summary.csv")
