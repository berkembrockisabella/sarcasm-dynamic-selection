"""
Experimento:
OLA, KNORA-E e KNORA-U, cada um sem e com roteamento por incongruência
A pool é sempre a mesma (7 especialistas)
    - sem roteamento: os k vizinhos da instancia de teste sao procurados no DSEL inteiro
    - com roteamento: os k vizinhos sao procurados só entre as instancias do
    DSEL do mesmo grupo (A = as 4 modalidades tem a mesma polaridade,
    B = pelo menos uma difere; coluna "grupo" de incongruence.csv)
Referências: OLA (Woods et al., 1997) e KNORA-E/U (Ko, Sabourin & Britto,
2008), como descritos em Cruz et al. (2018) e Britto et al. (2014)
"""

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from pool_selection.feature_loader import FeatureStore
from pool_training.load_frozen_pool import load_frozen_pool

K = 7
MODALIDADES = ["text", "context", "audio", "visual"]
METODOS = ["OLA", "KNORA-E", "KNORA-U"]
ROTEAMENTOS = ["sem_roteamento", "com_roteamento"]
GRUPOS = ["A", "B"]

OUT_DIR = Path("data/processed/selection")
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


# Predicoes da pool no DSEL e no teste
store = FeatureStore()
pool = load_frozen_pool()
MEMBROS = [membro["member_id"] for membro in pool]

pred_dsel = np.zeros((len(dsel), len(pool)), dtype=int)
pred_teste = np.zeros((len(teste), len(pool)), dtype=int)
proba_teste = np.zeros((len(teste), len(pool)))

for j, membro in enumerate(pool):
    combo = tuple(parte.strip() for parte in str(membro["modalities"]).split("+"))
    X_d = store.matrix(dsel["KEY"].tolist(), combo)
    X_t = store.matrix(teste["KEY"].tolist(), combo)
    pred_dsel[:, j] = membro["model"].predict(X_d)
    pred_teste[:, j] = membro["model"].predict(X_t)
    proba_teste[:, j] = membro["model"].predict_proba(X_t)[:, 1]

# Espaco da regiao de competencia: as 4 modalidades concatenadas.
# Cada modalidade é padronizada (scaler ajustado so no treino)
# e normalizada (L2) para ter peso parecido na distância.
blocos_dsel = []
blocos_teste = []
for modalidade in MODALIDADES:
    scaler = StandardScaler().fit(store.matrix(treino["KEY"].tolist(), (modalidade,)))
    bloco_d = scaler.transform(store.matrix(dsel["KEY"].tolist(), (modalidade,)))
    bloco_t = scaler.transform(store.matrix(teste["KEY"].tolist(), (modalidade,)))
    blocos_dsel.append(normalize(bloco_d))
    blocos_teste.append(normalize(bloco_t))
X_dsel = np.hstack(blocos_dsel)
X_teste = np.hstack(blocos_teste)
grupo_dsel = dsel["grupo"].to_numpy()
grupo_teste = teste["grupo"].to_numpy()


# ===== SELECAO DINAMICA =====
acerto_dsel = (pred_dsel == y_dsel[:, None]).astype(int)
acuracia_global = acerto_dsel.mean(axis=0)

# Vizinhos de cada instância de teste, nos dois roteamentos. Os índices
# guardados são sempre do DSEL inteiro, para usar acerto_dsel direto
vizinhos = {"com_roteamento": []}
busca = NearestNeighbors(n_neighbors=K).fit(X_dsel)
vizinhos["sem_roteamento"] = list(busca.kneighbors(X_teste, return_distance=False))

for grupo in GRUPOS:
    indices_grupo = np.where(grupo_dsel == grupo)[0]
    k_grupo = min(K, len(indices_grupo))
    busca_grupo = NearestNeighbors(n_neighbors=k_grupo).fit(X_dsel[indices_grupo])
    vizinhos[f"grupo_{grupo}"] = (indices_grupo, busca_grupo)

for i in range(len(teste)):
    indices_grupo, busca_grupo = vizinhos[f"grupo_{grupo_teste[i]}"]
    locais = busca_grupo.kneighbors(X_teste[i : i + 1], return_distance=False)[0]
    vizinhos["com_roteamento"].append(indices_grupo[locais])

# Voto majoritário entre os membros escolhidos. Empate (número par de votos)
# é decidido pela média das probabilidades de sarcasmo desses membros (com limiar 0.5)
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
        
        linha = {
            "KEY": teste.loc[i, "KEY"],
            "grupo": grupo_teste[i],
            "Sarcasm": y_teste[i],
            "roteamento": roteamento,
            "k_usado": len(viz),
        }
        
        # OLA: o membro com mais acertos entre os k vizinhos decide sozinho.
        # Empate -> maior acurácia global no DSEL -> ordem da pool
        competencia = acertos_locais.mean(axis=0)
        melhor = max(TODOS, key=lambda j: (competencia[j], acuracia_global[j], -j))
        linha["OLA"] = int(pred_teste[i, melhor])
        linha["OLA_score"] = float(proba_teste[i, melhor])
        linha["OLA_membros"] = MEMBROS[melhor]

        # KNORA-E: membros que acertam TODOS os vizinhos; se nenhum, tira o
        # vizinho mais distante e tenta de novo. Se nenhum acerta nem o mais
        # próximo, usa a pool inteira (Ko et al., 2008)
        escolhidos = TODOS
        for k_atual in range(len(viz), 0, -1):
            acertam_todos = np.where(acertos_locais[:k_atual].all(axis=0))[0]
            if len(acertam_todos) > 0:
                escolhidos = acertam_todos
                break
        linha["KNORA-E"] = voto(i, escolhidos)
        linha["KNORA-E_score"] = float(proba_teste[i, escolhidos].mean())
        linha["KNORA-E_membros"] = "+".join(MEMBROS[j] for j in escolhidos)

        # KNORA-U: cada membro vota com peso == número de vizinhos que ele
        # acertou. Se ninguém acertou nenhum, voto simples da pool inteira
        pesos = acertos_locais.sum(axis=0)
        if pesos.sum() > 0:
            usados = np.where(pesos > 0)[0]
            pesos_usados = pesos[usados].astype(float)
        else:
            usados = TODOS
            pesos_usados = np.ones(len(TODOS), dtype=float)
        
        linha["KNORA-U"] = voto(i, usados, pesos_usados)
        linha["KNORA-U_score"] = float(np.average(proba_teste[i, usados], weights=pesos_usados))
        linha["KNORA-U_membros"] = "+".join(MEMBROS[j] for j in usados)
        
        linhas.append(linha)

resultado = pd.DataFrame(linhas)

# Referências, iguais nos dois roteamentos: melhor membro no DSEL sozinho,
# voto majoritário da pool inteira e oráculo (algum membro acerta)
melhor_global = int(np.argmax(acuracia_global))
referencias = {
    f"Melhor membro ({MEMBROS[melhor_global]})": {
        "pred": pred_teste[:, melhor_global],
        "score": proba_teste[:, melhor_global],
    },
    "Voto da pool inteira": {
        "pred": np.array([voto(i, TODOS) for i in range(len(teste))]),
        "score": proba_teste.mean(axis=1),
    },
}


def calcular_metricas(y_true, y_pred, y_score):
    """Métricas de classificação"""
    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": (roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan),
    }
    return metricas

resumo = []
for roteamento in ROTEAMENTOS:
    parte = resultado[resultado["roteamento"] == roteamento]
    for metodo in METODOS:
        resumo.append({
            "metodo": f"{metodo} {roteamento}",
            **calcular_metricas(
                parte["Sarcasm"].to_numpy(),
                parte[metodo].to_numpy(),
                parte[f"{metodo}_score"].to_numpy(),
            ),
        })
        
        # APAGAR
        linha = {"metodo": f"{metodo} {roteamento}"}
        linha["accuracy"] = accuracy_score(parte["Sarcasm"], parte[metodo])
        linha ["precision"] = precision_score(parte["Sarcasm"], parte[metodo], zero_division=0)
        linha ["recall"] = recall_score(parte["Sarcasm"], parte[metodo], zero_division=0)
        linha ["f1"] = f1_score(parte["Sarcasm"], parte[metodo], zero_division=0)
        linha["f1_macro"] = f1_score(parte["Sarcasm"], parte[metodo], average="macro")
        linha["roc_auc"] = roc_auc_score(parte["Sarcasm"], prob) if len(np.unique(parte["Sarcasm"])) == 2 else np.nan
        
        
        

for nome, dados in referencias.items():
    resumo.append({
        "metodo": nome,
        **calcular_metricas(y_teste, dados["pred"], dados["score"]),
    })

# Oráculo: fração das falas em que pelo menos um membro acerta -- teto de
# acurácia de qualquer seleção feita com esta pool (não tem F1)
oraculo = (pred_teste == y_teste[:, None]).any(axis=1).mean()
resumo.append({"metodo": "Oraculo (algum membro acerta)", "accuracy": oraculo})
resumo = pd.DataFrame(resumo)

# Métricas por grupo (A e B)
metricas_grupo = []
for roteamento in ROTEAMENTOS:
    parte = resultado[resultado["roteamento"] == roteamento]
    for grupo in GRUPOS:
        g = parte[parte["grupo"] == grupo]
        for metodo in METODOS:
            metricas_grupo.append({
                "metodo": f"{metodo} {roteamento}",
                "grupo": grupo,
                **calcular_metricas(
                    g["Sarcasm"].to_numpy(),
                    g[metodo].to_numpy(),
                    g[f"{metodo}_score"].to_numpy(),
                ),
            })

for nome, dados in referencias.items():
    for grupo in GRUPOS:
        mascara = grupo_teste == grupo
        metricas_grupo.append({
            "metodo": nome,
            "grupo": grupo,
            **calcular_metricas(
                y_teste[mascara],
                dados["pred"][mascara],
                dados["score"][mascara],
            ),
        })

# Acurácia do oráculo por grupo
for grupo in GRUPOS:
    mascara = grupo_teste == grupo
    acuracia_oraculo_grupo = ((pred_teste[mascara] == y_teste[mascara, None]).any(axis=1).mean() if mascara.any() else np.nan)
    metricas_grupo.append({
        "metodo": "Oráculo (algum membro acerta)",
        "grupo": grupo,
        "accuracy": acuracia_oraculo_grupo,
    })

metricas_grupo = pd.DataFrame(metricas_grupo, columns=[
    "metodo", "roteamento", "grupo", "accuracy", "precision", "recall",
    "f1", "f1_macro", "f1_score", "roc_auc"
])

OUT_DIR.mkdir(parents=True, exist_ok=True)
resultado.to_csv(OUT_DIR / "ds_routing_test.csv", index=False)
resumo.to_csv(OUT_DIR / "ds_routing_summary.csv", index=False)
metricas_grupo.to_csv(OUT_DIR / "ds_routing_group.csv", index=False)

print("Pool:", ", ".join(f"{m['member_id']}={m['modalities']}/{m['classifier']}" for m in pool))
print("Acuracia global de cada membro no DSEL:", dict(zip(MEMBROS, acuracia_global.round(3))))
print("DSEL por grupo:", dsel["grupo"].value_counts().to_dict(), "| teste por grupo:", teste["grupo"].value_counts().to_dict())
print(f"k = {K} | oraculo no teste: {oraculo:.3f}")
print("\nResultados no teste:")
print(resumo.round(4).to_string(index=False))
print("\nSalvo em: data/processed/selection/ds_routing_test.csv")
print("Salvo em: data/processed/selection/ds_routing_summary.csv")
print("Salvo em: data/processed/selection/ds_routing_group.csv")