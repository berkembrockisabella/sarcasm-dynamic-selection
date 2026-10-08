import sys
from pathlib import Path

# Mesmo esquema de ds_routing.py para importar os pacotes irmaos
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

# Analise de sensibilidade ao tamanho da regiao de competencia (k).
# Mesma logica de ds_routing.py (mesma pool, mesmo espaco, mesmas regras de
# OLA / KNORA-E / KNORA-U); so varia o k. Nenhum k e "escolhido" olhando o
# teste -- a tabela mostra todos, e o k=7 do experimento principal serve de
# conferencia (deve reproduzir ds_routing_summary.csv)
# Grupo A tem so 16 falas no DSEL: com roteamento, k maior que isso e cortado
# para o tamanho do grupo (coluna k_efetivo_grupo_A)
KS = [3, 5, 7, 9, 11, 15, 21, 31]
MODALIDADES = ["text", "context", "audio", "visual"]
METODOS = ["OLA", "KNORA-E", "KNORA-U"]
ROTEAMENTOS = ["sem_roteamento", "com_roteamento"]

treino = pd.read_csv("data/splits/train.csv")
dsel = pd.read_csv("data/splits/dsel.csv")
teste = pd.read_csv("data/splits/test.csv")
grupos = pd.read_csv("data/processed/incongruence.csv")[["KEY", "grupo"]]
dsel = dsel.merge(grupos, on="KEY", how="left")
teste = teste.merge(grupos, on="KEY", how="left")

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

# Espaco da regiao de competencia (igual ao ds_routing.py)
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

acerto_dsel = (pred_dsel == y_dsel[:, None]).astype(int)
acuracia_global = acerto_dsel.mean(axis=0)
TODOS = np.arange(len(pool))
n_dsel = len(dsel)
indices_grupo = {g: np.where(grupo_dsel == g)[0] for g in ["A", "B"]}

# Vizinhos ordenados por distancia: busca-se uma vez o maximo de k e, para
# cada k, usa-se o prefixo (os k mais proximos) -- equivale a refazer a busca
k_max = max(KS)
ordem_sem = NearestNeighbors(n_neighbors=min(k_max, n_dsel)).fit(X_dsel).kneighbors(X_teste, return_distance=False)
ordem_grupo = {}
for g in ["A", "B"]:
    idx = indices_grupo[g]
    busca_g = NearestNeighbors(n_neighbors=min(k_max, len(idx))).fit(X_dsel[idx])
    ordem_grupo[g] = busca_g

linhas = []
predicoes = []

for k in KS:
    for roteamento in ROTEAMENTOS:
        # Vizinhos de cada fala de teste para este k e este roteamento
        vizinhos = []
        for i in range(len(teste)):
            if roteamento == "sem_roteamento":
                vizinhos.append(ordem_sem[i][:k])
            else:
                g = grupo_teste[i]
                idx = indices_grupo[g]
                locais = ordem_grupo[g].kneighbors(X_teste[i : i + 1], return_distance=False)[0]
                vizinhos.append(idx[locais][:k])

        pred_metodo = {m: [] for m in METODOS}
        for i in range(len(teste)):
            viz = vizinhos[i]
            acertos_locais = acerto_dsel[viz]

            # OLA: melhor acerto local; empate -> acuracia global -> ordem da pool
            competencia = acertos_locais.mean(axis=0)
            melhor = max(TODOS, key=lambda j: (competencia[j], acuracia_global[j], -j))
            pred_metodo["OLA"].append(int(pred_teste[i, melhor]))

            # KNORA-E: quem acerta todos os vizinhos; senao tira o mais distante
            escolhidos = TODOS
            for k_atual in range(len(viz), 0, -1):
                acertam_todos = np.where(acertos_locais[:k_atual].all(axis=0))[0]
                if len(acertam_todos) > 0:
                    escolhidos = acertam_todos
                    break
            votos_sarc = pred_teste[i, escolhidos].sum()
            votos_nao = len(escolhidos) - votos_sarc
            if votos_sarc != votos_nao:
                pred_metodo["KNORA-E"].append(int(votos_sarc > votos_nao))
            else:
                pred_metodo["KNORA-E"].append(int(proba_teste[i, escolhidos].mean() >= 0.5))

            # KNORA-U: voto ponderado pelo numero de vizinhos acertados
            pesos = acertos_locais.sum(axis=0)
            usados = np.where(pesos > 0)[0] if pesos.sum() > 0 else TODOS
            p = pesos[usados] if pesos.sum() > 0 else np.ones(len(usados))
            votos_sarc = (p * pred_teste[i, usados]).sum()
            votos_nao = (p * (1 - pred_teste[i, usados])).sum()
            if votos_sarc != votos_nao:
                pred_metodo["KNORA-U"].append(int(votos_sarc > votos_nao))
            else:
                pred_metodo["KNORA-U"].append(int(proba_teste[i, usados].mean() >= 0.5))

        for metodo in METODOS:
            predicao = np.array(pred_metodo[metodo])
            linha = {"k": k, "metodo": metodo, "roteamento": roteamento}
            linha["accuracy"] = accuracy_score(y_teste, predicao)
            linha["f1_macro"] = f1_score(y_teste, predicao, average="macro")
            for g in ["A", "B"]:
                mascara = grupo_teste == g
                linha[f"f1_macro_grupo_{g}"] = f1_score(y_teste[mascara], predicao[mascara], average="macro")
            linha["k_efetivo_grupo_A"] = min(k, len(indices_grupo["A"])) if roteamento == "com_roteamento" else k
            linhas.append(linha)

resumo = pd.DataFrame(linhas)
Path("data/processed/selection").mkdir(parents=True, exist_ok=True)
resumo.to_csv("data/processed/selection/ds_routing_k_sweep.csv", index=False)

# Tabela e grafico: F1 macro de cada metodo em funcao de k, sem e com roteamento
tabela = resumo.pivot_table(index="k", columns=["metodo", "roteamento"], values="f1_macro")
print("F1 macro no teste por k:")
print(tabela.round(4).to_string())
print("\nAcuracia no teste por k:")
print(resumo.pivot_table(index="k", columns=["metodo", "roteamento"], values="accuracy").round(4).to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

figura, eixos = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for eixo, metodo in zip(eixos, METODOS):
    for roteamento, estilo in [("sem_roteamento", "--o"), ("com_roteamento", "-o")]:
        parte = resumo[(resumo["metodo"] == metodo) & (resumo["roteamento"] == roteamento)]
        eixo.plot(parte["k"], parte["f1_macro"], estilo, label=roteamento.replace("_", " "))
    eixo.axvline(7, color="gray", linestyle=":", linewidth=1)
    eixo.set_title(metodo)
    eixo.set_xlabel("k (tamanho da regiao de competencia)")
    eixo.grid(alpha=0.3)
eixos[0].set_ylabel("F1 macro no teste")
eixos[0].legend()
figura.tight_layout()
figura.savefig("data/processed/selection/ds_routing_k_sweep.png", dpi=150)

print("\nSalvo em: data/processed/selection/ds_routing_k_sweep.csv")
print("Salvo em: data/processed/selection/ds_routing_k_sweep.png")
