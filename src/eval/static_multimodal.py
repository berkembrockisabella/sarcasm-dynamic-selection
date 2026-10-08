import json
import sys
from pathlib import Path

# Permite importar os pacotes irmaos (pool_selection) rodando este arquivo
# direto da raiz do repositorio -- mesmo esquema de train_frozen_pool.py
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from pool_selection.classifiers import build_classifier
from pool_selection.feature_loader import FeatureStore

# Linha de base estatica multimodal: classificadores que recebem as 4
# modalidades juntas (fusao precoce: embeddings de texto, contexto e audio +
# features visuais concatenados), treinados so no treino e aplicados igual a
# todas as falas do teste -- sem DSEL e sem selecao dinamica. Serve para
# responder se OLA/KNORA (com e sem roteamento, em ds_routing.py) diferem de
# um classificador estatico que ve todas as modalidades.
#
# Entram os 3 classificadores avaliados na formacao da pool (SVM, regressao
# logistica e random forest) e o voto majoritario dos 3. Para nao ajustar
# nada no teste, os hiperparametros vem da validacao cruzada no treino que
# build_pool.py ja fez (cv_results.csv, combinacao text+context+audio+visual)
TODAS = "text+context+audio+visual"
NOMES = {"svm": "SVM", "logistic_regression": "Regressao logistica", "random_forest": "Random forest"}

treino = pd.read_csv("data/splits/train.csv")
teste = pd.read_csv("data/splits/test.csv")
cv = pd.read_csv("data/results/pool/pool_selection/cv_results.csv")
ds = pd.read_csv("data/processed/selection/ds_routing_test.csv")
y_treino = treino["Sarcasm"].astype(int).to_numpy()
y_teste = teste["Sarcasm"].astype(int).to_numpy()

escolhidos = []
for _, linha in cv[cv["modalities"] == TODAS].sort_values("mean_f1", ascending=False).iterrows():
    escolhidos.append((f"Estatico {NOMES[linha['classifier']]}", linha))
if len(escolhidos) != 3:
    raise ValueError(f"cv_results.csv deveria ter 3 candidatos {TODAS}, tem {len(escolhidos)}")

store = FeatureStore()
predicoes = pd.DataFrame({"KEY": teste["KEY"], "Sarcasm": y_teste})
resumo = []

for nome, linha in escolhidos:
    combo = tuple(parte.strip() for parte in linha["modalities"].split("+"))
    modelo = build_classifier(linha["classifier"])
    modelo.set_params(**json.loads(linha["best_params"]))
    modelo.fit(store.matrix(treino["KEY"].tolist(), combo), y_treino)
    predicoes[nome] = modelo.predict(store.matrix(teste["KEY"].tolist(), combo))
    resumo.append({
        "metodo": nome,
        "candidato": linha["candidate_id"],
        "f1_macro_cv_treino": linha["mean_f1"],
        "accuracy": accuracy_score(y_teste, predicoes[nome]),
        "f1_macro": f1_score(y_teste, predicoes[nome], average="macro"),
    })

# Voto majoritario dos 3 estaticos (com 3 votos nao ha empate)
nomes_individuais = [nome for nome, _ in escolhidos]
predicoes["Estatico voto dos 3"] = (predicoes[nomes_individuais].sum(axis=1) >= 2).astype(int)
resumo.append({
    "metodo": "Estatico voto dos 3", "candidato": "voto SVM + RL + RF", "f1_macro_cv_treino": None,
    "accuracy": accuracy_score(y_teste, predicoes["Estatico voto dos 3"]),
    "f1_macro": f1_score(y_teste, predicoes["Estatico voto dos 3"], average="macro"),
})

# Junta as 6 configuracoes de selecao dinamica (uma coluna por metodo x
# roteamento), na mesma ordem das falas do teste
METODOS_DS = ["OLA", "KNORA-E", "KNORA-U"]
for roteamento in ["sem_roteamento", "com_roteamento"]:
    parte = ds[ds["roteamento"] == roteamento].set_index("KEY").reindex(teste["KEY"])
    if parte[METODOS_DS].isna().any().any():
        raise ValueError("ds_routing_test.csv nao cobre todas as falas do teste -- rode ds_routing.py de novo")
    for metodo in METODOS_DS:
        nome = f"{metodo} {roteamento.replace('_', ' ')}"
        predicoes[nome] = parte[metodo].astype(int).to_numpy()
        resumo.append({
            "metodo": nome, "candidato": "pool de 7", "f1_macro_cv_treino": None,
            "accuracy": accuracy_score(y_teste, predicoes[nome]),
            "f1_macro": f1_score(y_teste, predicoes[nome], average="macro"),
        })
resumo = pd.DataFrame(resumo)

Path("data/processed/eval").mkdir(parents=True, exist_ok=True)
predicoes.to_csv("data/processed/eval/static_multimodal_test.csv", index=False)
resumo.to_csv("data/processed/eval/static_multimodal_summary.csv", index=False)

print("Estaticos multimodais (hiperparametros da validacao cruzada no treino):")
for nome, linha in escolhidos:
    print(f"  {nome:24s} {linha['candidate_id']:40s} F1-macro CV = {linha['mean_f1']:.3f}")
print("\nResultados no teste (", len(teste), "falas ):")
print(resumo[["metodo", "accuracy", "f1_macro"]].round(4).to_string(index=False))
print("\nSalvo em: data/processed/eval/static_multimodal_test.csv")
print("Salvo em: data/processed/eval/static_multimodal_summary.csv")
