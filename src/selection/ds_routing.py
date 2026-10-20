"""
Experimento:
OLA, KNORA-E e KNORA-U, cada um sem e com roteamento por incongruência.

A pool é sempre a mesma (7 especialistas).

- sem roteamento:
  os k vizinhos da instância de teste são procurados
  no DSEL inteiro.

- com roteamento:
  os k vizinhos são procurados somente entre as
  instâncias do DSEL do mesmo grupo.

Grupos:
A = as 4 modalidades têm a mesma polaridade.
B = pelo menos uma modalidade difere.

A coluna "grupo" vem de incongruence.csv.

Referências:
OLA (Woods et al., 1997)
KNORA-E / KNORA-U (Ko, Sabourin & Britto, 2008)
"""

import sys
from pathlib import Path

# ============================================================
# IMPORTS ENTRE PACOTES DE SRC
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from pool_selection.feature_loader import FeatureStore
from pool_training.load_frozen_pool import load_frozen_pool


# ============================================================
# CONFIGURAÇÕES
# ============================================================

K = 7

MODALIDADES = [
    "text",
    "context",
    "audio",
    "visual",
]

METODOS = [
    "OLA",
    "KNORA-E",
    "KNORA-U",
]

ROTEAMENTOS = [
    "sem_roteamento",
    "com_roteamento",
]

GRUPOS = [
    "A",
    "B",
]

OUT_DIR = Path(
    "data/processed/selection"
)


# ============================================================
# CARREGAMENTO DOS SPLITS
# ============================================================

treino = pd.read_csv(
    "data/splits/train.csv"
)

dsel = pd.read_csv(
    "data/splits/dsel.csv"
)

teste = pd.read_csv(
    "data/splits/test.csv"
)

grupos = pd.read_csv(
    "data/processed/incongruence.csv"
)[
    [
        "KEY",
        "grupo",
    ]
]


# ============================================================
# ADICIONA GRUPOS
# ============================================================

dsel = dsel.merge(
    grupos,
    on="KEY",
    how="left",
)

teste = teste.merge(
    grupos,
    on="KEY",
    how="left",
)


# Sem grupo não é possível fazer o roteamento corretamente.
for nome, tabela in [
    ("DSEL", dsel),
    ("teste", teste),
]:

    if tabela["grupo"].isna().any():
        raise ValueError(
            f"{nome}: instâncias sem grupo "
            "em incongruence.csv"
        )


# ============================================================
# LABELS
# ============================================================

y_dsel = (
    dsel["Sarcasm"]
    .astype(int)
    .to_numpy()
)

y_teste = (
    teste["Sarcasm"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# POOL CONGELADA
# ============================================================

store = FeatureStore()

pool = load_frozen_pool()

MEMBROS = [
    membro["member_id"]
    for membro in pool
]


# ============================================================
# PREDIÇÕES DOS MEMBROS
# ============================================================

pred_dsel = np.zeros(
    (
        len(dsel),
        len(pool),
    ),
    dtype=int,
)

pred_teste = np.zeros(
    (
        len(teste),
        len(pool),
    ),
    dtype=int,
)

proba_teste = np.zeros(
    (
        len(teste),
        len(pool),
    ),
    dtype=float,
)


for j, membro in enumerate(pool):

    combo = tuple(
        parte.strip()
        for parte in str(
            membro["modalities"]
        ).split("+")
    )

    X_d = store.matrix(
        dsel["KEY"].tolist(),
        combo,
    )

    X_t = store.matrix(
        teste["KEY"].tolist(),
        combo,
    )

    pred_dsel[:, j] = (
        membro["model"]
        .predict(X_d)
    )

    pred_teste[:, j] = (
        membro["model"]
        .predict(X_t)
    )

    if not hasattr(
        membro["model"],
        "predict_proba",
    ):
        raise TypeError(
            f"O membro {membro['member_id']} "
            "não possui predict_proba()."
        )

    proba = (
        membro["model"]
        .predict_proba(X_t)
    )

    classes = np.asarray(
        membro["model"].classes_
    )

    positive_index = np.where(
        classes == 1
    )[0]

    if len(positive_index) != 1:
        raise ValueError(
            "Classe positiva 1 não encontrada "
            f"no membro {membro['member_id']}."
        )

    proba_teste[:, j] = (
        proba[
            :,
            positive_index[0]
        ]
    )


# ============================================================
# REGIÃO DE COMPETÊNCIA
# ============================================================

# As quatro modalidades são concatenadas.
#
# Para cada modalidade:
# - StandardScaler ajustado somente no TRAIN.
# - transformação em DSEL e TEST.
# - normalização L2.
#
# Isso evita que uma modalidade domine a distância apenas
# por possuir maior escala.

blocos_dsel = []
blocos_teste = []


for modalidade in MODALIDADES:

    X_train_mod = store.matrix(
        treino["KEY"].tolist(),
        (modalidade,),
    )

    X_dsel_mod = store.matrix(
        dsel["KEY"].tolist(),
        (modalidade,),
    )

    X_test_mod = store.matrix(
        teste["KEY"].tolist(),
        (modalidade,),
    )

    scaler = StandardScaler().fit(
        X_train_mod
    )

    bloco_d = scaler.transform(
        X_dsel_mod
    )

    bloco_t = scaler.transform(
        X_test_mod
    )

    blocos_dsel.append(
        normalize(bloco_d)
    )

    blocos_teste.append(
        normalize(bloco_t)
    )


X_dsel = np.hstack(
    blocos_dsel
)

X_teste = np.hstack(
    blocos_teste
)

grupo_dsel = (
    dsel["grupo"]
    .to_numpy()
)

grupo_teste = (
    teste["grupo"]
    .to_numpy()
)


# ============================================================
# COMPETÊNCIA NO DSEL
# ============================================================

acerto_dsel = (
    pred_dsel
    ==
    y_dsel[:, None]
).astype(int)

acuracia_global = (
    acerto_dsel.mean(
        axis=0
    )
)


# ============================================================
# VIZINHOS
# ============================================================

vizinhos = {
    "com_roteamento": []
}


# ------------------------------------------------------------
# Sem roteamento
# ------------------------------------------------------------

busca = NearestNeighbors(
    n_neighbors=K
).fit(
    X_dsel
)

vizinhos[
    "sem_roteamento"
] = list(
    busca.kneighbors(
        X_teste,
        return_distance=False,
    )
)


# ------------------------------------------------------------
# Com roteamento
# ------------------------------------------------------------

for grupo in GRUPOS:

    indices_grupo = np.where(
        grupo_dsel == grupo
    )[0]

    if len(indices_grupo) == 0:
        raise ValueError(
            f"Não existem instâncias "
            f"do grupo {grupo} no DSEL."
        )

    k_grupo = min(
        K,
        len(indices_grupo),
    )

    busca_grupo = (
        NearestNeighbors(
            n_neighbors=k_grupo
        )
        .fit(
            X_dsel[
                indices_grupo
            ]
        )
    )

    vizinhos[
        f"grupo_{grupo}"
    ] = (
        indices_grupo,
        busca_grupo,
    )


for i in range(
    len(teste)
):

    grupo_atual = (
        grupo_teste[i]
    )

    indices_grupo, busca_grupo = (
        vizinhos[
            f"grupo_{grupo_atual}"
        ]
    )

    locais = (
        busca_grupo
        .kneighbors(
            X_teste[
                i:i + 1
            ],
            return_distance=False,
        )[0]
    )

    vizinhos[
        "com_roteamento"
    ].append(
        indices_grupo[
            locais
        ]
    )


# ============================================================
# VOTO
# ============================================================

def voto(
    i,
    escolhidos,
    pesos=None,
):
    """
    Voto majoritário entre os membros escolhidos.

    Se houver empate, usa a média das probabilidades
    de sarcasmo com limiar 0.5.
    """

    escolhidos = np.asarray(
        escolhidos,
        dtype=int,
    )

    if pesos is None:
        pesos = np.ones(
            len(escolhidos),
            dtype=float,
        )

    pesos = np.asarray(
        pesos,
        dtype=float,
    )

    votos_sarc = (
        pesos
        *
        pred_teste[
            i,
            escolhidos
        ]
    ).sum()

    votos_nao = (
        pesos
        *
        (
            1
            -
            pred_teste[
                i,
                escolhidos
            ]
        )
    ).sum()

    if votos_sarc != votos_nao:
        return int(
            votos_sarc
            >
            votos_nao
        )

    prob_media = (
        proba_teste[
            i,
            escolhidos
        ]
        .mean()
    )

    return int(
        prob_media >= 0.5
    )


# ============================================================
# SELEÇÃO DINÂMICA
# ============================================================

TODOS = np.arange(
    len(pool)
)

linhas = []


for roteamento in ROTEAMENTOS:

    for i in range(
        len(teste)
    ):

        viz = (
            vizinhos[
                roteamento
            ][i]
        )

        acertos_locais = (
            acerto_dsel[
                viz
            ]
        )

        linha = {
            "KEY":
                teste.loc[
                    i,
                    "KEY",
                ],

            "grupo":
                grupo_teste[i],

            "Sarcasm":
                int(
                    y_teste[i]
                ),

            "roteamento":
                roteamento,

            "k_usado":
                len(viz),
        }

        # ====================================================
        # OLA
        # ====================================================

        competencia = (
            acertos_locais.mean(
                axis=0
            )
        )

        melhor = max(
            TODOS,
            key=lambda j: (
                competencia[j],
                acuracia_global[j],
                -j,
            ),
        )

        linha["OLA"] = int(
            pred_teste[
                i,
                melhor
            ]
        )

        linha[
            "OLA_score"
        ] = float(
            proba_teste[
                i,
                melhor
            ]
        )

        linha[
            "OLA_membros"
        ] = MEMBROS[
            melhor
        ]

        # ====================================================
        # KNORA-E
        # ====================================================

        escolhidos = TODOS

        for k_atual in range(
            len(viz),
            0,
            -1,
        ):

            acertam_todos = np.where(
                acertos_locais[
                    :k_atual
                ].all(
                    axis=0
                )
            )[0]

            if len(
                acertam_todos
            ) > 0:

                escolhidos = (
                    acertam_todos
                )

                break

        linha[
            "KNORA-E"
        ] = voto(
            i,
            escolhidos,
        )

        linha[
            "KNORA-E_score"
        ] = float(
            proba_teste[
                i,
                escolhidos
            ].mean()
        )

        linha[
            "KNORA-E_membros"
        ] = "+".join(
            MEMBROS[j]
            for j in escolhidos
        )

        # ====================================================
        # KNORA-U
        # ====================================================

        pesos = (
            acertos_locais.sum(
                axis=0
            )
        )

        if pesos.sum() > 0:

            usados = np.where(
                pesos > 0
            )[0]

            pesos_usados = (
                pesos[
                    usados
                ]
                .astype(float)
            )

        else:

            usados = TODOS

            pesos_usados = np.ones(
                len(TODOS),
                dtype=float,
            )

        linha[
            "KNORA-U"
        ] = voto(
            i,
            usados,
            pesos_usados,
        )

        linha[
            "KNORA-U_score"
        ] = float(
            np.average(
                proba_teste[
                    i,
                    usados
                ],
                weights=pesos_usados,
            )
        )

        linha[
            "KNORA-U_membros"
        ] = "+".join(
            MEMBROS[j]
            for j in usados
        )

        linhas.append(
            linha
        )


resultado = pd.DataFrame(
    linhas
)


# ============================================================
# REFERÊNCIAS
# ============================================================

melhor_global = int(
    np.argmax(
        acuracia_global
    )
)

referencias = {
    (
        f"Melhor membro "
        f"({MEMBROS[melhor_global]})"
    ): {
        "pred":
            pred_teste[
                :,
                melhor_global
            ],

        "score":
            proba_teste[
                :,
                melhor_global
            ],
    },

    "Voto da pool inteira": {
        "pred":
            np.array(
                [
                    voto(
                        i,
                        TODOS
                    )
                    for i
                    in range(
                        len(teste)
                    )
                ]
            ),

        "score":
            proba_teste.mean(
                axis=1
            ),
    },
}


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(
    y_true,
    y_pred,
    y_score,
):
    """
    Calcula métricas de classificação.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    y_score = np.asarray(
        y_score
    )

    metricas = {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1_macro":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "roc_auc":
            (
                roc_auc_score(
                    y_true,
                    y_score,
                )
                if len(
                    np.unique(
                        y_true
                    )
                ) == 2
                else np.nan
            ),
    }

    return metricas


# ============================================================
# RESUMO GLOBAL
# ============================================================

resumo = []


for roteamento in ROTEAMENTOS:

    parte = resultado[
        resultado[
            "roteamento"
        ]
        ==
        roteamento
    ]

    for metodo in METODOS:

        resumo.append(
            {
                "metodo":
                    f"{metodo} "
                    f"{roteamento}",

                **calcular_metricas(
                    parte[
                        "Sarcasm"
                    ].to_numpy(),

                    parte[
                        metodo
                    ].to_numpy(),

                    parte[
                        f"{metodo}_score"
                    ].to_numpy(),
                ),
            }
        )


# ============================================================
# REFERÊNCIAS NO RESUMO
# ============================================================

for nome, dados in (
    referencias.items()
):

    resumo.append(
        {
            "metodo":
                nome,

            **calcular_metricas(
                y_teste,
                dados["pred"],
                dados["score"],
            ),
        }
    )


# ============================================================
# ORÁCULO
# ============================================================

oraculo_por_instancia = (
    pred_teste
    ==
    y_teste[:, None]
).any(
    axis=1
)

oraculo = (
    oraculo_por_instancia
    .mean()
)

resumo.append(
    {
        "metodo":
            "Oraculo (algum membro acerta)",

        "accuracy":
            oraculo,

        "precision":
            np.nan,

        "recall":
            np.nan,

        "f1":
            np.nan,

        "f1_macro":
            np.nan,

        "roc_auc":
            np.nan,
    }
)


resumo = pd.DataFrame(
    resumo
)


# ============================================================
# MÉTRICAS POR GRUPO
# ============================================================

metricas_grupo = []


for roteamento in ROTEAMENTOS:

    parte = resultado[
        resultado[
            "roteamento"
        ]
        ==
        roteamento
    ]

    for grupo in GRUPOS:

        g = parte[
            parte[
                "grupo"
            ]
            ==
            grupo
        ]

        if len(g) == 0:
            continue

        for metodo in METODOS:

            metricas_grupo.append(
                {
                    "metodo":
                        f"{metodo} "
                        f"{roteamento}",

                    "grupo":
                        grupo,

                    **calcular_metricas(
                        g[
                            "Sarcasm"
                        ].to_numpy(),

                        g[
                            metodo
                        ].to_numpy(),

                        g[
                            f"{metodo}_score"
                        ].to_numpy(),
                    ),
                }
            )


# ============================================================
# REFERÊNCIAS POR GRUPO
# ============================================================

for nome, dados in (
    referencias.items()
):

    for grupo in GRUPOS:

        mascara = (
            grupo_teste
            ==
            grupo
        )

        if not mascara.any():
            continue

        metricas_grupo.append(
            {
                "metodo":
                    nome,

                "grupo":
                    grupo,

                **calcular_metricas(
                    y_teste[
                        mascara
                    ],

                    dados[
                        "pred"
                    ][
                        mascara
                    ],

                    dados[
                        "score"
                    ][
                        mascara
                    ],
                ),
            }
        )


# ============================================================
# ORÁCULO POR GRUPO
# ============================================================

for grupo in GRUPOS:

    mascara = (
        grupo_teste
        ==
        grupo
    )

    if mascara.any():

        acuracia_oraculo_grupo = (
            (
                pred_teste[
                    mascara
                ]
                ==
                y_teste[
                    mascara,
                    None
                ]
            )
            .any(
                axis=1
            )
            .mean()
        )

    else:

        acuracia_oraculo_grupo = (
            np.nan
        )

    metricas_grupo.append(
        {
            "metodo":
                "Oraculo "
                "(algum membro acerta)",

            "grupo":
                grupo,

            "accuracy":
                acuracia_oraculo_grupo,

            "precision":
                np.nan,

            "recall":
                np.nan,

            "f1":
                np.nan,

            "f1_macro":
                np.nan,

            "roc_auc":
                np.nan,
        }
    )


metricas_grupo = pd.DataFrame(
    metricas_grupo,
    columns=[
        "metodo",
        "grupo",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "roc_auc",
    ],
)


# ============================================================
# SALVAR RESULTADOS
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

resultado.to_csv(
    OUT_DIR
    / "ds_routing_test.csv",
    index=False,
)

resumo.to_csv(
    OUT_DIR
    / "ds_routing_summary.csv",
    index=False,
)

metricas_grupo.to_csv(
    OUT_DIR
    / "ds_routing_group.csv",
    index=False,
)


# ============================================================
# SAÍDA NO TERMINAL
# ============================================================

print(
    "Pool:",
    ", ".join(
        (
            f"{m['member_id']}="
            f"{m['modalities']}/"
            f"{m['classifier']}"
        )
        for m in pool
    )
)

print(
    "\nAcurácia global "
    "de cada membro no DSEL:"
)

print(
    dict(
        zip(
            MEMBROS,
            acuracia_global.round(3),
        )
    )
)

print(
    "\nDSEL por grupo:",
    dsel[
        "grupo"
    ]
    .value_counts()
    .to_dict(),
)

print(
    "Teste por grupo:",
    teste[
        "grupo"
    ]
    .value_counts()
    .to_dict(),
)

print(
    f"\nk = {K}"
)

print(
    f"Oráculo no teste: "
    f"{oraculo:.4f}"
)

print(
    "\nResultados no teste:"
)

print(
    resumo
    .round(4)
    .to_string(
        index=False
    )
)

print(
    "\nResultados por grupo:"
)

print(
    metricas_grupo
    .round(4)
    .to_string(
        index=False
    )
)

print(
    "\nArquivos salvos:"
)

print(
    OUT_DIR
    / "ds_routing_test.csv"
)

print(
    OUT_DIR
    / "ds_routing_summary.csv"
)

print(
    OUT_DIR
    / "ds_routing_group.csv"
)