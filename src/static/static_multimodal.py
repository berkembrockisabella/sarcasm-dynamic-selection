"""
Avalia baselines estáticos multimodais.

Os classificadores recebem as quatro modalidades:
- texto
- contexto
- áudio
- visual

Os modelos são treinados somente no TRAIN e avaliados no TEST.

Os hiperparâmetros utilizados são os obtidos anteriormente pela
validação cruzada realizada no TRAIN durante a formação da pool.

Execução:

    python -m src.static.static_multimodal
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CAMINHOS E IMPORTS
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from pool_selection.classifiers import build_classifier
from pool_selection.feature_loader import FeatureStore


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATA = ROOT_DIR / "data"

SPLITS = DATA / "splits"

POOL_RESULTS = (
    DATA
    / "results"
    / "pool"
    / "pool_selection"
)

OUT = (
    DATA
    / "processed"
    / "static"
)

TODAS = "text+context+audio+visual"

NOMES = {
    "svm":
        "SVM",

    "logistic_regression":
        "Regressao logistica",

    "random_forest":
        "Random forest",
}


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(
    y_true,
    y_pred,
    y_score=None,
):
    """
    Calcula as métricas utilizadas nos experimentos.
    """

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
    }

    if (
        y_score is not None
        and len(np.unique(y_true)) == 2
    ):

        metricas[
            "roc_auc"
        ] = roc_auc_score(
            y_true,
            y_score,
        )

    else:

        metricas[
            "roc_auc"
        ] = np.nan

    return metricas


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Verificação dos arquivos
    # --------------------------------------------------------

    train_path = (
        SPLITS
        / "train.csv"
    )

    test_path = (
        SPLITS
        / "test.csv"
    )

    cv_path = (
        POOL_RESULTS
        / "cv_results.csv"
    )

    for path in (
        train_path,
        test_path,
        cv_path,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Arquivo ausente: {path}"
            )

    # --------------------------------------------------------
    # Dados
    # --------------------------------------------------------

    treino = pd.read_csv(
        train_path
    )

    teste = pd.read_csv(
        test_path
    )

    cv = pd.read_csv(
        cv_path
    )

    for frame in (
        treino,
        teste,
    ):

        frame[
            "KEY"
        ] = (
            frame[
                "KEY"
            ]
            .astype(str)
        )

        frame[
            "Sarcasm"
        ] = (
            frame[
                "Sarcasm"
            ]
            .astype(int)
        )

    y_treino = (
        treino[
            "Sarcasm"
        ]
        .to_numpy()
    )

    y_teste = (
        teste[
            "Sarcasm"
        ]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Seleciona os três classificadores estáticos
    # --------------------------------------------------------

    candidatos = (
        cv[
            cv[
                "modalities"
            ]
            ==
            TODAS
        ]
        .sort_values(
            "mean_f1",
            ascending=False,
        )
    )

    escolhidos = []

    for _, linha in (
        candidatos.iterrows()
    ):

        classifier = (
            linha[
                "classifier"
            ]
        )

        if classifier not in NOMES:
            continue

        escolhidos.append(
            (
                f"Estatico "
                f"{NOMES[classifier]}",
                linha,
            )
        )

    if len(escolhidos) != 3:

        raise ValueError(
            "cv_results.csv deveria ter "
            f"3 candidatos para {TODAS}, "
            f"mas foram encontrados "
            f"{len(escolhidos)}."
        )

    # --------------------------------------------------------
    # Feature Store
    # --------------------------------------------------------

    store = FeatureStore()

    # --------------------------------------------------------
    # Predições
    # --------------------------------------------------------

    predicoes = pd.DataFrame(
        {
            "KEY":
                teste[
                    "KEY"
                ],

            "Sarcasm":
                y_teste,
        }
    )

    resumo = []

    # ========================================================
    # MODELOS INDIVIDUAIS
    # ========================================================

    for nome, linha in escolhidos:

        combo = tuple(
            parte.strip()
            for parte
            in linha[
                "modalities"
            ].split("+")
        )

        # ----------------------------------------------------
        # Modelo
        # ----------------------------------------------------

        modelo = build_classifier(
            linha[
                "classifier"
            ]
        )

        best_params = json.loads(
            linha[
                "best_params"
            ]
        )

        modelo.set_params(
            **best_params
        )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        X_train = store.matrix(
            treino[
                "KEY"
            ].tolist(),
            combo,
        )

        X_test = store.matrix(
            teste[
                "KEY"
            ].tolist(),
            combo,
        )

        # ----------------------------------------------------
        # Treinamento
        # ----------------------------------------------------

        modelo.fit(
            X_train,
            y_treino,
        )

        # ----------------------------------------------------
        # Predições
        # ----------------------------------------------------

        pred = modelo.predict(
            X_test
        )

        predicoes[
            nome
        ] = pred

        # ----------------------------------------------------
        # Probabilidade para ROC-AUC
        # ----------------------------------------------------

        prob = None

        if hasattr(
            modelo,
            "predict_proba",
        ):

            classes = np.asarray(
                modelo.classes_
            )

            indice_positivo = np.where(
                classes == 1
            )[0]

            if (
                len(
                    indice_positivo
                )
                ==
                1
            ):

                prob = (
                    modelo
                    .predict_proba(
                        X_test
                    )[
                        :,
                        indice_positivo[
                            0
                        ]
                    ]
                )

        # ----------------------------------------------------
        # Métricas
        # ----------------------------------------------------

        metricas = calcular_metricas(
            y_teste,
            pred,
            prob,
        )

        resumo.append(
            {
                "metodo":
                    (
                        f"{nome} "
                        f"({linha['candidate_id']})"
                    ),

                "f1_macro_cv_treino":
                    linha[
                        "mean_f1"
                    ],

                **metricas,
            }
        )

    # ========================================================
    # VOTO MAJORITÁRIO DOS 3
    # ========================================================

    nomes_individuais = [
        nome
        for nome, _
        in escolhidos
    ]

    voto = (
        predicoes[
            nomes_individuais
        ]
        .sum(
            axis=1
        )
        >=
        2
    ).astype(int)

    predicoes[
        "Estatico voto dos 3"
    ] = voto

    # Média das probabilidades não é necessária para o voto
    # majoritário. ROC-AUC permanece NaN para esse baseline.

    metricas_voto = calcular_metricas(
        y_teste,
        voto,
        None,
    )

    resumo.append(
        {
            "metodo":
                (
                    "Estatico voto dos 3 "
                    "(voto SVM + RL + RF)"
                ),

            "f1_macro_cv_treino":
                np.nan,

            **metricas_voto,
        }
    )

    # ========================================================
    # DATAFRAME DE RESULTADOS
    # ========================================================

    resumo = pd.DataFrame(
        resumo
    )

    # ========================================================
    # SAÍDAS
    # ========================================================

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Predições por instância
    # --------------------------------------------------------

    predicoes.to_csv(
        OUT
        / "static_multimodal_test.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Resultado detalhado
    #
    # Mantém a métrica da validação cruzada do TRAIN.
    # --------------------------------------------------------

    resumo.to_csv(
        OUT
        / "static_multimodal.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Resumo padronizado
    #
    # Remove somente f1_macro_cv_treino.
    # --------------------------------------------------------

    summary = resumo.drop(
        columns=[
            "f1_macro_cv_treino"
        ]
    )

    summary.to_csv(
        OUT
        / "static_multimodal_summary.csv",
        index=False,
    )

    # ========================================================
    # TERMINAL
    # ========================================================

    print(
        "Estaticos multimodais "
        "(hiperparametros da validacao "
        "cruzada no treino):"
    )

    for nome, linha in escolhidos:

        print(
            f"  {nome:24s} "
            f"{linha['candidate_id']:40s} "
            f"F1-macro CV = "
            f"{linha['mean_f1']:.3f}"
        )

    print(
        f"\nResultados no teste "
        f"({len(teste)} falas):"
    )

    print(
        summary[
            [
                "metodo",
                "accuracy",
                "f1_macro",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nArquivos salvos:"
    )

    print(
        OUT
        / "static_multimodal_test.csv"
    )

    print(
        OUT
        / "static_multimodal.csv"
    )

    print(
        OUT
        / "static_multimodal_summary.csv"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()