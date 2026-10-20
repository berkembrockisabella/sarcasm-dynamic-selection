"""
Treina e avalia o baseline Soft-MoE usando a pool congelada.

O gate é treinado no DSEL.
O TESTE é usado exclusivamente na avaliação final.

O MoE não recebe features explícitas de incongruência.

Execute na raiz:

    python -m src.moe.run_moe
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
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


from pool_selection.feature_loader import FeatureStore
from pool_training.load_frozen_pool import load_frozen_pool
from moe.moe import MixtureOfExperts
from moe.moe_expert_predictions import get_expert_probabilities


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATA = ROOT_DIR / "data"
SPLITS = DATA / "splits"
OUT = DATA / "processed" / "moe"

MODALITIES = (
    "text",
    "context",
    "audio",
    "visual",
)


# ============================================================
# FEATURES DO GATE
# ============================================================

def gate_matrix(
    store,
    keys,
    scalers=None,
    fit=False,
):
    """
    Monta a matriz multimodal usada pelo gate.

    Cada modalidade é padronizada separadamente.
    O scaler deve ser ajustado apenas no TRAIN.
    """

    blocks = []
    new_scalers = {}

    for modality in MODALITIES:

        X = store.matrix(
            keys=keys,
            combo=(modality,),
        )

        scaler = StandardScaler()

        if scalers is not None:

            scaler = scalers[
                modality
            ]

            block = scaler.transform(
                X
            )

        elif fit:

            block = scaler.fit_transform(
                X
            )

        else:

            raise ValueError(
                "Informe scalers ou fit=True."
            )

        blocks.append(
            block.astype(
                np.float32
            )
        )

        new_scalers[
            modality
        ] = scaler

    return (
        np.hstack(blocks),
        new_scalers,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Verificação dos splits
    # --------------------------------------------------------

    for name in (
        "train",
        "dsel",
        "test",
    ):

        path = (
            SPLITS
            / f"{name}.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Partição ausente: {path}"
            )

    # --------------------------------------------------------
    # Carregamento dos dados
    # --------------------------------------------------------

    train = pd.read_csv(
        SPLITS
        / "train.csv"
    )

    dsel = pd.read_csv(
        SPLITS
        / "dsel.csv"
    )

    test = pd.read_csv(
        SPLITS
        / "test.csv"
    )

    # Normaliza os identificadores e labels
    for frame in (
        train,
        dsel,
        test,
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

    # --------------------------------------------------------
    # Features do gate
    # --------------------------------------------------------

    store = FeatureStore()

    # O scaler é ajustado somente no TRAIN.
    _, scalers = gate_matrix(
        store,
        train[
            "KEY"
        ].tolist(),
        fit=True,
    )

    X_dsel, _ = gate_matrix(
        store,
        dsel[
            "KEY"
        ].tolist(),
        scalers=scalers,
    )

    X_test, _ = gate_matrix(
        store,
        test[
            "KEY"
        ].tolist(),
        scalers=scalers,
    )

    # --------------------------------------------------------
    # Pool congelada
    # --------------------------------------------------------

    pool = load_frozen_pool()

    # Probabilidades produzidas por cada especialista
    # no DSEL e no TEST.
    P_dsel, member_ids = (
        get_expert_probabilities(
            pool,
            dsel[
                "KEY"
            ].tolist(),
            store,
        )
    )

    P_test, test_member_ids = (
        get_expert_probabilities(
            pool,
            test[
                "KEY"
            ].tolist(),
            store,
        )
    )

    if (
        member_ids
        != test_member_ids
    ):

        raise ValueError(
            "A ordem dos especialistas "
            "difere entre DSEL e TEST."
        )

    # --------------------------------------------------------
    # Soft-MoE
    # --------------------------------------------------------

    model = MixtureOfExperts(
        input_dim=X_dsel.shape[1],
        n_experts=P_dsel.shape[1],
        hidden_dim=32,
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        max_epochs=200,
        patience=20,
        random_state=42,
    )

    model.fit(
        X_dsel,
        P_dsel,
        dsel[
            "Sarcasm"
        ].to_numpy(),
    )

    # --------------------------------------------------------
    # Avaliação no TEST
    # --------------------------------------------------------

    prob = model.predict_proba(
        X_test,
        P_test,
    )

    pred = (
        prob >= 0.5
    ).astype(int)

    y = (
        test[
            "Sarcasm"
        ]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    metrics = {
        "metodo":
            "Soft-MoE",

        "n_experts":
            len(pool),

        "n_dsel":
            len(dsel),

        "n_test":
            len(test),

        "accuracy":
            accuracy_score(
                y,
                pred,
            ),

        "precision":
            precision_score(
                y,
                pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                pred,
                zero_division=0,
            ),

        "f1_macro":
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            ),

        "roc_auc":
            (
                roc_auc_score(
                    y,
                    prob,
                )
                if len(
                    np.unique(y)
                ) == 2
                else np.nan
            ),
    }

    # --------------------------------------------------------
    # Pesos do gate
    # --------------------------------------------------------

    weights = model.get_gate_weights(
        X_test
    )

    # --------------------------------------------------------
    # Diretório de saída
    # --------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Predições por instância
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "KEY":
                test[
                    "KEY"
                ],

            "Sarcasm":
                y,

            "prediction":
                pred,

            "prob_sarcasm":
                prob,

            **{
                f"weight_{mid}":
                    weights[
                        :,
                        i
                    ]

                for i, mid
                in enumerate(
                    member_ids
                )
            },
        }
    ).to_csv(
        OUT
        / "moe_test.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Resultados completos
    # --------------------------------------------------------

    pd.DataFrame(
        [metrics]
    ).to_csv(
        OUT
        / "moe_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    summary_metrics = {
        key: value
        for key, value
        in metrics.items()
        if key
        not in {
            "n_experts",
            "n_dsel",
            "n_test",
        }
    }

    pd.DataFrame(
        [summary_metrics]
    ).to_csv(
        OUT
        / "moe_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Histórico de treinamento
    # --------------------------------------------------------

    pd.DataFrame(
        model.history
    ).to_csv(
        OUT
        / "moe_training_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Configuração
    # --------------------------------------------------------

    config = {
        "architecture":
            "soft_mixture_of_experts",

        "gate_input_modalities":
            list(
                MODALITIES
            ),

        "explicit_incongruence":
            False,

        "gate_training_split":
            "DSEL",

        "evaluation_split":
            "TEST",

        "expert_ids":
            member_ids,

        "threshold":
            0.5,

        "random_state":
            42,
    }

    (
        OUT
        / "moe_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Terminal
    # --------------------------------------------------------

    print(
        "Soft-MoE concluído"
    )

    print(
        pd.DataFrame(
            [summary_metrics]
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        f"Resultados: {OUT}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()