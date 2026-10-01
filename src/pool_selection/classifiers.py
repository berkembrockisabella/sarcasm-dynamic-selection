'''
Monta o pipeline dos classificadores: SVM, Regressão Logística e Random Forest
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import RANDOM_STATE

def build_classifier(name):
    match name:
        case "svm":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE,)),
            ])

        case "logistic_regression":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE,)),
            ])

        case "random_forest":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,)),
            ])

        case _:
            raise ValueError(f"Classificador desconhecido: {name}")
