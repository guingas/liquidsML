"""
============================================================
 MODELO: XGBOOST CLASSIFIER
============================================================
 Extreme Gradient Boosting — estado da arte para dados tabulares.

 Por quê usar:
  - Melhor performance em competições de ML com dados tabulares
  - Captura interações complexas entre features (ex: a relação
    entre condutividade e temperatura)
  - Regularização embutida (evita overfitting)
  - Também fornece feature importance

 Referência: Chen, T., & Guestrin, C. (2016). "XGBoost: A
             Scalable Tree Boosting System." KDD 2016.
============================================================
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ML_PARAMS, RANDOM_SEED


def treinar_xgboost(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray) -> dict:
    """
    Treina um classificador XGBoost.

    Args:
        X_train: Features de treino (normalizadas)
        y_train: Labels de treino
        X_val:   Features de validação
        y_val:   Labels de validação

    Returns:
        Dicionário com o modelo treinado e métricas
    """
    params = ML_PARAMS["xgboost"]

    print("\n  [XGBoost] Treinando...")
    print(f"    Parâmetros: {params}")

    modelo = XGBClassifier(**params, use_label_encoder=False)

    inicio = time.time()
    modelo.fit(X_train, y_train, verbose=False)
    tempo_treino = time.time() - inicio

    # Avaliação
    y_pred_train = modelo.predict(X_train)
    y_pred_val = modelo.predict(X_val)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_val, y_pred_val)

    print(f"    Acurácia treino:    {acc_train:.4f}")
    print(f"    Acurácia validação: {acc_val:.4f}")
    print(f"    Tempo de treino:    {tempo_treino:.2f}s")

    return {
        "nome": "XGBoost",
        "modelo": modelo,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "tempo_treino": tempo_treino,
        "feature_importances": modelo.feature_importances_,
    }
