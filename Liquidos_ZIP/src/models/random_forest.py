"""
============================================================
 MODELO: RANDOM FOREST CLASSIFIER
============================================================
 Floresta Aleatória — conjunto de árvores de decisão.

 Por quê usar:
  - Robusto a ruído nos dados (sensores baratos têm ruído)
  - Não exige normalização (mas usamos mesmo assim)
  - Fornece "feature importance" nativa → mostra quais
    sensores são mais úteis para a classificação
  - Excelente baseline para dados tabulares

 Referência: Breiman, L. (2001). "Random Forests."
             Machine Learning, 45(1), 5-32.
============================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ML_PARAMS, RANDOM_SEED


def treinar_random_forest(X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray) -> dict:
    """
    Treina um classificador Random Forest.

    Args:
        X_train: Features de treino (normalizadas)
        y_train: Labels de treino
        X_val:   Features de validação
        y_val:   Labels de validação

    Returns:
        Dicionário com o modelo treinado e métricas
    """
    params = ML_PARAMS["random_forest"]

    print("\n  [Random Forest] Treinando...")
    print(f"    Parâmetros: {params}")

    modelo = RandomForestClassifier(**params)

    inicio = time.time()
    modelo.fit(X_train, y_train)
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
        "nome": "Random Forest",
        "modelo": modelo,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "tempo_treino": tempo_treino,
        "feature_importances": modelo.feature_importances_,
    }
