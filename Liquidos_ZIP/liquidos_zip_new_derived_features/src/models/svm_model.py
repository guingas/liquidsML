"""
============================================================
 MODELO: SUPPORT VECTOR MACHINE (SVM)
============================================================
 Máquina de Vetores de Suporte — encontra o hiperplano ótimo.

 Por quê usar:
  - Excelente para datasets de tamanho médio
  - Funciona muito bem em alta dimensionalidade
  - O kernel RBF captura fronteiras de decisão não-lineares
  - Se funciona aqui, em ESP32 pode ser implementado com
    pouquíssima memória (basta salvar os vetores de suporte)

 Referência: Cortes, C., & Vapnik, V. (1995). "Support-vector
             networks." Machine Learning, 20(3), 273-297.
============================================================
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ML_PARAMS, RANDOM_SEED


def treinar_svm(X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray) -> dict:
    """
    Treina um classificador SVM com kernel RBF.

    Args:
        X_train: Features de treino (normalizadas — essencial para SVM!)
        y_train: Labels de treino
        X_val:   Features de validação
        y_val:   Labels de validação

    Returns:
        Dicionário com o modelo treinado e métricas
    """
    params = ML_PARAMS["svm"]

    print("\n  [SVM] Treinando...")
    print(f"    Parâmetros: {params}")

    modelo = SVC(**params, probability=True)

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
    print(f"    Vetores de suporte: {modelo.n_support_}")
    print(f"    Tempo de treino:    {tempo_treino:.2f}s")

    return {
        "nome": "SVM (RBF)",
        "modelo": modelo,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "tempo_treino": tempo_treino,
    }
