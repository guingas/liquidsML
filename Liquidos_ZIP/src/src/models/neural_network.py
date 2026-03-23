"""
============================================================
 MODELO: REDE NEURAL (MULTILAYER PERCEPTRON - MLP)
============================================================
 Rede Neural Densa — aprende representações não-lineares.

 Por quê usar:
  - Captura padrões complexos que modelos lineares perdem
  - Escalável para muitas features
  - Pode ser convertida para TensorFlow Lite e rodar no ESP32
  - Boa para quando temos muitos dados de sensores

 Arquitetura:
   Input(N_features) → Dense(128) → Dense(64) → Dense(32) → Output(N_classes)
   Com BatchNorm e Dropout para regularização.

 Referência: Goodfellow, I. et al. (2016). "Deep Learning." MIT Press.
============================================================
"""

import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ML_PARAMS, RANDOM_SEED


def treinar_rede_neural(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        n_classes: int) -> dict:
    """
    Treina uma Rede Neural (MLP) com TensorFlow/Keras.

    Args:
        X_train:   Features de treino
        y_train:   Labels de treino
        X_val:     Features de validação
        y_val:     Labels de validação
        n_classes: Número de classes (2 para tipo, 7 para subtipo)

    Returns:
        Dicionário com o modelo treinado e métricas
    """
    # Import do TensorFlow aqui para não travar se não estiver instalado
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(RANDOM_SEED)

    params = ML_PARAMS["neural_network"]
    n_features = X_train.shape[1]

    print("\n  [Rede Neural MLP] Treinando...")
    print(f"    Arquitetura: {n_features} → {params['hidden_layers']} → {n_classes}")

    # --- Construção do modelo ---
    modelo = Sequential(name="MLP_LiquidClassifier")
    modelo.add(Input(shape=(n_features,)))

    for i, units in enumerate(params["hidden_layers"]):
        modelo.add(Dense(units, activation="relu", name=f"dense_{i}"))
        modelo.add(BatchNormalization(name=f"bn_{i}"))
        modelo.add(Dropout(0.3, name=f"dropout_{i}"))

    # Camada de saída
    if n_classes == 2:
        modelo.add(Dense(1, activation="sigmoid", name="output"))
        loss = "binary_crossentropy"
    else:
        modelo.add(Dense(n_classes, activation="softmax", name="output"))
        loss = "sparse_categorical_crossentropy"

    modelo.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss=loss,
        metrics=["accuracy"],
    )

    # --- Treinamento com Early Stopping ---
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )

    inicio = time.time()
    historico = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[early_stop],
        verbose=0,
    )
    tempo_treino = time.time() - inicio

    # --- Avaliação ---
    _, acc_train = modelo.evaluate(X_train, y_train, verbose=0)
    _, acc_val = modelo.evaluate(X_val, y_val, verbose=0)

    print(f"    Acurácia treino:    {acc_train:.4f}")
    print(f"    Acurácia validação: {acc_val:.4f}")
    print(f"    Épocas treinadas:   {len(historico.history['loss'])}")
    print(f"    Tempo de treino:    {tempo_treino:.2f}s")

    return {
        "nome": "Rede Neural (MLP)",
        "modelo": modelo,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "tempo_treino": tempo_treino,
        "historico": historico.history,
    }
