"""
============================================================
 MODELO: REDE NEURAL CONVOLUCIONAL 1D (CNN-1D)
============================================================
 CNN aplicada a dados de sensores (não imagens).

 Por quê usar:
  - Os 11 canais espectrais do AS7341 são um "sinal 1D"
    similar a um espectro de áudio ou frequência.
  - A CNN detecta padrões locais no espectro (ex: pico no
    verde que indica cerveja lager).
  - Demonstra que convoluções não são só para imagens.
  - Também pode ser convertida para TensorFlow Lite (ESP32).

 Arquitetura:
   Input(N_features, 1) → Conv1D(32) → Conv1D(64) → Flatten → Dense → Output

 Referência: Kiranyaz, S. et al. (2021). "1D Convolutional
             Neural Networks and Applications." Mech. Syst.
             Signal Process.
============================================================
"""

import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ML_PARAMS, RANDOM_SEED


def treinar_cnn(X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                n_classes: int) -> dict:
    """
    Treina uma CNN 1D para classificação de líquidos.

    Os dados dos sensores são tratados como um "sinal 1D" onde
    cada feature é um ponto no "espectro" do líquido.

    Args:
        X_train:   Features de treino (2D: amostras × features)
        y_train:   Labels de treino
        X_val:     Features de validação
        y_val:     Labels de validação
        n_classes: Número de classes

    Returns:
        Dicionário com o modelo treinado e métricas
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
        BatchNormalization, Input,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(RANDOM_SEED)

    params = ML_PARAMS["cnn"]
    n_features = X_train.shape[1]

    print("\n  [CNN 1D] Treinando...")
    print(f"    Input shape: ({n_features}, 1)")
    print(f"    Filtros: {params['filters']}, Kernel: {params['kernel_size']}")

    # --- Reshape para CNN: (amostras, features, 1 canal) ---
    X_train_cnn = X_train.reshape(-1, n_features, 1)
    X_val_cnn = X_val.reshape(-1, n_features, 1)

    # --- Construção do modelo ---
    modelo = Sequential(name="CNN1D_LiquidClassifier")
    modelo.add(Input(shape=(n_features, 1)))

    for i, n_filters in enumerate(params["filters"]):
        modelo.add(Conv1D(n_filters, params["kernel_size"],
                          activation="relu", padding="same",
                          name=f"conv1d_{i}"))
        modelo.add(BatchNormalization(name=f"bn_{i}"))
        # Só aplica MaxPool se a dimensão for grande o suficiente
        if n_features // (2 ** (i + 1)) >= 2:
            modelo.add(MaxPooling1D(pool_size=2, name=f"maxpool_{i}"))

    modelo.add(Flatten(name="flatten"))
    modelo.add(Dense(64, activation="relu", name="dense_0"))
    modelo.add(Dropout(0.3, name="dropout_0"))

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

    # --- Treinamento ---
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )

    inicio = time.time()
    historico = modelo.fit(
        X_train_cnn, y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[early_stop],
        verbose=0,
    )
    tempo_treino = time.time() - inicio

    # --- Avaliação ---
    _, acc_train = modelo.evaluate(X_train_cnn, y_train, verbose=0)
    _, acc_val = modelo.evaluate(X_val_cnn, y_val, verbose=0)

    print(f"    Acurácia treino:    {acc_train:.4f}")
    print(f"    Acurácia validação: {acc_val:.4f}")
    print(f"    Épocas treinadas:   {len(historico.history['loss'])}")
    print(f"    Tempo de treino:    {tempo_treino:.2f}s")

    return {
        "nome": "CNN 1D",
        "modelo": modelo,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "tempo_treino": tempo_treino,
        "historico": historico.history,
    }
