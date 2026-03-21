"""
============================================================
 PIPELINE DE PRÉ-PROCESSAMENTO
 Limpeza, normalização e divisão dos dados
============================================================

 Este módulo é o "filtro" entre os dados brutos dos sensores
 e os modelos de Machine Learning.

 Etapas:
   1. Compensação de temperatura na condutividade
   2. Engenharia de features (razões espectrais)
   3. Normalização (StandardScaler)
   4. Divisão treino / validação / teste
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    SENSOR_NAMES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    DATA_PROCESSED,
)


def compensar_temperatura_condutividade(df: pd.DataFrame, alpha: float = 0.02) -> pd.DataFrame:
    """
    Normaliza a condutividade elétrica para 25°C.

    A condutividade varia ~2% por grau Celsius. Para que o ML
    não confunda "cerveja gelada" com "água quente", normalizamos.

    Fórmula: σ_25 = σ_T / (1 + α × (T - 25))

    Args:
        df:    DataFrame com colunas 'condutividade_uS' e 'temperatura_C'
        alpha: Coeficiente de temperatura (~0.02 para água)

    Returns:
        DataFrame com coluna 'condutividade_25C' adicionada
    """
    df = df.copy()
    temp = df["temperatura_C"]
    cond = df["condutividade_uS"]
    df["condutividade_25C"] = cond / (1 + alpha * (temp - 25))
    return df


def criar_features_espectrais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas a partir dos canais espectrais do AS7341.

    Razões entre canais realçam diferenças de cor que são
    mais robustas a variações de intensidade luminosa.

    Features criadas:
        - ratio_azul_vermelho: F3(480nm) / F7(630nm) → separa água de cerveja escura
        - ratio_verde_vermelho: F4(515nm) / F7(630nm) → destaca tons âmbar
        - ratio_nir_clear: NIR / Clear → indica teor de álcool/açúcar
        - spectral_mean: média de todos os canais → turbidez geral
        - spectral_std: desvio padrão dos canais → "forma" do espectro
    """
    df = df.copy()

    # Evita divisão por zero adicionando epsilon
    eps = 1e-6

    df["ratio_azul_vermelho"] = df["spec_F3_480nm"] / (df["spec_F7_630nm"] + eps)
    df["ratio_verde_vermelho"] = df["spec_F4_515nm"] / (df["spec_F7_630nm"] + eps)
    df["ratio_nir_clear"] = df["spec_NIR"] / (df["spec_Clear"] + eps)

    spec_cols = [c for c in df.columns if c.startswith("spec_")]
    df["spectral_mean"] = df[spec_cols].mean(axis=1)
    df["spectral_std"] = df[spec_cols].std(axis=1)

    return df


def preparar_dados(df: pd.DataFrame,
                   target_col: str = "tipo",
                   seed: int = RANDOM_SEED):
    """
    Pipeline completo de pré-processamento.

    Args:
        df:         DataFrame com dados brutos dos sensores
        target_col: Coluna alvo ("tipo" para água/cerveja,
                    "subtipo" para marca específica)
        seed:       Semente para reprodutibilidade

    Returns:
        Dicionário com:
            X_train, X_val, X_test:     Features normalizadas
            y_train, y_val, y_test:     Labels codificadas
            feature_names:              Nomes das features usadas
            scaler:                     StandardScaler ajustado
            label_encoder:              LabelEncoder ajustado
    """
    print("\n" + "=" * 60)
    print(" PRÉ-PROCESSAMENTO DOS DADOS")
    print("=" * 60)

    # --- Etapa 1: Compensação de temperatura ---
    print("  [1/5] Compensando condutividade para 25°C...")
    df = compensar_temperatura_condutividade(df)

    # --- Etapa 2: Engenharia de features ---
    print("  [2/5] Criando features espectrais derivadas...")
    df = criar_features_espectrais(df)

    # --- Etapa 3: Selecionar features e target ---
    print("  [3/5] Selecionando features...")

    # Features = sensores originais + features derivadas (exceto labels)
    colunas_excluir = ["tipo", "subtipo", "potabilidade", "potabilidade_label"]
    feature_names = [c for c in df.columns if c not in colunas_excluir and c != target_col]
    X = df[feature_names].values
    y_raw = df[target_col].values

    # Codifica labels: "agua" → 0, "cerveja" → 1, etc.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"        Classes: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # --- Etapa 4: Dividir dados (treino / validação / teste) ---
    print(f"  [4/5] Dividindo dados ({TRAIN_RATIO:.0%} treino / "
          f"{VAL_RATIO:.0%} validação / {TEST_RATIO:.0%} teste)...")

    # Primeira divisão: treino vs. (validação + teste)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, random_state=seed, stratify=y
    )

    # Segunda divisão: validação vs. teste
    test_fraction = TEST_RATIO / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_fraction, random_state=seed, stratify=y_temp
    )

    print(f"        Treino:    {X_train.shape[0]} amostras")
    print(f"        Validação: {X_val.shape[0]} amostras")
    print(f"        Teste:     {X_test.shape[0]} amostras")

    # --- Etapa 5: Normalização ---
    print("  [5/5] Normalizando com StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)    # Ajusta no treino
    X_val = scaler.transform(X_val)            # Aplica no validação
    X_test = scaler.transform(X_test)          # Aplica no teste

    # Salva o scaler para uso futuro (deploy no ESP32)
    scaler_path = os.path.join(DATA_PROCESSED, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    encoder_path = os.path.join(DATA_PROCESSED, "label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)

    print(f"        Scaler salvo em: {scaler_path}")
    print("=" * 60)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }
