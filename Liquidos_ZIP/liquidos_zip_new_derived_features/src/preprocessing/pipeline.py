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
    Cria features derivadas a partir dos sensores do dispositivo S.

    Features baseadas em literatura de ciência de alimentos, espectroscopia
    e eletroquímica para maximizar a discriminação entre tipos de bebidas.

    Referências:
        - Meydav et al. (1977), Food Chemistry — Browning Index
        - Siebert (2009), J. ASBC — Turbidez em bebidas
        - Lichtenthaler (1987), Methods in Enzymology — Clorofila
        - Britton et al. (2004), Carotenoids Handbook — Carotenóides
        - Cozzolino et al. (2011), Food Res. Intl. — NIR e açúcares
        - Resa et al. (2005), J. Food Eng. — Velocidade do som e álcool
        - Ashurst (2005), Chemistry of Soft Drinks — Índice de diluição
        - Kammerer et al. (2004), Eur. Food Res. Tech. — Razões espectrais
        - Boss et al. (2007), Talanta — Slope espectral
        - Lachenmeier et al. (2007), Reg. Tox. Pharm. — Adulteração
    """
    df = df.copy()
    eps = 1e-6

    # ----- Razões espectrais originais (mantidas) -----

    # Razão azul/vermelho: separa água (>1) de cerveja escura (<1)
    df["ratio_azul_vermelho"] = df["spec_F3_480nm"] / (df["spec_F7_630nm"] + eps)

    # Razão verde/vermelho: destaca tons âmbar (cerveja, chá preto)
    df["ratio_verde_vermelho"] = df["spec_F4_515nm"] / (df["spec_F7_630nm"] + eps)

    # Razão NIR/Clear: indica teor de álcool/açúcar vs turbidez
    df["ratio_nir_clear"] = df["spec_NIR"] / (df["spec_Clear"] + eps)

    # Estatísticas espectrais: perfil geral
    spec_cols = [c for c in df.columns if c.startswith("spec_")]
    df["spectral_mean"] = df[spec_cols].mean(axis=1)
    df["spectral_std"] = df[spec_cols].std(axis=1)

    # ----- Novas features derivadas (literatura) -----

    # 1. Browning Index — marcador de reação de Maillard
    #    Alto para café (melanoidinas), chá preto (tearubiginas), cerveja escura
    #    Ref: Meydav et al. 1977
    df["browning_index"] = (
        (df["spec_F6_590nm"] - df["spec_F3_480nm"]) /
        (df["spec_F6_590nm"] + df["spec_F3_480nm"] + eps)
    )

    # 2. Turbidity Index — partículas em suspensão
    #    Alto para suco natural (polpa), café espresso (crema), stout
    #    Ref: Siebert, 2009
    df["turbidity_index"] = 1.0 - df["spec_Clear"]

    # 3. Chlorophyll Proxy — absorção da clorofila-a em 680nm
    #    Alto para chá verde, chá mate, sucos verdes
    #    Ref: Lichtenthaler, 1987
    df["chlorophyll_proxy"] = (
        (df["spec_F7_630nm"] - df["spec_F8_680nm"]) /
        (df["spec_F7_630nm"] + df["spec_F8_680nm"] + eps)
    )

    # 4. Carotenoid Proxy — carotenóides absorvem violeta, refletem laranja
    #    Alto para suco de laranja, suco de manga
    #    Ref: Britton et al., 2004
    df["carotenoid_proxy"] = (
        (df["spec_F6_590nm"] - df["spec_F1_415nm"]) /
        (df["spec_F6_590nm"] + df["spec_F1_415nm"] + eps)
    )

    # 5. Sugar Proxy (NIR) — NIR correlaciona com Brix/açúcar
    #    Diferencia suco natural (alto Brix) de refresco (diluído)
    #    Ref: Cozzolino et al., 2011
    df["sugar_proxy_nir"] = df["spec_NIR"] * (1.0 - df["spec_Clear"])

    # 6. Alcohol Proxy — velocidade do som diminui com etanol
    #    Alto para cerveja e especialmente IPA (mais álcool)
    #    Ref: Resa et al., 2005
    cond_col = "condutividade_25C" if "condutividade_25C" in df.columns else "condutividade_uS"
    df["alcohol_proxy"] = (1500.0 - df["acustico_freq_Hz"]) / (df[cond_col] + eps)

    # 7. Spectral Slope — inclinação linear no espectro visível
    #    Positivo = mais vermelho (cerveja, chá preto)
    #    Negativo = mais azul (água limpa)
    #    Ref: Boss et al., 2007
    visible_channels = [
        "spec_F1_415nm", "spec_F2_445nm", "spec_F3_480nm",
        "spec_F4_515nm", "spec_F5_555nm", "spec_F6_590nm",
        "spec_F7_630nm", "spec_F8_680nm",
    ]
    wavelengths = np.array([415, 445, 480, 515, 555, 590, 630, 680], dtype=float)
    wavelengths_norm = (wavelengths - wavelengths.mean()) / wavelengths.std()
    spec_matrix = df[visible_channels].values
    df["spectral_slope"] = spec_matrix @ wavelengths_norm / (wavelengths_norm @ wavelengths_norm)

    # 8. Dilution Index — clareza relativa à condutividade
    #    Alto para refresco (transparente + baixa condutividade)
    #    Baixo para suco natural (opaco + alta condutividade)
    #    Ref: Ashurst, 2005
    df["dilution_index"] = df["spec_Clear"] / (df[cond_col] / 1000.0 + eps)

    # 9. Ratio Violeta/Vermelho — discriminação de pigmentos profundos
    #    Baixo para líquidos pigmentados (suco de uva, stout)
    #    Ref: Kammerer et al., 2004
    df["ratio_violeta_vermelho"] = df["spec_F1_415nm"] / (df["spec_F7_630nm"] + eps)

    # 10. pH-Conductivity Interaction — feature cross-sensor
    #     Combina acidez com mineralização para separar clusters
    #     Ref: Bhatia et al., 2017, IEEE Sensors Journal
    df["ph_cond_interaction"] = df["pH"] * df[cond_col] / 10000.0

    # 11. Acoustic Anomaly Index — desvio do baseline da água pura
    #     Alto para cerveja adulterada (DEG/metanol alteram propagação)
    #     Ref: McClements, 1997, CRC Critical Reviews Food Science
    df["acoustic_anomaly"] = np.abs(df["acustico_freq_Hz"] - 1482.0) / 50.0

    # ----- Features espectrais de ordem superior (proposta validada) -----

    # 12. Absorption Index 415nm (Lei de Beer-Lambert)
    #     Quantifica absorção no violeta — alto para café, suco de uva, stout
    #     Ref: Burns & Ciurczak (2007), Handbook of NIR Analysis
    df["absorption_index_415nm"] = np.log10(
        (df["spec_Clear"] + eps) / (df["spec_F1_415nm"] + eps)
    )

    # 13. Absorption Index 680nm (Lei de Beer-Lambert)
    #     Quantifica absorção no vermelho profundo — alto se clorofila presente
    #     Ref: Lichtenthaler & Buschmann, 2001, Current Protocols
    df["absorption_index_680nm"] = np.log10(
        (df["spec_Clear"] + eps) / (df["spec_F8_680nm"] + eps)
    )

    # 14. Spectral Entropy — mede aleatoriedade da distribuição espectral
    #     Baixo = espectro com picos dominantes (líquido pigmentado)
    #     Alto = espectro uniforme (água limpa)
    #     Ref: Rencher & Christensen, 2012, Methods of Multivariate Analysis
    spec_visible = df[visible_channels].values
    spec_sum = spec_visible.sum(axis=1, keepdims=True)
    spec_prob = spec_visible / (spec_sum + eps)
    # Clip para evitar log(0)
    spec_prob = np.clip(spec_prob, eps, 1.0)
    df["spectral_entropy"] = -(spec_prob * np.log2(spec_prob)).sum(axis=1)

    # 15. Spectral Skewness — assimetria do perfil espectral
    #     Positivo = mais energia em comprimentos de onda longos (vermelho)
    #     Negativo = mais energia em curtos (azul/violeta)
    #     Ref: Joanes & Gill, 1998, The Statistician
    spec_mean_vals = spec_visible.mean(axis=1, keepdims=True)
    spec_std_vals = spec_visible.std(axis=1, keepdims=True) + eps
    df["spectral_skewness"] = (
        ((spec_visible - spec_mean_vals) / spec_std_vals) ** 3
    ).mean(axis=1)

    # 16. Spectral Kurtosis — "peakedness" do perfil espectral
    #     Alto = picos pronunciados em poucas bandas
    #     Baixo = distribuição uniforme (plana)
    #     Ref: DeCarlo, 1997, Psychological Methods
    df["spectral_kurtosis"] = (
        ((spec_visible - spec_mean_vals) / spec_std_vals) ** 4
    ).mean(axis=1) - 3.0  # Excess kurtosis (normal = 0)

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
    colunas_excluir = ["tipo", "subtipo", "potabilidade", "potabilidade_label", "adulteracao"]
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
