"""
============================================================
 ANÁLISE DE IMPORTÂNCIA DOS SENSORES
============================================================
 O módulo mais importante para o objetivo do projeto:
 DETERMINAR QUAIS SENSORES COMPRAR PARA O MVP.

 Técnicas utilizadas:
   1. Feature Importance nativa do Random Forest e XGBoost
   2. Permutation Importance (model-agnostic)
   3. Ranking consolidado de sensores

 A saída deste módulo responde à pergunta:
   "Se eu só pudesse comprar 3 sensores, quais seriam?"
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import FIGURES_DIR, RESULTS_DIR, RANDOM_SEED


def extrair_importancia_arvores(modelos_treinados: list,
                                feature_names: list) -> pd.DataFrame:
    """
    Extrai feature importance dos modelos baseados em árvores
    (Random Forest e XGBoost).

    Returns:
        DataFrame com a importância de cada feature por modelo
    """
    df_list = []

    for m in modelos_treinados:
        if "feature_importances" in m:
            df_temp = pd.DataFrame({
                "sensor": feature_names,
                "importancia": m["feature_importances"],
                "modelo": m["nome"],
            })
            df_list.append(df_temp)

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()


def calcular_permutation_importance(modelo, X_test: np.ndarray,
                                     y_test: np.ndarray,
                                     feature_names: list,
                                     nome_modelo: str,
                                     is_keras: bool = False) -> pd.DataFrame:
    """
    Calcula Permutation Importance (embaralha uma feature por vez
    e mede quanto a acurácia cai).

    Esta é a técnica mais confiável para medir a importância real
    de cada sensor, independente do modelo.
    """
    if is_keras:
        # Para Keras, precisamos de um wrapper
        from sklearn.metrics import accuracy_score

        def keras_predict(X):
            n_features = X.shape[1]
            if "CNN" in nome_modelo:
                X_in = X.reshape(-1, n_features, 1)
            else:
                X_in = X
            y_proba = modelo.predict(X_in, verbose=0)
            if y_proba.shape[1] == 1:
                return (y_proba.flatten() > 0.5).astype(int)
            return np.argmax(y_proba, axis=1)

        # Implementação manual simplificada para Keras
        baseline_acc = accuracy_score(y_test, keras_predict(X_test))
        importancias = []
        rng = np.random.default_rng(RANDOM_SEED)

        for i in range(X_test.shape[1]):
            X_permuted = X_test.copy()
            X_permuted[:, i] = rng.permutation(X_permuted[:, i])
            permuted_acc = accuracy_score(y_test, keras_predict(X_permuted))
            importancias.append(baseline_acc - permuted_acc)

        return pd.DataFrame({
            "sensor": feature_names,
            "importancia": importancias,
            "modelo": f"{nome_modelo} (Permutation)",
        })
    else:
        result = permutation_importance(
            modelo, X_test, y_test,
            n_repeats=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        return pd.DataFrame({
            "sensor": feature_names,
            "importancia": result.importances_mean,
            "modelo": f"{nome_modelo} (Permutation)",
        })


def plotar_importancia_por_modelo(df_importancia: pd.DataFrame, sufixo: str = ""):
    """Gráfico de barras da importância de features por modelo."""
    modelos = df_importancia["modelo"].unique()

    for mod in modelos:
        df_mod = df_importancia[df_importancia["modelo"] == mod].copy()
        df_mod = df_mod.sort_values("importancia", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        cores = sns.color_palette("viridis", len(df_mod))
        ax.barh(df_mod["sensor"], df_mod["importancia"], color=cores)
        ax.set_xlabel("Importância")
        ax.set_title(f"Importância dos Sensores — {mod}")
        plt.tight_layout()

        nome_arquivo = f"importancia_{mod.replace(' ', '_').replace('(', '').replace(')', '')}{sufixo}.png"
        caminho = os.path.join(FIGURES_DIR, nome_arquivo)
        fig.savefig(caminho, dpi=150)
        plt.close(fig)


def gerar_ranking_consolidado(df_importancia: pd.DataFrame,
                               feature_names: list,
                               sufixo: str = "") -> pd.DataFrame:
    """
    Gera um ranking consolidado normalizando as importâncias
    de todos os modelos e calculando a média.

    Isso responde: "No geral, quais sensores são mais importantes?"
    """
    # Normaliza importância para [0, 1] dentro de cada modelo
    df_norm = df_importancia.copy()
    for mod in df_norm["modelo"].unique():
        mask = df_norm["modelo"] == mod
        vals = df_norm.loc[mask, "importancia"]
        min_val, max_val = vals.min(), vals.max()
        rng = max_val - min_val if max_val != min_val else 1.0
        df_norm.loc[mask, "importancia_norm"] = (vals - min_val) / rng

    # Média normalizada por sensor
    ranking = (
        df_norm.groupby("sensor")["importancia_norm"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranking.columns = ["Sensor", "Importância Média Normalizada"]
    ranking["Rank"] = range(1, len(ranking) + 1)
    ranking = ranking[["Rank", "Sensor", "Importância Média Normalizada"]]

    # Identifica o tipo físico do sensor
    def tipo_sensor(nome):
        if "spec_" in nome:
            return "Óptico (AS7341)"
        elif "condutividade" in nome:
            return "Condutividade (PCB)"
        elif "pH" in nome:
            return "pH (E-201-C)"
        elif "temperatura" in nome:
            return "Temperatura (DS18B20)"
        elif "acustico" in nome:
            return "Acústico (Piezo)"
        elif nome in ("ratio_azul_vermelho", "ratio_verde_vermelho", "ratio_nir_clear",
                       "ratio_violeta_vermelho", "spectral_mean", "spectral_std",
                       "spectral_slope"):
            return "Derivada Espectral"
        elif nome in ("browning_index", "turbidity_index", "chlorophyll_proxy",
                       "carotenoid_proxy"):
            return "Derivada Colorimétrica"
        elif nome in ("sugar_proxy_nir", "alcohol_proxy", "dilution_index",
                       "ph_cond_interaction", "acoustic_anomaly"):
            return "Derivada Multi-Sensor"
        else:
            return "Outro"

    ranking["Tipo Sensor"] = ranking["Sensor"].apply(tipo_sensor)

    # Plota ranking final
    fig, ax = plt.subplots(figsize=(12, 8))
    cores_tipo = {
        "Óptico (AS7341)": "#2196F3",
        "Condutividade (PCB)": "#FF9800",
        "pH (E-201-C)": "#4CAF50",
        "Temperatura (DS18B20)": "#F44336",
        "Acústico (Piezo)": "#9C27B0",
        "Derivada Espectral": "#607D8B",
        "Derivada Colorimétrica": "#00BCD4",
        "Derivada Multi-Sensor": "#795548",
    }
    cores = [cores_tipo.get(t, "#999999") for t in ranking["Tipo Sensor"]]

    ax.barh(ranking["Sensor"], ranking["Importância Média Normalizada"], color=cores)
    ax.set_xlabel("Importância Média Normalizada")
    ax.set_title("RANKING CONSOLIDADO — Quais Sensores Comprar?")
    ax.invert_yaxis()

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in cores_tipo.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    caminho = os.path.join(FIGURES_DIR, f"ranking_sensores{sufixo}.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Ranking de sensores salvo: {caminho}")

    return ranking


def executar_analise_sensores(modelos_treinados: list,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               feature_names: list,
                               sufixo: str = "") -> pd.DataFrame:
    """
    Pipeline completo de análise de importância dos sensores.

    Args:
        modelos_treinados: Lista de modelos treinados
        X_test:            Features de teste
        y_test:            Labels de teste
        feature_names:     Nomes das features
        sufixo:            Sufixo para arquivos

    Returns:
        DataFrame com o ranking consolidado
    """
    print("\n" + "=" * 60)
    print(f" ANÁLISE DE IMPORTÂNCIA DOS SENSORES{sufixo.upper()}")
    print("=" * 60)

    # 1. Importância nativa das árvores
    print("\n  [1/3] Extraindo importância nativa (RF, XGBoost)...")
    df_arvores = extrair_importancia_arvores(modelos_treinados, feature_names)

    # 2. Permutation Importance para o melhor modelo de árvore
    print("  [2/3] Calculando Permutation Importance...")
    df_permutation_list = []
    for m in modelos_treinados:
        nome = m["nome"]
        modelo = m["modelo"]
        is_keras = "Rede Neural" in nome or "CNN" in nome

        # Pular permutation para CNN (muito lento em dados pequenos, pouco ganho)
        if "CNN" in nome:
            continue

        print(f"        → {nome}...")
        df_perm = calcular_permutation_importance(
            modelo, X_test, y_test, feature_names, nome, is_keras
        )
        df_permutation_list.append(df_perm)

    df_permutation = pd.concat(df_permutation_list, ignore_index=True) if df_permutation_list else pd.DataFrame()

    # 3. Consolidação
    print("  [3/3] Gerando ranking consolidado...")
    df_todas = pd.concat([df_arvores, df_permutation], ignore_index=True)

    # Plotar importância individual
    plotar_importancia_por_modelo(df_todas, sufixo)

    # Ranking final
    ranking = gerar_ranking_consolidado(df_todas, feature_names, sufixo)

    print("\n" + "=" * 60)
    print(" RANKING FINAL DE SENSORES — DECISÃO DE COMPRA")
    print("=" * 60)
    print(ranking.to_string(index=False))

    # Salvar
    csv_path = os.path.join(RESULTS_DIR, f"ranking_sensores{sufixo}.csv")
    ranking.to_csv(csv_path, index=False)
    print(f"\n  Ranking salvo: {csv_path}")

    # Recomendação
    top5 = ranking.head(5)
    tipos_necessarios = top5["Tipo Sensor"].unique()
    print("\n  ╔══════════════════════════════════════════════════╗")
    print("  ║  RECOMENDAÇÃO PARA O MVP:                       ║")
    print("  ║  Compre os sensores dos seguintes tipos:         ║")
    for t in tipos_necessarios:
        print(f"  ║    → {t:<44s}║")
    print("  ╚══════════════════════════════════════════════════╝")

    return ranking
