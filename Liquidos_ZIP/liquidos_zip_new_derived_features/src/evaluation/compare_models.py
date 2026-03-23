"""
============================================================
 AVALIAÇÃO E COMPARAÇÃO DOS MODELOS
============================================================
 Gera métricas, matrizes de confusão e gráficos comparativos
 para todos os modelos treinados.

 Objetivo: Determinar qual combinação modelo + sensores é a
 melhor para o MVP do dispositivo S.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sem GUI (salva em arquivo)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import FIGURES_DIR, RESULTS_DIR


def avaliar_no_teste(modelo, X_test: np.ndarray, y_test: np.ndarray,
                     nome_modelo: str, is_keras: bool = False) -> dict:
    """
    Avalia um modelo no conjunto de teste final.

    Args:
        modelo:       Modelo treinado (sklearn ou keras)
        X_test:       Features de teste
        y_test:       Labels de teste
        nome_modelo:  Nome para exibição
        is_keras:     True se for modelo TensorFlow/Keras

    Returns:
        Dicionário com métricas de teste
    """
    if is_keras:
        n_features = X_test.shape[1]
        # CNN precisa de reshape
        if "CNN" in nome_modelo:
            X_input = X_test.reshape(-1, n_features, 1)
        else:
            X_input = X_test

        y_proba = modelo.predict(X_input, verbose=0)
        if y_proba.shape[1] == 1:
            y_pred = (y_proba.flatten() > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    n_classes = len(np.unique(y_test))
    avg = "binary" if n_classes == 2 else "weighted"

    precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "nome": nome_modelo,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def gerar_tabela_comparativa(resultados: list, label_encoder) -> pd.DataFrame:
    """
    Cria uma tabela comparativa de todos os modelos.

    Args:
        resultados: Lista de dicionários retornados por avaliar_no_teste()
        label_encoder: LabelEncoder para nomes das classes

    Returns:
        DataFrame com a comparação
    """
    rows = []
    for r in resultados:
        rows.append({
            "Modelo": r["nome"],
            "Acurácia (%)": f"{r['accuracy'] * 100:.2f}",
            "Precisão (%)": f"{r['precision'] * 100:.2f}",
            "Recall (%)": f"{r['recall'] * 100:.2f}",
            "F1-Score (%)": f"{r['f1_score'] * 100:.2f}",
        })

    df = pd.DataFrame(rows)
    return df


def plotar_matriz_confusao(cm: np.ndarray, classes: list,
                           nome_modelo: str, sufixo: str = ""):
    """
    Plota e salva a matriz de confusão de um modelo.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {nome_modelo}")
    plt.tight_layout()

    nome_arquivo = f"confusion_matrix_{nome_modelo.replace(' ', '_').replace('(', '').replace(')', '')}{sufixo}.png"
    caminho = os.path.join(FIGURES_DIR, nome_arquivo)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"    Salvo: {caminho}")


def plotar_comparacao_modelos(resultados: list, sufixo: str = ""):
    """
    Gráfico de barras comparando acurácia de todos os modelos.
    """
    nomes = [r["nome"] for r in resultados]
    acuracias = [r["accuracy"] * 100 for r in resultados]

    # Ordena por acurácia
    indices = np.argsort(acuracias)[::-1]
    nomes = [nomes[i] for i in indices]
    acuracias = [acuracias[i] for i in indices]

    # Cores: melhor em verde, piores em gradiente
    cores = sns.color_palette("coolwarm_r", len(nomes))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(nomes, acuracias, color=cores)

    # Adiciona valor em cada barra
    for bar, acc in zip(bars, acuracias):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2f}%", va="center", fontweight="bold")

    ax.set_xlabel("Acurácia no Teste (%)")
    ax.set_title("Comparação de Modelos — Identificação de Líquidos")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    plt.tight_layout()

    caminho = os.path.join(FIGURES_DIR, f"comparacao_modelos{sufixo}.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Gráfico comparativo salvo: {caminho}")


def plotar_metricas_completas(resultados: list, sufixo: str = ""):
    """
    Gráfico de barras agrupadas com Acurácia, Precisão, Recall e F1.
    """
    nomes = [r["nome"] for r in resultados]
    metricas = {
        "Acurácia": [r["accuracy"] * 100 for r in resultados],
        "Precisão": [r["precision"] * 100 for r in resultados],
        "Recall": [r["recall"] * 100 for r in resultados],
        "F1-Score": [r["f1_score"] * 100 for r in resultados],
    }

    x = np.arange(len(nomes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metrica, valores) in enumerate(metricas.items()):
        ax.bar(x + i * width, valores, width, label=metrica)

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Porcentagem (%)")
    ax.set_title("Métricas Completas — Todos os Modelos")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(nomes, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()

    caminho = os.path.join(FIGURES_DIR, f"metricas_completas{sufixo}.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Gráfico de métricas salvo: {caminho}")


def executar_avaliacao_completa(modelos_treinados: list,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                label_encoder,
                                sufixo: str = "") -> pd.DataFrame:
    """
    Executa a avaliação completa de todos os modelos.

    Args:
        modelos_treinados: Lista de dicts retornados pelas funções de treino
        X_test:            Features de teste
        y_test:            Labels de teste
        label_encoder:     LabelEncoder para recuperar nomes das classes
        sufixo:            Sufixo para nomes dos arquivos (ex: "_tipo", "_subtipo")

    Returns:
        DataFrame com a tabela comparativa
    """
    print("\n" + "=" * 60)
    print(f" AVALIAÇÃO NO CONJUNTO DE TESTE{sufixo.upper()}")
    print("=" * 60)

    classes = list(label_encoder.classes_)
    resultados_teste = []

    for m in modelos_treinados:
        nome = m["nome"]
        modelo = m["modelo"]
        is_keras = "Rede Neural" in nome or "CNN" in nome

        print(f"\n  Avaliando: {nome}")
        resultado = avaliar_no_teste(modelo, X_test, y_test, nome, is_keras)
        resultados_teste.append(resultado)

        # Matriz de confusão individual
        plotar_matriz_confusao(resultado["confusion_matrix"], classes, nome, sufixo)

    # Tabela comparativa
    df_comparacao = gerar_tabela_comparativa(resultados_teste, label_encoder)

    print("\n" + "=" * 60)
    print(" TABELA COMPARATIVA DOS MODELOS")
    print("=" * 60)
    print(df_comparacao.to_string(index=False))

    # Salva CSV
    csv_path = os.path.join(RESULTS_DIR, f"comparacao_modelos{sufixo}.csv")
    df_comparacao.to_csv(csv_path, index=False)
    print(f"\n  Tabela salva: {csv_path}")

    # Gráficos
    plotar_comparacao_modelos(resultados_teste, sufixo)
    plotar_metricas_completas(resultados_teste, sufixo)

    return df_comparacao
