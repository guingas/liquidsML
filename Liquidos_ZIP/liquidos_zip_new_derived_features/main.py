"""
============================================================
 MAIN — PIPELINE PRINCIPAL DO DISPOSITIVO S
 Identificação de Líquidos por Sensores + Machine Learning
============================================================

 Este é o ponto de entrada do projeto. Executa o pipeline
 completo em sequência HIERÁRQUICA:

   ETAPA 1 — Classificar TIPO da bebida (água vs cerveja)
   ETAPA 2 — Para cada tipo, classificar PROPRIEDADES:
     • Água   → Potabilidade (potável / contaminada)
     • Água   → Variante (torneira, mineral, mineral_gas, contaminada)
     • Cerveja → Estilo (heineken, budweiser, ipa, stout)

 Cada etapa: treina 5 modelos, avalia, analisa sensores.

 Como executar:
   python main.py
============================================================
"""

import os
import sys
import warnings

# Suprime warnings do TensorFlow para saída limpa
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Adiciona raiz do projeto ao path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --- Imports do projeto ---
from config.settings import RESULTS_DIR
from src.data_generation.synthetic_sensors import gerar_dataset_completo, MAPA_POTABILIDADE, MAPA_ADULTERACAO
from src.preprocessing.pipeline import preparar_dados
from src.models.random_forest import treinar_random_forest
from src.models.xgboost_model import treinar_xgboost
from src.models.neural_network import treinar_rede_neural
from src.models.cnn_model import treinar_cnn
from src.models.svm_model import treinar_svm
from src.evaluation.compare_models import executar_avaliacao_completa
from src.feature_analysis.sensor_importance import executar_analise_sensores


def treinar_todos_modelos(dados: dict, n_classes: int) -> list:
    """
    Treina todos os 5 modelos de ML nos dados fornecidos.

    Args:
        dados:     Dicionário retornado pelo pré-processamento
        n_classes: Número de classes do problema

    Returns:
        Lista de dicionários com modelos treinados e métricas
    """
    X_train = dados["X_train"]
    y_train = dados["y_train"]
    X_val = dados["X_val"]
    y_val = dados["y_val"]

    print("\n" + "=" * 60)
    print(" TREINAMENTO DOS MODELOS DE MACHINE LEARNING")
    print(f" (n_classes={n_classes}, n_features={X_train.shape[1]},"
          f" n_treino={X_train.shape[0]})")
    print("=" * 60)

    modelos = []

    # 1. Random Forest
    resultado_rf = treinar_random_forest(X_train, y_train, X_val, y_val)
    modelos.append(resultado_rf)

    # 2. XGBoost
    resultado_xgb = treinar_xgboost(X_train, y_train, X_val, y_val)
    modelos.append(resultado_xgb)

    # 3. Rede Neural (MLP)
    resultado_nn = treinar_rede_neural(X_train, y_train, X_val, y_val, n_classes)
    modelos.append(resultado_nn)

    # 4. CNN 1D
    resultado_cnn = treinar_cnn(X_train, y_train, X_val, y_val, n_classes)
    modelos.append(resultado_cnn)

    # 5. SVM
    resultado_svm = treinar_svm(X_train, y_train, X_val, y_val)
    modelos.append(resultado_svm)

    # Resumo rápido
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │    RESUMO DO TREINAMENTO (Validação)         │")
    print("  ├──────────────────────┬──────────────────────┤")
    print("  │ Modelo               │ Acurácia Validação   │")
    print("  ├──────────────────────┼──────────────────────┤")
    for m in modelos:
        nome = m["nome"][:20].ljust(20)
        acc = f"{m['acc_val'] * 100:.2f}%".rjust(18)
        print(f"  │ {nome} │ {acc}   │")
    print("  └──────────────────────┴──────────────────────┘")

    return modelos


def executar_pipeline(df, target_col: str, nome_problema: str):
    """
    Executa o pipeline completo para um problema de classificação.

    Args:
        df:              DataFrame com dados
        target_col:      Coluna alvo
        nome_problema:   Nome descritivo para os arquivos de saída
    """
    sufixo = f"_{nome_problema}"

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print(f"║  PROBLEMA: {nome_problema.upper():<46s}║")
    print(f"║  Target:   {target_col:<46s}║")
    print("╚" + "═" * 58 + "╝")

    # --- Etapa 1: Pré-processamento ---
    dados = preparar_dados(df, target_col=target_col)

    n_classes = len(dados["label_encoder"].classes_)

    # --- Etapa 2: Treinamento ---
    modelos = treinar_todos_modelos(dados, n_classes)

    # --- Etapa 3: Avaliação ---
    df_comparacao = executar_avaliacao_completa(
        modelos_treinados=modelos,
        X_test=dados["X_test"],
        y_test=dados["y_test"],
        label_encoder=dados["label_encoder"],
        sufixo=sufixo,
    )

    # --- Etapa 4: Análise de Sensores ---
    ranking = executar_analise_sensores(
        modelos_treinados=modelos,
        X_test=dados["X_test"],
        y_test=dados["y_test"],
        feature_names=dados["feature_names"],
        sufixo=sufixo,
    )

    return df_comparacao, ranking


def main():
    """Ponto de entrada principal."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║                                                          ║")
    print("║   DISPOSITIVO S — IDENTIFICAÇÃO DE LÍQUIDOS              ║")
    print("║   Pipeline Hierárquico de Classificação                  ║")
    print("║                                                          ║")
    print("║   Sensores: AS7341 (óptico) + DS18B20 (temp) +          ║")
    print("║             TDS (condutividade) + pH + Piezo (acústico)  ║")
    print("║                                                          ║")
    print("║   Modelos: Random Forest | XGBoost | MLP | CNN | SVM    ║")
    print("║                                                          ║")
    print("║   ETAPA 1: Tipo da bebida (6 classes)                    ║")
    print("║     água | cerveja | café | chá | suco | refresco        ║")
    print("║                                                          ║")
    print("║   ETAPA 2: Propriedades por tipo                         ║")
    print("║     • Água     → Potabilidade + Variante                 ║")
    print("║     • Cerveja  → Estilo + Adulteração                    ║")
    print("║     • Café     → Tipo (espresso/filtrado/cappuccino/...) ║")
    print("║     • Chá      → Tipo (verde/preto/camomila/mate)       ║")
    print("║     • Suco     → Fruta (laranja/uva/manga)               ║")
    print("║     • Refresco → Sabor (laranja/uva)                     ║")
    print("║                                                          ║")
    print("╚" + "═" * 58 + "╝")

    # =========================================================
    # GERAR DADOS SINTÉTICOS
    # =========================================================
    df = gerar_dataset_completo()

    # =========================================================
    # ETAPA 1: Classificar TIPO da bebida (6 classes)
    # =========================================================
    print("\n" + "=" * 60)
    print("  📌 ETAPA 1 — CLASSIFICAÇÃO DO TIPO DA BEBIDA")
    print("     água | cerveja | café | chá | suco | refresco")
    print("=" * 60)
    df_comp_tipo, ranking_tipo = executar_pipeline(
        df, target_col="tipo", nome_problema="etapa1_tipo_bebida"
    )

    # =========================================================
    # ETAPA 2: Propriedades por tipo
    # =========================================================
    print("\n" + "=" * 60)
    print("  📌 ETAPA 2 — PROPRIEDADES ESPECÍFICAS POR TIPO")
    print("=" * 60)

    # --- 2a: Água → Potabilidade ---
    df_agua = df[df["tipo"] == "agua"].copy()
    df_agua["potabilidade_label"] = df_agua["potabilidade"].map(
        {1: "potavel", 0: "contaminada"}
    )
    df_agua = df_agua.dropna(subset=["potabilidade_label"])

    if len(df_agua) > 0:
        df_comp_pot, ranking_pot = executar_pipeline(
            df_agua,
            target_col="potabilidade_label",
            nome_problema="etapa2_agua_potabilidade",
        )

    # --- 2b: Água → Variante (torneira, mineral, mineral_gas, contaminada) ---
    df_agua_full = df[df["tipo"] == "agua"].copy()
    if len(df_agua_full) > 0:
        df_comp_var, ranking_var = executar_pipeline(
            df_agua_full,
            target_col="subtipo",
            nome_problema="etapa2_agua_variante",
        )

    # --- 2c: Cerveja → Estilo/Marca (apenas cervejas normais) ---
    df_cerveja_normal = df[
        (df["tipo"] == "cerveja") & (df["adulteracao"] == "normal")
    ].copy()
    if len(df_cerveja_normal) > 0:
        df_comp_cerv, ranking_cerv = executar_pipeline(
            df_cerveja_normal,
            target_col="subtipo",
            nome_problema="etapa2_cerveja_estilo",
        )

    # --- 2d: Cerveja → Detecção de Adulteração ---
    df_cerveja = df[df["tipo"] == "cerveja"].copy()
    if len(df_cerveja) > 0:
        df_comp_adult, ranking_adult = executar_pipeline(
            df_cerveja,
            target_col="adulteracao",
            nome_problema="etapa2_cerveja_adulteracao",
        )

    # --- 2e: Café → Tipo (espresso, filtrado, cappuccino, solúvel) ---
    df_cafe = df[df["tipo"] == "cafe"].copy()
    if len(df_cafe) > 0:
        df_comp_cafe, ranking_cafe = executar_pipeline(
            df_cafe,
            target_col="subtipo",
            nome_problema="etapa2_cafe_tipo",
        )

    # --- 2f: Chá → Tipo (verde, preto, camomila, mate) ---
    df_cha = df[df["tipo"] == "cha"].copy()
    if len(df_cha) > 0:
        df_comp_cha, ranking_cha = executar_pipeline(
            df_cha,
            target_col="subtipo",
            nome_problema="etapa2_cha_tipo",
        )

    # --- 2g: Suco → Fruta (laranja, uva, manga) ---
    df_suco = df[df["tipo"] == "suco"].copy()
    if len(df_suco) > 0:
        df_comp_suco, ranking_suco = executar_pipeline(
            df_suco,
            target_col="subtipo",
            nome_problema="etapa2_suco_fruta",
        )

    # --- 2h: Refresco → Sabor (laranja, uva) ---
    df_refresco = df[df["tipo"] == "refresco"].copy()
    if len(df_refresco) > 0:
        df_comp_refr, ranking_refr = executar_pipeline(
            df_refresco,
            target_col="subtipo",
            nome_problema="etapa2_refresco_sabor",
        )

    # =========================================================
    # RESUMO FINAL
    # =========================================================
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║              PIPELINE CONCLUÍDO COM SUCESSO              ║")
    print("╠" + "═" * 58 + "╣")
    print("║  Etapa 1:  TIPO da bebida (6 classes)                    ║")
    print("║  Etapa 2a: Água → Potabilidade                           ║")
    print("║  Etapa 2b: Água → Variante                               ║")
    print("║  Etapa 2c: Cerveja → Estilo / Marca                      ║")
    print("║  Etapa 2d: Cerveja → Adulteração (metanol / DEG)         ║")
    print("║  Etapa 2e: Café → Tipo                                    ║")
    print("║  Etapa 2f: Chá → Tipo                                     ║")
    print("║  Etapa 2g: Suco → Fruta                                   ║")
    print("║  Etapa 2h: Refresco → Sabor                               ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Resultados salvos em: {RESULTS_DIR}")
    print(f"  Gráficos salvos em:   {os.path.join(RESULTS_DIR, 'figures')}")
    print("\n  Arquivos gerados:")

    # Lista arquivos gerados
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            caminho = os.path.join(root, f)
            tamanho = os.path.getsize(caminho)
            print(f"    → {os.path.relpath(caminho, RESULTS_DIR):<50s} ({tamanho:>8,d} bytes)")

    print("\n  Próximo passo: Analise os gráficos em results/figures/")
    print("  para decidir quais sensores comprar para o MVP!")
    print()


if __name__ == "__main__":
    main()
