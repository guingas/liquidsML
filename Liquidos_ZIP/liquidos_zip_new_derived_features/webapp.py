"""
============================================================
 WEBAPP INTERATIVO — Dispositivo S
 Identificação de Líquidos — Pipeline Integrado
============================================================
 Execução:  streamlit run webapp.py

 Pipeline automático e completo:
   1. Recebe amostra (sensores)
   2. Classifica TIPO (água / cerveja)
   3. Automaticamente classifica PROPRIEDADES:
      • Água   → Potabilidade + Variante
      • Cerveja → Estilo / Marca
============================================================
"""

import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from src.data_generation.synthetic_sensors import (
    gerar_dataset_completo,
    gerar_amostras_liquido,
    PERFIS_SENSORES,
    MAPA_TIPO,
    MAPA_POTABILIDADE,
    MAPA_ADULTERACAO,
)
from src.preprocessing.pipeline import (
    compensar_temperatura_condutividade,
    criar_features_espectrais,
)
from src.data_generation.dataset_loaders import (
    carregar_csv_url,
    validar_url_dataset,
    resolver_url_dataset,
    carregar_do_catalogo,
    DOMINIOS_PERMITIDOS,
    CATALOGO_DATASETS,
    categorias_catalogo,
    listar_catalogo,
)
from config.settings import SENSOR_NAMES, RANDOM_SEED

# ============================================================
# Configuração da Página
# ============================================================
st.set_page_config(
    page_title="Dispositivo S — Identificação de Líquidos",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Definição dos classificadores do pipeline
# ============================================================
CLASSIFICADORES = {
    "tipo":               {"nome": "Tipo da Bebida",            "target": "tipo",               "filtro": None},
    "agua_potabilidade":  {"nome": "Potabilidade (Água)",       "target": "potabilidade_label", "filtro": "agua"},
    "agua_variante":      {"nome": "Variante (Água)",           "target": "subtipo",            "filtro": "agua"},
    "cerveja_estilo":     {"nome": "Estilo (Cerveja)",          "target": "subtipo",            "filtro": "cerveja_normal"},
    "cerveja_adulteracao":{"nome": "Adulteração (Cerveja)",     "target": "adulteracao",        "filtro": "cerveja"},
    "cafe_tipo":          {"nome": "Tipo (Café)",               "target": "subtipo",            "filtro": "cafe"},
    "cha_tipo":           {"nome": "Tipo (Chá)",                "target": "subtipo",            "filtro": "cha"},
    "suco_fruta":         {"nome": "Fruta (Suco)",              "target": "subtipo",            "filtro": "suco"},
    "refresco_sabor":     {"nome": "Sabor (Refresco)",          "target": "subtipo",            "filtro": "refresco"},
}

NOMES_MODELOS = ["Random Forest", "XGBoost", "SVM", "Rede Neural (MLP)", "CNN 1D"]

# ============================================================
# Descrição dos Sensores do Dispositivo S
# ============================================================
GRUPOS_SENSORES = {
    "🌡️ Temperatura (DS18B20)": {
        "colunas": ["temperatura_C"],
        "descricao": (
            "Sensor digital à prova d'água DS18B20. Mede temperatura de -55°C a +125°C "
            "com precisão de ±0.5°C. Comunicação 1-Wire. Fundamental para compensar a "
            "condutividade (que varia ~2%/°C) e diferenciar bebidas geladas de quentes."
        ),
        "derivadas": ["condutividade_25C"],
        "hardware": "DS18B20 em encapsulamento à prova d'água",
        "custo_aprox": "R$ 8",
        "features_detalhadas": {
            "temperatura_C": {
                "nome": "Temperatura",
                "unidade": "°C",
                "descricao": "Temperatura do líquido. Serve como feature direta (cerveja ~4-8°C, café ~60-80°C) e como insumo para compensar a condutividade.",
                "faixa_tipica": "0 – 100 °C",
            },
        },
        "derivadas_detalhadas": {
            "condutividade_25C": {
                "nome": "Condutividade compensada a 25°C",
                "formula": "σ_25 = σ_T / (1 + 0.02 × (T - 25))",
                "descricao": "Remove o efeito da temperatura sobre a condutividade elétrica. Sem essa compensação, o modelo poderia confundir 'cerveja gelada' com 'água quente'. Depende de: `temperatura_C` + `condutividade_uS`.",
                "depende_de": ["temperatura_C", "condutividade_uS"],
            },
        },
    },
    "⚡ Condutividade (Eletrodos PCB)": {
        "colunas": ["condutividade_uS"],
        "descricao": (
            "Eletrodos interdigitados gravados na PCB do próprio dispositivo. Medem a "
            "condutividade elétrica do líquido em µS/cm, que reflete a concentração de "
            "íons dissolvidos (sais, minerais, ácidos). Água destilada ≈ 0 µS, água "
            "mineral ≈ 200-800 µS, cerveja ≈ 1000-3000 µS."
        ),
        "derivadas": ["condutividade_25C"],
        "hardware": "Eletrodos interdigitados de cobre na PCB + circuito AC",
        "custo_aprox": "R$ 2 (incluso na PCB)",
        "features_detalhadas": {
            "condutividade_uS": {
                "nome": "Condutividade Elétrica",
                "unidade": "µS/cm",
                "descricao": "Mede a facilidade com que a corrente elétrica passa pelo líquido. Quanto mais íons dissolvidos (sais, minerais, ácidos), maior o valor. Um dos sensores mais discriminativos para separar tipos de líquido.",
                "faixa_tipica": "Água destilada ≈ 0 | Mineral ≈ 200-800 | Cerveja ≈ 1000-3000",
            },
        },
        "derivadas_detalhadas": {
            "condutividade_25C": {
                "nome": "Condutividade compensada a 25°C",
                "formula": "σ_25 = σ_T / (1 + 0.02 × (T - 25))",
                "descricao": "Normaliza a condutividade para 25°C, eliminando variações térmicas. Depende de: `temperatura_C` + `condutividade_uS`.",
                "depende_de": ["temperatura_C", "condutividade_uS"],
            },
        },
    },
    "🧪 pH (Módulo E-201-C)": {
        "colunas": ["pH"],
        "descricao": (
            "Eletrodo de pH com amplificador integrado. Mede a acidez/alcalinidade de "
            "0 a 14. Diferencia claramente cerveja (pH 4.0-4.5) de água mineral (pH 7.0-8.0), "
            "água contaminada (pH variável), e detecta adulteração em vinhos."
        ),
        "derivadas": [],
        "hardware": "Módulo pH E-201-C + amplificador BNC",
        "custo_aprox": "R$ 35",
        "features_detalhadas": {
            "pH": {
                "nome": "pH (Acidez/Alcalinidade)",
                "unidade": "escala 0-14",
                "descricao": "Mede a concentração de íons H⁺. pH < 7 = ácido, pH = 7 = neutro, pH > 7 = básico. Cerveja ≈ 4.0-4.5 | Água mineral ≈ 7.0-8.0 | Vinho ≈ 3.0-3.5 | Leite ≈ 6.5-6.8. Um dos sensores mais discriminativos do dispositivo.",
                "faixa_tipica": "Cerveja 4.0–4.5 | Água 6.5–8.0 | Vinho 3.0–3.5",
            },
        },
        "derivadas_detalhadas": {},
    },
    "🌈 Espectral (AS7341)": {
        "colunas": [
            "spec_F1_415nm", "spec_F2_445nm", "spec_F3_480nm", "spec_F4_515nm",
            "spec_F5_555nm", "spec_F6_590nm", "spec_F7_630nm", "spec_F8_680nm",
            "spec_Clear", "spec_NIR",
        ],
        "descricao": (
            "Sensor espectral multi-canal AS7341 da ams-OSRAM. 8 canais visíveis (415-680nm), "
            "1 canal Clear (luz total) e 1 NIR (infravermelho próximo). Cada canal mede a "
            "intensidade da luz em uma faixa estreita de comprimento de onda. Identifica cor, "
            "turbidez e composição — cerveja escura absorve mais no azul (415-480nm), água "
            "mineral limpa tem espectro quase plano, leite espalha fortemente."
        ),
        "derivadas": [
            "ratio_azul_vermelho", "ratio_verde_vermelho",
            "ratio_nir_clear", "spectral_mean", "spectral_std",
            "browning_index", "turbidity_index", "chlorophyll_proxy",
            "carotenoid_proxy", "sugar_proxy_nir", "spectral_slope",
            "ratio_violeta_vermelho",
            "absorption_index_415nm", "absorption_index_680nm",
            "spectral_entropy", "spectral_skewness", "spectral_kurtosis",
        ],
        "hardware": "AS7341 breakout board (I²C, 3.3V)",
        "custo_aprox": "R$ 45",
        "features_detalhadas": {
            "spec_F1_415nm": {
                "nome": "Canal F1 — Violeta",
                "unidade": "contagem (ADC)",
                "descricao": "Faixa 415nm (violeta). Absorção forte por compostos aromáticos e flavonoides — cervejas escuras e vinhos tintos absorvem muito neste canal.",
                "faixa_tipica": "Água: alto | Cerveja escura: baixo",
            },
            "spec_F2_445nm": {
                "nome": "Canal F2 — Azul escuro",
                "unidade": "contagem (ADC)",
                "descricao": "Faixa 445nm (azul escuro). Sensível a pigmentos como carotenoides e turbidez por partículas finas em suspensão.",
                "faixa_tipica": "Água limpa: alto | Líquido turvo: baixo",
            },
            "spec_F3_480nm": {
                "nome": "Canal F3 — Azul",
                "unidade": "contagem (ADC)",
                "descricao": "Faixa 480nm (azul). Forte absorção em cervejas âmbar/escuras; água limpa transmite bem. Usado na razão azul/vermelho.",
                "faixa_tipica": "Água: alto | IPA/Stout: baixo",
            },
            "spec_F4_515nm": {
                "nome": "Canal F4 — Verde",
                "unidade": "contagem (ADC)",
                "descricao": "Faixa 515nm (verde). Detecta clorofila (se presente) e diferencia líquidos turvos. Usado na razão verde/vermelho.",
                "faixa_tipica": "Água: alto | Cerveja: médio",
            },
            "spec_F5_555nm": {
                "nome": "Canal F5 — Verde-amarelo",
                "unidade": "contagem (ADC)",
                "descricao": "Faixa 555nm — pico de sensibilidade do olho humano. Bom indicador de cor geral e luminosidade percebida.",
                "faixa_tipica": "Transparente: alto | Opaco: baixo",
            },
            "spec_F6_590nm": {
                "nome": "Canal F6 — Laranja",
                "unidade": "contagem (ADC)",
                "descricao": "Faixa 590nm (laranja). Tom âmbar de cervejas, chá, sucos. Forte em IPAs e pale ales.",
                "faixa_tipica": "IPA/Pale Ale: absorção média | Água: alto",
            },
            "spec_F7_630nm": {
                "nome": "Canal F7 — Vermelho",
                "unidade": "contagem (ADC)",
                "descricao": "Faixa 630nm (vermelho). Absorção baixa em água, alta em líquidos escuros (stout, café). Denominador nas razões espectrais.",
                "faixa_tipica": "Água: alto | Stout/Café: baixo",
            },
            "spec_F8_680nm": {
                "nome": "Canal F8 — Vermelho escuro",
                "unidade": "contagem (ADC)",
                "descricao": "Faixa 680nm— clorofila A absorve fortemente aqui. Útil para detectar contaminação biológica (algas) em água.",
                "faixa_tipica": "Água limpa: alto | Água contaminada (algas): baixo",
            },
            "spec_Clear": {
                "nome": "Canal Clear — Luz visível total",
                "unidade": "contagem (ADC)",
                "descricao": "Banda larga cobrindo todo o visível. Mede a intensidade TOTAL de luz que atravessa o líquido. Proxy direto de turbidez — líquido turvo = menos luz.",
                "faixa_tipica": "Transparente: alto | Leite/Stout: muito baixo",
            },
            "spec_NIR": {
                "nome": "Canal NIR — Infravermelho próximo",
                "unidade": "contagem (ADC)",
                "descricao": "Faixa ~910nm (infravermelho). Água, álcool e açúcar absorvem em comprimentos de onda diferentes no NIR. Indica teor alcoólico e concentração de açúcares.",
                "faixa_tipica": "Água pura: referência | Cerveja: absorção por álcool+açúcar",
            },
        },
        "derivadas_detalhadas": {
            "ratio_azul_vermelho": {
                "nome": "Razão Azul/Vermelho",
                "formula": "spec_F3_480nm / spec_F7_630nm",
                "descricao": "Razão entre canal azul (480nm) e vermelho (630nm). ALTA em água limpa (transmite azul e vermelho igualmente), BAIXA em cerveja escura (absorve mais azul que vermelho). Excelente para separar tipo de líquido. É invariante à intensidade total da luz (LED).",
                "depende_de": ["spec_F3_480nm", "spec_F7_630nm"],
            },
            "ratio_verde_vermelho": {
                "nome": "Razão Verde/Vermelho",
                "formula": "spec_F4_515nm / spec_F7_630nm",
                "descricao": "Razão entre verde (515nm) e vermelho (630nm). Destaca tons ÂMBAR — IPAs e pale ales têm ratio diferente de stouts. Útil para classificação de estilos de cerveja.",
                "depende_de": ["spec_F4_515nm", "spec_F7_630nm"],
            },
            "ratio_nir_clear": {
                "nome": "Razão NIR/Clear",
                "formula": "spec_NIR / spec_Clear",
                "descricao": "Proporção de infravermelho em relação à luz visível total. Indica TEOR DE ÁLCOOL/AÇÚCAR relativo à turbidez — líquidos com mais álcool ou açúcar absorvem mais NIR proporcionalmente.",
                "depende_de": ["spec_NIR", "spec_Clear"],
            },
            "spectral_mean": {
                "nome": "Média Espectral",
                "formula": "mean(F1, F2, …, F8, Clear, NIR)",
                "descricao": "Média aritmética de TODOS os 10 canais espectrais. Indica o BRILHO MÉDIO do líquido — líquidos transparentes têm média alta, opacos têm baixa. Proxy robusto de turbidez geral.",
                "depende_de": ["spec_F1_415nm", "spec_F2_445nm", "spec_F3_480nm", "spec_F4_515nm",
                               "spec_F5_555nm", "spec_F6_590nm", "spec_F7_630nm", "spec_F8_680nm",
                               "spec_Clear", "spec_NIR"],
            },
            "spectral_std": {
                "nome": "Desvio Padrão Espectral",
                "formula": "std(F1, F2, …, F8, Clear, NIR)",
                "descricao": "Desvio padrão entre os 10 canais. Revela a 'FORMA' do espectro — líquidos COLORIDOS têm desvio alto (picos em algumas faixas), água LIMPA tem desvio baixo (espectro plano). Fundamental para subclassificação.",
                "depende_de": ["spec_F1_415nm", "spec_F2_445nm", "spec_F3_480nm", "spec_F4_515nm",
                               "spec_F5_555nm", "spec_F6_590nm", "spec_F7_630nm", "spec_F8_680nm",
                               "spec_Clear", "spec_NIR"],
            },
        },
    },
    "🔊 Acústico (Piezoelétrico)": {
        "colunas": ["acustico_freq_Hz"],
        "descricao": (
            "Disco piezoelétrico de 27mm excitado em frequência de ressonância. A frequência "
            "muda conforme a densidade e viscosidade do líquido em contato. Líquidos mais "
            "densos (cerveja, leite) abaixam a frequência; líquidos leves (água) mantêm mais "
            "alta. Complementa outros sensores especialmente para diferenciar subtipos."
        ),
        "derivadas": ["acoustic_anomaly"],
        "hardware": "Disco piezo 27mm + driver oscilador",
        "custo_aprox": "R$ 3",
        "features_detalhadas": {
            "acustico_freq_Hz": {
                "nome": "Frequência Acústica de Ressonância",
                "unidade": "Hz",
                "descricao": "Frequência de ressonância do disco piezoelétrico em contato com o líquido. A frequência DIMINUI com densidade e viscosidade: água ≈ freq. alta, leite/cerveja densa ≈ freq. baixa. Complementa os sensores ópticos para subtipos similares em cor.",
                "faixa_tipica": "Água: ~2800 Hz | Cerveja: ~2500 Hz | Leite: ~2300 Hz",
            },
        },
        "derivadas_detalhadas": {},
    },
}

# Mapeamento inverso: coluna → grupo
_COL_PARA_GRUPO = {}
for _g, _info in GRUPOS_SENSORES.items():
    for _c in _info["colunas"]:
        _COL_PARA_GRUPO[_c] = _g


# ============================================================
# Funções de dados
# ============================================================

@st.cache_data
def carregar_sintetico(n_amostras: int, seed: int):
    return gerar_dataset_completo(n_amostras_por_classe=n_amostras, seed=seed, salvar=False)


def preprocessar(df: pd.DataFrame, target_col: str):
    """Preprocessa um DataFrame para um classificador específico."""
    df = compensar_temperatura_condutividade(df)
    df = criar_features_espectrais(df)

    colunas_excluir = ["tipo", "subtipo", "potabilidade", "potabilidade_label", "adulteracao"]
    feature_names = [c for c in df.columns if c not in colunas_excluir and c != target_col]
    X = df[feature_names].values
    y_raw = df[target_col].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_names, "scaler": scaler, "label_encoder": le,
    }


def preprocessar_amostra(df_amostra, scaler, feature_names):
    """Preprocessa uma amostra nova para inferência."""
    df_amostra = compensar_temperatura_condutividade(df_amostra)
    df_amostra = criar_features_espectrais(df_amostra)
    X = df_amostra[feature_names].values
    return scaler.transform(X)


def preparar_todos_datasets(df_completo):
    """Prepara os datasets (um por classificador) a partir do dataset completo."""
    datasets = {}

    # 1. Tipo (dataset completo — 6 classes)
    datasets["tipo"] = preprocessar(df_completo, "tipo")

    # 2. Potabilidade da água
    df_agua = df_completo[df_completo["tipo"] == "agua"].copy()
    df_agua["potabilidade_label"] = df_agua["potabilidade"].map({1: "potavel", 0: "contaminada"})
    df_agua = df_agua.dropna(subset=["potabilidade_label"])
    datasets["agua_potabilidade"] = preprocessar(df_agua, "potabilidade_label")

    # 3. Variante da água
    df_agua2 = df_completo[df_completo["tipo"] == "agua"].copy()
    datasets["agua_variante"] = preprocessar(df_agua2, "subtipo")

    # 4. Estilo da cerveja (apenas cervejas normais, sem adulteradas)
    df_cerveja_normal = df_completo[
        (df_completo["tipo"] == "cerveja") & (df_completo["adulteracao"] == "normal")
    ].copy()
    datasets["cerveja_estilo"] = preprocessar(df_cerveja_normal, "subtipo")

    # 5. Adulteração da cerveja (normal vs metanol vs dietilenoglicol)
    df_cerveja = df_completo[df_completo["tipo"] == "cerveja"].copy()
    datasets["cerveja_adulteracao"] = preprocessar(df_cerveja, "adulteracao")

    # 6. Tipo de café
    df_cafe = df_completo[df_completo["tipo"] == "cafe"].copy()
    datasets["cafe_tipo"] = preprocessar(df_cafe, "subtipo")

    # 7. Tipo de chá
    df_cha = df_completo[df_completo["tipo"] == "cha"].copy()
    datasets["cha_tipo"] = preprocessar(df_cha, "subtipo")

    # 8. Fruta do suco
    df_suco = df_completo[df_completo["tipo"] == "suco"].copy()
    datasets["suco_fruta"] = preprocessar(df_suco, "subtipo")

    # 9. Sabor do refresco
    df_refresco = df_completo[df_completo["tipo"] == "refresco"].copy()
    datasets["refresco_sabor"] = preprocessar(df_refresco, "subtipo")

    return datasets


# ============================================================
# Funções de modelo
# ============================================================

def treinar_modelo(nome, params, X_train, y_train, n_classes):
    t0 = time.time()

    if nome == "Random Forest":
        modelo = RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"], random_state=RANDOM_SEED,
        )
        modelo.fit(X_train, y_train)

    elif nome == "XGBoost":
        modelo = XGBClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], random_state=RANDOM_SEED,
            eval_metric="mlogloss", use_label_encoder=False,
        )
        modelo.fit(X_train, y_train, verbose=False)

    elif nome == "SVM":
        modelo = SVC(
            kernel=params["kernel"], C=params["C"], gamma=params["gamma"],
            random_state=RANDOM_SEED, probability=True,
        )
        modelo.fit(X_train, y_train)

    elif nome == "Rede Neural (MLP)":
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(RANDOM_SEED)
        n_features = X_train.shape[1]
        modelo = Sequential(name="MLP")
        modelo.add(Input(shape=(n_features,)))
        for units in params["hidden_layers"]:
            modelo.add(Dense(units, activation="relu"))
            modelo.add(BatchNormalization())
            modelo.add(Dropout(0.3))
        if n_classes == 2:
            modelo.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            modelo.add(Dense(n_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
        modelo.compile(optimizer=Adam(learning_rate=params["learning_rate"]),
                       loss=loss, metrics=["accuracy"])
        modelo.fit(X_train, y_train, epochs=params["epochs"],
                   batch_size=params["batch_size"], verbose=0,
                   callbacks=[EarlyStopping(monitor="loss", patience=10,
                                            restore_best_weights=True)])

    elif nome == "CNN 1D":
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import (
            Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input,
        )
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(RANDOM_SEED)
        n_features = X_train.shape[1]
        X_train_cnn = X_train.reshape(-1, n_features, 1)
        modelo = Sequential(name="CNN1D")
        modelo.add(Input(shape=(n_features, 1)))
        for nf in params["filters"]:
            modelo.add(Conv1D(nf, params["kernel_size"], activation="relu", padding="same"))
            modelo.add(BatchNormalization())
            if n_features // 2 >= 2:
                modelo.add(MaxPooling1D(pool_size=2))
                n_features = n_features // 2
        modelo.add(Flatten())
        modelo.add(Dense(64, activation="relu"))
        modelo.add(Dropout(0.3))
        if n_classes == 2:
            modelo.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            modelo.add(Dense(n_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
        modelo.compile(optimizer=Adam(learning_rate=params["learning_rate"]),
                       loss=loss, metrics=["accuracy"])
        modelo.fit(X_train_cnn, y_train, epochs=params["epochs"],
                   batch_size=params["batch_size"], verbose=0,
                   callbacks=[EarlyStopping(monitor="loss", patience=10,
                                            restore_best_weights=True)])

    return modelo, time.time() - t0


def predict_modelo(modelo, nome, X):
    is_keras = nome in ("Rede Neural (MLP)", "CNN 1D")
    if is_keras:
        X_in = X.reshape(-1, X.shape[1], 1) if nome == "CNN 1D" else X
        y_proba = modelo.predict(X_in, verbose=0)
        if y_proba.shape[-1] == 1:
            y_pred = (y_proba.flatten() > 0.5).astype(int)
            y_proba_full = np.column_stack([1 - y_proba.flatten(), y_proba.flatten()])
        else:
            y_pred = np.argmax(y_proba, axis=1)
            y_proba_full = y_proba
    else:
        y_pred = modelo.predict(X)
        y_proba_full = modelo.predict_proba(X)
    return y_pred, y_proba_full


def calcular_feature_importance(modelo, nome, X_test, y_test, feature_names):
    is_keras = nome in ("Rede Neural (MLP)", "CNN 1D")

    if not is_keras and hasattr(modelo, "feature_importances_"):
        return pd.DataFrame({
            "Sensor": feature_names, "Importância": modelo.feature_importances_,
        }).sort_values("Importância", ascending=False).reset_index(drop=True)

    if not is_keras:
        result = permutation_importance(modelo, X_test, y_test,
                                        n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)
        return pd.DataFrame({
            "Sensor": feature_names, "Importância": result.importances_mean,
        }).sort_values("Importância", ascending=False).reset_index(drop=True)

    baseline_acc = accuracy_score(y_test, predict_modelo(modelo, nome, X_test)[0])
    importancias = []
    rng = np.random.default_rng(RANDOM_SEED)
    for i in range(X_test.shape[1]):
        X_perm = X_test.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_acc = accuracy_score(y_test, predict_modelo(modelo, nome, X_perm)[0])
        importancias.append(baseline_acc - perm_acc)
    return pd.DataFrame({
        "Sensor": feature_names, "Importância": importancias,
    }).sort_values("Importância", ascending=False).reset_index(drop=True)


# ============================================================
# Parâmetros default para cada modelo
# ============================================================

PARAMS_DEFAULT = {
    "Random Forest": {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5},
    "XGBoost": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
    "SVM": {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
    "Rede Neural (MLP)": {"hidden_layers": [128, 64, 32], "epochs": 100, "batch_size": 32, "learning_rate": 0.001},
    "CNN 1D": {"filters": [32, 64], "kernel_size": 3, "epochs": 100, "batch_size": 32, "learning_rate": 0.001},
}


# ============================================================
# Plots
# ============================================================

def plotar_importancia(df_imp, nome):
    df_plot = df_imp.sort_values("Importância", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_plot["Sensor"], df_plot["Importância"], color=sns.color_palette("viridis", len(df_plot)))
    ax.set_xlabel("Importância")
    ax.set_title(f"Top Sensores — {nome}")
    plt.tight_layout()
    return fig


def plotar_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    plt.tight_layout()
    return fig


# ============================================================
# Pipeline integrado de inferência
# ============================================================

def executar_pipeline_completo(X_amostra_raw, df_amostra, nome_modelo):
    """
    Executa o pipeline integrado para uma amostra:
      1. Classifica TIPO
      2. Com base no tipo, classifica PROPRIEDADES
    Retorna dict com todos os resultados.
    """
    pipeline = st.session_state.pipeline
    resultado = {}

    # --- ETAPA 1: Tipo ---
    dados_tipo = pipeline["datasets"]["tipo"]
    modelo_tipo = pipeline["modelos"]["tipo"][nome_modelo]["modelo"]
    X_tipo = preprocessar_amostra(df_amostra.copy(), dados_tipo["scaler"], dados_tipo["feature_names"])
    y_pred_tipo, y_proba_tipo = predict_modelo(modelo_tipo, nome_modelo, X_tipo)
    le_tipo = dados_tipo["label_encoder"]
    tipo_pred = le_tipo.inverse_transform(y_pred_tipo)[0]
    resultado["tipo"] = tipo_pred
    resultado["tipo_proba"] = {le_tipo.classes_[i]: float(y_proba_tipo[0][i])
                                for i in range(len(le_tipo.classes_))}

    # --- ETAPA 2: Propriedades ---
    if tipo_pred == "agua":
        # 2a: Potabilidade
        dados_pot = pipeline["datasets"]["agua_potabilidade"]
        modelo_pot = pipeline["modelos"]["agua_potabilidade"][nome_modelo]["modelo"]
        X_pot = preprocessar_amostra(df_amostra.copy(), dados_pot["scaler"], dados_pot["feature_names"])
        y_pred_pot, y_proba_pot = predict_modelo(modelo_pot, nome_modelo, X_pot)
        le_pot = dados_pot["label_encoder"]
        resultado["potabilidade"] = le_pot.inverse_transform(y_pred_pot)[0]
        resultado["potabilidade_proba"] = {le_pot.classes_[i]: float(y_proba_pot[0][i])
                                            for i in range(len(le_pot.classes_))}

        # 2b: Variante
        dados_var = pipeline["datasets"]["agua_variante"]
        modelo_var = pipeline["modelos"]["agua_variante"][nome_modelo]["modelo"]
        X_var = preprocessar_amostra(df_amostra.copy(), dados_var["scaler"], dados_var["feature_names"])
        y_pred_var, y_proba_var = predict_modelo(modelo_var, nome_modelo, X_var)
        le_var = dados_var["label_encoder"]
        resultado["variante"] = le_var.inverse_transform(y_pred_var)[0]
        resultado["variante_proba"] = {le_var.classes_[i]: float(y_proba_var[0][i])
                                        for i in range(len(le_var.classes_))}

    elif tipo_pred == "cerveja":
        # 2c: Estilo (apenas estilos normais)
        dados_est = pipeline["datasets"]["cerveja_estilo"]
        modelo_est = pipeline["modelos"]["cerveja_estilo"][nome_modelo]["modelo"]
        X_est = preprocessar_amostra(df_amostra.copy(), dados_est["scaler"], dados_est["feature_names"])
        y_pred_est, y_proba_est = predict_modelo(modelo_est, nome_modelo, X_est)
        le_est = dados_est["label_encoder"]
        resultado["estilo"] = le_est.inverse_transform(y_pred_est)[0]
        resultado["estilo_proba"] = {le_est.classes_[i]: float(y_proba_est[0][i])
                                      for i in range(len(le_est.classes_))}

        # 2d: Adulteração
        dados_adult = pipeline["datasets"]["cerveja_adulteracao"]
        modelo_adult = pipeline["modelos"]["cerveja_adulteracao"][nome_modelo]["modelo"]
        X_adult = preprocessar_amostra(df_amostra.copy(), dados_adult["scaler"], dados_adult["feature_names"])
        y_pred_adult, y_proba_adult = predict_modelo(modelo_adult, nome_modelo, X_adult)
        le_adult = dados_adult["label_encoder"]
        resultado["adulteracao"] = le_adult.inverse_transform(y_pred_adult)[0]
        resultado["adulteracao_proba"] = {le_adult.classes_[i]: float(y_proba_adult[0][i])
                                           for i in range(len(le_adult.classes_))}

    elif tipo_pred == "cafe":
        dados_cafe = pipeline["datasets"]["cafe_tipo"]
        modelo_cafe = pipeline["modelos"]["cafe_tipo"][nome_modelo]["modelo"]
        X_cafe = preprocessar_amostra(df_amostra.copy(), dados_cafe["scaler"], dados_cafe["feature_names"])
        y_pred_cafe, y_proba_cafe = predict_modelo(modelo_cafe, nome_modelo, X_cafe)
        le_cafe = dados_cafe["label_encoder"]
        resultado["tipo_cafe"] = le_cafe.inverse_transform(y_pred_cafe)[0]
        resultado["tipo_cafe_proba"] = {le_cafe.classes_[i]: float(y_proba_cafe[0][i])
                                         for i in range(len(le_cafe.classes_))}

    elif tipo_pred == "cha":
        dados_cha = pipeline["datasets"]["cha_tipo"]
        modelo_cha = pipeline["modelos"]["cha_tipo"][nome_modelo]["modelo"]
        X_cha = preprocessar_amostra(df_amostra.copy(), dados_cha["scaler"], dados_cha["feature_names"])
        y_pred_cha, y_proba_cha = predict_modelo(modelo_cha, nome_modelo, X_cha)
        le_cha = dados_cha["label_encoder"]
        resultado["tipo_cha"] = le_cha.inverse_transform(y_pred_cha)[0]
        resultado["tipo_cha_proba"] = {le_cha.classes_[i]: float(y_proba_cha[0][i])
                                        for i in range(len(le_cha.classes_))}

    elif tipo_pred == "suco":
        dados_suco = pipeline["datasets"]["suco_fruta"]
        modelo_suco = pipeline["modelos"]["suco_fruta"][nome_modelo]["modelo"]
        X_suco = preprocessar_amostra(df_amostra.copy(), dados_suco["scaler"], dados_suco["feature_names"])
        y_pred_suco, y_proba_suco = predict_modelo(modelo_suco, nome_modelo, X_suco)
        le_suco = dados_suco["label_encoder"]
        resultado["fruta"] = le_suco.inverse_transform(y_pred_suco)[0]
        resultado["fruta_proba"] = {le_suco.classes_[i]: float(y_proba_suco[0][i])
                                     for i in range(len(le_suco.classes_))}

    elif tipo_pred == "refresco":
        dados_refr = pipeline["datasets"]["refresco_sabor"]
        modelo_refr = pipeline["modelos"]["refresco_sabor"][nome_modelo]["modelo"]
        X_refr = preprocessar_amostra(df_amostra.copy(), dados_refr["scaler"], dados_refr["feature_names"])
        y_pred_refr, y_proba_refr = predict_modelo(modelo_refr, nome_modelo, X_refr)
        le_refr = dados_refr["label_encoder"]
        resultado["sabor"] = le_refr.inverse_transform(y_pred_refr)[0]
        resultado["sabor_proba"] = {le_refr.classes_[i]: float(y_proba_refr[0][i])
                                     for i in range(len(le_refr.classes_))}

    return resultado


def exibir_resultado_pipeline(resultado, nome_modelo):
    """Mostra o resultado completo de um pipeline para uma amostra."""
    tipo = resultado["tipo"]
    conf_tipo = max(resultado["tipo_proba"].values()) * 100

    ICONES_TIPO = {
        "agua": "💧", "cerveja": "🍺", "cafe": "☕",
        "cha": "🍵", "suco": "🧃", "refresco": "🥤",
    }
    icone = ICONES_TIPO.get(tipo, "🧪")

    if tipo == "agua":
        pot = resultado.get("potabilidade", "?")
        variante = resultado.get("variante", "?")
        conf_pot = max(resultado.get("potabilidade_proba", {0: 0}).values()) * 100
        conf_var = max(resultado.get("variante_proba", {0: 0}).values()) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("1️⃣ Tipo", f"{icone} {tipo.upper()}", f"{conf_tipo:.1f}%")
        col2.metric("2️⃣ Potabilidade", pot.upper(), f"{conf_pot:.1f}%")
        col3.metric("3️⃣ Variante", variante.replace("agua_", "").replace("_", " ").title(), f"{conf_var:.1f}%")

        if pot == "contaminada":
            st.error("⚠️ **ALERTA: Água classificada como CONTAMINADA — NÃO POTÁVEL!**")
        else:
            st.success("✅ Água classificada como **POTÁVEL**.")

    elif tipo == "cerveja":
        estilo = resultado.get("estilo", "?")
        adult = resultado.get("adulteracao", "?")
        conf_est = max(resultado.get("estilo_proba", {0: 0}).values()) * 100
        conf_adult = max(resultado.get("adulteracao_proba", {0: 0}).values()) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("1️⃣ Tipo", f"{icone} {tipo.upper()}", f"{conf_tipo:.1f}%")
        col2.metric("2️⃣ Estilo", estilo.replace("cerveja_", "").title(), f"{conf_est:.1f}%")
        col3.metric("3️⃣ Adulteração", adult.upper(), f"{conf_adult:.1f}%")

        if adult != "normal":
            st.error(
                f"🚨 **ALERTA DE ADULTERAÇÃO!** Contaminante detectado: **{adult.upper()}**\n\n"
                f"Referência: Caso Backer/Belorizontina (2019-2020) — {adult} é tóxico!"
            )
        else:
            st.success("✅ Cerveja sem sinais de adulteração.")

    elif tipo == "cafe":
        tipo_cafe = resultado.get("tipo_cafe", "?")
        conf_cafe = max(resultado.get("tipo_cafe_proba", {0: 0}).values()) * 100
        col1, col2 = st.columns(2)
        col1.metric("1️⃣ Tipo", f"{icone} {tipo.upper()}", f"{conf_tipo:.1f}%")
        col2.metric("2️⃣ Tipo de Café", tipo_cafe.replace("cafe_", "").title(), f"{conf_cafe:.1f}%")

    elif tipo == "cha":
        tipo_cha = resultado.get("tipo_cha", "?")
        conf_cha = max(resultado.get("tipo_cha_proba", {0: 0}).values()) * 100
        col1, col2 = st.columns(2)
        col1.metric("1️⃣ Tipo", f"{icone} {tipo.upper()}", f"{conf_tipo:.1f}%")
        col2.metric("2️⃣ Tipo de Chá", tipo_cha.replace("cha_", "").title(), f"{conf_cha:.1f}%")

    elif tipo == "suco":
        fruta = resultado.get("fruta", "?")
        conf_fruta = max(resultado.get("fruta_proba", {0: 0}).values()) * 100
        col1, col2 = st.columns(2)
        col1.metric("1️⃣ Tipo", f"{icone} SUCO NATURAL", f"{conf_tipo:.1f}%")
        col2.metric("2️⃣ Fruta", fruta.replace("suco_", "").title(), f"{conf_fruta:.1f}%")

    elif tipo == "refresco":
        sabor = resultado.get("sabor", "?")
        conf_sabor = max(resultado.get("sabor_proba", {0: 0}).values()) * 100
        col1, col2 = st.columns(2)
        col1.metric("1️⃣ Tipo", f"{icone} REFRESCO", f"{conf_tipo:.1f}%")
        col2.metric("2️⃣ Sabor", sabor.replace("refresco_", "").title(), f"{conf_sabor:.1f}%")
        st.info("ℹ️ **Refresco** = bebida pronta diluída (20-30% suco), diferente de suco natural integral.")

    else:
        st.metric("1️⃣ Tipo", f"{icone} {tipo.upper()}", f"{conf_tipo:.1f}%")


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def main():
    # --- Logo UFBA no sidebar ---
    logo_path = os.path.join(PROJECT_ROOT, "assets", "ufbalogo.webp")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=180)
        st.sidebar.markdown("---")

    st.title("🧪 Dispositivo S — Identificação de Líquidos")
    st.caption(
        "Pipeline **integrado**: a amostra passa automaticamente por todas as etapas "
        "(tipo → propriedades específicas)."
    )

    # --- Diagrama do pipeline ---
    with st.expander("📐 Pipeline Hierárquico", expanded=False):
        st.markdown("""
```
              ┌──────────────────────────────────┐
              │  AMOSTRA DE LÍQUIDO (sensores)   │
              └───────────────┬──────────────────┘
                              ▼
              ┌──────────────────────────────────┐
              │  ETAPA 1: Qual TIPO de bebida?   │
              │  água/cerveja/café/chá/suco/refr.│
              └──┬──────┬──────┬───┬───┬────┬────┘
                 ▼      ▼      ▼   ▼   ▼    ▼
         ┌──────┐ ┌──────┐ ┌───┐ ┌──┐ ┌───┐ ┌─────┐
         │ ÁGUA │ │CERVEJA│ │CAFÉ│ │CHÁ│ │SUCO│ │REFR.│
         └┬──┬──┘ └┬──┬──┘ └─┬─┘ └─┬┘ └─┬─┘ └──┬──┘
          ▼  ▼     ▼  ▼      ▼     ▼    ▼      ▼
    ┌─────┐┌────┐┌────┐┌─────┐┌──┐┌──┐┌─────┐┌─────┐
    │Potab.││Var.││Est.││Adult.││  ││  ││Fruta││Sabor│
    └─────┘└────┘└────┘└─────┘└──┘└──┘└─────┘└─────┘
```
        """)

    # --- Sidebar: Configurações ---
    st.sidebar.header("⚙️ Configurações")
    n_amostras = st.sidebar.slider("Amostras por classe", 100, 2000, 500, step=100)

    modelo_escolhido = st.sidebar.selectbox(
        "Modelo de ML",
        NOMES_MODELOS,
        index=0,
    )

    # --- Inicializar pipeline ---
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = {"treinado": False, "modelos": {}, "datasets": {}, "acuracias": {}}

    # --- Carregar e preparar dados ---
    df_completo = carregar_sintetico(n_amostras, RANDOM_SEED)

    # --- Abas principais ---
    tab_treino, tab_analisar, tab_detalhes, tab_sensores, tab_ablation, tab_import = st.tabs([
        "🏋️ Treinar Pipeline",
        "🔬 Analisar Amostra",
        "📊 Detalhes dos Classificadores",
        "🏆 Importância dos Sensores",
        "🔬 Ablação de Sensores",
        "📥 Importar Dataset",
    ])

    # ==========================================================
    # ABA 1: TREINAR PIPELINE COMPLETO
    # ==========================================================
    with tab_treino:
        st.header("🏋️ Treinar Pipeline Completo")
        st.markdown(
            f"Treina **{modelo_escolhido}** para os **{len(CLASSIFICADORES)} classificadores** do pipeline de uma vez."
        )

        # Parâmetros ajustáveis
        st.subheader("Hiperparâmetros")
        params = _render_params(modelo_escolhido)

        if st.button("🚀 Treinar Pipeline Inteiro", key="btn_treinar", width="stretch"):
            datasets = preparar_todos_datasets(df_completo)
            st.session_state.pipeline["datasets"] = datasets

            progress = st.progress(0, text="Preparando...")
            modelos_treinados = {}
            acuracias = {}
            total_steps = len(CLASSIFICADORES)

            for idx, (clf_key, clf_info) in enumerate(CLASSIFICADORES.items()):
                progress.progress(
                    (idx) / total_steps,
                    text=f"Treinando: {clf_info['nome']}..."
                )
                dados = datasets[clf_key]
                n_classes = len(dados["label_encoder"].classes_)

                modelo, tempo = treinar_modelo(
                    modelo_escolhido, params,
                    dados["X_train"], dados["y_train"], n_classes,
                )
                y_pred_test, _ = predict_modelo(modelo, modelo_escolhido, dados["X_test"])
                acc = accuracy_score(dados["y_test"], y_pred_test)

                imp = calcular_feature_importance(
                    modelo, modelo_escolhido,
                    dados["X_test"], dados["y_test"],
                    dados["feature_names"],
                )

                modelos_treinados[clf_key] = {
                    modelo_escolhido: {
                        "modelo": modelo, "acc": acc, "tempo": tempo,
                        "y_pred": y_pred_test, "importancia": imp,
                    }
                }
                acuracias[clf_key] = acc

            # Mesclar com modelos existentes de outros algoritmos
            for clf_key in CLASSIFICADORES:
                if clf_key not in st.session_state.pipeline["modelos"]:
                    st.session_state.pipeline["modelos"][clf_key] = {}
                st.session_state.pipeline["modelos"][clf_key].update(modelos_treinados[clf_key])

            st.session_state.pipeline["acuracias"] = acuracias
            st.session_state.pipeline["treinado"] = True
            st.session_state.pipeline["ultimo_modelo"] = modelo_escolhido

            progress.progress(1.0, text="Pipeline treinado!")

            # Mostrar resumo
            st.success("✅ Pipeline completo treinado com sucesso!")
            n_cols = 3
            cols = st.columns(n_cols)
            for i, (clf_key, clf_info) in enumerate(CLASSIFICADORES.items()):
                acc = acuracias[clf_key]
                cols[i % n_cols].metric(clf_info["nome"], f"{acc*100:.1f}%")

        # Mostrar status atual
        if st.session_state.pipeline["treinado"]:
            st.markdown("---")
            st.subheader("Status do Pipeline")

            modelos_disponiveis = set()
            for clf_key in CLASSIFICADORES:
                mods = st.session_state.pipeline["modelos"].get(clf_key, {})
                modelos_disponiveis.update(mods.keys())

            rows = []
            for clf_key, clf_info in CLASSIFICADORES.items():
                mods = st.session_state.pipeline["modelos"].get(clf_key, {})
                for nm in sorted(modelos_disponiveis):
                    if nm in mods:
                        rows.append({
                            "Classificador": clf_info["nome"],
                            "Modelo": nm,
                            "Acurácia (%)": f"{mods[nm]['acc']*100:.2f}",
                            "Tempo (s)": f"{mods[nm]['tempo']:.2f}",
                        })

            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ==========================================================
    # ABA 2: ANALISAR AMOSTRA (PIPELINE COMPLETO AUTOMÁTICO)
    # ==========================================================
    with tab_analisar:
        st.header("🔬 Analisar Amostra")

        if not st.session_state.pipeline["treinado"]:
            st.warning("⚠️ Treine o pipeline primeiro na aba **Treinar Pipeline**.")
        else:
            st.markdown(
                "A amostra passa automaticamente por **todas as etapas**: "
                "tipo → potabilidade/variante (água) ou estilo (cerveja)."
            )

            # Selecionar modelo para inferência
            modelos_treinados_keys = set()
            for clf_key in CLASSIFICADORES:
                mods = st.session_state.pipeline["modelos"].get(clf_key, {})
                modelos_treinados_keys.update(mods.keys())
            # Só mostrar modelos que foram treinados para TODOS os classificadores
            modelos_completos = [nm for nm in modelos_treinados_keys
                                  if all(nm in st.session_state.pipeline["modelos"].get(k, {})
                                         for k in CLASSIFICADORES)]

            if not modelos_completos:
                st.warning("Nenhum modelo treinado para todos os classificadores.")
            else:
                modelo_inf = st.selectbox("Modelo para inferência", modelos_completos, key="modelo_inf")

                st.markdown("---")
                st.markdown("### Opção 1: Gerar amostras sintéticas")

                col1, col2 = st.columns([1, 1])
                with col1:
                    liquido = st.selectbox(
                        "Líquido", list(PERFIS_SENSORES.keys()),
                        format_func=lambda x: x.replace("_", " ").title(),
                    )
                    n_test = st.slider("Quantidade de amostras", 1, 20, 3, key="ns")

                with col2:
                    st.markdown(""); st.markdown("")
                    gerar = st.button("🧪 Analisar", key="bg", width="stretch")

                if gerar:
                    rng = np.random.default_rng(int(time.time()) % 2**31)
                    df_am = gerar_amostras_liquido(liquido, n_test, rng)

                    # Info real
                    real_tipo = MAPA_TIPO[liquido]
                    real_pot = {1: "potavel", 0: "contaminada"}.get(MAPA_POTABILIDADE.get(liquido, -1), "n/a")

                    st.markdown("---")
                    st.markdown(f"#### Líquido real: **{liquido.replace('_',' ').title()}**")
                    st.caption(f"Tipo: {real_tipo} | Subtipo: {liquido} | Potabilidade: {real_pot}")

                    for idx in range(n_test):
                        df_single = df_am.iloc[[idx]].copy()
                        resultado = executar_pipeline_completo(None, df_single, modelo_inf)

                        with st.expander(f"Amostra #{idx+1}", expanded=(idx == 0)):
                            exibir_resultado_pipeline(resultado, modelo_inf)

                st.markdown("---")
                st.markdown("### Opção 2: Inserir leitura dos sensores manualmente")
                with st.form("manual_input"):
                    st.markdown("Preencha os valores de cada sensor:")
                    cols_form = st.columns(3)
                    vals = {}
                    for i, s in enumerate(SENSOR_NAMES):
                        with cols_form[i % 3]:
                            if "temperatura" in s:
                                vals[s] = st.number_input(s, value=20.0, format="%.2f", key=f"m_{s}")
                            elif "condutividade" in s:
                                vals[s] = st.number_input(s, value=500.0, format="%.1f", key=f"m_{s}")
                            elif s == "pH":
                                vals[s] = st.number_input(s, value=7.0, min_value=0.0, max_value=14.0,
                                                          format="%.2f", key=f"m_{s}")
                            elif s.startswith("spec_"):
                                vals[s] = st.number_input(s, value=0.5, min_value=0.0, max_value=1.0,
                                                          format="%.3f", key=f"m_{s}")
                            elif "acustico" in s:
                                vals[s] = st.number_input(s, value=1470.0, format="%.1f", key=f"m_{s}")
                            else:
                                vals[s] = st.number_input(s, value=0.0, format="%.4f", key=f"m_{s}")
                    submitted = st.form_submit_button("🔍 Analisar Amostra", width="stretch")

                if submitted:
                    df_m = pd.DataFrame([vals])
                    df_m["tipo"] = ""; df_m["subtipo"] = ""
                    df_m["potabilidade"] = -1; df_m["potabilidade_label"] = ""

                    st.markdown("---")
                    st.subheader("Resultado Completo do Pipeline")
                    resultado = executar_pipeline_completo(None, df_m, modelo_inf)
                    exibir_resultado_pipeline(resultado, modelo_inf)

    # ==========================================================
    # ABA 3: DETALHES DOS CLASSIFICADORES
    # ==========================================================
    with tab_detalhes:
        st.header("📊 Detalhes dos Classificadores")

        if not st.session_state.pipeline["treinado"]:
            st.warning("⚠️ Treine o pipeline primeiro.")
        else:
            datasets = st.session_state.pipeline["datasets"]
            modelos_pipeline = st.session_state.pipeline["modelos"]

            for clf_key, clf_info in CLASSIFICADORES.items():
                with st.expander(f"**{clf_info['nome']}**", expanded=False):
                    dados = datasets[clf_key]
                    classes = list(dados["label_encoder"].classes_)
                    mods = modelos_pipeline.get(clf_key, {})

                    for nm, info in mods.items():
                        st.markdown(f"**{nm}** — Acurácia: **{info['acc']*100:.2f}%** | Tempo: {info['tempo']:.2f}s")
                        tab_cm, tab_imp = st.tabs(["Matriz de Confusão", "Importância Sensores"])
                        with tab_cm:
                            st.pyplot(plotar_confusion_matrix(dados["y_test"], info["y_pred"], classes))
                        with tab_imp:
                            st.pyplot(plotar_importancia(info["importancia"], f"{nm} — {clf_info['nome']}"))
                            st.dataframe(info["importancia"], width="stretch", hide_index=True)

    # ==========================================================
    # ABA 4: IMPORTÂNCIA DOS SENSORES (CONSOLIDADA)
    # ==========================================================
    with tab_sensores:
        st.header("🏆 Importância dos Sensores — Consolidada")

        if not st.session_state.pipeline["treinado"]:
            st.warning("⚠️ Treine o pipeline primeiro.")
        else:
            st.markdown("Ranking consolidado dos sensores mais importantes, "
                        "combinando todas as etapas e modelos treinados.")

            all_imps = []
            for clf_key, clf_info in CLASSIFICADORES.items():
                mods = st.session_state.pipeline["modelos"].get(clf_key, {})
                for nm, info in mods.items():
                    imp = info["importancia"].copy()
                    mx = imp["Importância"].max()
                    if mx > 0:
                        imp["Importância"] = imp["Importância"] / mx
                    imp["Classificador"] = clf_info["nome"]
                    imp["Modelo"] = nm
                    all_imps.append(imp)

            if all_imps:
                df_all = pd.concat(all_imps, ignore_index=True)
                consenso = (df_all.groupby("Sensor")["Importância"]
                            .mean()
                            .sort_values(ascending=False)
                            .reset_index())
                consenso.columns = ["Sensor", "Importância Média Normalizada"]

                fig, ax = plt.subplots(figsize=(10, 6))
                df_plot = consenso.sort_values("Importância Média Normalizada", ascending=True).tail(15)
                colors = sns.color_palette("viridis", len(df_plot))
                ax.barh(df_plot["Sensor"], df_plot["Importância Média Normalizada"], color=colors)
                ax.set_xlabel("Importância Média Normalizada")
                ax.set_title("Top Sensores — Consolidado (Todas Etapas + Modelos)")
                plt.tight_layout()
                st.pyplot(fig)

                st.dataframe(consenso, width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("🛒 Recomendação para o MVP")
                top5 = consenso.head(5)
                for _, row in top5.iterrows():
                    st.markdown(f"- **{row['Sensor']}** (importância: {row['Importância Média Normalizada']:.3f})")

    # ==========================================================
    # ABA 5: ABLAÇÃO DE SENSORES
    # ==========================================================
    with tab_ablation:
        _render_tab_ablation(df_completo, modelo_escolhido)

    # ==========================================================
    # ABA 6: IMPORTAR DATASET VIA URL
    # ==========================================================
    with tab_import:
        _render_tab_importar()

    # --- Sidebar resumo ---
    _render_sidebar_resumo()


# ============================================================
# Tab: Ablação de Sensores
# ============================================================

def _render_tab_ablation(df_completo, modelo_escolhido):
    st.header("🔬 Ablação de Sensores")
    st.markdown(
        "Desative sensores individualmente ou em grupo para medir o impacto na "
        "acurácia. Isso ajuda a decidir **quais sensores são realmente "
        "necessários** no hardware final e reduzir custo."
    )

    # --- Referência detalhada + Seleção integrada ---
    st.subheader("⚙️ Sensores — Referência & Seleção")
    st.caption(
        "Cada card abaixo descreve **um grupo de sensor**, suas features brutas, "
        "features derivadas e custo. **Desmarque** o grupo que deseja REMOVER "
        "do treinamento para medir o impacto na acurácia."
    )

    sensores_ativos = {}

    for grupo, info in GRUPOS_SENSORES.items():
        with st.expander(grupo, expanded=False):
            # --- Checkbox de ativação no topo do card ---
            ativo = st.checkbox(
                "✅ Sensor ATIVO no treinamento",
                value=True,
                key=f"abl_{grupo}",
            )
            sensores_ativos[grupo] = ativo

            st.markdown(f"**{info['descricao']}**")
            st.markdown(
                f"🔧 Hardware: {info['hardware']}  \n"
                f"💰 Custo estimado: {info['custo_aprox']}"
            )

            # --- Tabela de features brutas ---
            st.markdown("##### 📊 Features brutas")
            feat_rows = []
            for col, fd in info.get("features_detalhadas", {}).items():
                feat_rows.append({
                    "Coluna": col,
                    "Nome": fd["nome"],
                    "Unidade": fd.get("unidade", "—"),
                    "Faixa típica": fd.get("faixa_tipica", "—"),
                    "O que mede": fd["descricao"],
                })
            if feat_rows:
                st.dataframe(
                    pd.DataFrame(feat_rows),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                cols_str = ", ".join(f"`{c}`" for c in info["colunas"])
                st.markdown(f"Colunas: {cols_str}")

            # --- Features derivadas (com fórmula e explicação) ---
            derivadas_det = info.get("derivadas_detalhadas", {})
            if derivadas_det:
                st.markdown("##### 🧮 Features derivadas (calculadas a partir deste sensor)")
                for dcol, dd in derivadas_det.items():
                    deps = ", ".join(f"`{d}`" for d in dd.get("depende_de", []))
                    st.markdown(
                        f"**`{dcol}`** — {dd['nome']}  \n"
                        f"Fórmula: `{dd['formula']}`  \n"
                        f"Depende de: {deps}  \n"
                        f"{dd['descricao']}"
                    )
                st.info(
                    "⚠️ Ao remover este sensor, as features derivadas acima "
                    "também serão removidas do treinamento."
                )

    # Calcular quais colunas ficarão
    colunas_removidas = []
    derivadas_removidas = []
    for grupo, info in GRUPOS_SENSORES.items():
        if not sensores_ativos[grupo]:
            colunas_removidas.extend(info["colunas"])
            derivadas_removidas.extend(info["derivadas"])

    n_ativos = sum(1 for v in sensores_ativos.values() if v)
    n_total = len(GRUPOS_SENSORES)

    if n_ativos == 0:
        st.error("❌ Selecione pelo menos 1 grupo de sensores.")
        return

    if colunas_removidas:
        st.info(
            f"🗑️ **Removendo {len(colunas_removidas)} colunas brutas + "
            f"{len(derivadas_removidas)} derivadas:** "
            f"{', '.join(f'`{c}`' for c in colunas_removidas + derivadas_removidas)}"
        )
    else:
        st.success("✅ Todos os sensores ativos — pipeline completo.")

    # --- Botão de treino ablação ---
    st.markdown("---")
    classificadores_ablacao = st.multiselect(
        "Classificadores a testar",
        list(CLASSIFICADORES.keys()),
        default=list(CLASSIFICADORES.keys()),
        format_func=lambda k: CLASSIFICADORES[k]["nome"],
        key="abl_clfs",
    )

    if st.button("🚀 Treinar com sensores selecionados", key="btn_ablation"):
        if not classificadores_ablacao:
            st.error("Selecione pelo menos 1 classificador.")
            return

        # Preprocessar removendo colunas
        all_remove = set(colunas_removidas + derivadas_removidas)
        results = {}

        progress = st.progress(0, text="Preparando ablação…")
        total = len(classificadores_ablacao)

        for idx, clf_key in enumerate(classificadores_ablacao):
            clf_info = CLASSIFICADORES[clf_key]
            progress.progress(idx / total, text=f"Treinando {clf_info['nome']}…")

            # Preparar dataset específico
            if clf_info["filtro"] is None:
                df_subset = df_completo.copy()
            elif clf_info["filtro"] == "cerveja_normal":
                df_subset = df_completo[
                    (df_completo["tipo"] == "cerveja") & (df_completo["adulteracao"] == "normal")
                ].copy()
            else:
                df_subset = df_completo[df_completo["tipo"] == clf_info["filtro"]].copy()

            if clf_key == "agua_potabilidade":
                df_subset["potabilidade_label"] = df_subset["potabilidade"].map(
                    {1: "potavel", 0: "contaminada"}
                )
                df_subset = df_subset.dropna(subset=["potabilidade_label"])

            target_col = clf_info["target"]

            # Aplicar preprocessing
            df_proc = compensar_temperatura_condutividade(df_subset)
            df_proc = criar_features_espectrais(df_proc)

            colunas_excluir = {"tipo", "subtipo", "potabilidade", "potabilidade_label", "adulteracao", target_col}
            feature_names = [
                c for c in df_proc.columns
                if c not in colunas_excluir and c not in all_remove
            ]

            if len(feature_names) < 1:
                results[clf_key] = {"acc": 0.0, "n_features": 0, "erro": "Sem features"}
                continue

            X = df_proc[feature_names].values
            y_raw = df_proc[target_col].values

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            n_classes = len(le.classes_)

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y,
            )
            _, X_test, _, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp,
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Usar parâmetros default rápidos
            import copy
            params_rapidos = copy.deepcopy(PARAMS_DEFAULT[modelo_escolhido])
            # Reduzir épocas para treino rápido de ablação
            if "epochs" in params_rapidos:
                params_rapidos["epochs"] = min(params_rapidos["epochs"], 30)

            modelo, tempo = treinar_modelo(
                modelo_escolhido, params_rapidos, X_train, y_train, n_classes,
            )
            y_pred, _ = predict_modelo(modelo, modelo_escolhido, X_test)
            acc = accuracy_score(y_test, y_pred)

            results[clf_key] = {
                "acc": acc,
                "n_features": len(feature_names),
                "features": feature_names,
                "tempo": tempo,
            }

        progress.progress(1.0, text="Concluído!")

        # --- Salvar no session state para comparação ---
        config_key = ",".join(sorted(g for g, a in sensores_ativos.items() if a))
        if "ablation_history" not in st.session_state:
            st.session_state.ablation_history = []

        entry = {
            "config": config_key,
            "sensores_ativos": {g: a for g, a in sensores_ativos.items()},
            "modelo": modelo_escolhido,
            "n_sensores": n_ativos,
            "resultados": {k: v["acc"] for k, v in results.items()},
            "media": np.mean([v["acc"] for v in results.values()]),
        }
        st.session_state.ablation_history.append(entry)

        # --- Mostrar resultados ---
        st.success(f"✅ Ablação concluída — **{modelo_escolhido}** com **{n_ativos}/{n_total}** grupos de sensores")

        cols_res = st.columns(len(results))
        for i, (clf_key, res) in enumerate(results.items()):
            nome = CLASSIFICADORES[clf_key]["nome"]
            if "erro" in res:
                cols_res[i].metric(nome, "❌")
            else:
                # Comparar com pipeline completo se disponível
                delta = None
                baseline = st.session_state.pipeline.get("acuracias", {}).get(clf_key)
                if baseline is not None:
                    delta = f"{(res['acc'] - baseline)*100:+.1f}pp"
                cols_res[i].metric(
                    nome,
                    f"{res['acc']*100:.1f}%",
                    delta=delta,
                    delta_color="normal",
                )

        st.caption(
            f"Features usadas: {sum(r.get('n_features', 0) for r in results.values()) // max(len(results), 1)} "
            f"| Sensores removidos: {', '.join(g.split(' ')[1] for g, a in sensores_ativos.items() if not a) or 'nenhum'}"
        )

    # --- Histórico de ablações ---
    if "ablation_history" in st.session_state and st.session_state.ablation_history:
        st.markdown("---")
        st.subheader("📈 Histórico de Ablações")

        hist = st.session_state.ablation_history
        rows = []
        for i, h in enumerate(hist):
            ativos = [g.split(" ")[1] for g, a in h["sensores_ativos"].items() if a]
            removidos = [g.split(" ")[1] for g, a in h["sensores_ativos"].items() if not a]
            row = {
                "#": i + 1,
                "Modelo": h["modelo"],
                "Sensores Ativos": " + ".join(ativos),
                "Removidos": ", ".join(removidos) if removidos else "—",
                "Grupos": f"{h['n_sensores']}/{n_total}",
            }
            for clf_key in CLASSIFICADORES:
                nome = CLASSIFICADORES[clf_key]["nome"]
                acc = h["resultados"].get(clf_key)
                row[nome] = f"{acc*100:.1f}%" if acc is not None else "—"
            row["Média"] = f"{h['media']*100:.1f}%"
            rows.append(row)

        df_hist = pd.DataFrame(rows)
        st.dataframe(df_hist, hide_index=True)

        # Gráfico de comparação
        if len(hist) >= 2:
            fig, ax = plt.subplots(figsize=(10, 5))
            x_labels = [f"#{r['#']}\n{r['Removidos']}" for r in rows]
            clf_nomes = [CLASSIFICADORES[k]["nome"] for k in CLASSIFICADORES]
            x = np.arange(len(rows))
            width = 0.8 / len(clf_nomes)

            for j, clf_key in enumerate(CLASSIFICADORES):
                nome = CLASSIFICADORES[clf_key]["nome"]
                vals = []
                for h in hist:
                    acc = h["resultados"].get(clf_key, 0)
                    vals.append(acc * 100)
                ax.bar(x + j * width, vals, width, label=nome, alpha=0.85)

            ax.set_xlabel("Configuração (sensores removidos)")
            ax.set_ylabel("Acurácia (%)")
            ax.set_title("Comparação de Ablações")
            ax.set_xticks(x + width * (len(clf_nomes) - 1) / 2)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 105)
            plt.tight_layout()
            st.pyplot(fig)

        if st.button("🗑️ Limpar histórico", key="btn_clear_abl"):
            st.session_state.ablation_history = []
            st.rerun()


# ============================================================
# Tab: Importar Dataset
# ============================================================

def _render_tab_importar():
    st.header("📥 Importar Dataset")
    st.markdown(
        "Escolha um dataset do **catálogo curado** ou cole **qualquer link** "
        "de um dataset público. O sistema resolve automaticamente o download."
    )

    # ==========================================================
    # SEÇÃO 1: CATÁLOGO DE DATASETS
    # ==========================================================
    st.subheader("📚 Catálogo de Datasets")

    categorias = categorias_catalogo()
    cat_selecionada = st.selectbox(
        "Filtrar por categoria",
        ["Todas"] + categorias,
        key="cat_catalogo",
    )

    cat_filtro = None if cat_selecionada == "Todas" else cat_selecionada
    datasets_filtrados = listar_catalogo(cat_filtro)

    for ds_id, ds_info in datasets_filtrados.items():
        relevancia_curta = ds_info['relevancia_dispositivo_s'][:50]
        with st.expander(
            f"{ds_info['categoria']} **{ds_info['nome']}** — "
            f"~{ds_info['n_classes_aprox']} classes",
            expanded=False,
        ):
            st.markdown(f"**{ds_info['descricao']}**")
            st.markdown(f"- **Coluna alvo:** `{ds_info['target_col']}`")
            st.markdown(f"- **Separador:** `{ds_info['separador']}`")
            feats_str = ', '.join(f'`{f}`' for f in ds_info['features_chave'][:8])
            if len(ds_info['features_chave']) > 8:
                feats_str += f" … (+{len(ds_info['features_chave'])-8})"
            st.markdown(f"- **Features-chave:** {feats_str}")
            st.markdown(f"- **Relevância p/ Dispositivo S:** {ds_info['relevancia_dispositivo_s']}")
            if ds_info.get("notas"):
                st.caption(f"📝 {ds_info['notas']}")

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(f"[🔗 Página do dataset]({ds_info['url']})")
            with col_b:
                if st.button("📥 Carregar", key=f"btn_cat_{ds_id}"):
                    with st.spinner(f"Baixando {ds_info['nome']}…"):
                        try:
                            ds = carregar_do_catalogo(ds_id)
                            st.session_state.dataset_externo = ds
                            st.success(
                                f"✅ **{ds_info['nome']}** — "
                                f"{ds['n_amostras']} amostras | "
                                f"{ds['n_features']} features | "
                                f"{ds['n_classes']} classes"
                            )
                            st.rerun()
                        except ImportError:
                            st.error(
                                "❌ **kagglehub** não instalado. "
                                "Execute: `pip install kagglehub` e configure "
                                "suas credenciais Kaggle (~/.kaggle/kaggle.json)."
                            )
                        except Exception as e:
                            st.error(f"❌ {e}")

    # ==========================================================
    # SEÇÃO 2: URL LIVRE
    # ==========================================================
    st.markdown("---")
    st.subheader("🔗 Importar via URL livre")
    st.caption(
        "Cole qualquer link — página do GitHub, Google Drive, "
        "HuggingFace, Kaggle, Dropbox ou link direto de CSV."
    )

    with st.form("import_url_form"):
        url = st.text_input(
            "URL (página ou link direto)",
            placeholder="https://github.com/.../dataset.csv  ou  https://drive.google.com/file/d/.../view",
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            target_col_input = st.text_input(
                "🎯 Coluna alvo (target/label)",
                placeholder="ex: Potability, species, quality",
            )
        with col2:
            sep = st.selectbox("Separador", [",", ";", "\\t", "|"], index=0, key="csv_sep",
                               help="Detectado automaticamente para .tsv")
        with col3:
            max_rows = st.number_input("Máx. linhas", 100, 50000, 10000, step=1000, key="max_rows")

        submitted = st.form_submit_button("📥 Carregar e Analisar", type="primary")

    if submitted:
        if not url.strip():
            st.error("Cole uma URL válida.")
            return
        if not target_col_input.strip():
            st.error("Informe o nome da coluna alvo.")
            return

        with st.spinner("🔍 Resolvendo URL…"):
            info = resolver_url_dataset(url.strip())

        if info['resolvido']:
            st.info(f"🔗 **{info['tipo'].upper()}** — {info['mensagem']}")
            if info['url_direta'] and info['url_direta'] != url.strip():
                url_display = info['url_direta'][:120]
                if len(info['url_direta'] or '') > 120:
                    url_display += '…'
                st.caption(f"URL de download: `{url_display}`")
        else:
            st.warning(f"⚠️ {info['mensagem']}")
            if info['tipo'] == 'kaggle':
                st.info(
                    "💡 **Dica Kaggle**: Instale `kagglehub` (`pip install kagglehub`) "
                    "e configure suas credenciais Kaggle para download automático."
                )
                return
            st.info("Tentando usar a URL original como link direto…")

        real_sep = "\t" if sep == "\\t" else sep
        if info.get('separador_sugerido') and sep == ",":
            real_sep = info['separador_sugerido']
            sep_display = "\\t" if real_sep == "\t" else real_sep
            st.caption(f"Separador auto-detectado: `{sep_display}`")

        with st.spinner("Baixando e processando dataset…"):
            try:
                ds = carregar_csv_url(
                    url.strip(),
                    target_col=target_col_input.strip(),
                    separator=real_sep,
                    max_rows=max_rows,
                )
            except ValueError as e:
                st.error(f"❌ {e}")
                if info.get('alternativas'):
                    st.markdown("**Links alternativos encontrados na página:**")
                    for alt in info['alternativas']:
                        st.code(alt)
                return
            except Exception as e:
                st.error(f"❌ Erro inesperado: {e}")
                return

        st.session_state.dataset_externo = ds
        st.success(
            f"✅ **{ds['descricao']}** carregado! "
            f"{ds['n_amostras']} amostras | {ds['n_features']} features | "
            f"{ds['n_classes']} classes"
        )

    # ==========================================================
    # SEÇÃO 3: PREVIEW + TREINO (quando tem dataset carregado)
    # ==========================================================
    if "dataset_externo" in st.session_state:
        ds = st.session_state.dataset_externo
        st.markdown("---")

        st.subheader(f"📊 {ds['descricao']}")
        st.dataframe(ds["df"].head(10), hide_index=True)

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Amostras", ds["n_amostras"])
        col_info2.metric("Features", ds["n_features"])
        col_info3.metric("Classes", ds["n_classes"])

        classes_list = ds['classes']
        st.markdown(f"**Classes:** {', '.join(classes_list[:20])}"
                    + (f" … (+{len(classes_list)-20})" if len(classes_list) > 20 else ""))
        feats_list = ds['feature_names']
        st.markdown(f"**Features:** {', '.join(feats_list[:15])}"
                    + (f" … (+{len(feats_list)-15})" if len(feats_list) > 15 else ""))

        st.markdown("---")
        st.subheader("🏋️ Treinar Modelo no Dataset Importado")

        modelo_ext = st.selectbox("Modelo", NOMES_MODELOS, key="modelo_ext")
        params_ext = _render_params_ext(modelo_ext)

        if st.button("🚀 Treinar", key="btn_treinar_ext"):
            X = ds["X"]
            y = ds["y"]
            classes = ds["classes"]
            n_classes = len(classes)

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y,
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp,
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            with st.spinner(f"Treinando {modelo_ext}…"):
                modelo, tempo = treinar_modelo(modelo_ext, params_ext, X_train, y_train, n_classes)

            y_pred_test, y_proba_test = predict_modelo(modelo, modelo_ext, X_test)
            acc = accuracy_score(y_test, y_pred_test)

            imp = calcular_feature_importance(
                modelo, modelo_ext, X_test, y_test, ds["feature_names"],
            )

            st.success(f"✅ **{modelo_ext}** — Acurácia: **{acc*100:.2f}%** | Tempo: {tempo:.2f}s")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.pyplot(plotar_confusion_matrix(y_test, y_pred_test, classes))
            with col_r2:
                st.pyplot(plotar_importancia(imp, f"{modelo_ext} — Dataset Importado"))

            st.dataframe(imp, hide_index=True)

    # Domínios aceitos
    st.markdown("---")
    with st.expander("🔒 Domínios aceitos para download"):
        for d in sorted(DOMINIOS_PERMITIDOS):
            st.markdown(f"- `{d}`")


def _render_params_ext(nome_modelo):
    """Parâmetros para o modelo do dataset externo (keys únicos para evitar conflito)."""
    defaults = PARAMS_DEFAULT[nome_modelo]

    if nome_modelo == "Random Forest":
        return {
            "n_estimators": st.number_input("n_estimators", 10, 1000, defaults["n_estimators"], step=10, key="ext_rf_n"),
            "max_depth": st.number_input("max_depth", 1, 50, defaults["max_depth"], key="ext_rf_d"),
            "min_samples_split": st.number_input("min_samples_split", 2, 50, defaults["min_samples_split"], key="ext_rf_s"),
        }
    elif nome_modelo == "XGBoost":
        return {
            "n_estimators": st.number_input("n_estimators", 10, 1000, defaults["n_estimators"], step=10, key="ext_xg_n"),
            "max_depth": st.number_input("max_depth", 1, 20, defaults["max_depth"], key="ext_xg_d"),
            "learning_rate": st.number_input("learning_rate", 0.001, 1.0, defaults["learning_rate"],
                                             step=0.01, format="%.3f", key="ext_xg_l"),
        }
    elif nome_modelo == "SVM":
        return {
            "kernel": st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="ext_svk"),
            "C": st.number_input("C", 0.01, 1000.0, defaults["C"], step=1.0, format="%.2f", key="ext_svc"),
            "gamma": st.selectbox("Gamma", ["scale", "auto"], key="ext_svg"),
        }
    elif nome_modelo == "Rede Neural (MLP)":
        l1 = st.number_input("Camada 1", 8, 512, defaults["hidden_layers"][0], step=8, key="ext_nn1")
        l2 = st.number_input("Camada 2", 8, 256, defaults["hidden_layers"][1], step=8, key="ext_nn2")
        l3 = st.number_input("Camada 3", 8, 128, defaults["hidden_layers"][2], step=8, key="ext_nn3")
        return {
            "hidden_layers": [l1, l2, l3],
            "epochs": st.number_input("Épocas", 10, 500, defaults["epochs"], step=10, key="ext_nne"),
            "batch_size": st.selectbox("Batch size", [16, 32, 64, 128], index=1, key="ext_nnb"),
            "learning_rate": st.number_input("LR", 0.0001, 0.1, defaults["learning_rate"],
                                             step=0.0001, format="%.4f", key="ext_nnl"),
        }
    elif nome_modelo == "CNN 1D":
        return {
            "filters": [
                st.number_input("Filtros 1", 8, 256, defaults["filters"][0], step=8, key="ext_cf1"),
                st.number_input("Filtros 2", 8, 256, defaults["filters"][1], step=8, key="ext_cf2"),
            ],
            "kernel_size": st.selectbox("Kernel", [2, 3, 5, 7], index=1, key="ext_cks"),
            "epochs": st.number_input("Épocas", 10, 500, defaults["epochs"], step=10, key="ext_cep"),
            "batch_size": st.selectbox("Batch size", [16, 32, 64, 128], index=1, key="ext_cbs"),
            "learning_rate": st.number_input("LR", 0.0001, 0.1, defaults["learning_rate"],
                                             step=0.0001, format="%.4f", key="ext_clr"),
        }


# ============================================================
# Widgets de parâmetros por modelo
# ============================================================

def _render_params(nome_modelo):
    defaults = PARAMS_DEFAULT[nome_modelo]

    if nome_modelo == "Random Forest":
        return {
            "n_estimators": st.number_input("n_estimators", 10, 1000, defaults["n_estimators"], step=10, key="rf_n"),
            "max_depth": st.number_input("max_depth", 1, 50, defaults["max_depth"], key="rf_d"),
            "min_samples_split": st.number_input("min_samples_split", 2, 50, defaults["min_samples_split"], key="rf_s"),
        }
    elif nome_modelo == "XGBoost":
        return {
            "n_estimators": st.number_input("n_estimators", 10, 1000, defaults["n_estimators"], step=10, key="xg_n"),
            "max_depth": st.number_input("max_depth", 1, 20, defaults["max_depth"], key="xg_d"),
            "learning_rate": st.number_input("learning_rate", 0.001, 1.0, defaults["learning_rate"],
                                             step=0.01, format="%.3f", key="xg_l"),
        }
    elif nome_modelo == "SVM":
        return {
            "kernel": st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svk"),
            "C": st.number_input("C", 0.01, 1000.0, defaults["C"], step=1.0, format="%.2f", key="svc"),
            "gamma": st.selectbox("Gamma", ["scale", "auto"], key="svg"),
        }
    elif nome_modelo == "Rede Neural (MLP)":
        l1 = st.number_input("Camada 1", 8, 512, defaults["hidden_layers"][0], step=8, key="nn1")
        l2 = st.number_input("Camada 2", 8, 256, defaults["hidden_layers"][1], step=8, key="nn2")
        l3 = st.number_input("Camada 3", 8, 128, defaults["hidden_layers"][2], step=8, key="nn3")
        return {
            "hidden_layers": [l1, l2, l3],
            "epochs": st.number_input("Épocas", 10, 500, defaults["epochs"], step=10, key="nne"),
            "batch_size": st.selectbox("Batch size", [16, 32, 64, 128], index=1, key="nnb"),
            "learning_rate": st.number_input("LR", 0.0001, 0.1, defaults["learning_rate"],
                                             step=0.0001, format="%.4f", key="nnl"),
        }
    elif nome_modelo == "CNN 1D":
        return {
            "filters": [
                st.number_input("Filtros 1", 8, 256, defaults["filters"][0], step=8, key="cf1"),
                st.number_input("Filtros 2", 8, 256, defaults["filters"][1], step=8, key="cf2"),
            ],
            "kernel_size": st.selectbox("Kernel", [2, 3, 5, 7], index=1, key="cks"),
            "epochs": st.number_input("Épocas", 10, 500, defaults["epochs"], step=10, key="cep"),
            "batch_size": st.selectbox("Batch size", [16, 32, 64, 128], index=1, key="cbs"),
            "learning_rate": st.number_input("LR", 0.0001, 0.1, defaults["learning_rate"],
                                             step=0.0001, format="%.4f", key="clr"),
        }


# ============================================================
# Sidebar Resumo
# ============================================================

def _render_sidebar_resumo():
    if not st.session_state.pipeline.get("treinado"):
        return

    st.sidebar.markdown("---")
    st.sidebar.header("📋 Pipeline Treinado")

    for clf_key, clf_info in CLASSIFICADORES.items():
        mods = st.session_state.pipeline["modelos"].get(clf_key, {})
        if mods:
            accs = [f"{info['acc']*100:.0f}%" for info in mods.values()]
            nomes = list(mods.keys())
            st.sidebar.markdown(f"**{clf_info['nome']}**")
            for nm, info in mods.items():
                st.sidebar.caption(f"  {nm}: {info['acc']*100:.1f}%")

    # Consenso top sensores
    st.sidebar.markdown("---")
    st.sidebar.header("🏆 Top Sensores")
    all_imps = []
    for clf_key in CLASSIFICADORES:
        mods = st.session_state.pipeline["modelos"].get(clf_key, {})
        for nm, info in mods.items():
            imp = info["importancia"].copy()
            mx = imp["Importância"].max()
            if mx > 0:
                imp["Importância"] = imp["Importância"] / mx
            all_imps.append(imp)
    if all_imps:
        df_all = pd.concat(all_imps)
        consenso = df_all.groupby("Sensor")["Importância"].mean().sort_values(ascending=False).head(7)
        for s, v in consenso.items():
            st.sidebar.markdown(f"- **{s}** ({v:.3f})")


if __name__ == "__main__":
    main()
