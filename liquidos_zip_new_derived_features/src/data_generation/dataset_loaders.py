"""
============================================================
 CARREGADORES DE DATASETS PÚBLICOS
============================================================

 Módulo responsável por baixar, carregar e padronizar datasets
 públicos para treinamento e validação do dispositivo S.

 Categorias:
   💧 Água          — Potabilidade, Qualidade, Contaminação
   🍺 Cerveja       — Estilos, Qualidade, Formulação
   🍷 Vinho         — Acidez, Densidade, Adulteração
   🥛 Leite         — Degradação e Pureza
   ☕ Café/Chá      — Perfil Sensorial e Químico
   🧪 Sensores/Geral— E-Tongue, Gas Sensor, Bebidas

 IMPORTANTE:
   Datasets Kaggle podem ser baixados via `kagglehub`
   (pip install kagglehub) ou manualmente para data/raw/.
   Datasets UCI e GitHub são baixados automaticamente.
============================================================
"""

import os
import sys
import io
import re
import zipfile
import numpy as np
import pandas as pd
from urllib.parse import urlparse, parse_qs, urljoin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import DATA_RAW, RANDOM_SEED


# ============================================================
# 1. WATER POTABILITY (Kaggle)
# ============================================================
# Fonte: https://www.kaggle.com/datasets/adityakadiwal/water-potability
# Colunas: ph, Hardness, Solids, Chloramines, Sulfate,
#          Conductivity, Organic_carbon, Trihalomethanes,
#          Turbidity, Potability (0 ou 1)
# ============================================================

WATER_POTABILITY_FILE = os.path.join(DATA_RAW, "water_potability.csv")


def carregar_water_potability(filepath: str = WATER_POTABILITY_FILE) -> dict:
    """
    Carrega o dataset de potabilidade da água.

    O dataset contém 9 features físico-químicas e uma label
    binária (0 = não potável, 1 = potável).

    Args:
        filepath: Caminho para o CSV

    Returns:
        Dicionário com X, y, feature_names, classes, descricao
        ou None se o arquivo não for encontrado
    """
    if not os.path.isfile(filepath):
        return None

    df = pd.read_csv(filepath)

    # Preencher NaN com a mediana (dataset tem bastante missing)
    df = df.fillna(df.median(numeric_only=True))

    feature_cols = [c for c in df.columns if c != "Potability"]
    X = df[feature_cols].values
    y = df["Potability"].values

    return {
        "X": X,
        "y": y,
        "feature_names": feature_cols,
        "classes": ["Não Potável", "Potável"],
        "descricao": "Water Potability (Kaggle)",
        "n_amostras": len(df),
        "n_features": len(feature_cols),
        "df": df,
    }


# ============================================================
# 2. BEER STYLE (Kaggle ML Olympiad)
# ============================================================
# Fonte: https://www.kaggle.com/competitions/ML-Olympiad-can-you-guess-the-beer-style/data
# Espera-se train.csv com colunas: brewery_name, review_*,
#   beer_abv, beer_style, etc.
# ============================================================

BEER_STYLE_FILE = os.path.join(DATA_RAW, "beer_style_train.csv")


def carregar_beer_style(filepath: str = BEER_STYLE_FILE, max_styles: int = 15) -> dict:
    """
    Carrega o dataset de estilos de cerveja (ML Olympiad).

    Usa features numéricas (ABV, scores de review) para
    classificar o estilo da cerveja.

    Args:
        filepath:    Caminho para o CSV (train.csv renomeado)
        max_styles:  Limita aos N estilos mais frequentes
                     (evita classes com poucas amostras)

    Returns:
        Dicionário com X, y, feature_names, classes, descricao
        ou None se o arquivo não for encontrado
    """
    if not os.path.isfile(filepath):
        return None

    df = pd.read_csv(filepath)

    # Identificar a coluna de estilo (pode variar)
    style_col = None
    for candidate in ["beer_style", "style", "Style", "beer_style "]:
        if candidate in df.columns:
            style_col = candidate
            break

    if style_col is None:
        # Tenta a última coluna não-numérica
        non_num = df.select_dtypes(include=["object"]).columns
        if len(non_num) > 0:
            style_col = non_num[-1]
        else:
            return None

    # Selecionar apenas features numéricas
    feature_candidates = [
        "beer_abv", "review_overall", "review_aroma", "review_appearance",
        "review_palate", "review_taste", "beer_beerid",
        # Nomes alternativos
        "ABV", "Overall", "Aroma", "Appearance", "Palate", "Taste",
    ]
    feature_cols = [c for c in feature_candidates if c in df.columns]

    # Se poucos, pega todas as numéricas (exceto id e target)
    if len(feature_cols) < 3:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c.lower() not in ("id", "index", "unnamed: 0")]

    if len(feature_cols) < 2:
        return None

    # Filtrar NaN e limitar estilos
    df = df.dropna(subset=feature_cols + [style_col])
    top_styles = df[style_col].value_counts().head(max_styles).index.tolist()
    df = df[df[style_col].isin(top_styles)].reset_index(drop=True)

    X = df[feature_cols].values
    y_raw = df[style_col].values

    # Codificar labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    return {
        "X": X,
        "y": y,
        "feature_names": feature_cols,
        "classes": list(le.classes_),
        "descricao": "Beer Style — ML Olympiad (Kaggle)",
        "n_amostras": len(df),
        "n_features": len(feature_cols),
        "df": df,
        "label_encoder": le,
    }


# ============================================================
# 3. GAS SENSOR ARRAY DRIFT (UCI)
# ============================================================
# Fonte: https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset
# 13910 amostras, 128 features de sensores, 6 gases
# Formato .dat (espaços), label:feature_id:valor
# ============================================================

GAS_SENSOR_DIR = os.path.join(DATA_RAW, "gas_sensor_drift")


def carregar_gas_sensor_drift(dirpath: str = GAS_SENSOR_DIR) -> dict:
    """
    Carrega o dataset UCI Gas Sensor Array Drift.

    São 10 batchs (.dat files) com leituras de 16 sensores
    (8 features por sensor: estado estacionário e transiente)
    para 6 gases diferentes.

    Relevância: simula o comportamento de uma "Língua Eletrônica"
    com drift temporal — exatamente o que nosso dispositivo S
    enfrentará no mundo real.

    Args:
        dirpath: Diretório contendo os arquivos batch*.dat

    Returns:
        Dicionário com X, y, feature_names, classes, descricao
        ou None se os arquivos não existirem
    """
    if not os.path.isdir(dirpath):
        return None

    # Procura arquivos .dat
    dat_files = sorted([f for f in os.listdir(dirpath) if f.endswith(".dat")])
    if not dat_files:
        return None

    gas_names = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
                 4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}

    all_X = []
    all_y = []

    for fname in dat_files:
        fpath = os.path.join(dirpath, fname)
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Primeiro elemento: label (1-6) possivelmente com ;
                label_str = parts[0].split(";")[0]
                try:
                    label = int(label_str)
                except ValueError:
                    continue

                # Features no formato id:valor
                features = []
                for part in parts[1:]:
                    if ":" in part:
                        try:
                            val = float(part.split(":")[1])
                        except (ValueError, IndexError):
                            val = 0.0
                        features.append(val)

                if features:
                    all_X.append(features)
                    all_y.append(label)

    if not all_X:
        return None

    # Padroniza tamanho (128 features padrão)
    max_len = max(len(x) for x in all_X)
    X = np.array([x + [0.0] * (max_len - len(x)) for x in all_X])
    y = np.array(all_y) - 1  # 0-indexed

    # Nomes genéricos dos sensores (8 features × 16 sensores)
    feature_names = []
    for s in range(1, 17):
        for feat in ["DR", "steady"]:
            for stat in ["max", "min", "mean", "var"]:
                feature_names.append(f"S{s}_{feat}_{stat}")
    # Ajusta se temos mais/menos
    while len(feature_names) < max_len:
        feature_names.append(f"feat_{len(feature_names)}")
    feature_names = feature_names[:max_len]

    classes = [gas_names.get(i + 1, f"Gas_{i+1}") for i in range(len(set(y)))]

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "classes": classes,
        "descricao": "Gas Sensor Array Drift (UCI)",
        "n_amostras": len(X),
        "n_features": max_len,
    }


# ============================================================
# CATÁLOGO COMPLETO DE DATASETS
# ============================================================
# Cada entrada contém metadados suficientes para download
# e carregamento automático: URL, coluna alvo, separador,
# slug Kaggle (para kagglehub), etc.
# ============================================================

CATALOGO_DATASETS = {

    # -------------------------------------------------------
    # 💧 ÁGUA — Potabilidade, Qualidade e Contaminação
    # -------------------------------------------------------
    "water_potability": {
        "nome": "Water Quality and Potability",
        "categoria": "💧 Água",
        "descricao": "O mais completo para potabilidade — 9 features físico-químicas (pH, dureza, sólidos, cloraminas, sulfato, condutividade, carbono orgânico, trihalometanos, turbidez). Classificação binária.",
        "url": "https://www.kaggle.com/datasets/adityakadiwal/water-potability",
        "kaggle_slug": "adityakadiwal/water-potability",
        "target_col": "Potability",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 2,
        "features_chave": ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Turbidity"],
        "relevancia_dispositivo_s": "Alta — features diretamente mapeáveis para sensores de pH, condutividade e turbidez do dispositivo S.",
    },
    "water_quality_prediction": {
        "nome": "Water Quality Prediction",
        "categoria": "💧 Água",
        "descricao": "Focado em contaminantes e minerais — alumínio, amônia, arsênio, bário, cádmio, cloro, cobre, flúor, ferro, chumbo, manganês, mercúrio, nitratos, selênio, prata, urânio, viruses, bactérias.",
        "url": "https://www.kaggle.com/datasets/mssmartypants/water-quality",
        "kaggle_slug": "mssmartypants/water-quality",
        "target_col": "is_safe",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 2,
        "features_chave": ["aluminium", "ammonia", "arsenic", "chloramine", "conductivity", "copper", "fluoride", "bacteria", "viruses", "lead", "nitrates", "mercury", "perchlorate", "silver", "uranium", "turbidity"],
        "relevancia_dispositivo_s": "Média — muitos contaminantes específicos (metais pesados) não são detectáveis diretamente pelo dispositivo S, mas turbidez e condutividade são relevantes.",
    },
    "indian_water_quality": {
        "nome": "Indian Water Quality Data",
        "categoria": "💧 Água",
        "descricao": "Dados de rios e água bruta em estações da Índia — excelente para cenários reais com dados ruidosos e faltantes. Contém pH, condutividade, OD, DBO, coliformes, nitrato.",
        "url": "https://www.kaggle.com/datasets/artimous/indian-water-quality-data",
        "kaggle_slug": "artimous/indian-water-quality-data",
        "target_col": "year",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 10,
        "features_chave": ["TEMP", "D.O.", "PH", "CONDUCTIVITY", "B.O.D.", "NITRATENAN", "FECALCOLIFORM", "TOTALCOLIFORM"],
        "relevancia_dispositivo_s": "Alta — pH, condutividade e temperatura mapeiam diretamente nos sensores do dispositivo S.",
        "notas": "Pode precisar de limpeza adicional. Considerar criar label binária (qualidade boa/ruim) a partir do pH e coliformes.",
    },
    "drinking_water_quality": {
        "nome": "Drinking Water Quality",
        "categoria": "💧 Água",
        "descricao": "Métricas de estações de tratamento de água potável — foco em turbidez, flúor, cloro residual e conformidade regulatória.",
        "url": "https://www.kaggle.com/datasets/wkirgsn/drinking-water-quality",
        "kaggle_slug": "wkirgsn/drinking-water-quality",
        "target_col": "Compliance",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 2,
        "features_chave": ["Turbidity", "Fluoride", "Coliform", "E.Coli", "Chloramine", "Odor", "Sulfate", "Conductivity"],
        "relevancia_dispositivo_s": "Alta — dados reais de estações de tratamento validam o pipeline de potabilidade.",
    },

    # -------------------------------------------------------
    # 🍺 CERVEJA — Estilos, Qualidade e Formulação
    # -------------------------------------------------------
    "beer_profile_ratings": {
        "nome": "Beer Profile and Ratings Data Set",
        "categoria": "🍺 Cerveja",
        "descricao": "Perfis completos de cervejas: ABV, IBU, cor (SRM/EBC), reviews sensoriais (aroma, aparência, paladar, sabor). ~3200 cervejas com avaliações agregadas.",
        "url": "https://www.kaggle.com/datasets/ruthgn/beer-profile-and-ratings-data-set",
        "kaggle_slug": "ruthgn/beer-profile-and-ratings-data-set",
        "target_col": "Style",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 100,
        "features_chave": ["ABV", "Min IBU", "Max IBU", "Astringency", "Body", "Alcohol", "Bitter", "Sweet", "Sour", "Salty", "Fruits", "Hoppy", "Spices", "Malty"],
        "relevancia_dispositivo_s": "Média-Alta — ABV e amargor podem ser inferidos por condutividade e sensores ópticos.",
    },
    "craft_cans": {
        "nome": "Craft Cans Dataset",
        "categoria": "🍺 Cerveja",
        "descricao": "Cervejas artesanais americanas em lata — ABV, IBU, estilo, cervejaria. Focado em cervejas artesanais e métricas básicas.",
        "url": "https://www.kaggle.com/datasets/nickhould/craft-cans",
        "kaggle_slug": "nickhould/craft-cans",
        "target_col": "Style",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 30,
        "features_chave": ["ABV", "IBU", "Ounces"],
        "relevancia_dispositivo_s": "Média — poucas features numéricas, mas útil como complemento.",
    },
    "beer_recipes": {
        "nome": "Beer Recipes / Brewing Data",
        "categoria": "🍺 Cerveja",
        "descricao": "Receitas de cerveja com propriedades pré e pós-fermentação: OG, FG, ABV, IBU, cor, pH da brassagem. ~75000 receitas.",
        "url": "https://www.kaggle.com/datasets/jtrofe/beer-recipes",
        "kaggle_slug": "jtrofe/beer-recipes",
        "target_col": "Style",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 100,
        "features_chave": ["OG", "FG", "ABV", "IBU", "Color", "BoilSize", "BoilTime", "BoilGravity", "Efficiency", "MashThickness", "PitchRate", "PrimaryTemp"],
        "relevancia_dispositivo_s": "Alta — OG, FG e pH mapeiam para condutividade e sensores de pH; cor para espectral (AS7341).",
    },
    "beer_ml_olympiad": {
        "nome": "ML Olympiad - Beer Style",
        "categoria": "🍺 Cerveja",
        "descricao": "Desafio de classificação de estilos — reviews numéricas e ABV para predizer o estilo da cerveja.",
        "url": "https://www.kaggle.com/competitions/ML-Olympiad-can-you-guess-the-beer-style/data",
        "kaggle_slug": None,
        "target_col": "beer_style",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 50,
        "features_chave": ["beer_abv", "review_overall", "review_aroma", "review_appearance", "review_palate", "review_taste"],
        "relevancia_dispositivo_s": "Média — reviews são subjetivas, mas ABV é mensurável.",
        "notas": "Dataset de competição — precisa aceitar regras no Kaggle.",
    },

    # -------------------------------------------------------
    # 🍷 VINHO — Acidez, Densidade e Adulteração
    # -------------------------------------------------------
    "wine_quality_uci": {
        "nome": "Wine Quality (UCI - Red & White)",
        "categoria": "🍷 Vinho",
        "descricao": "O clássico da UCI: acidez fixa/volátil, ácido cítrico, açúcar residual, cloretos, SO2, densidade, pH, sulfatos, álcool. Score de qualidade 0-10.",
        "url": "https://archive.ics.uci.edu/ml/datasets/wine+quality",
        "kaggle_slug": None,
        "target_col": "quality",
        "separador": ";",
        "tipo_problema": "classificação",
        "n_classes_aprox": 7,
        "features_chave": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        "relevancia_dispositivo_s": "Muito Alta — pH, acidez, densidade e álcool são todos mensuráveis pelo dispositivo S. Dataset referência.",
        "url_direta": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    },
    "wine_quality_kaggle": {
        "nome": "Wine Quality (Kaggle tratado)",
        "categoria": "🍷 Vinho",
        "descricao": "Versão Kaggle do dataset UCI com limpeza prévia — combina vinho tinto e branco com coluna 'type'.",
        "url": "https://www.kaggle.com/datasets/rajyellow46/wine-quality",
        "kaggle_slug": "rajyellow46/wine-quality",
        "target_col": "quality",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 7,
        "features_chave": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "density", "pH", "sulphates", "alcohol", "type"],
        "relevancia_dispositivo_s": "Muito Alta — mesmas features do UCI com separação tinto/branco (tipo).",
    },
    "red_wine_quality": {
        "nome": "Red Wine Quality (Cortez et al.)",
        "categoria": "🍷 Vinho",
        "descricao": "Apenas vinho tinto — 1599 amostras, 11 features físico-químicas e score de qualidade.",
        "url": "https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009",
        "kaggle_slug": "uciml/red-wine-quality-cortez-et-al-2009",
        "target_col": "quality",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 6,
        "features_chave": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "density", "pH", "sulphates", "alcohol"],
        "relevancia_dispositivo_s": "Muito Alta — subconjunto focado ideal para treino de modelos de qualidade.",
    },
    "wine_spoilage": {
        "nome": "Wine Spoilage Thresholds (Mendeley)",
        "categoria": "🍷 Vinho",
        "descricao": "Focado em degradação e acidez — limiares de deterioração do vinho com métricas microbiológicas e químicas.",
        "url": "https://data.mendeley.com/datasets/vpc887d53s/",
        "kaggle_slug": None,
        "target_col": "quality",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 3,
        "features_chave": ["acidity", "pH", "alcohol", "sugar", "SO2"],
        "relevancia_dispositivo_s": "Alta — dados de degradação são relevantes para detectar líquidos deteriorados.",
        "notas": "Mendeley Data — pode requerer download manual.",
    },

    # -------------------------------------------------------
    # 🥛 LEITE — Degradação e Pureza
    # -------------------------------------------------------
    "milk_quality_prediction": {
        "nome": "Milk Quality Prediction",
        "categoria": "🥛 Leite",
        "descricao": "Classificação de qualidade do leite em 3 graus (low/medium/high) usando pH, temperatura, sabor, odor, gordura, turbidez, cor.",
        "url": "https://www.kaggle.com/datasets/cpluzshrijayan/milkquality",
        "kaggle_slug": "cpluzshrijayan/milkquality",
        "target_col": "Grade",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 3,
        "features_chave": ["pH", "Temprature", "Taste", "Odor", "Fat", "Turbidity", "Colour"],
        "relevancia_dispositivo_s": "Alta — pH, temperatura, turbidez e cor mapeiam diretamente nos sensores do dispositivo S.",
    },
    "milk_quality_classification": {
        "nome": "Milk Quality Classification",
        "categoria": "🥛 Leite",
        "descricao": "Variáveis puramente físico-químicas do leite — sem features subjetivas. Classificação binária ou multiclasse.",
        "url": "https://www.kaggle.com/datasets/vigneshwaran10/milk-quality-classification",
        "kaggle_slug": "vigneshwaran10/milk-quality-classification",
        "target_col": "Grade",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 3,
        "features_chave": ["pH", "Temperature", "Fat", "Turbidity", "Color", "Taste", "Odor"],
        "relevancia_dispositivo_s": "Alta — foco em métricas objetivas que o dispositivo S pode medir.",
    },

    # -------------------------------------------------------
    # ☕ CAFÉ E CHÁ — Perfil Sensorial e Químico
    # -------------------------------------------------------
    "coffee_quality_cqi": {
        "nome": "Coffee Quality (CQI Database)",
        "categoria": "☕ Café/Chá",
        "descricao": "Acidez, corpo, umidade, aroma, sabor, equilíbrio, doçura, uniformidade — dados do Coffee Quality Institute. ~1300 amostras de café arábica e robusta.",
        "url": "https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi",
        "kaggle_slug": "volpatto/coffee-quality-database-from-cqi",
        "target_col": "Species",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 2,
        "features_chave": ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Uniformity", "Clean.Cup", "Sweetness", "Moisture", "Quakers"],
        "relevancia_dispositivo_s": "Média — perfil sensorial é subjetivo, mas acidez e umidade são mensuráveis.",
    },
    "coffee_quality_updated": {
        "nome": "Coffee Quality (Atualizado)",
        "categoria": "☕ Café/Chá",
        "descricao": "Versão atualizada do CQI — scores mais refinados e metadados de origem, altitude, processamento.",
        "url": "https://www.kaggle.com/datasets/fatiimaezzahra/coffee-quality-dataset-cqi",
        "kaggle_slug": "fatiimaezzahra/coffee-quality-dataset-cqi",
        "target_col": "Species",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 2,
        "features_chave": ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Sweetness", "Moisture", "altitude_mean_meters"],
        "relevancia_dispositivo_s": "Média — semelhante ao CQI original, com dados geográficos adicionais.",
    },
    "tea_quality": {
        "nome": "Tea Quality Dataset",
        "categoria": "☕ Café/Chá",
        "descricao": "Focado em fermentação e propriedades químicas do chá — cafeína, taninos, catequinas, aminoácidos, cor da infusão.",
        "url": "https://www.kaggle.com/datasets/tianhwu/tea-quality-dataset",
        "kaggle_slug": "tianhwu/tea-quality-dataset",
        "target_col": "Quality",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 3,
        "features_chave": ["Caffeine", "Tannin", "Catechin", "AminoAcid", "pH", "Moisture", "Color"],
        "relevancia_dispositivo_s": "Alta — pH, cor e propriedades químicas são diretamente mensuráveis.",
    },

    # -------------------------------------------------------
    # 🧪 SENSORES / BEBIDAS GERAIS — O OURO PARA O HARDWARE
    # -------------------------------------------------------
    "etongue_mineral_waters": {
        "nome": "Electronic Tongue - Mineral Waters (UCI)",
        "categoria": "🧪 Sensores",
        "descricao": "Leituras elétricas de sensores (língua eletrônica / e-tongue) imersos em diferentes águas minerais. DIRETAMENTE ANÁLOGO ao dispositivo S — array de eletrodos medindo impedância/voltametria.",
        "url": "https://archive.ics.uci.edu/dataset/501/electronic+tongue+dataset+for+the+classification+of+different+mineral+waters",
        "kaggle_slug": None,
        "target_col": "class",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 5,
        "features_chave": ["sensor_readings"],
        "relevancia_dispositivo_s": "MUITO ALTA — dataset de e-tongue é o mais próximo possível do hardware real do dispositivo S.",
        "notas": "UCI — pode precisar de download manual ou scraping. Formato pode variar.",
    },
    "gas_sensor_drift": {
        "nome": "Gas Sensor Array Drift (UCI)",
        "categoria": "🧪 Sensores",
        "descricao": "13910 amostras, 128 features de 16 sensores, 6 gases. Essencial para treinar algoritmos a lidar com drift (desgaste do sensor ao longo do tempo). 10 batches temporais.",
        "url": "https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset",
        "kaggle_slug": None,
        "target_col": "gas_label",
        "separador": " ",
        "tipo_problema": "classificação",
        "n_classes_aprox": 6,
        "features_chave": ["128 features de sensores — DR, steady-state × 16 sensores"],
        "relevancia_dispositivo_s": "MUITO ALTA — simula exatamente o problema de drift que o dispositivo S enfrentará.",
        "notas": "Loader próprio já implementado (carregar_gas_sensor_drift). Formato especial label:id:valor.",
    },
    "beverages_github": {
        "nome": "Beverages Dataset (GitHub)",
        "categoria": "🧪 Sensores",
        "descricao": "Propriedades físico-químicas variadas de bebidas industriais genéricas — pH, condutividade, densidade, açúcar, acidez.",
        "url": "https://github.com/BharatMaheshwari96/Beverages-Dataset",
        "kaggle_slug": None,
        "target_col": "Beverage",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 10,
        "features_chave": ["pH", "Conductivity", "Density", "Sugar", "Acidity", "Color"],
        "relevancia_dispositivo_s": "MUITO ALTA — propriedades mensuráveis de bebidas reais, diretamente aplicável ao dispositivo S.",
        "notas": "GitHub — resolver automático converte para raw.githubusercontent.com.",
    },
    "water_quality_sampling_gov": {
        "nome": "Water Quality Sampling Data (Data.gov)",
        "categoria": "🧪 Sensores",
        "descricao": "Amostras governamentais reais dos EUA — dados complexos de estações de monitoramento com múltiplos parâmetros incluindo pH, condutividade, OD, temperatura, turbidez.",
        "url": "https://catalog.data.gov/dataset/water-quality-sampling-data",
        "kaggle_slug": None,
        "target_col": "Result",
        "separador": ",",
        "tipo_problema": "classificação",
        "n_classes_aprox": 5,
        "features_chave": ["pH", "Temperature", "Conductivity", "DissolvedOxygen", "Turbidity"],
        "relevancia_dispositivo_s": "Alta — dados reais de campo, muito relevantes para cenários práticos.",
        "notas": "Data.gov — pode ter formato variável. Verificar qual CSV específico está disponível.",
    },
}


def listar_catalogo(categoria: str = None) -> dict:
    """
    Retorna o catálogo de datasets, opcionalmente filtrado por categoria.

    Args:
        categoria: ex. '💧 Água', '🍺 Cerveja', None para todos

    Returns:
        Dicionário {id: info} filtrado
    """
    if categoria is None:
        return CATALOGO_DATASETS
    return {k: v for k, v in CATALOGO_DATASETS.items()
            if v["categoria"] == categoria}


def categorias_catalogo() -> list:
    """Retorna lista de categorias únicas no catálogo."""
    cats = []
    seen = set()
    for v in CATALOGO_DATASETS.values():
        c = v["categoria"]
        if c not in seen:
            seen.add(c)
            cats.append(c)
    return cats


# ============================================================
# DOWNLOAD VIA KAGGLEHUB
# ============================================================

def baixar_kaggle_dataset(kaggle_slug: str, salvar_em: str = None) -> str:
    """
    Baixa um dataset do Kaggle usando kagglehub.

    Args:
        kaggle_slug: 'owner/dataset-name' (ex: 'adityakadiwal/water-potability')
        salvar_em:   Diretório de destino (default: DATA_RAW)

    Returns:
        Caminho do maior CSV encontrado no dataset baixado.

    Raises:
        ImportError se kagglehub não estiver instalado.
        ValueError se nenhum CSV for encontrado.
    """
    import kagglehub

    if salvar_em is None:
        salvar_em = DATA_RAW

    local_path = kagglehub.dataset_download(kaggle_slug)

    # Procurar CSVs no diretório
    csvs = []
    for root, dirs, files in os.walk(local_path):
        for f in files:
            if f.endswith('.csv'):
                csvs.append(os.path.join(root, f))

    if not csvs:
        raise ValueError(f"Nenhum CSV encontrado em {local_path}")

    # Retorna o maior CSV
    maior = max(csvs, key=os.path.getsize)
    return maior


def carregar_do_catalogo(dataset_id: str, max_rows: int = 50000) -> dict:
    """
    Carrega um dataset do catálogo automaticamente.

    Tenta na seguinte ordem:
      1. Arquivo local em data/raw/
      2. kagglehub (se slug disponível)
      3. Download direto (se URL direta disponível)
      4. URL resolver + download

    Args:
        dataset_id: Chave do CATALOGO_DATASETS (ex: 'water_potability')
        max_rows:   Limite de linhas

    Returns:
        Dicionário com X, y, feature_names, classes, etc.
    """
    if dataset_id not in CATALOGO_DATASETS:
        raise ValueError(f"Dataset '{dataset_id}' não encontrado no catálogo.")

    info = CATALOGO_DATASETS[dataset_id]

    # Loaders especializados
    if dataset_id == "gas_sensor_drift":
        resultado = carregar_gas_sensor_drift()
        if resultado is not None:
            return resultado
        raise ValueError(
            "Gas Sensor Drift não encontrado. "
            "Baixe de: https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset "
            "e extraia em data/raw/gas_sensor_drift/"
        )

    # Verificar arquivo local primeiro
    nomes_possiveis = [
        os.path.join(DATA_RAW, f"{dataset_id}.csv"),
        os.path.join(DATA_RAW, f"{info['nome'].lower().replace(' ', '_')}.csv"),
    ]
    for local_path in nomes_possiveis:
        if os.path.isfile(local_path):
            df = pd.read_csv(local_path, sep=info["separador"], nrows=max_rows)
            return _preparar_dataset_df(df, info["target_col"], info["url"])

    # Tentar kagglehub
    if info.get("kaggle_slug"):
        try:
            csv_path = baixar_kaggle_dataset(info["kaggle_slug"])
            df = pd.read_csv(csv_path, sep=info["separador"], nrows=max_rows)
            return _preparar_dataset_df(df, info["target_col"], info["url"])
        except ImportError:
            pass  # kagglehub não instalado — tentar URL
        except Exception:
            pass  # erro no download — tentar URL

    # Tentar download direto ou via resolver
    url = info.get("url_direta", info["url"])
    return carregar_csv_url(
        url,
        target_col=info["target_col"],
        separator=info["separador"],
        max_rows=max_rows,
    )


# ============================================================
# 4. RESOLUÇÃO INTELIGENTE DE URLs
# ============================================================
# Converte automaticamente links de páginas em links diretos
# de download para as principais plataformas de dados.
# ============================================================


def resolver_url_dataset(url: str) -> dict:
    """
    Recebe qualquer URL (página ou link direto) e tenta resolver
    para um link direto de download CSV.

    Plataformas suportadas:
      - GitHub (blob → raw)
      - Google Drive (sharing → direct download)
      - HuggingFace Datasets (página → download direto)
      - Dropbox (dl=0 → dl=1)
      - Links diretos (.csv, .tsv, .data, .txt)
      - Páginas genéricas (scraping de links CSV)

    Returns:
        {
            'url_original':  str,
            'url_direta':    str | None,
            'tipo':          str,  # 'github', 'gdrive', 'hf', 'dropbox', 'direto', 'scraping', 'desconhecido'
            'mensagem':      str,
            'separador_sugerido': str | None,
            'resolvido':     bool,
        }
    """
    resultado = {
        'url_original': url,
        'url_direta': None,
        'tipo': 'desconhecido',
        'mensagem': '',
        'separador_sugerido': None,
        'resolvido': False,
    }

    try:
        parsed = urlparse(url)
    except Exception:
        resultado['mensagem'] = 'URL inválida.'
        return resultado

    host = (parsed.hostname or '').lower()
    path = parsed.path or ''

    # -------------------------------------------------------
    # 1. GITHUB:  github.com/.../blob/...  →  raw.githubusercontent.com/...
    # -------------------------------------------------------
    if host == 'github.com':
        m = re.match(r'/([^/]+)/([^/]+)/blob/(.+)', path)
        if m:
            user, repo, rest = m.groups()
            raw_url = f'https://raw.githubusercontent.com/{user}/{repo}/{rest}'
            resultado.update(url_direta=raw_url, tipo='github', resolvido=True,
                             mensagem=f'GitHub blob → raw ({os.path.basename(rest)})')
            if rest.endswith('.tsv'):
                resultado['separador_sugerido'] = '\t'
            return resultado
        # Pode ser um raw url já formatado como /user/repo/raw/...
        m_raw = re.match(r'/([^/]+)/([^/]+)/raw/(.+)', path)
        if m_raw:
            user, repo, rest = m_raw.groups()
            raw_url = f'https://raw.githubusercontent.com/{user}/{repo}/{rest}'
            resultado.update(url_direta=raw_url, tipo='github', resolvido=True,
                             mensagem=f'GitHub raw → download direto')
            return resultado

    # raw.githubusercontent.com — já é direto
    if host == 'raw.githubusercontent.com':
        resultado.update(url_direta=url, tipo='github', resolvido=True,
                         mensagem='Link direto do GitHub (raw)')
        if path.endswith('.tsv'):
            resultado['separador_sugerido'] = '\t'
        return resultado

    # -------------------------------------------------------
    # 2. GOOGLE DRIVE:  drive.google.com/file/d/FILEID/...
    #    ou docs.google.com/spreadsheets/d/ID/export?format=csv
    # -------------------------------------------------------
    if host in ('drive.google.com', 'docs.google.com'):
        # drive.google.com/file/d/{file_id}/view
        m_drive = re.search(r'/file/d/([a-zA-Z0-9_-]+)', path)
        if m_drive:
            file_id = m_drive.group(1)
            direct = f'https://drive.google.com/uc?export=download&id={file_id}'
            resultado.update(url_direta=direct, tipo='gdrive', resolvido=True,
                             mensagem=f'Google Drive file → download direto (id: {file_id[:12]}...)')
            return resultado
        # Google Sheets export
        m_sheet = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', path)
        if m_sheet:
            sheet_id = m_sheet.group(1)
            direct = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
            resultado.update(url_direta=direct, tipo='gdrive', resolvido=True,
                             mensagem=f'Google Sheets → export CSV')
            return resultado
        # Fallback: ?id=XXX param
        qs = parse_qs(parsed.query)
        if 'id' in qs:
            file_id = qs['id'][0]
            direct = f'https://drive.google.com/uc?export=download&id={file_id}'
            resultado.update(url_direta=direct, tipo='gdrive', resolvido=True,
                             mensagem=f'Google Drive (?id) → download direto')
            return resultado

    # -------------------------------------------------------
    # 3. HUGGING FACE:  huggingface.co/datasets/OWNER/NAME
    #    → usa API ou resolve para download do primeiro CSV/parquet
    # -------------------------------------------------------
    if host == 'huggingface.co' or host == 'www.huggingface.co':
        # Padrão: /datasets/owner/name  ou  /datasets/owner/name/blob/main/file.csv
        m_ds = re.match(r'/datasets/([^/]+)/([^/]+)(?:/(?:blob|resolve)/([^/]+)/(.+))?', path)
        if m_ds:
            owner, name = m_ds.group(1), m_ds.group(2)
            ref = m_ds.group(3) or 'main'
            filepath = m_ds.group(4)
            if filepath:
                # Link para arquivo específico — resolve direto
                direct = f'https://huggingface.co/datasets/{owner}/{name}/resolve/{ref}/{filepath}'
                resultado.update(url_direta=direct, tipo='hf', resolvido=True,
                                 mensagem=f'HuggingFace → resolve/{ref}/{filepath}')
                if filepath.endswith('.tsv'):
                    resultado['separador_sugerido'] = '\t'
                return resultado
            else:
                # Página do dataset — tentar /resolve/main/ com nomes comuns
                resultado.update(tipo='hf',
                                 mensagem=f'Página do dataset HuggingFace: {owner}/{name}')
                import requests
                for candidate in ['train.csv', 'data.csv', 'dataset.csv',
                                  'test.csv', f'{name}.csv', 'train.parquet']:
                    test_url = f'https://huggingface.co/datasets/{owner}/{name}/resolve/main/{candidate}'
                    try:
                        r = requests.head(test_url, timeout=10, allow_redirects=True)
                        if r.status_code == 200:
                            resultado.update(url_direta=test_url, resolvido=True,
                                             mensagem=f'HuggingFace → encontrado {candidate}')
                            return resultado
                    except Exception:
                        continue
                resultado['mensagem'] += ' — nenhum CSV encontrado automaticamente.'
                return resultado

    # -------------------------------------------------------
    # 4. DROPBOX:  www.dropbox.com/...?dl=0  →  ?dl=1
    # -------------------------------------------------------
    if host in ('www.dropbox.com', 'dropbox.com'):
        clean = re.sub(r'[?&]dl=0', '', url)
        sep_char = '&' if '?' in clean else '?'
        direct = f'{clean}{sep_char}dl=1'
        resultado.update(url_direta=direct, tipo='dropbox', resolvido=True,
                         mensagem='Dropbox → download direto (dl=1)')
        return resultado

    # -------------------------------------------------------
    # 5. LINK DIRETO:  termina em .csv, .tsv, .data, .txt
    # -------------------------------------------------------
    ext_lower = os.path.splitext(path)[1].lower()
    if ext_lower in ('.csv', '.tsv', '.data', '.txt', '.dat'):
        resultado.update(url_direta=url, tipo='direto', resolvido=True,
                         mensagem=f'Link direto detectado ({ext_lower})')
        if ext_lower == '.tsv':
            resultado['separador_sugerido'] = '\t'
        return resultado

    # -------------------------------------------------------
    # 6. KAGGLE:  www.kaggle.com/datasets/OWNER/NAME
    #    Kaggle não permite download direto sem auth;
    #    tentamos kagglehub se instalado.
    # -------------------------------------------------------
    if host in ('www.kaggle.com', 'kaggle.com'):
        m_kg = re.match(r'/datasets/([^/]+)/([^/]+)', path)
        if m_kg:
            kg_owner, kg_name = m_kg.group(1), m_kg.group(2)
            resultado.update(tipo='kaggle',
                             mensagem=f'Dataset Kaggle: {kg_owner}/{kg_name}')
            try:
                import kagglehub
                local_path = kagglehub.dataset_download(f'{kg_owner}/{kg_name}')
                # Procurar CSV dentro do diretório baixado
                csvs = []
                for root, dirs, files in os.walk(local_path):
                    for f in files:
                        if f.endswith('.csv'):
                            csvs.append(os.path.join(root, f))
                if csvs:
                    # Pegar o maior CSV
                    maior = max(csvs, key=os.path.getsize)
                    resultado.update(url_direta=f'file://{maior}',
                                     resolvido=True,
                                     mensagem=f'Kaggle (kagglehub) → {os.path.basename(maior)}')
                    return resultado
                resultado['mensagem'] += ' — kagglehub baixou mas nenhum CSV encontrado.'
            except ImportError:
                resultado['mensagem'] += (
                    ' — instale `kagglehub` para download automático do Kaggle '
                    '(pip install kagglehub). Ou use o link direto do CSV.'
                )
            except Exception as e:
                resultado['mensagem'] += f' — erro kagglehub: {e}'
            return resultado

    # -------------------------------------------------------
    # 7. UCI ML Repository
    # -------------------------------------------------------
    if 'archive.ics.uci.edu' in host:
        resultado.update(tipo='uci', mensagem='Página UCI')
        if ext_lower in ('.csv', '.data', '.tsv', '.txt'):
            resultado.update(url_direta=url, resolvido=True,
                             mensagem=f'UCI → link direto ({ext_lower})')
            return resultado
        # Tentar scraping
        resultado = _tentar_scraping_csv(url, resultado)
        return resultado

    # -------------------------------------------------------
    # 8. SCRAPING GENÉRICO:  busca links .csv na página
    # -------------------------------------------------------
    resultado = _tentar_scraping_csv(url, resultado)
    return resultado


def _tentar_scraping_csv(url: str, resultado: dict) -> dict:
    """
    Faz GET em `url`, parseia o HTML e procura links que
    apontem para arquivos .csv. Retorna o resultado atualizado.
    """
    import requests

    try:
        resp = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (dataset-downloader)'
        })
        resp.raise_for_status()
    except Exception as e:
        resultado['mensagem'] += f' Erro ao acessar a página: {e}'
        return resultado

    content_type = resp.headers.get('Content-Type', '')

    # Se o conteúdo já for CSV/text, é um link direto disfarçado
    if 'text/csv' in content_type or 'application/csv' in content_type:
        resultado.update(url_direta=url, tipo='direto', resolvido=True,
                         mensagem='Content-Type indica CSV — link direto.')
        return resultado

    # Procurar links .csv no HTML
    csv_links = re.findall(
        r'(?:href|src)=["\']([^"\'>]+\.csv(?:\?[^"\'>]*)?)["\']',
        resp.text, re.IGNORECASE,
    )

    if not csv_links:
        # Tentar também .data e .tsv
        csv_links = re.findall(
            r'(?:href|src)=["\']([^"\'>]+\.(?:data|tsv|txt)(?:\?[^"\'>]*)?)["\']',
            resp.text, re.IGNORECASE,
        )

    if csv_links:
        # Resolver URLs relativas
        resolved = [urljoin(url, link) for link in csv_links]
        # Priorizar por tamanho de nome (geralmente o principal é mais curto)
        # e remover duplicatas
        seen = set()
        unique = []
        for link in resolved:
            if link not in seen:
                seen.add(link)
                unique.append(link)

        resultado.update(
            url_direta=unique[0],
            tipo='scraping',
            resolvido=True,
            mensagem=f'Encontrado(s) {len(unique)} arquivo(s) de dados na página.',
        )
        if len(unique) > 1:
            resultado['alternativas'] = unique[1:10]  # máx 10 alternativas
        return resultado

    resultado['mensagem'] += ' Nenhum link CSV/TSV/DATA encontrado na página.'
    return resultado


# ============================================================
# 5. CARREGAR CSV VIA URL
# ============================================================

DOMINIOS_PERMITIDOS = {
    "raw.githubusercontent.com",
    "github.com",
    "gist.githubusercontent.com",
    "storage.googleapis.com",
    "huggingface.co",
    "www.huggingface.co",
    "datasets-server.huggingface.co",
    "archive.ics.uci.edu",
    "www.kaggle.com",
    "kaggle.com",
    "docs.google.com",
    "drive.google.com",
    "www.dropbox.com",
    "dropbox.com",
    "data.world",
    "dataverse.harvard.edu",
    "zenodo.org",
    "figshare.com",
    "datahub.io",
    "openml.org",
    "catalog.data.gov",
    "data.mendeley.com",
}


def validar_url_dataset(url: str) -> tuple:
    """
    Valida se a URL é segura para download de datasets.

    Returns:
        (valida: bool, mensagem: str)
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "URL inválida."

    if parsed.scheme not in ("http", "https"):
        return False, "Apenas URLs HTTP/HTTPS são permitidas."

    dominio = parsed.hostname
    if dominio is None:
        return False, "URL sem domínio válido."

    # Verificar se o domínio está na lista de permitidos
    dominio_ok = any(dominio == d or dominio.endswith("." + d) for d in DOMINIOS_PERMITIDOS)
    if not dominio_ok:
        return False, (
            f"Domínio `{dominio}` não está na lista de fontes confiáveis. "
            f"Domínios aceitos: {', '.join(sorted(DOMINIOS_PERMITIDOS))}"
        )

    return True, "OK"


def carregar_csv_url(url: str, target_col: str, separator: str = ",",
                     max_rows: int = 50000, resolver_auto: bool = True) -> dict:
    """
    Baixa um CSV de uma URL pública e prepara para classificação.

    Se `resolver_auto=True` (padrão), tenta resolver automaticamente
    URLs de páginas (GitHub, Google Drive, HuggingFace, Kaggle…)
    para links diretos antes de baixar.

    Args:
        url:           URL (página ou link direto) para o CSV
        target_col:    Nome da coluna alvo (label)
        separator:     Separador do CSV (, ou ; ou \\t)
        max_rows:      Limite de linhas para evitar problemas de memória
        resolver_auto: Se True, usa resolver_url_dataset() primeiro

    Returns:
        Dicionário com X, y, feature_names, classes, descricao, etc.
    """
    import requests

    # --- Resolução automática de URL ---
    url_download = url
    info_resolucao = None
    if resolver_auto:
        info_resolucao = resolver_url_dataset(url)
        if info_resolucao['resolvido'] and info_resolucao['url_direta']:
            url_download = info_resolucao['url_direta']
            if info_resolucao.get('separador_sugerido') and separator == ',':
                separator = info_resolucao['separador_sugerido']

    # Kaggle local files (kagglehub)
    if url_download.startswith('file://'):
        local_path = url_download.replace('file://', '')
        df = pd.read_csv(local_path, sep=separator, nrows=max_rows)
        return _preparar_dataset_df(df, target_col, url, info_resolucao)

    # Validar URL de download
    valida, msg = validar_url_dataset(url_download)
    if not valida:
        raise ValueError(msg)

    # Download com timeout e limite de tamanho
    try:
        resp = requests.get(url_download, timeout=30, stream=True)
        resp.raise_for_status()

        # Limitar a 50MB
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > 50 * 1024 * 1024:
            raise ValueError("Arquivo muito grande (>50MB).")

        # Ler conteúdo com limite
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            chunks.append(chunk)
            total += len(chunk)
            if total > 50 * 1024 * 1024:
                raise ValueError("Download excedeu 50MB.")

        content = b"".join(chunks)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Erro no download: {e}")

    # Detectar encoding
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    # Ler CSV
    df = pd.read_csv(io.StringIO(text), sep=separator, nrows=max_rows)

    return _preparar_dataset_df(df, target_col, url, info_resolucao)


def _preparar_dataset_df(df: pd.DataFrame, target_col: str,
                         url_original: str, info_resolucao: dict = None) -> dict:
    """
    Limpa, valida e empacota um DataFrame já carregado como
    dicionário pronto para treino/avaliação.
    """
    from sklearn.preprocessing import LabelEncoder

    if target_col not in df.columns:
        raise ValueError(
            f"Coluna '{target_col}' não encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    # Separar features numéricas e target
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_col]

    if len(feature_cols) < 2:
        raise ValueError(
            f"Menos de 2 features numéricas encontradas. "
            f"Colunas numéricas: {list(df.select_dtypes(include=[np.number]).columns)}"
        )

    # Remover linhas com NaN nas features e target
    df = df.dropna(subset=feature_cols + [target_col])

    if len(df) < 10:
        raise ValueError(f"Apenas {len(df)} amostras válidas após remover NaN.")

    X = df[feature_cols].values
    y_raw = df[target_col].values

    # Codificar labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)

    # Limitar classes (evitar problemas com 1000+ classes)
    if len(classes) > 50:
        top = pd.Series(y_raw).value_counts().head(50).index.tolist()
        mask = pd.Series(y_raw).isin(top)
        df = df[mask.values]
        X = df[feature_cols].values
        y_raw = df[target_col].values
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        classes = list(le.classes_)

    parsed = urlparse(url_original)
    nome_arquivo = os.path.basename(parsed.path) or "dataset"

    result = {
        "X": X,
        "y": y,
        "feature_names": feature_cols,
        "classes": [str(c) for c in classes],
        "descricao": f"Dataset via URL: {nome_arquivo}",
        "n_amostras": len(df),
        "n_features": len(feature_cols),
        "n_classes": len(classes),
        "df": df,
        "label_encoder": le,
        "url": url_original,
    }

    if info_resolucao:
        result['info_resolucao'] = info_resolucao

    return result


# ============================================================
# Função auxiliar: verificar quais datasets estão disponíveis
# ============================================================

def listar_datasets_disponiveis() -> dict:
    """
    Verifica quais datasets públicos estão em data/raw/.

    Returns:
        Dicionário {nome: loader_function} dos disponíveis
    """
    disponiveis = {}

    if os.path.isfile(WATER_POTABILITY_FILE):
        disponiveis["💧 Water Potability (Kaggle)"] = carregar_water_potability
    if os.path.isfile(BEER_STYLE_FILE):
        disponiveis["🍺 Beer Style — ML Olympiad (Kaggle)"] = carregar_beer_style
    if os.path.isdir(GAS_SENSOR_DIR):
        disponiveis["🧫 Gas Sensor Drift (UCI)"] = carregar_gas_sensor_drift

    return disponiveis


def instrucoes_download() -> str:
    """Retorna instruções de download formatadas para o usuário."""
    linhas = [
        "### 📥 Como adicionar datasets\n",
        "**Opção 1 — Automático (recomendado):**\n",
        "Use a aba **📥 Importar Dataset** no webapp para baixar",
        "diretamente do catálogo ou via URL.\n",
        "**Opção 2 — Via kagglehub:**\n",
        "```bash",
        "pip install kagglehub",
        "```",
        "Configure ~/.kaggle/kaggle.json e o sistema baixa automaticamente.\n",
        "**Opção 3 — Manual:**\n",
        "Coloque os CSVs na pasta `data/raw/`.\n",
        f"📂 Pasta esperada: `{DATA_RAW}`\n",
        f"📚 **{len(CATALOGO_DATASETS)} datasets** disponíveis no catálogo:\n",
    ]
    for cat in categorias_catalogo():
        ds_cat = listar_catalogo(cat)
        linhas.append(f"\n**{cat}** ({len(ds_cat)}):")
        for ds_info in ds_cat.values():
            slug = ds_info.get('kaggle_slug', '')
            fonte = f"kaggle: {slug}" if slug else ds_info['url'][:60]
            linhas.append(f"- {ds_info['nome']} — `{fonte}`")
    return "\n".join(linhas)
