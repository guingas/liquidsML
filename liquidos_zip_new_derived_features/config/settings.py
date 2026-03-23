"""
============================================================
 CONFIGURAÇÕES GLOBAIS DO PROJETO
 Dispositivo S - Identificação de Líquidos
============================================================
 Todas as constantes e parâmetros centralizados aqui.
 Altere este arquivo para ajustar o comportamento do pipeline.
============================================================
"""

import os

# ------------------------------------------------------------
# Caminhos do Projeto
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")
DATA_SYNTHETIC = os.path.join(DATA_DIR, "synthetic")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Criar diretórios se não existirem
for d in [DATA_RAW, DATA_PROCESSED, DATA_SYNTHETIC, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------
# Parâmetros de Geração de Dados Sintéticos
# ------------------------------------------------------------
N_AMOSTRAS_POR_CLASSE = 500       # Quantidade de amostras por tipo de líquido
RANDOM_SEED = 42                   # Semente para reprodutibilidade
NOISE_LEVEL = 0.05                 # Nível de ruído nos sensores (5%)

# ------------------------------------------------------------
# Definição dos Líquidos e Suas Classes
# ------------------------------------------------------------
# Nível 1: Tipo do líquido (classificação multiclasse)
TIPOS_LIQUIDO = ["agua", "cerveja", "cafe", "cha", "suco", "refresco"]

# Nível 2: Subtipos (classificação fina)
SUBTIPOS = {
    "agua": ["torneira", "mineral", "mineral_gas"],
    "cerveja": ["heineken", "budweiser", "ipa", "stout",
                "contaminada_metanol", "contaminada_deg"],
    "cafe": ["espresso", "filtrado", "cappuccino", "soluvel"],
    "cha": ["verde", "preto", "camomila", "mate"],
    "suco": ["laranja", "uva", "manga"],
    "refresco": ["laranja", "uva"],
}

# ------------------------------------------------------------
# Nomes dos Sensores (Features do vetor x~)
# ------------------------------------------------------------
# Estes são os sensores que compõem o vetor de features reduzido
SENSOR_NAMES = [
    # Temperatura (DS18B20)
    "temperatura_C",
    # Condutividade elétrica / TDS (Eletrodos interdigitados PCB)
    "condutividade_uS",
    # pH (Módulo pH E-201-C)
    "pH",
    # Espectro óptico - AS7341 (11 canais)
    "spec_F1_415nm",   # Violeta
    "spec_F2_445nm",   # Azul escuro
    "spec_F3_480nm",   # Azul
    "spec_F4_515nm",   # Verde
    "spec_F5_555nm",   # Verde-amarelo
    "spec_F6_590nm",   # Laranja
    "spec_F7_630nm",   # Vermelho
    "spec_F8_680nm",   # Vermelho escuro
    "spec_Clear",      # Luz visível total
    "spec_NIR",        # Infravermelho próximo
    # Resposta acústica (Disco piezoelétrico)
    "acustico_freq_Hz",
]

# Total de features
N_FEATURES = len(SENSOR_NAMES)

# ------------------------------------------------------------
# Parâmetros de Divisão dos Dados
# ------------------------------------------------------------
TRAIN_RATIO = 0.70    # 70% treino
VAL_RATIO = 0.15      # 15% validação
TEST_RATIO = 0.15     # 15% teste

# ------------------------------------------------------------
# Parâmetros dos Modelos de Machine Learning
# ------------------------------------------------------------
ML_PARAMS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_SEED,
        "eval_metric": "mlogloss",
    },
    "neural_network": {
        "hidden_layers": [128, 64, 32],
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
    "cnn": {
        "filters": [32, 64],
        "kernel_size": 3,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
    "svm": {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
        "random_state": RANDOM_SEED,
    },
}
