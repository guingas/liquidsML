"""
============================================================
 GERADOR DE DADOS SINTÉTICOS DE SENSORES
 Simula leituras realistas dos sensores do dispositivo S
============================================================

 Este módulo gera dados que imitam as leituras reais que os
 sensores produziriam ao analisar diferentes líquidos.

 Cada líquido possui um "perfil de sensor" baseado em suas
 propriedades físico-químicas conhecidas na literatura:
   - Condutividade elétrica (sais dissolvidos)
   - pH (acidez)
   - Espectro óptico (cor / turbidez)
   - Resposta acústica (densidade)
   - Temperatura

 Referências:
   - Propriedades da água: WHO Guidelines for Drinking-water Quality
   - Propriedades da cerveja: BJCP Style Guidelines 2021
   - Velocidade do som em líquidos: CRC Handbook of Chemistry & Physics
============================================================
"""

import numpy as np
import pandas as pd
import os
import sys

# Adiciona o diretório raiz ao path para importar config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    N_AMOSTRAS_POR_CLASSE,
    RANDOM_SEED,
    NOISE_LEVEL,
    SENSOR_NAMES,
    DATA_SYNTHETIC,
)


# ============================================================
# PERFIS FÍSICO-QUÍMICOS DOS LÍQUIDOS
# ============================================================
# Cada perfil define [média, desvio_padrão] para cada sensor.
# Valores baseados em literatura científica e datasheets.
#
# Ordem das features (mesmo de SENSOR_NAMES):
#   [temp, condutividade, pH,
#    F1(415), F2(445), F3(480), F4(515), F5(555), F6(590),
#    F7(630), F8(680), Clear, NIR,
#    acustico]
# ============================================================

PERFIS_SENSORES = {
    # ---------------------------------------------------------
    # ÁGUA DA TORNEIRA
    # Condutividade: 200-800 µS/cm (sais minerais variáveis)
    # pH: 6.5-8.0 (tratamento com cloro eleva ligeiramente)
    # Espectro: alta transmissão, quase transparente
    # Acústico: ~1482 m/s (velocidade do som em água pura a 20°C)
    # ---------------------------------------------------------
    "agua_torneira": {
        "temperatura_C":     [22.0,  2.0],
        "condutividade_uS":  [450.0, 150.0],
        "pH":                [7.2,   0.4],
        "spec_F1_415nm":     [0.90,  0.05],   # Alta transmissão (transparente)
        "spec_F2_445nm":     [0.92,  0.04],
        "spec_F3_480nm":     [0.93,  0.04],
        "spec_F4_515nm":     [0.91,  0.04],
        "spec_F5_555nm":     [0.90,  0.05],
        "spec_F6_590nm":     [0.89,  0.05],
        "spec_F7_630nm":     [0.88,  0.05],
        "spec_F8_680nm":     [0.85,  0.06],
        "spec_Clear":        [0.91,  0.04],
        "spec_NIR":          [0.70,  0.08],   # Água absorve NIR
        "acustico_freq_Hz":  [1482,  8.0],
    },
    # ---------------------------------------------------------
    # ÁGUA MINERAL (SEM GÁS)
    # Condutividade mais baixa (menos tratamento químico)
    # pH levemente alcalino
    # ---------------------------------------------------------
    "agua_mineral": {
        "temperatura_C":     [20.0,  2.0],
        "condutividade_uS":  [250.0, 80.0],
        "pH":                [7.5,   0.3],
        "spec_F1_415nm":     [0.93,  0.03],
        "spec_F2_445nm":     [0.94,  0.03],
        "spec_F3_480nm":     [0.95,  0.03],
        "spec_F4_515nm":     [0.94,  0.03],
        "spec_F5_555nm":     [0.93,  0.03],
        "spec_F6_590nm":     [0.92,  0.03],
        "spec_F7_630nm":     [0.91,  0.04],
        "spec_F8_680nm":     [0.88,  0.05],
        "spec_Clear":        [0.93,  0.03],
        "spec_NIR":          [0.72,  0.07],
        "acustico_freq_Hz":  [1480,  6.0],
    },
    # ---------------------------------------------------------
    # ÁGUA MINERAL COM GÁS
    # CO2 dissolvido reduz pH e altera resposta acústica
    # Bolhas causam espalhamento óptico (turbidez aparente)
    # ---------------------------------------------------------
    "agua_mineral_gas": {
        "temperatura_C":     [8.0,   3.0],    # Geralmente servida gelada
        "condutividade_uS":  [350.0, 100.0],  # CO2 eleva levemente
        "pH":                [5.5,   0.5],     # CO2 -> ácido carbônico
        "spec_F1_415nm":     [0.80,  0.08],    # Bolhas reduzem transmissão
        "spec_F2_445nm":     [0.82,  0.07],
        "spec_F3_480nm":     [0.83,  0.07],
        "spec_F4_515nm":     [0.82,  0.07],
        "spec_F5_555nm":     [0.81,  0.08],
        "spec_F6_590nm":     [0.80,  0.08],
        "spec_F7_630nm":     [0.78,  0.08],
        "spec_F8_680nm":     [0.75,  0.09],
        "spec_Clear":        [0.80,  0.07],
        "spec_NIR":          [0.65,  0.10],
        "acustico_freq_Hz":  [1450,  20.0],    # CO2 altera propagação
    },
    # ---------------------------------------------------------
    # HEINEKEN (Lager / Pilsner)
    # Cor: dourado claro (SRM 2-4)
    # ABV: ~5.0%
    # Condutividade: alta (malte, lúpulo, minerais)
    # ---------------------------------------------------------
    "cerveja_heineken": {
        "temperatura_C":     [5.0,   2.0],     # Servida gelada
        "condutividade_uS":  [1800,  200.0],   # Alta mineralização
        "pH":                [4.2,   0.2],      # Ácida (fermentação)
        "spec_F1_415nm":     [0.30,  0.08],     # Absorve violeta
        "spec_F2_445nm":     [0.40,  0.08],
        "spec_F3_480nm":     [0.55,  0.08],
        "spec_F4_515nm":     [0.70,  0.07],     # Pico no verde (cerveja dourada)
        "spec_F5_555nm":     [0.78,  0.06],     # Pico verde-amarelo
        "spec_F6_590nm":     [0.72,  0.07],
        "spec_F7_630nm":     [0.50,  0.08],
        "spec_F8_680nm":     [0.35,  0.08],
        "spec_Clear":        [0.55,  0.07],
        "spec_NIR":          [0.45,  0.08],
        "acustico_freq_Hz":  [1460,  10.0],     # Álcool reduz velocidade
    },
    # ---------------------------------------------------------
    # BUDWEISER (American Lager)
    # Cor: dourado pálido (SRM 1.5-3)
    # ABV: ~5.0%
    # Mais leve que Heineken
    # ---------------------------------------------------------
    "cerveja_budweiser": {
        "temperatura_C":     [5.0,   2.0],
        "condutividade_uS":  [1650,  180.0],
        "pH":                [4.1,   0.2],
        "spec_F1_415nm":     [0.35,  0.08],
        "spec_F2_445nm":     [0.45,  0.07],
        "spec_F3_480nm":     [0.60,  0.07],
        "spec_F4_515nm":     [0.72,  0.06],
        "spec_F5_555nm":     [0.82,  0.05],     # Mais clara → mais transmissão
        "spec_F6_590nm":     [0.75,  0.06],
        "spec_F7_630nm":     [0.55,  0.07],
        "spec_F8_680nm":     [0.38,  0.08],
        "spec_Clear":        [0.60,  0.06],
        "spec_NIR":          [0.48,  0.07],
        "acustico_freq_Hz":  [1462,  10.0],
    },
    # ---------------------------------------------------------
    # IPA (India Pale Ale)
    # Cor: âmbar / cobre (SRM 6-14)
    # ABV: ~6.5%
    # Mais amarga, mais escura, maior teor alcoólico
    # ---------------------------------------------------------
    "cerveja_ipa": {
        "temperatura_C":     [7.0,   2.0],     # Servida menos gelada
        "condutividade_uS":  [2100,  250.0],   # Mais malte → mais minerais
        "pH":                [4.0,   0.2],
        "spec_F1_415nm":     [0.15,  0.06],    # Absorve muito em curtos λ
        "spec_F2_445nm":     [0.22,  0.07],
        "spec_F3_480nm":     [0.35,  0.08],
        "spec_F4_515nm":     [0.50,  0.08],
        "spec_F5_555nm":     [0.60,  0.07],
        "spec_F6_590nm":     [0.68,  0.07],    # Pico no laranja (âmbar)
        "spec_F7_630nm":     [0.60,  0.08],
        "spec_F8_680nm":     [0.45,  0.09],
        "spec_Clear":        [0.42,  0.08],
        "spec_NIR":          [0.40,  0.09],
        "acustico_freq_Hz":  [1448,  12.0],    # Mais álcool → menor velocidade
    },
    # ---------------------------------------------------------
    # STOUT (Guinness-like)
    # Cor: muito escura / preta (SRM 30-40+)
    # ABV: ~5.5%
    # Cevada torrada = espectro muito absorvente
    # ---------------------------------------------------------
    "cerveja_stout": {
        "temperatura_C":     [10.0,  2.5],     # Servida menos gelada
        "condutividade_uS":  [2400,  300.0],   # Muito malte
        "pH":                [4.3,   0.3],
        "spec_F1_415nm":     [0.05,  0.03],    # Quase opaca no violeta
        "spec_F2_445nm":     [0.08,  0.04],
        "spec_F3_480nm":     [0.12,  0.05],
        "spec_F4_515nm":     [0.18,  0.06],
        "spec_F5_555nm":     [0.22,  0.07],
        "spec_F6_590nm":     [0.28,  0.07],
        "spec_F7_630nm":     [0.35,  0.08],    # Alguma transmissão no vermelho
        "spec_F8_680nm":     [0.30,  0.08],
        "spec_Clear":        [0.18,  0.06],
        "spec_NIR":          [0.25,  0.08],
        "acustico_freq_Hz":  [1455,  12.0],
    },
    # ---------------------------------------------------------
    # ÁGUA CONTAMINADA (não potável)
    # Alta condutividade (excesso de sais / metais pesados)
    # pH fora do padrão, turbidez alta
    # Referência: WHO Guidelines — limites de potabilidade
    # ---------------------------------------------------------
    "agua_contaminada": {
        "temperatura_C":     [24.0,  4.0],
        "condutividade_uS":  [1200.0, 400.0],   # Muito acima do normal
        "pH":                [5.5,   1.2],       # Ácida ou alcalina
        "spec_F1_415nm":     [0.55,  0.15],      # Absorção por partículas
        "spec_F2_445nm":     [0.58,  0.14],
        "spec_F3_480nm":     [0.60,  0.14],
        "spec_F4_515nm":     [0.62,  0.14],
        "spec_F5_555nm":     [0.60,  0.14],
        "spec_F6_590nm":     [0.57,  0.14],
        "spec_F7_630nm":     [0.53,  0.14],
        "spec_F8_680nm":     [0.48,  0.15],
        "spec_Clear":        [0.55,  0.13],
        "spec_NIR":          [0.50,  0.12],
        "acustico_freq_Hz":  [1475,  15.0],
    },
}

# Mapeamento: subtipo → tipo principal
MAPA_TIPO = {
    "agua_torneira":     "agua",
    "agua_mineral":      "agua",
    "agua_mineral_gas":  "agua",
    "agua_contaminada":  "agua",
    "cerveja_heineken":  "cerveja",
    "cerveja_budweiser": "cerveja",
    "cerveja_ipa":       "cerveja",
    "cerveja_stout":     "cerveja",
}

# Mapeamento: subtipo → potabilidade (1 = potável, 0 = não potável)
MAPA_POTABILIDADE = {
    "agua_torneira":     1,
    "agua_mineral":      1,
    "agua_mineral_gas":  1,
    "agua_contaminada":  0,
    "cerveja_heineken":  -1,   # -1 = não se aplica
    "cerveja_budweiser": -1,
    "cerveja_ipa":       -1,
    "cerveja_stout":     -1,
}


def gerar_amostras_liquido(subtipo: str, n_amostras: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Gera N amostras sintéticas para um subtipo de líquido.

    Cada amostra simula uma leitura completa de todos os sensores,
    com ruído gaussiano realista adicionado aos valores médios.

    Args:
        subtipo:    Nome do subtipo (ex: "cerveja_heineken")
        n_amostras: Número de amostras a gerar
        rng:        Gerador de números aleatórios (NumPy)

    Returns:
        DataFrame com n_amostras linhas e todas as features + labels
    """
    perfil = PERFIS_SENSORES[subtipo]
    dados = {}

    for sensor_name in SENSOR_NAMES:
        media, desvio = perfil[sensor_name]
        # Gera valores com distribuição normal (simulando ruído real do sensor)
        valores = rng.normal(loc=media, scale=desvio, size=n_amostras)

        # Garante que valores físicos não fiquem negativos
        if sensor_name in ["condutividade_uS", "acustico_freq_Hz"]:
            valores = np.clip(valores, 0, None)
        elif sensor_name == "pH":
            valores = np.clip(valores, 0, 14)
        elif sensor_name.startswith("spec_"):
            valores = np.clip(valores, 0, 1)  # Transmissão normalizada [0, 1]

        dados[sensor_name] = valores

    # Adiciona as labels (rótulos)
    dados["subtipo"] = subtipo
    dados["tipo"] = MAPA_TIPO[subtipo]
    dados["potabilidade"] = MAPA_POTABILIDADE.get(subtipo, -1)

    return pd.DataFrame(dados)


def gerar_dataset_completo(n_amostras_por_classe: int = N_AMOSTRAS_POR_CLASSE,
                           seed: int = RANDOM_SEED,
                           salvar: bool = True) -> pd.DataFrame:
    """
    Gera o dataset completo com todos os tipos de líquidos.

    Args:
        n_amostras_por_classe: Amostras por subtipo de líquido
        seed:                  Semente para reprodutibilidade
        salvar:                Se True, salva o CSV em data/synthetic/

    Returns:
        DataFrame com todas as amostras concatenadas e embaralhadas
    """
    rng = np.random.default_rng(seed)
    frames = []

    print("=" * 60)
    print(" GERANDO DADOS SINTÉTICOS DE SENSORES")
    print("=" * 60)

    for subtipo in PERFIS_SENSORES:
        df_sub = gerar_amostras_liquido(subtipo, n_amostras_por_classe, rng)
        frames.append(df_sub)
        print(f"  ✓ {subtipo:<25s} → {n_amostras_por_classe} amostras geradas")

    # Concatena e embaralha
    df_completo = pd.concat(frames, ignore_index=True)
    df_completo = df_completo.sample(frac=1, random_state=seed).reset_index(drop=True)

    total = len(df_completo)
    n_features = len(SENSOR_NAMES)
    print(f"\n  Total: {total} amostras | {n_features} features por amostra")
    print(f"  Classes (tipo):    {df_completo['tipo'].value_counts().to_dict()}")
    print(f"  Classes (subtipo): {df_completo['subtipo'].value_counts().to_dict()}")

    if salvar:
        caminho = os.path.join(DATA_SYNTHETIC, "dataset_sensores.csv")
        df_completo.to_csv(caminho, index=False)
        print(f"\n  Salvo em: {caminho}")

    print("=" * 60)
    return df_completo


# ============================================================
# Execução direta para teste
# ============================================================
if __name__ == "__main__":
    df = gerar_dataset_completo()
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    print(f"\nEstatísticas descritivas:")
    print(df.describe().round(2))
