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

    # =========================================================
    # CAFÉ — Perfis baseados em:
    #   - Illy & Viani (2005), Espresso Coffee: Science of Quality
    #   - Specialty Coffee Association (SCA) Protocols
    #   - Conductivity: Petracco (2001), Tecnologia del Café
    # =========================================================

    # ---------------------------------------------------------
    # CAFÉ ESPRESSO
    # Extração sob pressão: 25-30 mL, altíssima concentração
    # TDS ~8-12%, condutividade elevada, pH ~4.9
    # Cor: marrom escuro intenso com crema (opaco)
    # ---------------------------------------------------------
    "cafe_espresso": {
        "temperatura_C":     [65.0,  10.0],     # Servido quente
        "condutividade_uS":  [1400,  200.0],    # Alta concentração de sólidos
        "pH":                [4.9,   0.3],       # Ácidos orgânicos (clorogênico, cítrico)
        "spec_F1_415nm":     [0.04,  0.03],      # Quase opaco no violeta (melanoidinas)
        "spec_F2_445nm":     [0.06,  0.03],
        "spec_F3_480nm":     [0.08,  0.04],
        "spec_F4_515nm":     [0.12,  0.05],
        "spec_F5_555nm":     [0.16,  0.05],
        "spec_F6_590nm":     [0.22,  0.06],      # Leve transmissão marrom
        "spec_F7_630nm":     [0.28,  0.07],      # Marrom transmite vermelho
        "spec_F8_680nm":     [0.25,  0.07],
        "spec_Clear":        [0.14,  0.05],       # Muito opaco
        "spec_NIR":          [0.32,  0.08],       # Cafeína + compostos orgânicos
        "acustico_freq_Hz":  [1488,  8.0],
    },
    # ---------------------------------------------------------
    # CAFÉ FILTRADO (Coado)
    # Método de percolação: menos concentrado que espresso
    # TDS ~1.2-1.5%, pH ~5.0
    # ---------------------------------------------------------
    "cafe_filtrado": {
        "temperatura_C":     [60.0,  10.0],
        "condutividade_uS":  [1000,  150.0],
        "pH":                [5.0,   0.3],
        "spec_F1_415nm":     [0.10,  0.05],
        "spec_F2_445nm":     [0.14,  0.05],
        "spec_F3_480nm":     [0.18,  0.06],
        "spec_F4_515nm":     [0.24,  0.06],
        "spec_F5_555nm":     [0.30,  0.07],
        "spec_F6_590nm":     [0.38,  0.07],
        "spec_F7_630nm":     [0.42,  0.08],
        "spec_F8_680nm":     [0.36,  0.08],
        "spec_Clear":        [0.25,  0.06],
        "spec_NIR":          [0.38,  0.08],
        "acustico_freq_Hz":  [1486,  8.0],
    },
    # ---------------------------------------------------------
    # CAFÉ CAPPUCCINO (com leite vaporizado)
    # Leite eleva pH, adiciona espalhamento óptico (emulsão)
    # Gordura do leite: forte absorção NIR
    # ---------------------------------------------------------
    "cafe_cappuccino": {
        "temperatura_C":     [55.0,  10.0],
        "condutividade_uS":  [900,   150.0],    # Diluído pelo leite
        "pH":                [5.5,   0.4],        # Leite eleva pH (~6.6)
        "spec_F1_415nm":     [0.20,  0.06],       # Leite espalha luz (Mie scattering)
        "spec_F2_445nm":     [0.25,  0.06],
        "spec_F3_480nm":     [0.30,  0.07],
        "spec_F4_515nm":     [0.35,  0.07],
        "spec_F5_555nm":     [0.38,  0.07],
        "spec_F6_590nm":     [0.42,  0.07],
        "spec_F7_630nm":     [0.40,  0.08],
        "spec_F8_680nm":     [0.35,  0.08],
        "spec_Clear":        [0.32,  0.07],       # Opacidade do leite
        "spec_NIR":          [0.42,  0.08],       # Gordura do leite absorve NIR
        "acustico_freq_Hz":  [1492,  8.0],        # Leite aumenta densidade
    },
    # ---------------------------------------------------------
    # CAFÉ SOLÚVEL (instantâneo)
    # Processo de liofilização ou spray-drying
    # Menor complexidade química que café fresco
    # ---------------------------------------------------------
    "cafe_soluvel": {
        "temperatura_C":     [70.0,  12.0],
        "condutividade_uS":  [800,   150.0],
        "pH":                [5.1,   0.3],
        "spec_F1_415nm":     [0.08,  0.04],
        "spec_F2_445nm":     [0.12,  0.05],
        "spec_F3_480nm":     [0.16,  0.05],
        "spec_F4_515nm":     [0.20,  0.06],
        "spec_F5_555nm":     [0.26,  0.06],
        "spec_F6_590nm":     [0.34,  0.07],
        "spec_F7_630nm":     [0.38,  0.07],
        "spec_F8_680nm":     [0.32,  0.07],
        "spec_Clear":        [0.20,  0.06],
        "spec_NIR":          [0.35,  0.08],
        "acustico_freq_Hz":  [1487,  8.0],
    },

    # =========================================================
    # CHÁ — Perfis baseados em:
    #   - Harbowy et al. (1997), Tea Chemistry, CRC Critical Reviews
    #   - ISO 3103 (Tea preparation for sensory analysis)
    #   - Liang et al. (2003), J. Science of Food & Agriculture
    # =========================================================

    # ---------------------------------------------------------
    # CHÁ VERDE (Camellia sinensis - não oxidado)
    # Catequinas preservadas, clorofila presente
    # pH neutro (~6.5), condutividade baixa
    # Espectro: tom verde-amarelado, clorofila absorve 680nm
    # ---------------------------------------------------------
    "cha_verde": {
        "temperatura_C":     [70.0,  10.0],
        "condutividade_uS":  [200,   60.0],
        "pH":                [6.5,   0.5],
        "spec_F1_415nm":     [0.62,  0.07],       # Moderada absorção violeta
        "spec_F2_445nm":     [0.68,  0.06],
        "spec_F3_480nm":     [0.74,  0.06],
        "spec_F4_515nm":     [0.80,  0.05],       # Pico verde (clorofila reflete)
        "spec_F5_555nm":     [0.78,  0.06],
        "spec_F6_590nm":     [0.72,  0.06],
        "spec_F7_630nm":     [0.68,  0.07],
        "spec_F8_680nm":     [0.58,  0.08],       # Clorofila absorve 680nm
        "spec_Clear":        [0.70,  0.06],
        "spec_NIR":          [0.58,  0.08],
        "acustico_freq_Hz":  [1485,  6.0],
    },
    # ---------------------------------------------------------
    # CHÁ PRETO (Camellia sinensis - totalmente oxidado)
    # Teaflavinas e tearubiginas → cor âmbar/marrom
    # pH ácido (~4.8), condutividade moderada
    # ---------------------------------------------------------
    "cha_preto": {
        "temperatura_C":     [80.0,  10.0],
        "condutividade_uS":  [350,   80.0],
        "pH":                [4.8,   0.4],
        "spec_F1_415nm":     [0.35,  0.07],       # Absorção forte em curtos λ
        "spec_F2_445nm":     [0.40,  0.07],
        "spec_F3_480nm":     [0.48,  0.07],
        "spec_F4_515nm":     [0.55,  0.07],
        "spec_F5_555nm":     [0.62,  0.06],
        "spec_F6_590nm":     [0.65,  0.07],       # Pico âmbar
        "spec_F7_630nm":     [0.58,  0.07],
        "spec_F8_680nm":     [0.50,  0.08],
        "spec_Clear":        [0.50,  0.07],
        "spec_NIR":          [0.52,  0.08],
        "acustico_freq_Hz":  [1484,  6.0],
    },
    # ---------------------------------------------------------
    # CHÁ DE CAMOMILA (Matricaria chamomilla)
    # Infusão leve, amarelo pálido, flavonóides (apigenina)
    # pH ~ 6.0, condutividade muito baixa
    # ---------------------------------------------------------
    "cha_camomila": {
        "temperatura_C":     [70.0,  10.0],
        "condutividade_uS":  [150,   50.0],
        "pH":                [6.0,   0.4],
        "spec_F1_415nm":     [0.70,  0.06],
        "spec_F2_445nm":     [0.73,  0.05],
        "spec_F3_480nm":     [0.78,  0.05],
        "spec_F4_515nm":     [0.82,  0.05],
        "spec_F5_555nm":     [0.85,  0.04],       # Pico amarelo-verde
        "spec_F6_590nm":     [0.83,  0.05],
        "spec_F7_630nm":     [0.78,  0.06],
        "spec_F8_680nm":     [0.72,  0.07],
        "spec_Clear":        [0.78,  0.05],
        "spec_NIR":          [0.62,  0.07],
        "acustico_freq_Hz":  [1483,  6.0],
    },
    # ---------------------------------------------------------
    # CHÁ MATE (Ilex paraguariensis)
    # Popular no Brasil, Argentina, Uruguai
    # Contém saponinas, polifenóis, cafeína
    # Cor verde-amarronzada, sabor amargo
    # Ref: Heck & de Mejia (2007), J. Food Science
    # ---------------------------------------------------------
    "cha_mate": {
        "temperatura_C":     [60.0,  15.0],       # Chimarrão quente ou tereré frio
        "condutividade_uS":  [400,   100.0],
        "pH":                [5.5,   0.5],
        "spec_F1_415nm":     [0.42,  0.08],
        "spec_F2_445nm":     [0.48,  0.07],
        "spec_F3_480nm":     [0.52,  0.07],
        "spec_F4_515nm":     [0.58,  0.07],       # Tom esverdeado
        "spec_F5_555nm":     [0.56,  0.07],
        "spec_F6_590nm":     [0.52,  0.08],
        "spec_F7_630nm":     [0.48,  0.08],
        "spec_F8_680nm":     [0.42,  0.09],
        "spec_Clear":        [0.50,  0.07],
        "spec_NIR":          [0.48,  0.08],
        "acustico_freq_Hz":  [1484,  7.0],
    },

    # =========================================================
    # SUCO NATURAL — Perfis baseados em:
    #   - USDA FoodData Central (composição nutricional)
    #   - Sánchez-Moreno et al. (2003), Food Chemistry
    #   - Meléndez-Martínez et al. (2007), Food Chemistry
    #   - Legislação brasileira: IN nº 37/2018 MAPA
    #     (suco = 100% fruta, sem adição de açúcar/água)
    # =========================================================

    # ---------------------------------------------------------
    # SUCO DE LARANJA (Natural, integral)
    # Rico em carotenóides (absorve azul, transmite laranja)
    # Brix: 10-14°, alta polpa óptica
    # Condutividade alta: ácido cítrico + minerais (K, Ca)
    # ---------------------------------------------------------
    "suco_laranja": {
        "temperatura_C":     [8.0,   3.0],        # Servido gelado
        "condutividade_uS":  [1200,  200.0],      # Rico em íons (citrato, K+)
        "pH":                [3.5,   0.3],         # Ácido cítrico dominante
        "spec_F1_415nm":     [0.08,  0.04],        # Carotenóides absorvem azul/violeta
        "spec_F2_445nm":     [0.10,  0.04],
        "spec_F3_480nm":     [0.14,  0.05],
        "spec_F4_515nm":     [0.25,  0.06],
        "spec_F5_555nm":     [0.50,  0.07],        # Transição verde-amarelo
        "spec_F6_590nm":     [0.70,  0.06],        # Pico laranja (carotenóides)
        "spec_F7_630nm":     [0.55,  0.07],
        "spec_F8_680nm":     [0.40,  0.08],
        "spec_Clear":        [0.30,  0.07],        # Turbidez alta (polpa)
        "spec_NIR":          [0.48,  0.08],        # Açúcar absorve NIR
        "acustico_freq_Hz":  [1470,  10.0],        # Denso (açúcar + polpa)
    },
    # ---------------------------------------------------------
    # SUCO DE UVA (Natural, integral)
    # Antocianinas: absorção forte no verde (cor roxa/púrpura)
    # Brix: 14-18° (muito doce), alta turbidez
    # Ref: Nixdorf & Hermosín-Gutiérrez (2010), J. Agric. Food Chem.
    # ---------------------------------------------------------
    "suco_uva": {
        "temperatura_C":     [8.0,   3.0],
        "condutividade_uS":  [1100,  200.0],
        "pH":                [3.3,   0.3],         # Ácido tartárico
        "spec_F1_415nm":     [0.05,  0.03],        # Antocianinas absorvem violeta-verde
        "spec_F2_445nm":     [0.06,  0.03],
        "spec_F3_480nm":     [0.08,  0.04],
        "spec_F4_515nm":     [0.10,  0.04],        # Forte absorção no verde
        "spec_F5_555nm":     [0.12,  0.05],
        "spec_F6_590nm":     [0.18,  0.06],
        "spec_F7_630nm":     [0.32,  0.07],        # Alguma transmissão vermelho
        "spec_F8_680nm":     [0.35,  0.08],
        "spec_Clear":        [0.12,  0.05],        # Muito opaco (antocianinas + polpa)
        "spec_NIR":          [0.42,  0.08],        # Alto açúcar
        "acustico_freq_Hz":  [1468,  10.0],
    },
    # ---------------------------------------------------------
    # SUCO DE MANGA (Natural, integral)
    # Carotenóides (β-caroteno): absorve azul, transmite amarelo-laranja
    # Brix: 12-16°, polpa densa
    # Ref: Schieber et al. (2000), Trends in Food Sci. & Tech.
    # ---------------------------------------------------------
    "suco_manga": {
        "temperatura_C":     [8.0,   3.0],
        "condutividade_uS":  [900,   180.0],
        "pH":                [3.8,   0.3],
        "spec_F1_415nm":     [0.10,  0.05],
        "spec_F2_445nm":     [0.14,  0.05],
        "spec_F3_480nm":     [0.18,  0.06],
        "spec_F4_515nm":     [0.32,  0.07],
        "spec_F5_555nm":     [0.60,  0.07],        # Pico amarelo
        "spec_F6_590nm":     [0.68,  0.06],        # Pico laranja
        "spec_F7_630nm":     [0.52,  0.07],
        "spec_F8_680nm":     [0.38,  0.08],
        "spec_Clear":        [0.33,  0.07],        # Turbidez por polpa
        "spec_NIR":          [0.45,  0.08],
        "acustico_freq_Hz":  [1472,  10.0],
    },

    # =========================================================
    # REFRESCO (bebida pronta diluída) — Perfis baseados em:
    #   - Legislação brasileira: Decreto nº 6.871/2009
    #     Refresco = mín. 20-30% de suco diluído em água + açúcar
    #   - Ashurst (2005), Chemistry and Technology of Soft Drinks
    #   - Diferenças-chave vs suco natural:
    #       • Condutividade muito menor (diluição)
    #       • Espectro mais claro (menos pigmento)
    #       • Turbidez menor (sem polpa ou pouca polpa)
    #       • Brix similar (açúcar adicionado) mas matrix diferente
    # =========================================================

    # ---------------------------------------------------------
    # REFRESCO DE LARANJA
    # ~25% suco + água + açúcar + acidulante + corante
    # ---------------------------------------------------------
    "refresco_laranja": {
        "temperatura_C":     [8.0,   3.0],
        "condutividade_uS":  [350,   100.0],      # Muito diluído
        "pH":                [3.0,   0.3],          # Acidulante adicionado (ácido cítrico)
        "spec_F1_415nm":     [0.42,  0.08],         # Bem mais claro que suco natural
        "spec_F2_445nm":     [0.48,  0.07],
        "spec_F3_480nm":     [0.55,  0.07],
        "spec_F4_515nm":     [0.62,  0.07],
        "spec_F5_555nm":     [0.72,  0.06],
        "spec_F6_590nm":     [0.78,  0.05],         # Tom laranja mas mais claro
        "spec_F7_630nm":     [0.72,  0.06],
        "spec_F8_680nm":     [0.62,  0.07],
        "spec_Clear":        [0.65,  0.06],         # Muito mais transparente
        "spec_NIR":          [0.58,  0.07],
        "acustico_freq_Hz":  [1478,  8.0],          # Próximo à água (diluído)
    },
    # ---------------------------------------------------------
    # REFRESCO DE UVA
    # ~20% suco + água + açúcar + corante artificial (carmim/tartrazina)
    # ---------------------------------------------------------
    "refresco_uva": {
        "temperatura_C":     [8.0,   3.0],
        "condutividade_uS":  [300,   100.0],
        "pH":                [2.9,   0.3],
        "spec_F1_415nm":     [0.28,  0.07],         # Corante absorve, mas menos que natural
        "spec_F2_445nm":     [0.32,  0.07],
        "spec_F3_480nm":     [0.36,  0.07],
        "spec_F4_515nm":     [0.40,  0.07],
        "spec_F5_555nm":     [0.42,  0.07],
        "spec_F6_590nm":     [0.50,  0.07],
        "spec_F7_630nm":     [0.58,  0.07],
        "spec_F8_680nm":     [0.52,  0.08],
        "spec_Clear":        [0.40,  0.07],         # Mais claro que suco natural
        "spec_NIR":          [0.55,  0.07],
        "acustico_freq_Hz":  [1479,  8.0],
    },

    # =========================================================
    # CERVEJA CONTAMINADA — Perfis baseados em:
    #   - Caso Backer / Belorizontina (Belo Horizonte, 2019-2020)
    #     Contaminação por Dietilenoglicol (DEG) e Metanol
    #   - Lachenmeier et al. (2007), Regulatory Toxicology & Pharmacology
    #   - Rehm et al. (2014), BMC Public Health (methanol toxicity)
    #   - ANVISA RDC nº 275/2019 (limites de contaminantes)
    #
    #   O DEG vazou do sistema de refrigeração para os tanques
    #   de cerveja. É um glicol tóxico (DL50 ~1 mL/kg) que causa
    #   insuficiência renal aguda. O metanol é subproduto da
    #   degradação/reação do DEG com etanol.
    # =========================================================

    # ---------------------------------------------------------
    # CERVEJA CONTAMINADA COM METANOL
    # Base: lager similar à Budweiser, com metanol >100 mg/L
    # Metanol: d=0.792, absorção NIR diferente de etanol
    # pH levemente menor (oxidação → ácido fórmico)
    # Condutividade levemente elevada (ácido fórmico ioniza)
    # ---------------------------------------------------------
    "cerveja_contaminada_metanol": {
        "temperatura_C":     [5.0,   2.0],
        "condutividade_uS":  [1900,  250.0],      # Levemente elevada
        "pH":                [3.9,   0.3],          # Ácido fórmico reduz pH
        "spec_F1_415nm":     [0.33,  0.08],
        "spec_F2_445nm":     [0.43,  0.07],
        "spec_F3_480nm":     [0.58,  0.07],
        "spec_F4_515nm":     [0.70,  0.06],
        "spec_F5_555nm":     [0.80,  0.05],
        "spec_F6_590nm":     [0.73,  0.06],
        "spec_F7_630nm":     [0.53,  0.07],
        "spec_F8_680nm":     [0.36,  0.08],
        "spec_Clear":        [0.58,  0.07],
        "spec_NIR":          [0.52,  0.08],        # Metanol × etanol = NIR diferente
        "acustico_freq_Hz":  [1458,  12.0],        # Metanol altera densidade
    },
    # ---------------------------------------------------------
    # CERVEJA CONTAMINADA COM DIETILENOGLICOL (DEG)
    # DEG: C4H10O3, d=1.118, viscoso, doce
    # Aumenta significativamente condutividade e viscosidade
    # Absorção NIR distinta (ligações O-H e C-H do glicol)
    # Altera propagação acústica (viscosidade 5x maior que água)
    # ---------------------------------------------------------
    "cerveja_contaminada_deg": {
        "temperatura_C":     [5.0,   2.0],
        "condutividade_uS":  [2200,  300.0],      # DEG aumenta condutividade
        "pH":                [4.0,   0.2],
        "spec_F1_415nm":     [0.34,  0.08],         # Espectro visível similar a lager
        "spec_F2_445nm":     [0.44,  0.07],
        "spec_F3_480nm":     [0.58,  0.07],
        "spec_F4_515nm":     [0.70,  0.06],
        "spec_F5_555nm":     [0.78,  0.06],
        "spec_F6_590nm":     [0.72,  0.06],
        "spec_F7_630nm":     [0.52,  0.07],
        "spec_F8_680nm":     [0.36,  0.08],
        "spec_Clear":        [0.56,  0.07],
        "spec_NIR":          [0.56,  0.09],        # DEG tem absorção NIR distinta
        "acustico_freq_Hz":  [1440,  15.0],        # DEG viscoso → muda propagação
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
    "cerveja_contaminada_metanol": "cerveja",
    "cerveja_contaminada_deg":     "cerveja",
    "cafe_espresso":     "cafe",
    "cafe_filtrado":     "cafe",
    "cafe_cappuccino":   "cafe",
    "cafe_soluvel":      "cafe",
    "cha_verde":         "cha",
    "cha_preto":         "cha",
    "cha_camomila":      "cha",
    "cha_mate":          "cha",
    "suco_laranja":      "suco",
    "suco_uva":          "suco",
    "suco_manga":        "suco",
    "refresco_laranja":  "refresco",
    "refresco_uva":      "refresco",
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
    "cerveja_contaminada_metanol": -1,
    "cerveja_contaminada_deg":     -1,
    "cafe_espresso":     -1,
    "cafe_filtrado":     -1,
    "cafe_cappuccino":   -1,
    "cafe_soluvel":      -1,
    "cha_verde":         -1,
    "cha_preto":         -1,
    "cha_camomila":      -1,
    "cha_mate":          -1,
    "suco_laranja":      -1,
    "suco_uva":          -1,
    "suco_manga":        -1,
    "refresco_laranja":  -1,
    "refresco_uva":      -1,
}

# Mapeamento: subtipo de cerveja → adulteração
# "normal" = cerveja legítima, outros indicam tipo de contaminante
MAPA_ADULTERACAO = {
    "cerveja_heineken":            "normal",
    "cerveja_budweiser":           "normal",
    "cerveja_ipa":                 "normal",
    "cerveja_stout":               "normal",
    "cerveja_contaminada_metanol": "metanol",
    "cerveja_contaminada_deg":     "dietilenoglicol",
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
    dados["adulteracao"] = MAPA_ADULTERACAO.get(subtipo, "nao_aplica")

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
