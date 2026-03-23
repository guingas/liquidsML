# Dispositivo S — Identificação de Líquidos por Sensores + ML

## Sobre o Projeto

Sistema de **identificação automática de líquidos** usando um conjunto de sensores de baixo custo acoplados a um **ESP32**, com classificação por **Machine Learning**.

**Objetivo:** Validar via simulação quais sensores são mais importantes para o MVP (Minimum Viable Product), antes de comprar o hardware.

---

## Estrutura do Projeto

```
Líquidos/
├── main.py                          ← Executa o pipeline completo
├── requirements.txt                 ← Dependências Python
├── config/
│   └── settings.py                  ← Configurações globais (sensores, modelos, etc.)
├── src/
│   ├── data_generation/
│   │   └── synthetic_sensors.py     ← Gera dados sintéticos dos sensores
│   ├── preprocessing/
│   │   └── pipeline.py              ← Limpeza, normalização, split treino/val/teste
│   ├── models/
│   │   ├── random_forest.py         ← Classificador Random Forest
│   │   ├── xgboost_model.py         ← Classificador XGBoost
│   │   ├── neural_network.py        ← Rede Neural MLP (Keras/TensorFlow)
│   │   ├── cnn_model.py             ← CNN 1D (Keras/TensorFlow)
│   │   └── svm_model.py             ← Support Vector Machine
│   ├── evaluation/
│   │   └── compare_models.py        ← Avaliação e comparação de modelos
│   └── feature_analysis/
│       └── sensor_importance.py     ← Análise de importância dos sensores
├── data/
│   ├── raw/                         ← Datasets públicos (baixar manualmente)
│   ├── processed/                   ← Dados processados (scaler, encoder)
│   └── synthetic/                   ← Dados sintéticos gerados
└── results/
    └── figures/                     ← Gráficos e matrizes de confusão
```

---

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Rodar o pipeline completo

```bash
python main.py
```

O script:
1. Gera **3.500 amostras sintéticas** (500 por tipo de líquido)
2. Pré-processa: compensação de temperatura, features espectrais, normalização
3. Divide: **70% treino** / **15% validação** / **15% teste**
4. Treina **5 modelos** de ML
5. Avalia e compara no conjunto de teste
6. Gera **ranking de importância dos sensores**

---

## Sensores Simulados

| Sensor | Componente | Feature | O que mede |
|--------|-----------|---------|------------|
| Temperatura | DS18B20 | `temperatura_C` | Temperatura do líquido (°C) |
| Condutividade | Eletrodos PCB (ENIG) | `condutividade_uS` | Sais minerais dissolvidos (µS/cm) |
| pH | Módulo E-201-C | `pH` | Acidez/alcalinidade |
| Espectro Óptico | AS7341 (11 canais) | `spec_F1..F8, Clear, NIR` | Cor, turbidez, "impressão digital" do líquido |
| Acústico | Disco piezoelétrico | `acustico_freq_Hz` | Densidade (via velocidade do som) |

---

## Líquidos Simulados

### Tipo "água"
- **Torneira**: condutividade média, pH neutro, transparente
- **Mineral**: condutividade baixa, levemente alcalina
- **Mineral com gás**: pH ácido (CO2), espalhamento óptico por bolhas

### Tipo "cerveja"
- **Heineken** (Lager): dourada clara, ABV ~5%
- **Budweiser** (American Lager): dourada pálida, ABV ~5%
- **IPA**: âmbar/cobre, ABV ~6.5%, mais amarga
- **Stout**: escura/preta, ABV ~5.5%, cevada torrada

---

## Modelos de Machine Learning

| Modelo | Tipo | Por quê usar |
|--------|------|-------------|
| **Random Forest** | Ensemble (Árvores) | Robusto a ruído, feature importance nativa |
| **XGBoost** | Gradient Boosting | Estado da arte para dados tabulares |
| **Rede Neural (MLP)** | Deep Learning | Captura padrões complexos, convertível para TFLite |
| **CNN 1D** | Deep Learning | Detecta padrões no espectro óptico |
| **SVM (RBF)** | Kernel | Bom em alta dimensionalidade, leve para deploy |

---

## Datasets Públicos (para treino futuro com dados reais)

### Qualidade da Água
- **Kaggle — Water Quality and Potability**
  - https://www.kaggle.com/datasets/adityakadiwal/water-potability
  - Contém: pH, Dureza, TDS, Condutividade, Turbidez, Potabilidade

### Perfil de Cervejas
- **Kaggle — Beer Profile and Ratings**
  - https://www.kaggle.com/datasets/ruthgn/beer-profile-and-ratings-data-set
  - Contém: ABV, cor (SRM), amargor (IBU), estilos, marcas

### Língua Eletrônica / Sensores
- **UCI — Electronic Tongue / Gas Sensor Array**
  - https://archive.ics.uci.edu/ml/datasets.php
  - Contém: leituras de matrizes de sensores para classificação de líquidos

---

## Saídas do Pipeline

Após executar `python main.py`, verifique:

- `results/comparacao_modelos_tipo_liquido.csv` — Qual modelo classifica melhor água vs. cerveja
- `results/comparacao_modelos_subtipo_marca.csv` — Qual modelo identifica melhor a marca
- `results/ranking_sensores_tipo_liquido.csv` — Quais sensores mais importam para tipo
- `results/ranking_sensores_subtipo_marca.csv` — Quais sensores mais importam para marca
- `results/figures/` — Gráficos de barras, matrizes de confusão, etc.

---

## Próximos Passos (MVP Hardware)

1. **Analisar o ranking de sensores** → decidir quais comprar
2. **Comprar**: ESP32 + AS7341 + DS18B20 + eletrodos PCB + piezo (~$20)
3. **Imprimir** presilha 3D com haste invasiva
4. **Coletar dados reais** → substituir dados sintéticos
5. **Converter modelo** para TensorFlow Lite → rodar no ESP32
6. **Desenvolver app** Bluetooth para exibir resultados
