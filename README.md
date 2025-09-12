# AI_Pred_Rundeck — Análise Preditiva (ARIMA & Prophet)

Projeto para prever comportamento de métricas operacionais e **detectar anomalias** (falhas, picos, quedas) com ênfase em **sazonalidade** e **datas especiais** (feriados, fechamentos, eventos). Integração pensada para **execução via Rundeck** (jobs diários/por hora).

---

## 🎯 Objetivos
- Prever valores futuros de séries temporais (curto e médio prazo).
- Considerar **sazonalidades diárias/semanais/mensais** e **feriados brasileiros**.
- **Detectar anomalias** por resíduos e intervalos de previsão.
- Operacionalizar via **Rundeck** com relatórios e alertas.

---

## 📦 Dados (entrada)
- **Granularidade**: minuto, hora ou dia (definir por métrica).
- **Colunas mínimas**:  
  - `timestamp` (UTC ou America/Sao_Paulo, consistente)  
  - `value` (numérico)  
  - `tag` (opcional: serviço, cluster, job, etc.)
- **Qualidade**: tratar *gaps*, duplicidades, outliers extremos.

---

## 🧪 Metodologia
- **ARIMA/SARIMA**: captura autocorrelação e sazonalidade explícita (p,d,q)(P,D,Q)[s].
- **Prophet**: tendência + sazonalidades (dia/semana/ano) + **feriados (BR)** + regressoras.
- **Validação temporal** (backtesting *walk-forward*), métricas: **RMSE, MAE, MAPE**.
- **Anomalias**: resíduos padronizados (> 3σ), violações de **intervalos de previsão**, ESD/quantis.

---

## 🧰 Estrutura
```bash
AI_Pred_Rundeck/
├── data
│   └── dados.csv
├── docs
│   └── exec.txt
├── jobs
│   └── rundeck.yaml
├── notebooks
│   └── AI_Pred_Rundeck_Pipeline.ipynb
├── outputs
├── README.md
├── run_pipeline.sh
└── src
    ├── anomalies.py
    ├── etl.py
    ├── features.py
    ├── train_arima.py
    └── train_prophet.py
```
---

## 🛠️ Pipeline (alto nível)

```mermaid
graph TD
  A["Coleta de dados (CSV/API/DB)"] --> B["ETL & Qualidade"]
  B --> C["Feature Engineering<br/>Lags, feriados (BR), sazonalidade"]
  C --> D{"Treino"}
  D --> E["ARIMA/SARIMA"]
  D --> F["Prophet"]
  E --> G["Backtesting & Métricas"]
  F --> G
  G --> H["Previsão"]
  H --> I["Detecção de Anomalias<br/>resíduos · intervalos · ESD"]
  I --> J["Alertas/Relatórios"]
  J --> K["Rundeck Job (agendado)"]
