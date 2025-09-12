#!/usr/bin/env bash
###############################################################################
# run_pipeline.sh
#
# O QUE ESTE SCRIPT FAZ:
# 1) Prepara o ambiente Python (cria uma VM isolada chamada virtualenv)
# 2) Garante que as bibliotecas necessárias estão instaladas
# 3) Confere se existe um arquivo de dados (data/dados.csv). Se não existir,
#    cria um arquivo de exemplo automaticamente.
# 4) Executa, na ordem, os passos do pipeline de previsão:
#    - ETL (limpeza/organização dos dados)
#    - Features (criação de colunas auxiliares)
#    - Treino/Previsão com Prophet
#    - Treino/Previsão com ARIMA
#    - Detecção de anomalias (o que foge do padrão)
# 5) Mostra onde ficam os resultados (pasta outputs/)
#
# TERMOS SIMPLES:
# - "ETL": etapa que lê os dados brutos, ajusta datas, preenche falhas e salva limpo.
# - "Features": novas colunas que ajudam o modelo a entender padrões (ex.: dia da semana).
# - "Modelo Prophet" e "Modelo ARIMA": duas formas diferentes de prever o futuro
#   usando o histórico.
# - "Anomalias": pontos fora do comum (muito acima/abaixo do esperado).
#
###############################################################################

set -euo pipefail

############################
# CONFIGURAÇÕES DO USUÁRIO #
############################
DIR_BASE="/home/mbenedicto/Documents/CanopusAI/AI_Pred_Rundeck"
PYTHON_BIN="${/:-python3}"      # binário do Python
VENV_DIR="${/:-.venv}"            # pasta da "caixinha" Python
DATA_DIR="${data/:-data}"             # onde ficam os dados de entrada
OUT_DIR="${output/:-outputs}"            # onde ficarão os resultados
SRC_DIR="${src/:-src}"                # onde ficam os códigos .py
DATA_FILE="${data/:-data/dados.csv}"

# Horizonte de previsão (quantos períodos no futuro vamos prever)
HORIZON="${HORIZON:-30}"
# Frequência dos dados (D=diário, H=horário). Nosso exemplo usa diário.
FREQ="${FREQ:-D}"

# Se quiser instalar pacotes do sistema automaticamente (útil para Prophet), deixe "1".
INSTALL_SYS_DEPS="${INSTALL_SYS_DEPS:-1}"

#########################################
# FUNÇÕES SIMPLES PARA MENSAGENS/ERROS  #
#########################################
log()   { echo -e "🟦  $*"; }
ok()    { echo -e "✅  $*"; }
warn()  { echo -e "⚠️  $*"; }
error() { echo -e "❌  $*" >&2; }

trap 'error "Ocorreu um erro. Veja as mensagens acima e tente novamente."' ERR

#########################################
# 0) ESTRUTURA DE PASTAS DO PROJETO     #
#########################################
mkdir -p "$DATA_DIR" "$OUT_DIR" "$SRC_DIR"

#########################################
# 1) DEPENDÊNCIAS DO SISTEMA (Ubuntu)   #
#########################################
# Por que isso é necessário?
# - O Prophet compila um componente chamado CmdStan. Para compilar é bom ter:
#   gcc, g++, make, etc. (pacote build-essential) e o módulo de venv do Python.
if [[ "$INSTALL_SYS_DEPS" == "1" ]]; then
  log "Conferindo pacotes básicos do sistema (pode pedir senha do sudo)..."
  if ! command -v gcc >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y build-essential
  fi
  if ! dpkg -s python3-venv >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y python3-venv
  fi
  ok "Dependências de sistema OK."
else
  warn "Pulando instalação de pacotes do sistema (INSTALL_SYS_DEPS=0)."
fi

#########################################
# 2) AMBIENTE PYTHON ISOLADO (venv)     #
#########################################
# Por que usar venv?
# - Para não "bagunçar" seu Python do sistema. Tudo fica organizado dentro de .venv
if [[ ! -d "$VENV_DIR" ]]; then
  log "Criando ambiente Python isolado em ${VENV_DIR} ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  ok "Virtualenv criado."
fi

# Ativa a venv (a partir daqui, 'python' e 'pip' usam a caixinha isolada)
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Atualizando pip e instalando bibliotecas Python..."
python -m pip install --upgrade pip
# Bibliotecas:
# - pandas/numpy: manipulação numérica e de dados
# - pmdarima: Auto-ARIMA (modelo de séries temporais)
# - prophet: outro modelo de séries temporais (considera sazonalidade e feriados)
# - holidays[BR]: feriados do Brasil (ajuda o Prophet)
python -m pip install pandas numpy pmdarima prophet "holidays[BR]"
ok "Bibliotecas Python instaladas."

# (Opcional) Pré-instala o CmdStan para o Prophet evitar compilar na 1ª execução
log "Preparando CmdStan (pode demorar na primeira vez)..."
python - <<'PY'
try:
    import cmdstanpy as c
    c.install_cmdstan()
except Exception as e:
    print("Aviso: não foi possível instalar CmdStan agora. O Prophet tentará na primeira execução.", e)
PY
ok "Passo Prophet/CmdStan finalizado (ou será feito sob demanda)."

#########################################
# 3) DADOS DE ENTRADA                   #
#########################################
# Se você já tem dados reais:
# - Coloque seu arquivo CSV em data/dados.csv com duas colunas:
#   timestamp (data) e value (número).
# Se NÃO tiver, geramos um arquivo de exemplo (sintético) automaticamente.
if [[ ! -f "$DATA_FILE" ]]; then
  warn "Não encontrei ${DATA_FILE}. Vou gerar um arquivo de exemplo com dados sintéticos."
  python - <<PY
import numpy as np, pandas as pd
rng = pd.date_range("2024-01-01", "2025-08-31", freq="D")
n = len(rng); rs = np.random.RandomState(42)
trend = np.linspace(100, 140, n)
weekly = 10*np.sin(2*np.pi*(rng.dayofweek.values/7.0))
yearly = 15*np.sin(2*np.pi*(rng.dayofyear.values/365.25))
noise = rs.normal(0,5,n)
value = trend + weekly + yearly + noise
# alguns eventos e anomalias
anom_idx = rs.choice(n, size=10, replace=False)
value[anom_idx] += rs.choice([-40,40], size=10)
df = pd.DataFrame({"timestamp": rng, "value": np.maximum(value,0).round(2)})
df.to_csv("${DATA_FILE}", index=False)
print(f"Gerado exemplo em ${DATA_FILE} com {len(df)} linhas.")
PY
else
  ok "Arquivo de dados encontrado: ${DATA_FILE}"
fi

#########################################
# 4) CONFERINDO OS ARQUIVOS .py         #
#########################################
# Estes arquivos implementam as etapas do pipeline.
for f in etl.py features.py train_prophet.py train_arima.py anomalies.py; do
  if [[ ! -f "${SRC_DIR}/${f}" ]]; then
    error "Arquivo ausente: ${SRC_DIR}/${f}. Crie/baixe o código fonte antes de continuar."
    exit 1
  fi
done
ok "Arquivos de código encontrados em ${SRC_DIR}/."

#########################################
# 5) EXECUTANDO O PIPELINE              #
#########################################

# 5.1) ETL: limpa, ordena por data, preenche falhas e (opcional) limita outliers
log "Executando ETL (limpeza e organização dos dados)..."
python "${SRC_DIR}/etl.py" \
  --input "${DATA_FILE}" \
  --output "${OUT_DIR}/clean.csv" \
  --freq "${FREQ}" \
  --cap_outliers

# 5.2) FEATURES: cria colunas auxiliares (lags, médias móveis, dia da semana, etc.)
log "Gerando features (colunas auxiliares para os modelos)..."
python "${SRC_DIR}/features.py" \
  --input "${OUT_DIR}/clean.csv" \
  --output "${OUT_DIR}/features.csv" \
  --lags "1,7,28" \
  --roll 7

# 5.3) PROPHET: previsão usando sazonalidades e feriados do Brasil
log "Treinando e prevendo com Prophet..."
python "${SRC_DIR}/train_prophet.py" \
  --input "${OUT_DIR}/clean.csv" \
  --output "${OUT_DIR}/prophet_forecast.csv" \
  --horizon "${HORIZON}" \
  --freq "${FREQ}"

# 5.4) ARIMA: previsão usando Auto-ARIMA
log "Treinando e prevendo com ARIMA..."
python "${SRC_DIR}/train_arima.py" \
  --input "${OUT_DIR}/clean.csv" \
  --output "${OUT_DIR}/arima_forecast.csv" \
  --horizon "${HORIZON}" \
  --freq "${FREQ}"

# 5.5) ANOMALIAS: compara histórico vs previsão e marca pontos fora do previsto
log "Detectando anomalias (com base no Prophet)..."
python "${SRC_DIR}/anomalies.py" \
  --history "${OUT_DIR}/clean.csv" \
  --forecast "${OUT_DIR}/prophet_forecast.csv" \
  --output "${OUT_DIR}/anomalies_prophet.csv"

log "Detectando anomalias (com base no ARIMA)..."
python "${SRC_DIR}/anomalies.py" \
  --history "${OUT_DIR}/clean.csv" \
  --forecast "${OUT_DIR}/arima_forecast.csv" \
  --output "${OUT_DIR}/anomalies_arima.csv"

#########################################
# 6) RESUMO FINAL                       #
#########################################
ok "Pipeline concluído!"
echo
echo "🗂  Onde encontrar os resultados:"
echo "    - ${OUT_DIR}/clean.csv               -> dados limpos e preparados"
echo "    - ${OUT_DIR}/features.csv            -> dados com colunas auxiliares"
echo "    - ${OUT_DIR}/prophet_forecast.csv    -> previsão do Prophet"
echo "    - ${OUT_DIR}/arima_forecast.csv      -> previsão do ARIMA"
echo "    - ${OUT_DIR}/anomalies_prophet.csv   -> anomalias segundo Prophet"
echo "    - ${OUT_DIR}/anomalies_arima.csv     -> anomalias segundo ARIMA"
echo
echo "ℹ️  Dica: abra os CSVs no Excel/LibreOffice ou em um notebook Jupyter para ver gráficos."
echo "         Você pode agendar este script no Rundeck como um Job diário."

