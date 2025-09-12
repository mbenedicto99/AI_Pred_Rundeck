#!/usr/bin/env python3
"""
Treino/forecast com Prophet:
- Lê outputs/clean.csv (ds,y)
- Adiciona feriados BR
- Gera previsão para horizonte (--horizon) e salva CSV
"""
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/clean.csv")
    ap.add_argument("--output", default="outputs/prophet_forecast.csv")
    ap.add_argument("--horizon", type=int, default=30, help="Períodos futuros (ex.: dias)")
    ap.add_argument("--freq", default="D", help="Frequência (D/H)")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.rename(columns={"timestamp": "ds", "value": "y"})

    from prophet import Prophet
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,
        interval_width=0.95
    )
    # Feriados BR
    m.add_country_holidays(country_name='BR')

    m.fit(df)

    future = m.make_future_dataframe(periods=args.horizon, freq=args.freq)
    fcst = m.predict(future)

    # Salva somente colunas úteis
    cols = ["ds","yhat","yhat_lower","yhat_upper"]
    fcst[cols].to_csv(outp, index=False)
    print(f"✅ Forecast salvo em {outp} ({len(fcst)} linhas)")

if __name__ == "__main__":
    main()

