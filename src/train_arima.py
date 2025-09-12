#!/usr/bin/env python3
"""
Treino/forecast com Auto-ARIMA (pmdarima):
- Lê outputs/clean.csv
- Detecta sazonalidade semanal (m=7) para dados diários; para horário usa m=24
- Salva previsão com intervalo de confiança
"""
import argparse
import pandas as pd
from pathlib import Path
import pmdarima as pm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/clean.csv")
    ap.add_argument("--output", default="outputs/arima_forecast.csv")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--freq", default="D", help="D ou H")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, parse_dates=["timestamp"]).sort_values("timestamp")
    s = df.set_index("timestamp")["value"].asfreq(args.freq)

    m = 7 if args.freq.upper() == "D" else 24
    model = pm.auto_arima(
        s, seasonal=True, m=m,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", information_criterion="aic"
    )

    yhat, conf = model.predict(n_periods=args.horizon, return_conf_int=True, alpha=0.05)
    idx = pd.date_range(s.index[-1] + pd.tseries.frequencies.to_offset(args.freq),
                        periods=args.horizon, freq=args.freq)

    out = pd.DataFrame({
        "ds": idx,
        "yhat": yhat,
        "yhat_lower": conf[:,0],
        "yhat_upper": conf[:,1],
    })
    out.to_csv(outp, index=False)
    print(f"✅ Forecast salvo em {outp} ({len(out)} linhas)")

if __name__ == "__main__":
    main()

