#!/usr/bin/env python3
"""
Detecção de anomalias:
- Junta histórico (clean.csv) com forecast (Prophet/ARIMA)
- Marca anomalia se y estiver fora de [yhat_lower, yhat_upper]
- Se limites não existirem, usa z-score rolling (janela=30)
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def detect_by_pi(obs: pd.DataFrame) -> pd.DataFrame:
    cond_pi = (obs["y"] < obs["yhat_lower"]) | (obs["y"] > obs["yhat_upper"])
    obs["anomaly"] = cond_pi.astype(int)
    obs["residual"] = obs["y"] - obs["yhat"]
    return obs

def detect_by_zscore(obs: pd.DataFrame, window: int = 30, z: float = 3.0) -> pd.DataFrame:
    s = obs["y"]
    mu = s.rolling(window, min_periods=1).mean()
    sd = s.rolling(window, min_periods=1).std().fillna(0.0)
    zscore = (s - mu) / (sd.replace(0, np.nan))
    obs["yhat"] = mu
    obs["yhat_lower"] = mu - z * sd
    obs["yhat_upper"] = mu + z * sd
    obs["residual"] = s - mu
    obs["anomaly"] = (zscore.abs() > z).astype(int)
    return obs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="outputs/clean.csv")
    ap.add_argument("--forecast", required=True, help="CSV de forecast (prophet/arima)")
    ap.add_argument("--output", default="outputs/anomalies.csv")
    args = ap.parse_args()

    hist = pd.read_csv(args.history, parse_dates=["timestamp"])
    hist = hist.rename(columns={"timestamp": "ds", "value": "y"})

    fcst = pd.read_csv(args.forecast, parse_dates=["ds"])

    # Usa apenas período observado (interseção) para checagem de anomalias em histórico
    obs = hist.merge(fcst, on="ds", how="left")

    if {"yhat_lower","yhat_upper"}.issubset(obs.columns):
        result = detect_by_pi(obs)
    else:
        result = detect_by_zscore(obs)

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(outp, index=False)
    print(f"✅ Anomalias salvas em {outp} (total={result['anomaly'].sum()} marcadas)")

if __name__ == "__main__":
    main()

