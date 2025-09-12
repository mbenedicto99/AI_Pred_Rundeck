#!/usr/bin/env python3
"""
Geração de features para modelos:
- Lags: 1, 7, 28
- Janelas móveis: média/std 7 dias
- Sazonalidade cíclica: dia da semana e dia do ano (seno/cosseno)
- Flags de fim de semana e (opcional) feriado BR
Salva em outputs/features.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def try_holidays_br(dates: pd.Series) -> pd.Series:
    try:
        import holidays
        br = holidays.Brazil()
        return dates.dt.date.map(lambda d: 1 if d in br else 0).astype(int)
    except Exception:
        return pd.Series(0, index=dates.index, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/clean.csv")
    ap.add_argument("--output", default="outputs/features.csv")
    ap.add_argument("--lags", default="1,7,28")
    ap.add_argument("--roll", type=int, default=7, help="Janela rolling (dias)")
    args = ap.parse_args()

    path_in = Path(args.input)
    path_out = Path(args.output)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path_in, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.rename(columns={"timestamp": "ds", "value": "y"})

    # Calendário/Sazonalidade
    df["dow"] = df["ds"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["ds"].dt.month
    doy = df["ds"].dt.dayofyear

    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["yoy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["yoy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Feriados BR (opcional)
    df["is_holiday_br"] = try_holidays_br(df["ds"])

    # Lags
    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    for L in lags:
        df[f"lag_{L}"] = df["y"].shift(L)

    # Rolling
    roll = args.roll
    df[f"roll_mean_{roll}"] = df["y"].rolling(roll, min_periods=1).mean()
    df[f"roll_std_{roll}"]  = df["y"].rolling(roll, min_periods=1).std().fillna(0)

    # Remove linhas iniciais com NaN em lags
    df = df.dropna().reset_index(drop=True)

    df.to_csv(path_out, index=False)
    print(f"✅ Salvo: {path_out} ({len(df)} linhas)")

if __name__ == "__main__":
    main()

