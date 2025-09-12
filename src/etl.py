#!/usr/bin/env python3
"""
ETL básico para séries temporais.
- Lê data/dados.csv (ou caminho informado)
- Normaliza colunas, ordena e remove duplicidades
- Ajusta frequência (ex.: D/H), interpola e trata outliers (MAD clipping opcional)
- Salva em outputs/clean.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def mad_clip(s: pd.Series, k: float = 3.5) -> pd.Series:
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad == 0 or np.isnan(mad):
        return s
    lower = med - k * 1.4826 * mad
    upper = med + k * 1.4826 * mad
    return s.clip(lower, upper)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/dados.csv")
    ap.add_argument("--output", default="outputs/clean.csv")
    ap.add_argument("--freq", default="D", help="Frequência destino (ex.: D, H)")
    ap.add_argument("--cap_outliers", action="store_true", help="Ativa MAD clipping")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, parse_dates=["timestamp"])
    # Normaliza nomes
    cols = {c.lower(): c for c in df.columns}
    if "ds" in df.columns and "y" in df.columns:
        df = df.rename(columns={"ds": "timestamp", "y": "value"})
    elif "timestamp" not in cols or "value" not in cols:
        raise ValueError("Esperado colunas 'timestamp' e 'value' ou 'ds' e 'y'.")

    df = df[["timestamp", "value"]].dropna().sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"])

    # Resample para frequência alvo
    s = df.set_index("timestamp")["value"].asfreq(args.freq)
    # Interpola e preenche bordas
    s = s.interpolate("time").ffill().bfill()

    if args.cap_outliers:
        s = mad_clip(s)

    clean = s.reset_index().rename(columns={"index": "timestamp"})
    clean.to_csv(outp, index=False)
    print(f"✅ Salvo: {outp} ({len(clean)} linhas)")

if __name__ == "__main__":
    main()

