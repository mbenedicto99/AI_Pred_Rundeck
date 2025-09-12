import pandas as pd
import pmdarima as pm

df = pd.read_csv("data/series.csv", parse_dates=["timestamp"]).sort_values("timestamp")
df = df.set_index("timestamp")["value"].asfreq("D")  # ajuste a frequência conforme seu dado

# Auto-ARIMA com sazonalidade semanal (m=7) — ajuste conforme frequência
model = pm.auto_arima(
    df, seasonal=True, m=7,
    error_action="ignore", suppress_warnings=True, stepwise=True,
    information_criterion="aic"
)
n_forecast = 30
fcst = model.predict(n_periods=n_forecast)
pd.DataFrame({
    "ds": pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq="D"),
    "yhat": fcst
}).to_csv("outputs/arima_previsao.csv", index=False)

