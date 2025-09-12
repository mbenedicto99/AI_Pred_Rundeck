import pandas as pd
from prophet import Prophet
import holidays

# 1) Dados
df = pd.read_csv("data/series.csv", parse_dates=["timestamp"])
df = df.rename(columns={"timestamp":"ds", "value":"y"}).sort_values("ds")

# 2) Feriados BR
br_holidays = holidays.Brazil()
hol_df = pd.DataFrame({
    "ds": pd.to_datetime(list(br_holidays.keys())),
    "holiday": [str(v) for v in br_holidays.values()]
})

# 3) Modelo Prophet
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.5,
    interval_width=0.95
)
m = m.add_country_holidays(country_name='BR')
m.fit(df)

# 4) Previsão
future = m.make_future_dataframe(periods=30, freq="D")
fcst = m.predict(future)

# 5) Anomalias (resíduos no período observado)
obs = df.merge(fcst[["ds","yhat","yhat_lower","yhat_upper"]], on="ds", how="left")
obs["resid"] = obs["y"] - obs["yhat"]
sigma = obs["resid"].std(ddof=1)
obs["anomalia"] = (obs["resid"].abs() > 3*sigma) | ((obs["y"] < obs["yhat_lower"]) | (obs["y"] > obs["yhat_upper"]))
obs.to_csv("outputs/prophet_observado_com_anomalias.csv", index=False)
fcst.to_csv("outputs/prophet_previsao.csv", index=False)
