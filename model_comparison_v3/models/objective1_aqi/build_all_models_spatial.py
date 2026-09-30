"""Builds station-level spatial + testing-data tables for ALL 5 Objective 1
models (Linear Regression, Random Forest, XGBoost, ARIMA, CNN-LSTM), for the
Streamlit app. All 5 already have test_predictions.csv (station_id, date,
aqi, pred) saved from training - this just joins station lat/lon/city and
produces:
  1. {model}_spatial_latest.csv - one row per station, most recent test-date
     prediction, for the map (heatmap + markers).
  2. all_models_test_predictions.csv - every model's full test-set row,
     concatenated with a `model` column, for the testing-data table/metrics.

ARIMA note: it's deliberately univariate (station's own AQI history only,
no satellite/weather features - see artifacts/arima/metrics.json note
field), so its "spatial pattern" reflects each station's own historical
dynamics, not any spatial covariate. Still a valid station-located
prediction though, so it's included the same way as the other 4 - not
excluded, just captioned differently in the UI.
"""

from __future__ import annotations

import os

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts")
OUT_DIR = os.path.join(DATA_DIR, "all_models_spatial")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["linear_regression", "random_forest", "xgboost", "arima", "cnn_lstm"]
MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "arima": "ARIMA",
    "cnn_lstm": "CNN-LSTM",
}


def aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def main():
    stations = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"),
                            usecols=["station_id", "lat", "lon", "city"]).drop_duplicates("station_id")

    all_rows = []
    for model in MODELS:
        path = os.path.join(ARTIFACTS_DIR, model, "test_predictions.csv")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.merge(stations, on="station_id", how="left")
        missing = df["lat"].isna().sum()
        if missing:
            print(f"[{model}] warning: {missing} rows with no station lat/lon match")
        df["model"] = MODEL_LABELS[model]
        all_rows.append(df)

        latest = df.sort_values("date").groupby("station_id").tail(1).copy()
        latest["aqi_category"] = latest["pred"].apply(aqi_category)
        latest = latest.rename(columns={"pred": "predicted_aqi", "aqi": "actual_aqi"})
        latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
        out_path = os.path.join(OUT_DIR, f"{model}_spatial_latest.csv")
        latest[["station_id", "city", "lat", "lon", "date", "predicted_aqi", "actual_aqi", "aqi_category"]].to_csv(out_path, index=False)
        print(f"[{model}] {len(latest)} stations -> {out_path} "
              f"(predicted_aqi range {latest['predicted_aqi'].min():.1f}-{latest['predicted_aqi'].max():.1f})")

    combined = pd.concat(all_rows, ignore_index=True)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined_path = os.path.join(OUT_DIR, "all_models_test_predictions.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined testing-data table: {len(combined)} rows -> {combined_path}")


if __name__ == "__main__":
    main()
