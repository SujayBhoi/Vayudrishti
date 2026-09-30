"""ADDITIONAL, standalone deliverable - continuous, India-wide spatial
surface of AQI derived from satellite-predicted PM2.5, NOT scattered
station points.

Reuses existing infrastructure end to end, no new GEE extraction:
- Objective 1's already-downloaded national 0.25deg grid feature table
  (data_pipeline/load_objective1_grid.py: AOD x2 + 5 TROPOMI gases + ERA5,
  4447 grid points x 31 real 2025 dates - same GeoTIFFs already fetched for
  the existing spatial AQI map, models/objective1_aqi/artifacts/spatial_grid).
- features/merge_sources.derive_wind_met - the EXACT SAME function already
  used to build temp_c/rh_pct/wind_speed/wind_dir_deg for the two-stage
  model's own station-level training data (model_comparison_v3's
  build_clean101_daily_dataset.py imports this same function), so grid
  inference features are derived identically to training, no skew.
- The two-stage model's already-trained Stage 1 (gases+weather -> PM2.5,
  train_two_stage_pm25_aqi.py) - no retraining needed, same STAGE1_FEATURES.
- cpcb_process_v2.sub_index("PM2_5", ...) - the project's one CPCB
  breakpoint formula, applied directly to predicted PM2.5 (not through
  Stage 2's learned linear AQI mapping) per the explicit request to convert
  PM2.5 -> AQI "using our existing CPCB breakpoint formula".

Does not touch the primary CNN-LSTM/XGBoost pipeline, the existing spatial
AQI map (spatial_grid/spatial_aqi_predictions.csv), or Objective 2.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
TRIAL02_ROOT = os.path.dirname(V3_ROOT)
DATA_DIR = os.path.join(V3_ROOT, "data")

sys.path.insert(0, os.path.join(TRIAL02_ROOT, "data_pipeline"))
sys.path.insert(0, os.path.join(TRIAL02_ROOT, "features"))
from load_objective1_grid import build_grid_feature_table  # noqa: E402
from merge_sources import derive_wind_met  # noqa: E402
from cpcb_process_v2 import sub_index  # noqa: E402

TWO_STAGE_ARTIFACTS = os.path.join(V3_ROOT, "models", "objective1_aqi", "artifacts", "two_stage_pm25_aqi")
STAGE1_FEATURES = ["aod_047", "aod_055", "no2", "so2", "co", "o3", "hcho",
                    "temp_c", "rh_pct", "wind_speed", "wind_dir_deg", "month", "doy_sin", "doy_cos"]

OUT_PATH = os.path.join(DATA_DIR, "pm25_aqi_grid_predictions.csv")


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
    print("Loading Objective 1's national grid feature table (already-downloaded GeoTIFFs, no new extraction)...")
    df = build_grid_feature_table()
    print(f"  {len(df)} rows, {df['grid_id'].nunique()} grid points, {df['date'].nunique()} dates")

    df = derive_wind_met(df)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    print("Loading two-stage Stage 1 model (gases+weather -> PM2.5)...")
    stage1 = joblib.load(os.path.join(TWO_STAGE_ARTIFACTS, "stage1_gases_to_pm25.pkl"))
    imputer1 = joblib.load(os.path.join(TWO_STAGE_ARTIFACTS, "stage1_imputer.pkl"))

    missing = [c for c in STAGE1_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"grid feature table missing columns required by Stage 1: {missing}")

    X = imputer1.transform(df[STAGE1_FEATURES])
    pm25_pred = stage1.predict(X)
    print(f"Predicted PM2.5 across grid: min={pm25_pred.min():.1f}, mean={pm25_pred.mean():.1f}, max={pm25_pred.max():.1f}")

    print("Converting PM2.5 -> AQI via CPCB breakpoint formula (sub_index)...")
    aqi_from_pm25 = np.array([sub_index("PM2_5", v) for v in pm25_pred])

    # Keep the raw per-gas/weather grid values too (not just the final PM2.5/AQI
    # output) so downstream views can render each satellite input independently,
    # not just the derived prediction.
    raw_gas_cols = ["aod_047", "aod_055", "no2", "so2", "co", "o3", "hcho", "temp_c", "rh_pct", "wind_speed"]
    out = df[["grid_id", "lat", "lon", "date"] + raw_gas_cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["predicted_pm25"] = pm25_pred
    out["predicted_aqi"] = aqi_from_pm25
    out["aqi_category"] = out["predicted_aqi"].apply(aqi_category)

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows -> {OUT_PATH}")
    print(out.groupby("date")["predicted_aqi"].mean().round(1))


if __name__ == "__main__":
    main()
