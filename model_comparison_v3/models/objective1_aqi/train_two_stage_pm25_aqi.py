"""ADDITIONAL, standalone reference model - NOT the primary approach, does
NOT replace CNN-LSTM/XGBoost/etc (section 8's direct gases-to-AQI models
stay primary). This is a two-stage alternative, explicitly requested for
comparison:

  Stage 1: satellite gases (NO2, SO2, CO, O3, AOD) + weather -> predict PM2.5
  Stage 2: predicted PM2.5 -> predict AQI

Both stages trained on 2019-2024, evaluated on the 2025 test period - same
split, same station set, same eval_utils metrics as every other model in
this project, for direct comparability.

Stage 2 deliberately trains on Stage 1's OWN predictions for the training
period (not the real PM2.5), not the real training PM2.5 - this matches
what actually happens at inference/deployment time (real PM2.5 is never
known when this pipeline runs for real, only Stage 1's estimate of it), so
Stage 2 learns to correct for Stage 1's typical error pattern rather than
assuming a perfect PM2.5 input it will never actually get.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "two_stage_pm25_aqi")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate, per_station_metrics, timer, check_not_degenerate  # noqa: E402

# physically-grounded "gases + weather" inputs for Stage 1 - deliberately
# excludes station_aqi_target_enc (an AQI-specific leakage-prone feature)
# and lag/rolling features (keeps Stage 1 a clean, simple, same-day
# gases->PM2.5 mapping, matching the "simple two-stage reference model"
# framing this was requested as)
STAGE1_FEATURES = ["aod_047", "aod_055", "no2", "so2", "co", "o3", "hcho",
                    "temp_c", "rh_pct", "wind_speed", "wind_dir_deg", "month", "doy_sin", "doy_cos"]


def main():
    main_df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    pm25 = pd.read_csv(os.path.join(DATA_DIR, "pm25_reference_daily.csv"), parse_dates=["date"])

    df = main_df.merge(pm25, on=["station_id", "date"], how="inner")
    print(f"Merged (gases/weather + real PM2.5 + real AQI): {df.shape}")
    n_before = len(df)
    df = df[df["pm25"].notna()]
    print(f"Dropped {n_before - len(df)} rows with missing PM2.5 target -> {len(df)}")

    train = df[df.split == "train"]
    test = df[df.split == "test"]

    # ---- Stage 1: gases + weather -> PM2.5 ----
    imputer1 = SimpleImputer(strategy="median")
    X1_train = imputer1.fit_transform(train[STAGE1_FEATURES])
    X1_test = imputer1.transform(test[STAGE1_FEATURES])
    y1_train, y1_test = train["pm25"].values, test["pm25"].values

    stage1 = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    with timer() as t1_train:
        stage1.fit(X1_train, y1_train)
    with timer() as t1_infer:
        pm25_pred_test = stage1.predict(X1_test)
    pm25_pred_train = stage1.predict(X1_train)  # Stage 2 trains on THESE, not real train PM2.5 - see module docstring

    check_not_degenerate(pm25_pred_test, "Stage1(PM2.5)")
    stage1_metrics = {
        "train": evaluate(y1_train, pm25_pred_train),
        "test": evaluate(y1_test, pm25_pred_test),
        "train_seconds": t1_train["seconds"],
        "inference_seconds_total": t1_infer["seconds"],
        "n_features": len(STAGE1_FEATURES),
    }
    print("Stage 1 (gases+weather -> PM2.5):")
    print(json.dumps(stage1_metrics, indent=2))

    # ---- Stage 2: predicted PM2.5 -> AQI ----
    X2_train = pm25_pred_train.reshape(-1, 1)
    X2_test = pm25_pred_test.reshape(-1, 1)
    y2_train, y2_test = train["aqi"].values, test["aqi"].values

    stage2 = LinearRegression()
    with timer() as t2_train:
        stage2.fit(X2_train, y2_train)
    with timer() as t2_infer:
        aqi_pred_test = stage2.predict(X2_test)
    aqi_pred_train = stage2.predict(X2_train)

    check_not_degenerate(aqi_pred_test, "Stage2(AQI)")
    stage2_metrics = {
        "train": evaluate(y2_train, aqi_pred_train),
        "test": evaluate(y2_test, aqi_pred_test),
        "train_seconds": t2_train["seconds"],
        "inference_seconds_total": t2_infer["seconds"],
    }
    print("\nStage 2 (predicted PM2.5 -> AQI):")
    print(json.dumps(stage2_metrics, indent=2))

    # end-to-end (what a real deployment would see: only gases+weather in, AQI out)
    e2e_seconds = t1_train["seconds"] + t2_train["seconds"]
    e2e_infer_seconds = t1_infer["seconds"] + t2_infer["seconds"]
    e2e_metrics = {
        "test": evaluate(y2_test, aqi_pred_test),  # same as stage2 test - this IS the end-to-end test prediction
        "train_seconds_total": e2e_seconds,
        "inference_seconds_total": e2e_infer_seconds,
        "n": len(y2_test),
    }
    print("\nEnd-to-end (gases+weather -> PM2.5 -> AQI), test set:")
    print(json.dumps(e2e_metrics, indent=2))

    test_out = test[["station_id", "date", "aqi"]].copy()
    test_out["pred"] = aqi_pred_test
    test_out["pm25_actual"] = y1_test
    test_out["pm25_pred"] = pm25_pred_test
    per_station = per_station_metrics(test_out, "aqi", "pred")

    joblib.dump(stage1, os.path.join(ARTIFACTS_DIR, "stage1_gases_to_pm25.pkl"))
    joblib.dump(imputer1, os.path.join(ARTIFACTS_DIR, "stage1_imputer.pkl"))
    joblib.dump(stage2, os.path.join(ARTIFACTS_DIR, "stage2_pm25_to_aqi.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump({"stage1_pm25": stage1_metrics, "stage2_aqi": stage2_metrics, "end_to_end": e2e_metrics,
                   "stage1_features": STAGE1_FEATURES, "note": "ADDITIONAL reference model, not primary - see module docstring"}, f, indent=2)
    per_station.to_csv(os.path.join(ARTIFACTS_DIR, "per_station_metrics.csv"), index=False)
    test_out.to_csv(os.path.join(ARTIFACTS_DIR, "test_predictions.csv"), index=False)
    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
