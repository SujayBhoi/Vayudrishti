"""Linear Regression on the clean101 daily dataset - the classic linear
statistical baseline in the 5-model comparison. Same feature set as
RF/XGBoost (base + daily lag/rolling + station_aqi_target_enc), median-
imputed (LinearRegression can't handle NaN, same imputer pattern as the
existing RF baseline in train_extended_xgb.py).
"""

import json
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "linear_regression")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(V3_ROOT, "data_pipeline"))
from build_clean101_daily_dataset import feature_cols  # noqa: E402

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate, per_station_metrics, timer, check_not_degenerate  # noqa: E402


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    cols = feature_cols(df)
    train = df[df.split == "train"]
    test = df[df.split == "test"]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train[cols])
    X_test = imputer.transform(test[cols])
    y_train, y_test = train["aqi"].values, test["aqi"].values

    model = LinearRegression()
    with timer() as t_train:
        model.fit(X_train, y_train)
    with timer() as t_infer:
        preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    check_not_degenerate(preds_test, "LinearRegression")

    metrics = {
        "train": evaluate(y_train, preds_train),
        "test": evaluate(y_test, preds_test),
        "train_seconds": t_train["seconds"],
        "inference_seconds_total": t_infer["seconds"],
        "inference_seconds_per_row": t_infer["seconds"] / len(test),
        "n_features": len(cols),
    }
    print(json.dumps(metrics, indent=2))

    test_out = test[["station_id", "date", "aqi"]].copy()
    test_out["pred"] = preds_test
    per_station = per_station_metrics(test_out, "aqi", "pred")

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "linear_regression.pkl"))
    joblib.dump(imputer, os.path.join(ARTIFACTS_DIR, "imputer.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    per_station.to_csv(os.path.join(ARTIFACTS_DIR, "per_station_metrics.csv"), index=False)
    test_out.to_csv(os.path.join(ARTIFACTS_DIR, "test_predictions.csv"), index=False)
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(cols, f, indent=2)

    # standardized coefficients, for the feature-importance comparison (Phase 2)
    import numpy as np
    X_train_std = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-9)
    std_model = LinearRegression().fit(X_train_std, y_train)
    coef_df = pd.DataFrame({"feature": cols, "standardized_coef": std_model.coef_}).sort_values("standardized_coef", key=abs, ascending=False)
    coef_df.to_csv(os.path.join(ARTIFACTS_DIR, "standardized_coefficients.csv"), index=False)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
