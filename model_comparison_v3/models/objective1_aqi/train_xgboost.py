"""XGBoost on the clean101 daily dataset - same RandomizedSearchCV/time-
based-CV pattern as the existing train_extended_xgb.py, retargeted at the
daily, overpass-matched, 101-station dataset + fixed 2019-2024/2025 split.
XGBoost handles NaN natively - no imputation, unlike RF/LinearRegression."""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "xgboost")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(V3_ROOT, "data_pipeline"))
from build_clean101_daily_dataset import feature_cols  # noqa: E402

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate, per_station_metrics, timer, check_not_degenerate, time_based_cv_folds  # noqa: E402


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    cols = feature_cols(df)
    train = df[df.split == "train"]
    test = df[df.split == "test"]

    X_train, X_test = train[cols], test[cols]
    y_train, y_test = train["aqi"].values, test["aqi"].values

    # recency-weighted sample weights (half-life 365 days), same rationale as v1/v2
    days_ago = (train["date"].max() - train["date"]).dt.days
    sample_weight = 0.5 ** (days_ago / 365)

    param_dist = {
        "n_estimators": [300, 500, 700, 900, 1200],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "min_child_weight": [1, 3, 5, 8],
        "reg_alpha": [0, 0.1, 1, 5],
        "reg_lambda": [1, 5, 10, 20],
    }
    base_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    time_folds = time_based_cv_folds(train["date"], n_folds=4)
    search = RandomizedSearchCV(
        base_model, param_dist, n_iter=60, scoring="neg_root_mean_squared_error",
        cv=time_folds, random_state=42, n_jobs=-1, verbose=1,
    )
    with timer() as t_train:
        search.fit(X_train, y_train, sample_weight=sample_weight)
    model = search.best_estimator_

    with timer() as t_infer:
        preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    check_not_degenerate(preds_test, "XGBoost")

    metrics = {
        "train": evaluate(y_train, preds_train),
        "test": evaluate(y_test, preds_test),
        "train_seconds": t_train["seconds"],
        "inference_seconds_total": t_infer["seconds"],
        "inference_seconds_per_row": t_infer["seconds"] / len(test),
        "n_features": len(cols),
        "best_params": search.best_params_,
    }
    print(json.dumps(metrics, indent=2, default=str))

    test_out = test[["station_id", "date", "aqi"]].copy()
    test_out["pred"] = preds_test
    per_station = per_station_metrics(test_out, "aqi", "pred")

    importances = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "xgboost.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    per_station.to_csv(os.path.join(ARTIFACTS_DIR, "per_station_metrics.csv"), index=False)
    test_out.to_csv(os.path.join(ARTIFACTS_DIR, "test_predictions.csv"), index=False)
    importances.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importances.csv"), index=False)
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(cols, f, indent=2)

    # SHAP values, for Phase 2's feature-importance comparison
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.sample(min(2000, len(X_test)), random_state=42))
    np.save(os.path.join(ARTIFACTS_DIR, "shap_values_sample.npy"), shap_values)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
