"""Random Forest on the clean101 daily dataset - same feature set and split
as Linear Regression/XGBoost, median-imputed (RF, unlike XGBoost, needs
imputation - same as the existing train_extended_xgb.py's RF baseline)."""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "random_forest")
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

    model = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)
    with timer() as t_train:
        model.fit(X_train, y_train)
    with timer() as t_infer:
        preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    check_not_degenerate(preds_test, "RandomForest")

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

    importances = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "random_forest.pkl"))
    joblib.dump(imputer, os.path.join(ARTIFACTS_DIR, "imputer.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    per_station.to_csv(os.path.join(ARTIFACTS_DIR, "per_station_metrics.csv"), index=False)
    test_out.to_csv(os.path.join(ARTIFACTS_DIR, "test_predictions.csv"), index=False)
    importances.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importances.csv"), index=False)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
