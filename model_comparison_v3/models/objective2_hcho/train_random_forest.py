"""Random Forest for Objective 2 - same per-cell tabular setup as
train_linear_regression.py (24 lag features, same lookback/split), swapping
in a tree ensemble to check whether non-linear per-cell structure helps
over a plain linear fit, still without ConvLSTM's spatial/temporal
architecture.
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "random_forest")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate_masked, timer, check_not_degenerate  # noqa: E402
from tabular_features import build_tabular, predictions_to_grid, LOOKBACK  # noqa: E402


def main():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    tensor, mask = data["tensor"], data["mask"]
    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json")) as f:
        manifest = json.load(f)
    split_idx, n_months = manifest["split_idx"], manifest["n_months"]
    H, W = mask.shape

    train_targets = list(range(LOOKBACK, split_idx))
    test_targets = list(range(split_idx, n_months))
    X_train, y_train, meta_train = build_tabular(tensor, mask, train_targets)
    X_test, y_test, meta_test = build_tabular(tensor, mask, test_targets)
    print(f"Train rows: {X_train.shape}, Test rows: {X_test.shape}")

    model = RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
    with timer() as t_train:
        model.fit(X_train, y_train)
    with timer() as t_infer:
        preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    check_not_degenerate(preds_test, np.ones_like(preds_test, dtype=bool), "RandomForest(obj2)")

    test_month_offset = {m: i for i, m in enumerate(sorted(set(m for m, r, c in meta_test)))}
    train_month_offset = {m: i for i, m in enumerate(sorted(set(m for m, r, c in meta_train)))}
    preds_test_grid = predictions_to_grid(preds_test, meta_test, n_months, H, W, test_month_offset)
    targets_test_grid = predictions_to_grid(y_test, meta_test, n_months, H, W, test_month_offset)
    preds_train_grid = predictions_to_grid(preds_train, meta_train, n_months, H, W, train_month_offset)
    targets_train_grid = predictions_to_grid(y_train, meta_train, n_months, H, W, train_month_offset)

    feature_names = [f"{ch}_t-{LOOKBACK-k}" for k in range(LOOKBACK) for ch in ["hcho", "fire_count", "wind_u", "wind_v"]]
    import pandas as pd
    importances = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    metrics = {
        "train": evaluate_masked(targets_train_grid, preds_train_grid, mask),
        "test": evaluate_masked(targets_test_grid, preds_test_grid, mask),
        "train_seconds": t_train["seconds"],
        "inference_seconds_total": t_infer["seconds"],
        "inference_seconds_per_month": t_infer["seconds"] / len(test_month_offset),
        "n_features": X_train.shape[1],
        "lookback_months": LOOKBACK,
    }
    print(json.dumps(metrics, indent=2))

    np.savez_compressed(os.path.join(ARTIFACTS_DIR, "predictions.npz"), preds=preds_test_grid, targets=targets_test_grid)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    importances.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importances.csv"), index=False)
    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
