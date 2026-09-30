"""Linear Regression for Objective 2 - per-cell tabular model, same 6-month
lookback and train/test months as ConvLSTM, but flattened (no spatial
convolution, no recurrence) - each grid cell predicted from only its own
24 lag features (6 months x 4 channels). Directly answers "does
ConvLSTM's spatial+temporal structure actually help beyond a per-cell
linear fit of recent history."
"""

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "linear_regression")
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

    model = LinearRegression()
    with timer() as t_train:
        model.fit(X_train, y_train)
    with timer() as t_infer:
        preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    check_not_degenerate(preds_test, np.ones_like(preds_test, dtype=bool), "LinearRegression(obj2)")

    test_month_offset = {m: i for i, m in enumerate(sorted(set(m for m, r, c in meta_test)))}
    train_month_offset = {m: i for i, m in enumerate(sorted(set(m for m, r, c in meta_train)))}
    preds_test_grid = predictions_to_grid(preds_test, meta_test, n_months, H, W, test_month_offset)
    targets_test_grid = predictions_to_grid(y_test, meta_test, n_months, H, W, test_month_offset)
    preds_train_grid = predictions_to_grid(preds_train, meta_train, n_months, H, W, train_month_offset)
    targets_train_grid = predictions_to_grid(y_train, meta_train, n_months, H, W, train_month_offset)

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
    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
