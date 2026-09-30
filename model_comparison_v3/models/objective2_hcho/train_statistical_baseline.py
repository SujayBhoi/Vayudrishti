"""Objective 2 statistical baseline: per-grid-cell historical climatology.

predict(cell, month) = mean(HCHO at that exact cell across TRAINING months
only). Confirmed definition: "the cleanest 'no learning' reference point,
letting us isolate whether ConvLSTM's spatio-temporal learning genuinely
adds value over just knowing a cell's typical historical HCHO level."

Full national grid (all real India cells in the 111x116 lattice), matching
ConvLSTM's scope exactly for a fair comparison.
"""

import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "statistical_baseline")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate_masked, timer, check_not_degenerate  # noqa: E402


def main():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    tensor, mask = data["tensor"], data["mask"]
    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json")) as f:
        manifest = json.load(f)
    split_idx = manifest["split_idx"]

    hcho = tensor[:, 0, :, :]  # (n_months, H, W)
    train_hcho, test_hcho = hcho[:split_idx], hcho[split_idx:]

    with timer() as t_train:
        climatology = np.nanmean(train_hcho, axis=0)  # (H, W) - per-cell mean over training months only

    with timer() as t_infer:
        preds = np.broadcast_to(climatology, test_hcho.shape).copy()

    check_not_degenerate(preds, np.broadcast_to(mask, preds.shape), "StatisticalBaseline(climatology)")

    metrics_test = evaluate_masked(test_hcho, preds, mask)
    train_preds = np.broadcast_to(climatology, train_hcho.shape).copy()
    metrics_train = evaluate_masked(train_hcho, train_preds, mask)

    metrics = {
        "train": metrics_train,
        "test": metrics_test,
        "train_seconds": t_train["seconds"],
        "inference_seconds_total": t_infer["seconds"],
        "inference_seconds_per_month": t_infer["seconds"] / len(test_hcho),
        "n_test_months": len(test_hcho),
    }
    print(json.dumps(metrics, indent=2))

    np.savez_compressed(os.path.join(ARTIFACTS_DIR, "predictions.npz"), preds=preds, climatology=climatology)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
