"""Persistence (naive) baseline for Objective 2 - predict(cell, month) =
that same cell's HCHO value the PREVIOUS month. The classic, standard
time-series baseline (distinct from the statistical/climatology baseline,
which predicts each cell's historical MEAN, not its last observed value).
No training needed.
"""

import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "persistence")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate_masked, timer, check_not_degenerate  # noqa: E402


def main():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    tensor, mask = data["tensor"], data["mask"]
    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json")) as f:
        manifest = json.load(f)
    split_idx = manifest["split_idx"]
    hcho = tensor[:, 0, :, :]

    with timer() as t_train:
        pass  # nothing to fit

    with timer() as t_infer:
        preds_test = hcho[split_idx - 1:-1]   # month i-1 predicts month i, for i in [split_idx, n_months)
    targets_test = hcho[split_idx:]

    preds_train = hcho[0:split_idx - 1]
    targets_train = hcho[1:split_idx]

    check_not_degenerate(preds_test, np.broadcast_to(mask, preds_test.shape), "Persistence")

    metrics = {
        "train": evaluate_masked(targets_train, preds_train, mask),
        "test": evaluate_masked(targets_test, preds_test, mask),
        "train_seconds": t_train["seconds"],
        "inference_seconds_total": t_infer["seconds"],
        "inference_seconds_per_month": t_infer["seconds"] / len(preds_test),
        "n_test_months": len(preds_test),
    }
    print(json.dumps(metrics, indent=2))

    np.savez_compressed(os.path.join(ARTIFACTS_DIR, "predictions.npz"), preds=preds_test, targets=targets_test)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
