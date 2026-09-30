"""Objective 2 evaluation suite: corridor-vs-national breakdown for the
already-trained models, plus rolling-origin cross-validation (retraining
both models on 3 different chronological holdout windows) to confirm the
ConvLSTM-beats-baseline result isn't a lucky single split.

Corridor definition reused from the existing attribution pipeline
(Punjab-Haryana-Delhi-western-UP: lat 27-32N, lon 73-81E) - geographic only,
no season restriction here (detection/this comparison is year-round).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts")

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate_masked  # noqa: E402
import train_convlstm as clstm  # noqa: E402

CORRIDOR_LAT = (27, 32)
CORRIDOR_LON = (73, 81)


def corridor_mask(lats: np.ndarray, lons: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    lat_grid = lats[:, None] * np.ones((1, len(lons)))
    lon_grid = np.ones((len(lats), 1)) * lons[None, :]
    in_corridor = (lat_grid >= CORRIDOR_LAT[0]) & (lat_grid <= CORRIDOR_LAT[1]) & (lon_grid >= CORRIDOR_LON[0]) & (lon_grid <= CORRIDOR_LON[1])
    return base_mask & in_corridor


def regional_breakdown():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    mask, lats, lons = data["mask"], data["lats"], data["lons"]
    corr_mask = corridor_mask(lats, lons, mask)
    rest_mask = mask & ~corr_mask
    print(f"Corridor cells: {corr_mask.sum()}, rest-of-India cells: {rest_mask.sum()}")

    results = {}
    baseline_pred = np.load(os.path.join(ARTIFACTS_DIR, "statistical_baseline", "predictions.npz"))["preds"]
    convlstm_pred_data = np.load(os.path.join(ARTIFACTS_DIR, "convlstm", "predictions.npz"))
    convlstm_pred, convlstm_target = convlstm_pred_data["preds"], convlstm_pred_data["targets"]

    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json")) as f:
        manifest = json.load(f)
    split_idx = manifest["split_idx"]
    tensor = data["tensor"]
    baseline_target = tensor[split_idx:, 0]

    for name, mreg in [("corridor", corr_mask), ("rest_of_india", rest_mask), ("national_full", mask)]:
        results[name] = {
            "statistical_baseline": evaluate_masked(baseline_target, baseline_pred, mreg),
            "convlstm": evaluate_masked(convlstm_target, convlstm_pred, mreg),
        }
    return results


def rolling_origin_cv(n_folds: int = 3):
    """Retrain both models on n_folds different chronological holdout
    windows (progressively earlier test blocks), report mean+-std R2/RMSE -
    confirms the result isn't a lucky single split."""
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    tensor_raw, mask = data["tensor"], data["mask"]
    n_months = tensor_raw.shape[0]
    test_block = 12

    fold_results = {"statistical_baseline": [], "convlstm": []}
    for fold in range(n_folds):
        test_end = n_months - fold * test_block
        test_start = test_end - test_block
        train_end = test_start
        if train_end < clstm.LOOKBACK + 10:
            print(f"Fold {fold}: not enough training months left, stopping")
            break

        train_hcho = tensor_raw[:train_end, 0]
        test_hcho = tensor_raw[test_start:test_end, 0]
        climatology = np.nanmean(train_hcho, axis=0)
        baseline_preds = np.broadcast_to(climatology, test_hcho.shape)
        m_base = evaluate_masked(test_hcho, baseline_preds, mask)
        fold_results["statistical_baseline"].append(m_base)

        # ConvLSTM: retrain fresh on this fold's train window, evaluate on this fold's test window
        import torch
        H, W = mask.shape
        tensor_norm = tensor_raw.copy()
        means, stds = [], []
        for ch in range(4):
            train_vals = tensor_raw[:train_end, ch][np.broadcast_to(mask, (train_end, H, W))]
            m, s = np.nanmean(train_vals), np.nanstd(train_vals) + 1e-12
            means.append(m); stds.append(s)
            tensor_norm[:, ch] = (tensor_raw[:, ch] - m) / s
        tensor_norm = np.nan_to_num(tensor_norm, nan=0.0)

        train_targets = list(range(clstm.LOOKBACK, train_end))
        test_targets = list(range(test_start, test_end))
        X_train, y_train, _ = clstm.build_sequences(tensor_norm, mask, train_targets)
        X_test, y_test_norm, test_idx = clstm.build_sequences(tensor_norm, mask, test_targets)
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Fold {fold}: not enough sequences, skipping ConvLSTM")
            continue

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = clstm.ConvLSTMPredictor(4, clstm.HIDDEN_CHANNELS, H, W).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=clstm.LR)
        mask_t = torch.tensor(mask, device=device, dtype=torch.bool)
        Xt = torch.tensor(X_train, device=device)
        yt = torch.nan_to_num(torch.tensor(y_train, device=device), nan=0.0)
        for epoch in range(clstm.EPOCHS):
            model.train()
            opt.zero_grad()
            pred = model(Xt)
            loss = ((pred - yt) ** 2)[:, mask_t].mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_test_norm = model(torch.tensor(X_test, device=device)).cpu().numpy()
        pred_test = pred_test_norm * stds[0] + means[0]
        y_test_true = tensor_raw[test_idx, 0]
        m_conv = evaluate_masked(y_test_true, pred_test, mask)
        fold_results["convlstm"].append(m_conv)
        print(f"Fold {fold}: test months [{test_start}:{test_end}) baseline_r2={m_base['r2']:.4f} convlstm_r2={m_conv['r2']:.4f}")

    summary = {}
    for model_name, folds in fold_results.items():
        if not folds:
            continue
        r2s = [f["r2"] for f in folds]
        rmses = [f["rmse"] for f in folds]
        summary[model_name] = {
            "n_folds": len(folds), "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)), "fold_details": folds,
        }
    return summary


def main():
    print("=== Regional breakdown (corridor vs rest-of-India vs national) ===")
    breakdown = regional_breakdown()
    print(json.dumps(breakdown, indent=2))

    print("\n=== Rolling-origin CV (3 folds) ===")
    cv = rolling_origin_cv(n_folds=3)
    print(json.dumps(cv, indent=2))

    out = {"regional_breakdown": breakdown, "rolling_origin_cv": cv}
    with open(os.path.join(ARTIFACTS_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {os.path.join(ARTIFACTS_DIR, 'evaluation_summary.json')}")


if __name__ == "__main__":
    main()
