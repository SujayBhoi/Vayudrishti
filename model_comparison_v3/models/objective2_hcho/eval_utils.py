"""Shared evaluation utilities for the Objective 2 statistical baseline vs
ConvLSTM comparison - same metric conventions as Objective 1's eval_utils.py
(RMSE/MAE/R2/MAPE), but operating on masked grid arrays instead of tabular
rows."""

import time
from contextlib import contextmanager

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAPE_EPSILON = 1e-6


def evaluate_masked(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict:
    """y_true/y_pred: (n_months, H, W) or (H, W). mask: (H, W) bool, True = real India cell."""
    valid = mask & ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    denom = np.maximum(np.abs(yt), MAPE_EPSILON)
    mape = float(np.mean(np.abs(yt - yp) / denom) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "n": int(valid.sum())}


@contextmanager
def timer():
    t = {}
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t["seconds"] = time.perf_counter() - t0


def check_not_degenerate(preds: np.ndarray, mask: np.ndarray, label: str, min_std: float = 1e-9):
    std = float(np.nanstd(preds[mask]))
    if std < min_std:
        raise RuntimeError(f"{label}: predictions collapsed to a near-constant (std={std:.6g}) - investigate before trusting metrics.")
    return std
