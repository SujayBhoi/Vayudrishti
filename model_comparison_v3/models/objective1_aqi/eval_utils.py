"""Shared evaluation utilities for the clean101 5-model comparison.

Factored out so all 5 models (Linear Regression, Random Forest, XGBoost,
ARIMA/SARIMA, CNN-LSTM) produce directly comparable output - one evaluate()
function, one per_station_metrics() function, one timing convention. Fixes
the CNN-LSTM skeleton's original bug of computing its own independent split
instead of reusing a shared one (see model_comparison_v3/PROJECT_DOCUMENTATION.md).
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAPE_EPSILON = 1.0  # AQI can be legitimately 0 on very clean days; guard against div-by-~0


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), MAPE_EPSILON)
    mape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "n": int(mask.sum())}


def per_station_metrics(df: pd.DataFrame, y_true_col: str, y_pred_col: str, station_col: str = "station_id", min_test_rows: int = 5) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby(station_col):
        if len(g) < min_test_rows:
            continue
        m = evaluate(g[y_true_col], g[y_pred_col])
        m[station_col] = sid
        rows.append(m)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("r2", ascending=False).reset_index(drop=True)
    return out


@contextmanager
def timer():
    """Usage: with timer() as t: ...; t['seconds'] is populated after the block."""
    t = {}
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t["seconds"] = time.perf_counter() - t0


def time_based_cv_folds(dates, n_folds: int = 4):
    """Rolling-origin (expanding-window) time-series CV folds - NOT random
    k-fold, which would leak through lag/target-encoding features. Same
    pattern as the existing train_extended_xgb.py."""
    dates = pd.Series(dates).reset_index(drop=True)
    order = np.argsort(dates.values)
    sorted_dates = dates.values[order]
    edges = np.quantile(sorted_dates.astype("int64"), np.linspace(0, 1, n_folds + 1))
    folds = []
    for i in range(1, n_folds):
        train_idx = order[sorted_dates.astype("int64") <= edges[i]]
        val_idx = order[(sorted_dates.astype("int64") > edges[i]) & (sorted_dates.astype("int64") <= edges[i + 1])]
        if len(train_idx) and len(val_idx):
            folds.append((train_idx, val_idx))
    return folds


def check_not_degenerate(preds, label: str, min_std: float = 1e-6):
    """Project's own 5.11 lesson: a model that collapsed to predicting a
    constant can still show a plausible-looking R2 by coincidence - always
    check preds.std() before trusting a metric."""
    std = float(np.nanstd(preds))
    if std < min_std:
        raise RuntimeError(f"{label}: predictions collapsed to a near-constant (std={std:.6g}) - do not trust this model's metrics without investigating.")
    return std
