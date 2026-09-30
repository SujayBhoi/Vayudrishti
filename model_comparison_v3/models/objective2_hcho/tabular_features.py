"""Shared tabular feature builder for Objective 2's per-cell ML models
(Linear Regression, Random Forest) - flattens the same 6-month lookback
window ConvLSTM uses (per-cell, not spatial-convolved) into one row per
(month, grid cell): 6 months x 4 channels = 24 features, target = that
cell's HCHO at the target month.

This is the direct "does spatial structure (ConvLSTM's convolutions)
actually help beyond a cell's own history" comparison - same target
months, same lookback, same train/test split, just no neighbor
information and no temporal recurrence.
"""

import numpy as np

LOOKBACK = 6


def build_tabular(tensor: np.ndarray, mask: np.ndarray, target_indices: list[int], lookback: int = LOOKBACK):
    """tensor: (n_months, C, H, W). Returns X (n_rows, lookback*C), y (n_rows,),
    and (month_idx, row, col) metadata per row - real India cells only,
    rows with a NaN target dropped, NaN features filled with 0 (matches
    ConvLSTM's nan_to_num convention)."""
    n_months, C, H, W = tensor.shape
    rows_r, rows_c = np.where(mask)
    n_cells = len(rows_r)

    X_parts, y_parts, meta = [], [], []
    for i in target_indices:
        if i - lookback < 0:
            continue
        target_vals = tensor[i, 0, rows_r, rows_c]
        valid = ~np.isnan(target_vals)
        if valid.sum() == 0:
            continue
        window = tensor[i - lookback:i, :, rows_r, rows_c]  # (lookback, C, n_cells)
        window = np.nan_to_num(window, nan=0.0)
        feats = window.transpose(2, 0, 1).reshape(n_cells, lookback * C)  # (n_cells, lookback*C)
        X_parts.append(feats[valid])
        y_parts.append(target_vals[valid])
        for r, c in zip(rows_r[valid], rows_c[valid]):
            meta.append((i, r, c))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y, meta


def predictions_to_grid(preds: np.ndarray, meta: list, n_months: int, H: int, W: int, month_offset_map: dict) -> np.ndarray:
    """Reassemble flat (row-per-cell-per-month) predictions back into a
    (n_test_months, H, W) grid, NaN where not predicted, for reuse with
    eval_utils.evaluate_masked (keeps output format identical to
    statistical_baseline/convlstm's saved predictions.npz)."""
    n_test_months = len(month_offset_map)
    grid = np.full((n_test_months, H, W), np.nan, dtype=np.float32)
    for (m, r, c), p in zip(meta, preds):
        grid[month_offset_map[m], r, c] = p
    return grid
