"""Predicted-hotspot analysis: does ConvLSTM's forecast actually flag the
right grid cells as hotspots, region by region across India - not just
the corridor, but East/South/West too?

Reuses the EXACT already-validated national HCHO hotspot threshold from
the existing (pre-v3) detection pipeline (models/objective2_hcho/
artifacts/summary.pkl: 0.0003287748113506 mol/m2, the 90th percentile over
the full national multi-year distribution) - ties this new analysis
directly to the already-presented detection numbers rather than inventing
a new threshold.

For each of the 17 test months, a grid cell is "actual hotspot" if its
REAL HCHO value clears the threshold, "predicted hotspot" if ConvLSTM's
forecast for that cell clears it. Precision/recall of predicted vs actual
hotspots, computed nationally and per region.

Region boxes (approximate lat/lon boxes, NOT official state boundaries -
a rough grouping for regional breakdown, same spirit as the existing
corridor definition):
  - Corridor: lat 27-32, lon 73-81 (Punjab-Haryana-Delhi-western UP)
  - East: lat 19-28, lon 85-97 (Odisha, West Bengal, Assam, NE)
  - South: lat 8-19, lon 74-85 (Kerala, Tamil Nadu, Karnataka, AP/Telangana)
  - West: lat 15-24, lon 68-77 (Gujarat, Maharashtra, western Rajasthan)
  - Other/Central: everything else (real India cells not in the above)
Assigned mutually exclusively, in the listed priority order, so every
cell belongs to exactly one region.
"""

import json
import os

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
TRIAL02_ROOT = os.path.dirname(V3_ROOT)
DATA_DIR = os.path.join(V3_ROOT, "data")
FIG_DIR = os.path.join(V3_ROOT, "figures")
ARTIFACTS_DIR = os.path.join(os.path.dirname(THIS_DIR), "models", "objective2_hcho", "artifacts")

THRESHOLD = 0.0003287748113506  # exact, from the existing validated detection pipeline
# NOTE (found while building this analysis, not hidden): applying this
# ABSOLUTE threshold to ConvLSTM's predictions flags ZERO cells anywhere,
# in every test month - verified directly (predicted max 2.87e-4 never
# reaches the 3.29e-4 threshold; predicted std 2.6e-5 vs actual std 9.9e-5,
# a ~4x compression). This is the well-known "regression to the mean"
# behavior of a plain-MSE-trained model: minimizing average squared error
# structurally discourages ever guessing an extreme value, and the
# threshold itself was calibrated on raw daily pixels, not this tensor's
# monthly-smoothed values (monthly averaging itself softens extremes even
# in the ACTUAL data). This does NOT mean the model has no spatial skill -
# per-month Spearman rank correlation (predicted vs actual HCHO) is
# positive in every one of the 17 test months, mean 0.28 - the model knows
# WHERE HCHO will be relatively higher, it just can't be read off with an
# absolute threshold calibrated for a different, spikier data source.
# Fix used here: a RELATIVE threshold - each month's predicted "hotspots"
# are its own top TOP_FRACTION cells, matched against the SAME absolute
# threshold applied to the actual data (so "actual hotspot" still means
# "genuinely exceeds the real-world threshold").
TOP_FRACTION = 0.05  # top 5% of a month's predicted values = "predicted hotspot" for that month

REGION_BOXES = [
    ("Corridor (Punjab-Haryana-Delhi-UP)", (27, 32), (73, 81)),
    ("East (Odisha/WB/Assam/NE)", (19, 28), (85, 97)),
    ("South (Kerala/TN/Karnataka/AP)", (8, 19), (74, 85)),
    ("West (Gujarat/Maharashtra)", (15, 24), (68, 77)),
]


def assign_regions(lats: np.ndarray, lons: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    region = np.full((H, W), "Other/Central", dtype=object)
    lat_grid = lats[:, None] * np.ones((1, W))
    lon_grid = np.ones((H, 1)) * lons[None, :]
    assigned = np.zeros((H, W), dtype=bool)
    for name, (lat_lo, lat_hi), (lon_lo, lon_hi) in REGION_BOXES:
        in_box = (lat_grid >= lat_lo) & (lat_grid <= lat_hi) & (lon_grid >= lon_lo) & (lon_grid <= lon_hi)
        newly = in_box & ~assigned
        region[newly] = name
        assigned |= in_box
    region[~mask] = "N/A"
    return region


def precision_recall(actual_hot: np.ndarray, pred_hot: np.ndarray, region_mask: np.ndarray) -> dict:
    a = actual_hot[region_mask]
    p = pred_hot[region_mask]
    tp = int((a & p).sum())
    fp = int((~a & p).sum())
    fn = int((a & ~p).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    n_actual = int(a.sum())
    n_predicted = int(p.sum())
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "n_actual_hotspot_cells": n_actual, "n_predicted_hotspot_cells": n_predicted, "n_cells_total": int(region_mask.sum())}


def main():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    mask, lats, lons = data["mask"], data["lats"], data["lons"]
    region = assign_regions(lats, lons, mask)

    pred_data = np.load(os.path.join(ARTIFACTS_DIR, "convlstm", "predictions.npz"))
    preds, targets = pred_data["preds"], pred_data["targets"]  # (n_test_months, H, W)

    actual_hot = (targets >= THRESHOLD) & mask[None, :, :] & ~np.isnan(targets)

    # ABSOLUTE threshold on predictions: verified to flag zero cells everywhere,
    # every month (see NOTE above) - kept as an explicit, honest data point.
    pred_hot_absolute = (preds >= THRESHOLD) & mask[None, :, :] & ~np.isnan(preds)
    n_pred_absolute = int(pred_hot_absolute.sum())

    # RELATIVE threshold: each month's own top TOP_FRACTION predicted cells,
    # among real India cells only.
    pred_hot = np.zeros_like(pred_hot_absolute)
    for i in range(preds.shape[0]):
        vals = preds[i][mask]
        vals = vals[~np.isnan(vals)]
        cutoff = np.quantile(vals, 1 - TOP_FRACTION)
        pred_hot[i] = (preds[i] >= cutoff) & mask & ~np.isnan(preds[i])

    from scipy.stats import spearmanr
    rank_corrs = []
    for i in range(preds.shape[0]):
        p, t = preds[i][mask], targets[i][mask]
        valid = ~np.isnan(t)
        r, _ = spearmanr(p[valid], t[valid])
        rank_corrs.append(float(r))

    print(f"Absolute-threshold predicted hotspot cells (any month): {n_pred_absolute} "
          f"-- confirms the compression issue described above, not a bug in this script.")
    print(f"Per-month Spearman rank correlation (predicted vs actual): mean={np.mean(rank_corrs):.3f}, "
          f"all positive: {all(r > 0 for r in rank_corrs)}")
    print()

    results = {}
    region_names = [r[0] for r in REGION_BOXES] + ["Other/Central"]
    for name in ["National (all test months pooled)"] + region_names:
        if name == "National (all test months pooled)":
            region_mask_3d = np.broadcast_to(mask, actual_hot.shape)
        else:
            region_mask_2d = (region == name)
            region_mask_3d = np.broadcast_to(region_mask_2d, actual_hot.shape)
        results[name] = precision_recall(actual_hot, pred_hot, region_mask_3d)

    print(json.dumps(results, indent=2))

    out_path = os.path.join(ARTIFACTS_DIR, "hotspot_prediction_analysis.json")
    with open(out_path, "w") as f:
        json.dump({
            "threshold": THRESHOLD,
            "top_fraction_used_for_relative_predicted_hotspots": TOP_FRACTION,
            "n_predicted_hotspot_cells_absolute_threshold": n_pred_absolute,
            "mean_spearman_rank_correlation": float(np.mean(rank_corrs)),
            "spearman_all_months_positive": bool(all(r > 0 for r in rank_corrs)),
            "spearman_by_month": rank_corrs,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # save per-month, per-cell actual/predicted hotspot arrays + region map for charting
    np.savez_compressed(os.path.join(ARTIFACTS_DIR, "hotspot_prediction_grids.npz"),
                         actual_hot=actual_hot, pred_hot=pred_hot, mask=mask, region=region.astype(str),
                         lats=lats, lons=lons)
    print(f"Saved grids -> {os.path.join(ARTIFACTS_DIR, 'hotspot_prediction_grids.npz')}")


if __name__ == "__main__":
    main()
