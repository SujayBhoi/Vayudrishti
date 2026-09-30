"""Turns per-hotspot SHAP values (from train_attribution_xgb.py) into
plain-language explanations, mirroring v1's shap_explain.py pattern for
Objective 1 so both objectives explain themselves the same way in the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

LABELS = {
    "fire_dist_km": "distance to nearest fire",
    "fire_count_nearby": "number of nearby fires",
    "wind_alignment": "wind alignment from fire toward this location",
    "u": "east-west wind component",
    "v": "north-south wind component",
}


def explain_point(row_idx: int, X: pd.DataFrame, shap_values: shap.Explanation, top_n: int = 3) -> str:
    contribs = shap_values.values[row_idx]
    order = np.argsort(-np.abs(contribs))[:top_n]
    parts = []
    for i in order:
        col = X.columns[i]
        val = contribs[i]
        direction = "raised" if val > 0 else "lowered"
        parts.append(f"{LABELS.get(col, col)} {direction} the HCHO level (impact {val:+.4g})")
    return "; ".join(parts)


def explain_all(X: pd.DataFrame, shap_values: shap.Explanation, meta: pd.DataFrame) -> pd.DataFrame:
    """meta must be row-aligned with X (same index), containing at least
    [lat, lon, date, hcho]. Returns meta + an `explanation` text column."""
    explanations = [explain_point(i, X, shap_values) for i in range(len(X))]
    out = meta.reset_index(drop=True).copy()
    out["explanation"] = explanations
    return out
