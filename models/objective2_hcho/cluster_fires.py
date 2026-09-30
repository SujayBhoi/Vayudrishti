"""Objective 2, attribution input prep: DBSCAN clustering of fire points, and
per-(hotspot point, date) fire/wind attribution features.

Adapted from the teammate's notebook concept (Untitled1.ipynb cell 18:
get_fire_collection, cluster_fires, feature list), cleaned up to fix two
issues found during review:
  1. The notebook computed DBSCAN fire clusters but never used the labels
     downstream (fire_count_nearby was a separate, unrelated haversine count).
     Here, fire_count_nearby is still a direct radius count (that part was
     fine and reusable) but cluster membership is now actually attached to
     each fire point and available to callers, so it isn't dead code.
  2. No Colab/Drive-specific paths anywhere; runs against whatever DataFrame
     is passed in.

Feature set kept identical to the notebook (fire_dist_km, fire_count_nearby,
wind_alignment, u, v) since those were sound choices — the fix is in how
they're used (see train_attribution_xgb.py: explains only already-flagged
hotspots, target is not raw HCHO), not in the feature list itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

FIRE_DBSCAN_EPS_KM = 15
FIRE_DBSCAN_MIN_SAMPLES = 3
NEARBY_RADIUS_KM = 50


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def cluster_fires(fire_df: pd.DataFrame, eps_km: float = FIRE_DBSCAN_EPS_KM, min_samples: int = FIRE_DBSCAN_MIN_SAMPLES) -> pd.DataFrame:
    """fire_df must have [lat, lon]. Adds a `cluster` column (-1 = noise)."""
    if len(fire_df) < min_samples:
        out = fire_df.copy()
        out["cluster"] = -1
        return out
    coords = fire_df[["lat", "lon"]].to_numpy()
    labels = DBSCAN(eps=eps_km / 111.0, min_samples=min_samples).fit_predict(coords)
    out = fire_df.copy()
    out["cluster"] = labels
    return out


def build_attribution_features(points_df: pd.DataFrame, fire_df: pd.DataFrame, wind_df: pd.DataFrame, radius_km: float = NEARBY_RADIUS_KM) -> pd.DataFrame:
    """For each row in points_df (typically: hotspot points for one date),
    compute nearest-fire distance/count and wind alignment against fire_df
    (same-date fire detections) and wind_df (nearest wind grid sample).

    points_df: [lat, lon, ...]
    fire_df: [lat, lon] same-date fire detections
    wind_df: [lat, lon, u, v] same-date wind grid samples
    """
    out_rows = []
    for _, pt in points_df.iterrows():
        if len(fire_df):
            dists = _haversine_km(pt["lat"], pt["lon"], fire_df["lat"].to_numpy(), fire_df["lon"].to_numpy())
            fire_dist_km = float(dists.min())
            fire_count_nearby = int((dists <= radius_km).sum())
            nearest_fire_idx = int(np.argmin(dists))
        else:
            fire_dist_km, fire_count_nearby, nearest_fire_idx = np.nan, 0, None

        if len(wind_df):
            wdists = _haversine_km(pt["lat"], pt["lon"], wind_df["lat"].to_numpy(), wind_df["lon"].to_numpy())
            nearest_wind = wind_df.iloc[int(np.argmin(wdists))]
            u, v = float(nearest_wind["u"]), float(nearest_wind["v"])
        else:
            u, v = 0.0, 0.0

        wind_alignment = 0.0
        if fire_count_nearby and nearest_fire_idx is not None and (u != 0 or v != 0):
            fire_row = fire_df.iloc[nearest_fire_idx]
            to_point = np.array([pt["lon"] - fire_row["lon"], pt["lat"] - fire_row["lat"]])
            wind_vec = np.array([u, v])
            denom = (np.linalg.norm(to_point) * np.linalg.norm(wind_vec)) or 1e-9
            wind_alignment = float(np.dot(to_point, wind_vec) / denom)

        row = dict(pt)
        row.update(dict(fire_dist_km=fire_dist_km, fire_count_nearby=fire_count_nearby, wind_alignment=wind_alignment, u=u, v=v))
        out_rows.append(row)
    return pd.DataFrame(out_rows)
