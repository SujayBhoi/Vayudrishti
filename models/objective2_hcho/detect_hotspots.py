"""Objective 2, step 1: HCHO hotspot DETECTION (statistical, no fire/attribution
input at all) — this is the piece that was missing from both the teammate's
notebook (which only had an arbitrary top-30-by-predicted-value cutoff) and
is the direct fix the user asked for. Generalizes Trial v1's
hotspot_analysis.py (which worked, but was scoped to the Punjab-Haryana-Delhi
corridor) to run over any HCHO grid table, including the India-wide one.

Two independent detection methods, kept separate so either can be inspected
on its own:
  1. Percentile threshold (p90 by default) on raw HCHO value, exactly as v1.
  2. DBSCAN spatial clustering of the flagged (percentile-exceeding) points,
     to turn scattered pixels into contiguous hotspot regions — also as v1.

Input contract: a DataFrame with columns [grid_id or station-like id, lat,
lon, date (or period_start), hcho]. Output: the same frame with an
`is_hotspot` boolean column, plus a separate clusters DataFrame.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN

HCHO_HOTSPOT_PERCENTILE = 90
DBSCAN_EPS_DEG = 0.4          # ~44km, matches v1's corridor-scale clustering
DBSCAN_MIN_SAMPLES = 3


def flag_hotspots(df: pd.DataFrame, hcho_col: str = "hcho", percentile: float = HCHO_HOTSPOT_PERCENTILE) -> tuple[pd.DataFrame, float]:
    """Percentile threshold computed against the FULL national, multi-year
    baseline (not just burning-season rows), so a flagged point is genuinely
    anomalous relative to typical conditions everywhere/everywhen, not just
    relative to an already-selected hot window — same principle as v1."""
    threshold = df[hcho_col].quantile(percentile / 100)
    out = df.copy()
    out["is_hotspot"] = out[hcho_col] >= threshold
    return out, threshold


def cluster_hotspots_for_period(period_df: pd.DataFrame, eps_deg: float = DBSCAN_EPS_DEG, min_samples: int = DBSCAN_MIN_SAMPLES) -> pd.DataFrame:
    hot = period_df[period_df["is_hotspot"]]
    if len(hot) < min_samples:
        return pd.DataFrame(columns=["cluster_id", "lat", "lon", "n_points", "mean_hcho"])
    coords = hot[["lat", "lon"]].to_numpy()
    labels = DBSCAN(eps=eps_deg, min_samples=min_samples).fit_predict(coords)
    hot = hot.copy()
    hot["cluster_id"] = labels
    clusters = (
        hot[hot["cluster_id"] >= 0]
        .groupby("cluster_id")
        .agg(lat=("lat", "mean"), lon=("lon", "mean"), n_points=("lat", "size"), mean_hcho=("hcho", "mean"))
        .reset_index()
    )
    return clusters


def detect_all(df: pd.DataFrame, period_col: str = "date", hcho_col: str = "hcho") -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Full detection pass: flag + cluster per period. Returns (flagged_df,
    clusters_df, threshold). This is the complete "detect" half of the
    required two-step (detect, then explain) structure — no fire/wind/
    attribution features are used anywhere in this file."""
    flagged, threshold = flag_hotspots(df, hcho_col=hcho_col)
    all_clusters = []
    for period, pdf in flagged.groupby(period_col):
        clusters = cluster_hotspots_for_period(pdf)
        clusters[period_col] = period
        all_clusters.append(clusters)
    clusters_df = pd.concat(all_clusters, ignore_index=True) if all_clusters else pd.DataFrame()
    return flagged, clusters_df, threshold
