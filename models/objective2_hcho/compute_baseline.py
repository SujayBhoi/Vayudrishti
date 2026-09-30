"""Computes a regional HCHO baseline and per-point anomaly, so the
attribution model can predict "how far above/below normal for this area is
this point" instead of raw HCHO value.

Why this matters (diagnosed via 4 independent methods - heavy XGBoost,
regularized XGBoost, single-split linear regression, 5-fold CV linear
regression - all converging on ~0 R2): raw HCHO has large baseline
variation driven by latitude, humidity, vegetation density and regional
photochemistry that has nothing to do with fire. Predicting the raw value
among already-flagged (top-10%) points means the model is fighting that
baseline noise instead of isolating fire's marginal contribution. Removing
each point's own region's typical HCHO level should let the (real, if
modest - confirmed via raw correlation) fire-driven increment show through.

Baseline granularity: with only 84 sampled dates total (not dense enough
for a reliable per-grid-cell, per-month climatology - most individual grid
cells only appear a handful of times), the baseline is computed per
~2deg x 2deg region across ALL available dates (not month-specific) - a
practical middle ground between "too coarse to mean anything" (one national
number) and "too sparse to be reliable" (per-pixel per-month).
"""

import pandas as pd

REGION_BIN_DEG = 2.0


def add_regional_baseline(hcho_df: pd.DataFrame) -> pd.DataFrame:
    """hcho_df must have [lat, lon, hcho, ...] for ALL pixel-dates (not just
    flagged hotspots) - the baseline needs the full background distribution,
    not just the already-elevated subset, or it would just re-measure the
    same thing it's supposed to correct for."""
    df = hcho_df.copy()
    df["region_lat"] = (df["lat"] / REGION_BIN_DEG).round() * REGION_BIN_DEG
    df["region_lon"] = (df["lon"] / REGION_BIN_DEG).round() * REGION_BIN_DEG
    baseline = df.groupby(["region_lat", "region_lon"])["hcho"].mean().rename("hcho_baseline")
    df = df.merge(baseline, on=["region_lat", "region_lon"], how="left")
    df["hcho_anomaly"] = df["hcho"] - df["hcho_baseline"]
    return df
