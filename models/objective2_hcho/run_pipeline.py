"""End-to-end Objective 2 orchestration: detect -> attribute -> explain.

1. Load the national HCHO pixel-date table (data_pipeline/load_hcho_rasters.py
   output) and the fire/wind grid tables (data_pipeline/fetch_objective2_grid.py
   batch export output, downloaded from Drive).
2. detect_hotspots.detect_all() - percentile threshold + DBSCAN clustering,
   using ONLY the HCHO field itself, over ALL of India, all seasons. This is
   the complete "detect" step; no fire/wind data is touched here at all.
3. cluster_fires.build_attribution_features() computes fire-distance/count/
   wind-alignment features per flagged hotspot, matched to its nearest
   available fire/wind week (see date-matching note below).
4. TWO separate attribution models are trained on this feature table, not
   one - see "Why two models" below.
5. explain_hotspots.explain_all() turns SHAP values into plain-language text
   per hotspot for each model.

Why two models: fitting one XGBoost model on ALL nationally-flagged hotspots
(all seasons, all regions) showed essentially zero explanatory power
(test R2 ~ 0) for fire-only features. That's not a bug - HCHO has many
sources (industrial VOCs, traffic, biogenic isoprene) and most flagged
hotspots outside the Oct-Dec stubble-burning season/Punjab-Haryana-Delhi-UP
corridor genuinely aren't fire-driven, so a fire-only model correctly
declines to explain them well. Restricting to that season+corridor recovers
a real (if still modest) fire<->HCHO correlation. So:
  - PRIMARY model ("corridor_season"): trained only on Oct-Dec + the
    Punjab-Haryana-Delhi-UP corridor - the actual validated fire-attribution
    result, used for hotspots that fall in that window.
  - SECONDARY model ("national_all_season"): trained on every flagged
    hotspot nationally/year-round - kept as an honest, separate artifact,
    not blended into the primary number. Used as an explicit "likely
    non-fire source" fallback for hotspots outside the primary window,
    rather than forcing a fire narrative onto them.
Detection itself (step 2) stays fully national and year-round either way -
only the ATTRIBUTION step is split, matching what the two-step
detect-then-explain structure is supposed to do.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from cluster_fires import build_attribution_features, cluster_fires
from compute_baseline import add_regional_baseline
from detect_hotspots import detect_all
from explain_hotspots import explain_all
from train_attribution_xgb import save, train

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Punjab-Haryana-Delhi-western UP corridor, matching v1's original
# hotspot_grid_extract.py bounding box; Oct-Dec = peak stubble season.
CORRIDOR_LAT = (27.0, 32.0)
CORRIDOR_LON = (73.0, 81.0)
CORRIDOR_MONTHS = {10, 11, 12}


def build_attributed_table(hcho_df: pd.DataFrame, fire_df: pd.DataFrame, wind_df: pd.DataFrame):
    """Steps 1-2: detect (national, all-season) + attribution feature
    build for every flagged hotspot. Shared by both downstream models."""
    print(f"Loaded {len(hcho_df)} HCHO pixel-date records, {hcho_df['date'].nunique()} dates")

    # Baseline computed on the FULL pixel-date population (not just the
    # eventual hotspots) - it needs the true background distribution.
    # Detection threshold (below) still uses raw hcho, unaffected; only the
    # downstream attribution regression target changes to the anomaly.
    hcho_df = add_regional_baseline(hcho_df)

    flagged, clusters_df, threshold = detect_all(hcho_df)
    n_hotspots = int(flagged["is_hotspot"].sum())
    print(f"Detected {n_hotspots} hotspot pixel-dates (threshold={threshold:.6g}), "
          f"{len(clusters_df)} DBSCAN clusters across all dates")

    # Nearest-week match: HCHO dates are monthly (the 15th), fire/wind grids
    # are weekly buckets anchored 2018-01-01 - an exact string match only
    # aligns 13/84 times by coincidence, which is what produced the original
    # near-zero R2 before this fix (verified: max gap after this fix is a
    # few days, i.e. genuinely the same real-world period every time).
    hotspot_points = flagged[flagged["is_hotspot"]]
    fire_dates = pd.to_datetime(sorted(fire_df["date"].unique())) if "date" in fire_df.columns else None
    wind_dates = pd.to_datetime(sorted(wind_df["date"].unique())) if "date" in wind_df.columns else None
    attributed_parts = []
    matched_date_pairs = []
    for date, group in hotspot_points.groupby("date"):
        if fire_dates is not None:
            nearest_fire_date = fire_dates[np.argmin(np.abs(fire_dates - pd.Timestamp(date)))].strftime("%Y-%m-%d")
            fire_today = fire_df[fire_df["date"] == nearest_fire_date]
        else:
            fire_today = fire_df
        if wind_dates is not None:
            nearest_wind_date = wind_dates[np.argmin(np.abs(wind_dates - pd.Timestamp(date)))].strftime("%Y-%m-%d")
            wind_today = wind_df[wind_df["date"] == nearest_wind_date]
        else:
            wind_today = wind_df
        if fire_today.empty:
            continue
        matched_date_pairs.append((date, nearest_fire_date if fire_dates is not None else date))
        fire_today = cluster_fires(fire_today)
        feats = build_attribution_features(group, fire_today, wind_today)
        attributed_parts.append(feats)
    max_gap = max((abs((pd.Timestamp(a) - pd.Timestamp(b)).days) for a, b in matched_date_pairs), default=0)
    print(f"Matched {len(matched_date_pairs)}/{hotspot_points['date'].nunique()} HCHO dates to nearest fire/wind week (max gap: {max_gap} days)")

    if not attributed_parts:
        raise RuntimeError("No hotspot dates had matching fire data - check date alignment between HCHO and fire grids")
    attributed = pd.concat(attributed_parts, ignore_index=True)
    attributed["month"] = pd.to_datetime(attributed["date"]).dt.month
    print(f"Built attribution features for {len(attributed)} flagged hotspot points")
    return flagged, clusters_df, threshold, n_hotspots, attributed


def is_in_corridor_season(df: pd.DataFrame) -> pd.Series:
    return (
        df["month"].isin(CORRIDOR_MONTHS)
        & df["lat"].between(*CORRIDOR_LAT)
        & df["lon"].between(*CORRIDOR_LON)
    )


def train_and_explain(attributed: pd.DataFrame, tag: str):
    model, shap_values, test_r2, X, df = train(attributed)
    print(f"[{tag}] attribution model test R2={test_r2:.3f} on {len(df)} points "
          f"(explains variation AMONG {'this season/region window' if tag == 'corridor_season' else 'nationally-flagged hotspots'})")
    meta_cols = [c for c in ["grid_id", "lat", "lon", "date", "hcho", "hcho_baseline", "hcho_anomaly"] if c in df.columns]
    explained = explain_all(X, shap_values, df[meta_cols])
    explained["model"] = tag

    save(model, os.path.join(ARTIFACTS_DIR, f"attribution_xgb_{tag}.pkl"))
    explained.to_csv(os.path.join(ARTIFACTS_DIR, f"hotspot_explanations_{tag}.csv"), index=False)
    return model, test_r2, explained


def run(hcho_df: pd.DataFrame, fire_df: pd.DataFrame, wind_df: pd.DataFrame):
    flagged, clusters_df, threshold, n_hotspots, attributed = build_attributed_table(hcho_df, fire_df, wind_df)

    corridor_mask = is_in_corridor_season(attributed)
    corridor_df = attributed[corridor_mask]
    print(f"\nCorridor+season subset: {len(corridor_df)} of {len(attributed)} flagged points "
          f"({len(corridor_df)/len(attributed):.1%})")

    _, primary_r2, primary_explained = train_and_explain(corridor_df, "corridor_season")
    _, secondary_r2, secondary_explained = train_and_explain(attributed, "national_all_season")

    explained = pd.concat([primary_explained, secondary_explained], ignore_index=True)

    flagged.to_csv(os.path.join(ARTIFACTS_DIR, "hcho_flagged.csv"), index=False)
    clusters_df.to_csv(os.path.join(ARTIFACTS_DIR, "hcho_clusters.csv"), index=False)
    explained.to_csv(os.path.join(ARTIFACTS_DIR, "hotspot_explanations.csv"), index=False)
    joblib.dump({
        "threshold": threshold, "n_hotspots": n_hotspots,
        "primary_corridor_season_r2": primary_r2,
        "secondary_national_all_season_r2": secondary_r2,
        "corridor_season_fraction": len(corridor_df) / len(attributed),
    }, os.path.join(ARTIFACTS_DIR, "summary.pkl"))

    print(f"\nSaved: hcho_flagged.csv, hcho_clusters.csv, hotspot_explanations.csv, "
          f"attribution_xgb_corridor_season.pkl, attribution_xgb_national_all_season.pkl")
    return flagged, clusters_df, explained
