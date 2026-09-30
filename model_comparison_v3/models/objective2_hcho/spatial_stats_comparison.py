"""Objective 2 extension (mentor ask, relayed 2026-08-07 03:24 IST): compare
the existing anomaly-threshold hotspot method against two standard spatial
autocorrelation statistics - Getis-Ord Gi* (local hotspot z-scores) and
Anselin's Local Moran's I / LISA (cluster-outlier classification) - using
the real esda/libpysal implementations, not a hand-rolled approximation.

Applied to the monthly national HCHO grid (spatiotemporal_grid.npz, 4447
real India land cells, channel 0 = HCHO) for 2025-11, the month with the
most existing-method-flagged hotspots in the whole 2018-2025 record
(stubble-burning season, Nov/Dec consistently peak - a real, expected
pattern, not cherry-picked for effect).

Neighbors defined via k=8 nearest-neighbor weights on real land-cell
centroids (not raster rook/queen contiguity, since ~53% of the 111x116
grid is non-India/masked - KNN on valid centroids is the standard approach
for irregular/masked spatial units).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran_Local
from libpysal.weights import KNN

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts")
LEGACY_OBJ2_DIR = os.path.join(os.path.dirname(V3_ROOT), "models", "objective2_hcho", "artifacts")

TARGET_MONTH = "2025-11"
SEED = 42


def main():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    tensor, mask, lats, lons = data["tensor"], data["mask"], data["lats"], data["lons"]
    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json")) as f:
        manifest = json.load(f)
    month_idx = manifest["all_months"].index(TARGET_MONTH)

    hcho_grid = tensor[month_idx, 0]  # channel 0 = hcho
    lat_idx, lon_idx = np.where(mask)
    lat_vals = lats[lat_idx]
    lon_vals = lons[lon_idx]
    y = hcho_grid[lat_idx, lon_idx]
    valid = ~np.isnan(y)
    lat_vals, lon_vals, y = lat_vals[valid], lon_vals[valid], y[valid]
    coords = np.column_stack([lon_vals, lat_vals])
    print(f"Month {TARGET_MONTH}: {len(y)} valid India land cells (of {mask.sum()} total)")

    w = KNN.from_array(coords, k=8)
    w.transform = "r"

    print("Running Getis-Ord Gi*...")
    gi = G_Local(y, w, star=True, permutations=999, seed=SEED)
    gi_z = gi.Zs
    gi_p = gi.p_sim
    gi_hotspot = (gi_z > 1.96) & (gi_p < 0.05)
    gi_coldspot = (gi_z < -1.96) & (gi_p < 0.05)

    print("Running Anselin Local Moran's I (LISA)...")
    lisa = Moran_Local(y, w, permutations=999, seed=SEED)
    # quadrant: 1=HH (hot cluster), 2=LH, 3=LL (cold cluster), 4=HL
    lisa_hh = (lisa.q == 1) & (lisa.p_sim < 0.05)
    lisa_ll = (lisa.q == 3) & (lisa.p_sim < 0.05)

    out = pd.DataFrame({
        "lat": lat_vals, "lon": lon_vals, "hcho": y,
        "getis_ord_z": gi_z, "getis_ord_p": gi_p, "getis_ord_hotspot": gi_hotspot, "getis_ord_coldspot": gi_coldspot,
        "lisa_I": lisa.Is, "lisa_p": lisa.p_sim, "lisa_quadrant": lisa.q, "lisa_hh_cluster": lisa_hh, "lisa_ll_cluster": lisa_ll,
    })
    out["month"] = TARGET_MONTH

    # --- compare against the existing anomaly-threshold method for the same month ---
    flagged_path = os.path.join(LEGACY_OBJ2_DIR, "hcho_flagged.csv")
    existing = pd.read_csv(flagged_path, parse_dates=["date"])
    existing["month"] = existing["date"].dt.to_period("M").astype(str)
    existing_month = existing[existing["month"] == TARGET_MONTH]
    existing_hotspots = existing_month[existing_month["is_hotspot"] == True]  # noqa: E712
    print(f"Existing method: {len(existing_hotspots)} flagged station-days in {TARGET_MONTH} "
          f"(daily cadence, {existing_month['date'].nunique()} days)")

    def nearest_flag(lat, lon, pts, radius_deg=0.5):
        d = np.hypot(pts["lat"].to_numpy() - lat, pts["lon"].to_numpy() - lon)
        return (d < radius_deg).any()

    if len(existing_hotspots):
        out["existing_method_nearby_flag"] = [
            nearest_flag(r.lat, r.lon, existing_hotspots) for r in out.itertuples()
        ]
    else:
        out["existing_method_nearby_flag"] = False

    n_cells = len(out)
    agree_getis = ((out["getis_ord_hotspot"]) & (out["existing_method_nearby_flag"])).sum()
    agree_lisa = ((out["lisa_hh_cluster"]) & (out["existing_method_nearby_flag"])).sum()
    summary = {
        "month": TARGET_MONTH,
        "n_grid_cells": int(n_cells),
        "getis_ord": {
            "n_hotspot_cells": int(out["getis_ord_hotspot"].sum()),
            "n_coldspot_cells": int(out["getis_ord_coldspot"].sum()),
            "pct_hotspot": round(100 * out["getis_ord_hotspot"].sum() / n_cells, 2),
        },
        "lisa": {
            "n_hh_cluster_cells": int(out["lisa_hh_cluster"].sum()),
            "n_ll_cluster_cells": int(out["lisa_ll_cluster"].sum()),
            "pct_hh": round(100 * out["lisa_hh_cluster"].sum() / n_cells, 2),
        },
        "existing_anomaly_threshold": {
            "n_flagged_station_days": int(len(existing_hotspots)),
            "n_grid_cells_near_a_flag": int(out["existing_method_nearby_flag"].sum()),
        },
        "agreement": {
            "getis_ord_hotspot_AND_existing_flag_nearby": int(agree_getis),
            "lisa_hh_AND_existing_flag_nearby": int(agree_lisa),
            "note": "Existing method is daily/station-point-based (~403 stations' worth of pixels), "
                    "Getis-Ord/LISA run on the monthly 4447-cell national grid - agreement measured as "
                    "'is a statistically-significant hotspot cell within 0.5 degrees of any existing-method "
                    "flagged point that month', a reasonable but not exact correspondence given the "
                    "different spatial units.",
        },
    }
    print(json.dumps(summary, indent=2))

    out.to_csv(os.path.join(ARTIFACTS_DIR, "spatial_stats_comparison.csv"), index=False)
    with open(os.path.join(ARTIFACTS_DIR, "spatial_stats_comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
