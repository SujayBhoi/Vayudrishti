"""Precomputes, for every month, the flat (lat, lon, hcho, fire_count,
wind_u, wind_v) table + engineered features (today/yesterday/7-day-avg
HCHO, month, lat, lon) - the fast data the Streamlit month slider reads,
so no per-interaction recomputation is needed.

Feature engineering note: MODIS AOD, temperature, humidity, wind speed
direction already covered via wind_u/wind_v, pressure aren't in the
spatiotemporal_grid.npz tensor (only hcho/fire_count/wind_u/wind_v were
extracted for ConvLSTM) - those would need a new GEE extraction, out of
scope given remaining time. Built honestly from what's actually there.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts")


def main():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    tensor, mask, lats, lons = data["tensor"], data["mask"], data["lats"], data["lons"]
    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json")) as f:
        manifest = json.load(f)
    months = manifest["all_months"]

    lat_idx, lon_idx = np.where(mask)
    lat_vals, lon_vals = lats[lat_idx], lons[lon_idx]

    rows = []
    for i, month in enumerate(months):
        hcho = tensor[i, 0][lat_idx, lon_idx]
        fire = tensor[i, 1][lat_idx, lon_idx]
        wind_u = tensor[i, 2][lat_idx, lon_idx]
        wind_v = tensor[i, 3][lat_idx, lon_idx]
        hcho_prev = tensor[i - 1, 0][lat_idx, lon_idx] if i > 0 else np.full_like(hcho, np.nan)
        window = tensor[max(0, i - 6):i + 1, 0][:, lat_idx, lon_idx]
        hcho_7avg = np.nanmean(window, axis=0)

        df = pd.DataFrame({
            "month": month, "lat": lat_vals, "lon": lon_vals,
            "hcho_today": hcho, "hcho_prev_month": hcho_prev, "hcho_7month_avg": hcho_7avg,
            "fire_count": fire, "wind_u": wind_u, "wind_v": wind_v,
            "wind_speed": np.sqrt(wind_u ** 2 + wind_v ** 2),
            "wind_dir_deg": (np.degrees(np.arctan2(wind_v, wind_u)) + 360) % 360,
            "month_num": int(month[5:7]),
        })
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(ARTIFACTS_DIR, "month_grid_features.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows ({len(months)} months x {mask.sum()} cells) -> {out_path}")


if __name__ == "__main__":
    main()
