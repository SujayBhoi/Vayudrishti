"""Objective 2 extension (mentor ask): combine wind and fire data and
generate seasonal graphs, using the same spatiotemporal_grid.npz tensor
already built for ConvLSTM (channels: hcho, fire_count, wind_u, wind_v -
2018-01 through 2025-12, 96 months, national grid, 4447 real India cells).

Two views produced:
1. National monthly time series: mean fire_count, mean wind_speed, mean
   HCHO across all valid cells, per month - shows the raw co-evolution.
2. Seasonal climatology: same three variables averaged by IMD's standard
   4-season convention (Winter Dec-Feb, Summer Mar-May, Monsoon Jun-Sep,
   Post-Monsoon Oct-Nov - the stubble-burning season), across all 8 years
   - shows the repeating annual pattern, not just one year's noise.
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

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
}
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]


def main():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    tensor, mask = data["tensor"], data["mask"]
    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json")) as f:
        manifest = json.load(f)
    months = manifest["all_months"]

    rows = []
    for i, month in enumerate(months):
        hcho = tensor[i, 0][mask]
        fire = tensor[i, 1][mask]
        wind_u = tensor[i, 2][mask]
        wind_v = tensor[i, 3][mask]
        wind_speed = np.sqrt(wind_u ** 2 + wind_v ** 2)
        rows.append({
            "month": month,
            "year": int(month[:4]),
            "month_num": int(month[5:7]),
            "mean_hcho": float(np.nanmean(hcho)),
            "mean_fire_count": float(np.nanmean(fire)),
            "mean_wind_speed": float(np.nanmean(wind_speed)),
        })

    ts = pd.DataFrame(rows)
    ts["season"] = ts["month_num"].map(SEASON_MAP)
    ts.to_csv(os.path.join(ARTIFACTS_DIR, "wind_fire_monthly_timeseries.csv"), index=False)

    seasonal = ts.groupby("season")[["mean_hcho", "mean_fire_count", "mean_wind_speed"]].agg(["mean", "std"])
    seasonal.columns = ["_".join(c) for c in seasonal.columns]
    seasonal = seasonal.reindex(SEASON_ORDER).reset_index()
    seasonal.to_csv(os.path.join(ARTIFACTS_DIR, "wind_fire_seasonal_climatology.csv"), index=False)

    print("=== Monthly time series (head) ===")
    print(ts.head())
    print("\n=== Seasonal climatology (8-year average) ===")
    print(seasonal.to_string(index=False))

    corr_fire_hcho = ts["mean_fire_count"].corr(ts["mean_hcho"])
    corr_wind_hcho = ts["mean_wind_speed"].corr(ts["mean_hcho"])
    corr_fire_wind = ts["mean_fire_count"].corr(ts["mean_wind_speed"])
    print(f"\nNational monthly correlations: fire-vs-hcho={corr_fire_hcho:.3f}, "
          f"wind-vs-hcho={corr_wind_hcho:.3f}, fire-vs-wind={corr_fire_wind:.3f}")

    summary = {
        "correlations": {"fire_vs_hcho": corr_fire_hcho, "wind_vs_hcho": corr_wind_hcho, "fire_vs_wind": corr_fire_wind},
        "peak_fire_season": seasonal.loc[seasonal["mean_fire_count_mean"].idxmax(), "season"],
        "peak_hcho_season": seasonal.loc[seasonal["mean_hcho_mean"].idxmax(), "season"],
        "lowest_wind_season": seasonal.loc[seasonal["mean_wind_speed_mean"].idxmin(), "season"],
    }
    with open(os.path.join(ARTIFACTS_DIR, "wind_fire_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"\nSaved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
