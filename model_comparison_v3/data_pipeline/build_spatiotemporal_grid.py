"""Builds the dense national spatio-temporal tensor for Objective 2's
statistical baseline + ConvLSTM comparison.

Scope, explicit: full national grid, all 4,447 points - same scope as the
existing HCHO detection pipeline, not a regional subset (confirmed with the
user). No new GEE extraction needed - HCHO/fire/wind grids are already
fully downloaded and processed on disk from the existing Objective 2 work.

Steps:
1. Reconstruct the 4,447-point 0.25deg lattice as a dense masked (H,W) grid
   (111 lat x 116 lon - confirmed structure).
2. Spatial nearest-neighbor join (scipy.spatial.cKDTree) of the ~1.15M raw
   HCHO pixel rows onto the 4,447-point lattice - hcho_grid_national.csv's
   own grid_id is a meaningless per-row counter (confirmed), so this can't
   be a plain grid_id merge.
3. Bin HCHO into REGULAR MONTHLY buckets (confirmed decision) - the 351
   irregular HCHO dates (1-53 day gaps) get averaged per (grid cell,
   year-month), trading date-level resolution for a temporally valid
   sequence-model input.
4. Aggregate the already-regular weekly fire/wind grids to the same monthly
   buckets (fire: sum, wind: mean) - reusing load_objective2_grids.py's
   existing loaders, not reimplementing.
5. Assemble the dense tensor (n_months, channels=[hcho, fire_count, wind_u,
   wind_v], H=111, W=116) + a fixed India-boundary mask.
6. Chronological split: last ~15-20% of monthly steps held out as test.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
TRIAL02_ROOT = os.path.dirname(V3_ROOT)
DATA_DIR = os.path.join(V3_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(TRIAL02_ROOT, "data_pipeline"))
from load_objective2_grids import load_grid_lookup, load_wind_grid, load_fire_grid  # noqa: E402

HCHO_PATH = os.path.join(TRIAL02_ROOT, "data_pipeline", "legacy_v1_data", "processed", "hcho_grid_national.csv")
TEST_FRACTION = 0.18


def build_lattice_index(grid: pd.DataFrame):
    lats = np.sort(grid["lat"].unique())
    lons = np.sort(grid["lon"].unique())
    lat_to_row = {v: i for i, v in enumerate(lats)}
    lon_to_col = {v: i for i, v in enumerate(lons)}
    grid = grid.copy()
    grid["row"] = grid["lat"].map(lat_to_row)
    grid["col"] = grid["lon"].map(lon_to_col)
    H, W = len(lats), len(lons)
    mask = np.zeros((H, W), dtype=bool)
    mask[grid["row"].values, grid["col"].values] = True  # True = real India grid cell
    return grid, lats, lons, H, W, mask


def join_hcho_to_lattice(grid: pd.DataFrame) -> pd.DataFrame:
    print("Loading HCHO pixel data ...")
    hcho = pd.read_csv(HCHO_PATH, usecols=["lon", "lat", "hcho", "date"], parse_dates=["date"])
    print(f"  {hcho.shape}")

    tree = cKDTree(grid[["lat", "lon"]].values)
    _, idx = tree.query(hcho[["lat", "lon"]].values, k=1)
    hcho["grid_id"] = grid["grid_id"].values[idx]
    hcho["row"] = grid["row"].values[idx]
    hcho["col"] = grid["col"].values[idx]
    hcho["year_month"] = hcho["date"].dt.to_period("M").astype(str)

    monthly = hcho.groupby(["grid_id", "row", "col", "year_month"])["hcho"].mean().reset_index()
    print(f"  Monthly-binned HCHO: {monthly.shape}, {monthly.year_month.nunique()} unique months")
    return monthly


def monthly_aggregate(df: pd.DataFrame, value_col: str, agg: str, grid: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    out = df.groupby(["grid_id", "year_month"])[value_col].agg(agg).reset_index()
    out = out.merge(grid[["grid_id", "row", "col"]], on="grid_id", how="left")
    return out


def main():
    grid_raw = load_grid_lookup()
    grid, lats, lons, H, W, mask = build_lattice_index(grid_raw)
    print(f"Lattice: {H} lat x {W} lon = {H*W} cells, {mask.sum()} real India cells (of {len(grid)} grid points)")

    hcho_monthly = join_hcho_to_lattice(grid)

    print("Loading fire/wind grids ...")
    fire = load_fire_grid(min_count=0)  # dense - keep ALL cells, not just fire-active ones
    wind = load_wind_grid()
    fire_monthly = monthly_aggregate(fire, "count", "sum", grid)
    wind_u_monthly = monthly_aggregate(wind, "u", "mean", grid)
    wind_v_monthly = monthly_aggregate(wind, "v", "mean", grid)
    print(f"  Fire monthly: {fire_monthly.shape}, Wind monthly: {wind_u_monthly.shape}")

    all_months = sorted(set(hcho_monthly.year_month) | set(fire_monthly.year_month) | set(wind_u_monthly.year_month))
    print(f"Total distinct year-months across all sources: {len(all_months)}")

    n_months = len(all_months)
    tensor = np.full((n_months, 4, H, W), np.nan, dtype=np.float32)  # channels: hcho, fire_count, wind_u, wind_v
    month_to_idx = {m: i for i, m in enumerate(all_months)}

    def fill(df, value_col, channel):
        for row in df.itertuples(index=False):
            m_idx = month_to_idx[row.year_month]
            r, c = int(row.row), int(row.col)
            tensor[m_idx, channel, r, c] = getattr(row, value_col)

    fill(hcho_monthly, "hcho", 0)
    fill(fire_monthly, "count", 1)
    fill(wind_u_monthly, "u", 2)
    fill(wind_v_monthly, "v", 3)

    empty_months = [m for i, m in enumerate(all_months) if np.isnan(tensor[i, 0][mask]).all()]
    if empty_months:
        print(f"WARNING: {len(empty_months)} months have zero HCHO coverage on any India cell: {empty_months}")

    n_test = max(1, int(round(n_months * TEST_FRACTION)))
    split_idx = n_months - n_test
    print(f"Chronological split: {split_idx} train months, {n_test} test months")
    print(f"  Train: {all_months[0]} .. {all_months[split_idx-1]}")
    print(f"  Test:  {all_months[split_idx]} .. {all_months[-1]}")

    np.savez_compressed(
        os.path.join(DATA_DIR, "spatiotemporal_grid.npz"),
        tensor=tensor, mask=mask, lats=lats, lons=lons,
    )
    manifest = {
        "H": H, "W": W, "n_months": n_months, "channels": ["hcho", "fire_count", "wind_u", "wind_v"],
        "all_months": all_months, "split_idx": split_idx, "n_train_months": split_idx, "n_test_months": n_test,
        "empty_months": empty_months, "n_real_india_cells": int(mask.sum()),
    }
    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    grid[["grid_id", "lat", "lon", "row", "col"]].to_csv(os.path.join(DATA_DIR, "grid_lattice_index.csv"), index=False)

    print(f"\nSaved tensor {tensor.shape} -> {os.path.join(DATA_DIR, 'spatiotemporal_grid.npz')}")
    print(f"Saved manifest -> {os.path.join(DATA_DIR, 'spatiotemporal_grid_manifest.json')}")


if __name__ == "__main__":
    main()
