"""Loads the wind and fire India-wide grid exports (fetch_objective2_grid.py's
fetch_wind_grid/fetch_fire_grid batch outputs) and joins them against the
local grid_id->lat/lon table, since the exported .geo column carries a full
polygon per row for the fire grid (buffered 50km before reduceRegions) -
wasteful (1.85GB for what should be tens of MB) but not wrong; cheaper to
work around locally than re-export.

IMPORTANT schema note: this fire data is a per-0.25deg-cell WEEKLY COUNT of
fire pixels within a 50km buffer (matching v1's original grid-density
approach - legacy_v1_src/hotspot_grid_extract.py's fire_density via focal
sum), not individual fire-pixel detections like the teammate's notebook
used. cluster_fires.py's build_attribution_features() was written assuming
individual points; here each fire-active grid cell (count>0) for a given
date is treated as one fire "point" at its cell centroid - a reasonable,
scientifically defensible proxy (it's literally what v1's more mature
Objective 2 pipeline did), not a raw-pixel match.
"""

import os

import pandas as pd

DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "drive_downloads", "ps3_v2_exports")
GRID_PATH = os.path.join(os.path.dirname(__file__), "raw_v2", "grids", "india_grid_0.25deg.csv")


def load_grid_lookup() -> pd.DataFrame:
    return pd.read_csv(GRID_PATH)


def load_wind_grid() -> pd.DataFrame:
    path = os.path.join(DOWNLOAD_DIR, "v2_obj2_wind_grid_2018-01-01_2026-01-01.csv")
    df = pd.read_csv(path, usecols=["grid_id", "period_start", "u_component_of_wind_10m", "v_component_of_wind_10m"])
    df = df.rename(columns={"period_start": "date", "u_component_of_wind_10m": "u", "v_component_of_wind_10m": "v"})
    grid = load_grid_lookup()
    df = df.merge(grid, on="grid_id", how="left")
    return df


def load_fire_grid(min_count: int = 1) -> pd.DataFrame:
    path = os.path.join(DOWNLOAD_DIR, "v2_obj2_fire_grid_2018-01-01_2026-01-01.csv")
    df = pd.read_csv(path, usecols=["grid_id", "period_start", "count"])
    df = df.rename(columns={"period_start": "date"})
    df = df[df["count"] >= min_count]  # only fire-active cells become "fire points"
    grid = load_grid_lookup()
    df = df.merge(grid, on="grid_id", how="left")
    return df


if __name__ == "__main__":
    wind = load_wind_grid()
    print(f"Wind grid: {len(wind)} rows, {wind['date'].nunique()} dates")
    fire = load_fire_grid()
    print(f"Fire-active grid cells: {len(fire)} rows, {fire['date'].nunique()} dates")
    print(fire.head())
