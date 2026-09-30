"""Loads Objective 1's India-wide spatial feature grid (fetch_objective1_grid.py's
per-source multi-band GeoTIFF exports) and samples each source's rasters at
the shared 0.25deg national grid (raw_v2/grids/india_grid_0.25deg.csv, 4448
points), producing a wide (grid_id, lat, lon, date) x variable feature table
with the same column names as the station-level training data.

Only reads the CORRECTED batches per source (see PROJECT_DOCUMENTATION.md
section 8 for the full audit):
- aod, era5, hcho: the "topup" batch (n=250, real evenly-spaced daily dates -
  the fix for bug 5.12). The earlier "plain" batch (n=100, one-date-per-month
  due to the pre-fix bug) is intentionally NOT read here - superseded, and
  gases never got a valid "plain" batch at all, so mixing it in would create
  a non-uniform date set across sources.
- no2/so2/co/o3: the "fixed" batch (n=250, same date set as "topup", with the
  corrected cloud_fraction/qa_value config from bug 5.3).
Both the original successful files and any "_retry" relaunches (this
session's fix for the Byte/Float64 bug 5.4b and the empty-image-guard
failures) are picked up by the same glob, since a batch index only ever has
ONE successful output file (a FAILED task produces nothing in Drive).
"""

from __future__ import annotations

import os
import re
from glob import glob

import numpy as np
import pandas as pd
import rasterio

GRID_DIR = os.path.dirname(__file__)
GRID_PATH = os.path.join(GRID_DIR, "raw_v2", "grids", "india_grid_0.25deg.csv")
DOWNLOAD_ROOT = os.path.join(GRID_DIR, "drive_downloads")

SOURCE_VARIANT = {
    "aod": "topup", "era5": "topup", "hcho": "topup",
    "no2": "fixed", "so2": "fixed", "co": "fixed", "o3": "fixed",
}

# raw exported band variable -> training-schema column name (matches
# fetch_tropomi.py's out_col mapping and merge_sources.py's aod rename)
COLUMN_RENAME = {
    "Optical_Depth_047": "aod_047",
    "Optical_Depth_055": "aod_055",
    "tropospheric_NO2_column_number_density": "no2",
    "SO2_column_number_density": "so2",
    "CO_column_number_density": "co",
    "O3_column_number_density": "o3",
    "tropospheric_HCHO_column_number_density": "hcho",
}

BAND_DATE_RE = re.compile(r"^(.*)_(\d{4})_(\d{2})_(\d{2})$")


def load_grid_points() -> pd.DataFrame:
    return pd.read_csv(GRID_PATH)


def _source_files(source_name: str) -> list[str]:
    variant = SOURCE_VARIANT[source_name]
    folder = os.path.join(DOWNLOAD_ROOT, f"ps3_obj1_grid_{source_name}")
    pattern = os.path.join(folder, f"v2_obj1_grid_{source_name}_{variant}*.tif")
    return sorted(glob(pattern))


def load_source_grid(source_name: str, grid_points: pd.DataFrame) -> pd.DataFrame:
    """Returns a long table: grid_id, lat, lon, date, <renamed variable columns>."""
    files = _source_files(source_name)
    if not files:
        raise FileNotFoundError(
            f"No downloaded files for source '{source_name}' matching variant "
            f"'{SOURCE_VARIANT[source_name]}' - run download_from_drive.py "
            f"--folder ps3_obj1_grid_{source_name} first."
        )

    xs = grid_points["lon"].to_numpy()
    ys = grid_points["lat"].to_numpy()
    coords = list(zip(xs, ys))

    # date -> {var: np.array of len(grid_points)}
    by_date: dict[str, dict[str, np.ndarray]] = {}

    for path in files:
        with rasterio.open(path) as src:
            band_names = src.descriptions
            nodata = src.nodata
            samples = np.array(list(src.sample(coords)), dtype="float64")  # (n_points, n_bands)
        if nodata is not None:
            samples = np.where(samples == nodata, np.nan, samples)
        for i, desc in enumerate(band_names):
            m = BAND_DATE_RE.match(desc or "")
            if not m:
                continue
            var, y, mo, d = m.groups()
            date_str = f"{y}-{mo}-{d}"
            by_date.setdefault(date_str, {})[var] = samples[:, i]

    frames = []
    for date_str, var_arrays in by_date.items():
        df = grid_points.copy()
        df["date"] = date_str
        for var, arr in var_arrays.items():
            df[COLUMN_RENAME.get(var, var)] = arr
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_grid_feature_table() -> pd.DataFrame:
    """Merges all 7 sources on (grid_id, date) into one wide table.

    Uses an INNER merge deliberately, not outer: a prediction row is only
    meaningful if every source has real data for that (grid_id, date) pair.
    An outer merge would keep rows where some sources are missing entirely
    (e.g. one source's download is still incomplete, or a source's date list
    doesn't perfectly align with another's for some other reason) - those
    rows would carry no lat/lon (lat/lon only travels with the first-merged
    source) and no real feature values from the missing sources, producing
    silently-garbage prediction rows rather than cleanly excluding them.
    Confirmed via a partial-download test run (2026-08-05): with sources at
    different download-completeness, an outer merge doubled the row count -
    every grid point gained a second, NaN-lat/lon row for dates only some
    sources had. Inner merge is correct both for this partial-data case AND
    the eventual complete-data case (where all 7 sources should agree on the
    exact same ~242-date set, so inner vs outer shouldn't even differ then -
    but inner is the fail-safe choice if any future re-run departs from that
    assumption for any reason)."""
    grid_points = load_grid_points()
    df = None
    for source_name in ["aod", "era5", "no2", "so2", "co", "o3", "hcho"]:
        src_df = load_source_grid(source_name, grid_points)
        if df is None:
            df = src_df
        else:
            merge_cols = [c for c in src_df.columns if c not in ("lat", "lon")]
            df = df.merge(src_df[merge_cols], on=["grid_id", "date"], how="inner")
    return df


if __name__ == "__main__":
    df = build_grid_feature_table()
    print(f"Grid feature table: {len(df)} rows, {df['grid_id'].nunique()} grid points, "
          f"{df['date'].nunique()} dates")
    print(df.columns.tolist())
    print(df.head())
