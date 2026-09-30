"""Builds the fixed India-wide sample-point grid shared by:
  - Objective 2's HCHO hotspot detection (point-sampled, cheap per the quota
    analysis, so this can run at genuinely national scope)
  - Objective 1's spatial AQI prediction grid, at whatever resolution the
    EECU calibration in this checkpoint determines is affordable for full
    raster export (see models/objective1_aqi/ calibration notes)

Grid is generated in Python (not GEE) using the India boundary from GEE's
LSIB dataset, so the same point list is reusable across both objectives and
across many fetch scripts without repeated GEE geometry calls.
"""

import argparse
import os

import ee
import numpy as np
import pandas as pd

from gee_auth import init_ee

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw_v2", "grids")
os.makedirs(OUT_DIR, exist_ok=True)


def build_grid(resolution_deg: float) -> pd.DataFrame:
    init_ee()
    india = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "India")).geometry()
    bounds = india.bounds().getInfo()["coordinates"][0]
    lons = [c[0] for c in bounds]
    lats = [c[1] for c in bounds]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    lon_vals = np.arange(lon_min, lon_max, resolution_deg)
    lat_vals = np.arange(lat_min, lat_max, resolution_deg)
    grid_lon, grid_lat = np.meshgrid(lon_vals, lat_vals)
    candidate = pd.DataFrame({"lon": grid_lon.ravel(), "lat": grid_lat.ravel()})

    # Keep only points that actually fall inside India (bounds box includes ocean/neighbors)
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point(r.lon, r.lat), {"idx": i}) for i, r in enumerate(candidate.itertuples())
    ])
    inside = fc.filterBounds(india)
    kept_idx = [f["properties"]["idx"] for f in inside.getInfo()["features"]]
    grid = candidate.loc[kept_idx].reset_index(drop=True)
    grid["grid_id"] = [f"g{i:06d}" for i in range(len(grid))]
    return grid[["grid_id", "lat", "lon"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution-deg", type=float, default=0.25,
                         help="grid spacing in degrees; ~0.25deg ~= 28km, matches ERA5 native")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    grid = build_grid(args.resolution_deg)
    out = args.out or os.path.join(OUT_DIR, f"india_grid_{args.resolution_deg}deg.csv")
    grid.to_csv(out, index=False)
    print(f"Built {len(grid)} grid points at {args.resolution_deg} deg resolution -> {out}")
