"""v2: MODIS AOD (MCD19A2) weekly time series at all v2 CPCB station points.

Rewritten after a real incident (see CALIBRATION.md / conversation record):
the original batched Export.table.toDrive design burned 10.2 EECU-hr in
under a minute because compositing AOD granules across all 412 scattered
stations forces near-full-country compositing every week. A follow-up fix
attempt (per-station-scoped batch export) hit a separate, persistent
"Unable to compute intersection of geometries" projection error specific to
MCD19A2's sinusoidal tile grid in batch-export context, that survived
several fix attempts (explicit crs=, .reproject(), mosaic vs mean).

This version instead reuses v1's ORIGINAL, proven-working pattern
(legacy_v1_src/gee_extract.py: interactive .getInfo() per station, each
scoped to a small buffer via filterBounds) - see interactive_point_extract.py
- generalized from 13 cities to all 412 stations with light thread-pool
concurrency. Confirmed working at small scale (10 stations, 3 months, 29s)
before this full run.
"""

import argparse
import time

import ee

from fetch_cpcb import load_station_master
from gee_auth import init_ee
from interactive_point_extract import extract_all_stations

OUT_DIR = "data_pipeline/raw_v2/aod"
BUFFER_M = 10_000


def mask_aod(img):
    return img.updateMask(img.select("Optical_Depth_047").gte(0)).multiply(0.001).copyProperties(img, ["system:time_start"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    init_ee()
    stations = load_station_master()
    print(f"Extracting AOD for {len(stations)} stations, {args.start}..{args.end}")

    t0 = time.time()
    extract_all_stations(
        stations, args.start, args.end,
        collection="MODIS/061/MCD19A2_GRANULES",
        bands=["Optical_Depth_047", "Optical_Depth_055"],
        buffer_m=BUFFER_M, scale=1000,
        mask_fn=mask_aod,
        rename={"Optical_Depth_047": "aod_047", "Optical_Depth_055": "aod_055"},
        out_dir=OUT_DIR,
        n_workers=args.workers, label="AOD",
    )
    print(f"AOD extraction complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
