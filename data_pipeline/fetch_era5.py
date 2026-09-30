"""v2: ERA5-Land meteorology weekly time series at all v2 CPCB station points.

Uses the same proven-safe interactive per-station pattern as
fetch_modis_aod.py, for consistency - ERA5 is NOT a granule-heavy source
(confirmed cheap even at full national extent in Checkpoint 0's
calibration: 10.5 EECU-sec for a native-scale full-India export), so the
original batched national approach was likely fine for this source
specifically, but after the AOD incident "likely fine" isn't good enough -
every source now goes through the same proven path rather than being
individually re-justified.
"""

import argparse
import time

from fetch_cpcb import load_station_master
from gee_auth import init_ee
from interactive_point_extract import extract_all_stations

BUFFER_M = 10_000
BANDS = ["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m", "v_component_of_wind_10m"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    init_ee()
    stations = load_station_master()
    print(f"Extracting ERA5 for {len(stations)} stations, {args.start}..{args.end}")

    t0 = time.time()
    extract_all_stations(
        stations, args.start, args.end,
        collection="ECMWF/ERA5_LAND/DAILY_AGGR",
        bands=BANDS,
        buffer_m=BUFFER_M, scale=9000,
        mask_fn=None,
        rename=None,
        out_dir="data_pipeline/raw_v2/era5",
        n_workers=args.workers, label="ERA5",
    )
    print(f"ERA5 extraction complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
