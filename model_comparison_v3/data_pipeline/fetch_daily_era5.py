"""Fresh DAILY ERA5-Land extraction for the clean101 comparison.

The existing v2 ERA5 extraction (data_pipeline/data_pipeline/raw_v2/era5/)
is weekly-composited (verified directly: rows are exactly 7 days apart),
even though the source collection (ECMWF/ERA5_LAND/DAILY_AGGR) is itself
daily. This re-extracts at period_days=1 instead of 7, for the 101 clean
stations only. Output goes to a NEW directory (not the existing weekly
one) - the existing weekly extraction is never touched.

ERA5-Land is a simple gridded reanalysis product (no compositing-cost risk
like AOD/TROPOMI granules), and DAILY_AGGR is already a full-day aggregate
(no finer sub-day selection is possible from this product), so no
overpass-window filtering is applied - just a plain daily value per
station-day.

Year-chunked (same reason as fetch_overpass_windowed_satellite.py): a
single interactive call spanning the full ~7-year range at period_days=1
(~2,557 daily periods) hits GEE's "User memory limit exceeded" - the same
failure mode already confirmed for the satellite extraction. The existing
extract_one_station/extract_all_stations don't chunk by year internally, so
this reimplements the per-station loop directly (reusing _periodic_reduce/
_fc_to_df, same as the satellite script) rather than relying on that
higher-level, non-chunking API.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
TRIAL02_ROOT = os.path.dirname(V3_ROOT)

sys.path.insert(0, os.path.join(TRIAL02_ROOT, "data_pipeline"))
from gee_auth import init_ee  # noqa: E402
from interactive_point_extract import _periodic_reduce, _fc_to_df  # noqa: E402

BANDS = ["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m", "v_component_of_wind_10m"]
OUT_DIR = os.path.join(V3_ROOT, "data", "raw_daily_era5")
BUFFER_M = 10_000
COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
SCALE = 9000
N_WORKERS = 8


def _year_chunks(start: str, end: str):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    chunks, cur = [], s
    while cur < e:
        nxt = min(cur + pd.Timedelta(days=365), e)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return chunks


def extract_one_station(station_id, lat, lon, start, end, retries=3):
    os.makedirs(OUT_DIR, exist_ok=True)
    safe_id = "".join(c if c.isalnum() else "_" for c in station_id)
    fname = os.path.join(OUT_DIR, f"{safe_id}.csv")
    if os.path.exists(fname):
        return pd.read_csv(fname)

    region = ee.Geometry.Point(lon, lat).buffer(BUFFER_M)
    ic = ee.ImageCollection(COLLECTION).filterDate(start, end).filterBounds(region).select(BANDS)

    year_dfs = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        for attempt in range(retries):
            try:
                fc = _periodic_reduce(
                    ic, region, chunk_start, chunk_end, period_days=1,
                    combine_reducer=lambda sub: sub.mean(),
                    region_reducer=ee.Reducer.mean(), scale=SCALE,
                )
                df = _fc_to_df(fc, rename=None)
                year_dfs.append(df)
                break
            except Exception as e:
                print(f"  [retry {attempt+1}] {station_id}/{chunk_start}: {e}")
                time.sleep(5)
        else:
            raise RuntimeError(f"Extraction failed for {station_id}/{chunk_start}")

    full = pd.concat(year_dfs, ignore_index=True)
    full["station_id"] = station_id
    full.to_csv(fname, index=False)
    return full


def main():
    init_ee()
    with open(os.path.join(V3_ROOT, "data", "clean101_report.json")) as f:
        report = json.load(f)
    station_ids = report["final_stations"]
    master = pd.read_csv(os.path.join(TRIAL02_ROOT, "data_pipeline", "legacy_v1_data", "processed", "cpcb_v2_station_master.csv"))
    stations = master[master.station_id.isin(station_ids)][["station_id", "lat", "lon"]].reset_index(drop=True)
    print(f"Extracting daily ERA5 for {len(stations)} stations, year-chunked")

    t0 = time.time()
    total = len(stations)
    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(extract_one_station, row.station_id, row.lat, row.lon, "2019-01-01", "2026-01-01"): row.station_id for row in stations.itertuples(index=False)}
        for fut in as_completed(futures):
            sid = futures[fut]
            done += 1
            try:
                df = fut.result()
                if done % 10 == 0 or done == total:
                    print(f"[ERA5-daily] {done}/{total} done (last: {sid}, {len(df)} rows)")
            except Exception as e:
                print(f"[ERA5-daily] FAILED {sid}: {e}")
    print(f"Daily ERA5 extraction complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
