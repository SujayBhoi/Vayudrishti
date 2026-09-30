"""Proven-safe per-station extraction pattern for granule/orbit-heavy
sources (MODIS AOD, TROPOMI gases) - reuses v1's exact interactive
.getInfo() approach (legacy_v1_src/gee_extract.py: _periodic_reduce,
._fc_to_df), which never hit the batch-export projection bug documented in
batch_utils.build_scoped_periodic_point_table's docstring, because v1 never
used batch Export.table.toDrive for these sources - only interactive calls
scoped to one small station buffer at a time.

Generalizes v1's per-CITY loop to per-STATION (412 vs 13), with the same
disk-cache-per-station-per-variable resumability, plus light thread-pool
concurrency (GEE allows up to 40 concurrent interactive requests per
project) so 412 stations don't run fully sequentially.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import pandas as pd

PERIOD_DAYS = 7
N_WORKERS = 8


def _n_periods(start, end, period_days):
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    return int(-(-days // period_days))


def _periodic_reduce(ic, region, start, end, period_days, combine_reducer, region_reducer, scale):
    start_date = ee.Date(start)
    n = _n_periods(start, end, period_days)
    periods = ee.List.sequence(0, n - 1)

    def make_period(i):
        p_start = start_date.advance(ee.Number(i).multiply(period_days), "day")
        p_end = p_start.advance(period_days, "day")
        sub = ic.filterDate(p_start, p_end)
        img = combine_reducer(sub).set("period_start", p_start.format("YYYY-MM-dd"))
        return img

    imgs = ee.ImageCollection(periods.map(make_period))

    def reduce_fx(img):
        stats = img.reduceRegion(reducer=region_reducer, geometry=region, scale=scale, bestEffort=True, maxPixels=1e8)
        return ee.Feature(None, stats.set("date", img.get("period_start")))

    return ee.FeatureCollection(imgs.map(reduce_fx))


def _fc_to_df(fc, rename=None):
    info = fc.getInfo()
    rows = [f["properties"] for f in info["features"]]
    df = pd.DataFrame(rows)
    if rename:
        df = df.rename(columns=rename)
    return df


def extract_one_station(
    station_id: str, lat: float, lon: float, start: str, end: str,
    collection: str, bands: list[str], buffer_m: int, scale: int,
    mask_fn, rename: dict | None, out_dir: str, retries: int = 3,
) -> pd.DataFrame:
    safe_id = "".join(c if c.isalnum() else "_" for c in station_id)
    fname = os.path.join(out_dir, f"{safe_id}.csv")
    if os.path.exists(fname):
        return pd.read_csv(fname)

    region = ee.Geometry.Point(lon, lat).buffer(buffer_m)
    ic = ee.ImageCollection(collection).filterDate(start, end).filterBounds(region).select(bands)
    if mask_fn:
        ic = ic.map(mask_fn)

    for attempt in range(retries):
        try:
            fc = _periodic_reduce(
                ic, region, start, end, PERIOD_DAYS,
                combine_reducer=lambda sub: sub.mean(),
                region_reducer=ee.Reducer.mean(),
                scale=scale,
            )
            df = _fc_to_df(fc, rename=rename)
            df["station_id"] = station_id
            df.to_csv(fname, index=False)
            return df
        except Exception as e:
            print(f"  [retry {attempt+1}] {station_id}: {e}")
            time.sleep(5)
    raise RuntimeError(f"Extraction failed for {station_id}")


def extract_all_stations(
    stations: pd.DataFrame, start: str, end: str, collection: str, bands: list[str],
    buffer_m: int, scale: int, mask_fn, rename: dict | None, out_dir: str,
    n_workers: int = N_WORKERS, label: str = "",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    total = len(stations)
    done_count = 0

    def _run(row):
        return extract_one_station(row.station_id, row.lat, row.lon, start, end, collection, bands, buffer_m, scale, mask_fn, rename, out_dir)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run, row): row.station_id for row in stations.itertuples(index=False)}
        for fut in as_completed(futures):
            sid = futures[fut]
            done_count += 1
            try:
                df = fut.result()
                if done_count % 20 == 0 or done_count == total:
                    print(f"[{label}] {done_count}/{total} done (last: {sid}, {len(df)} rows)")
            except Exception as e:
                print(f"[{label}] FAILED {sid}: {e}")
