"""Overpass-windowed daily satellite extraction for the clean101 comparison.

Instead of interactive_point_extract.py's weekly ic.mean() composite, this
does a DAILY composite restricted to images whose acquisition time falls in
the confirmed overpass window (11:18-15:18 IST == 05:48-09:48 UTC, since IST
has no DST and is a fixed +5:30 offset - TROPOMI ~13:08 IST, MODIS/Aqua
~13:10-13:45 IST, verified via direct GEE system:time_start queries).

Reuses _periodic_reduce/_fc_to_df from interactive_point_extract.py (same
proven-safe per-station filterBounds pattern, same collections/bands/masks
as fetch_modis_aod.py/fetch_tropomi.py) with period_days=1 and a custom
combine_reducer that filters to the overpass window before averaging, with
an empty-window guard (a day with zero in-window images gets a masked
fallback, matching the project's existing 5.4 empty-collection fix -
including the 5.4b Byte/Float64 type-consistency fix).
"""

from __future__ import annotations

import argparse
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
from interactive_point_extract import _fc_to_df, _n_periods  # noqa: E402

# window: 11:18-15:18 IST -> UTC minute-of-day (IST = UTC+5:30, fixed offset)
WINDOW_START_UTC_MIN = 11 * 60 + 18 - 330  # 348 (05:48 UTC)
WINDOW_END_UTC_MIN = 15 * 60 + 18 - 330    # 588 (09:48 UTC)

N_WORKERS = 8

SOURCES = {
    "aod": dict(
        collection="MODIS/061/MCD19A2_GRANULES",
        bands=["Optical_Depth_047", "Optical_Depth_055"],
        scale=1000,
        mask_fn=lambda img: img.updateMask(img.select("Optical_Depth_047").gte(0)).multiply(0.001).copyProperties(img, ["system:time_start"]),
        rename={"Optical_Depth_047": "aod_047", "Optical_Depth_055": "aod_055"},
    ),
    "no2": dict(collection="COPERNICUS/S5P/OFFL/L3_NO2", bands=["tropospheric_NO2_column_number_density", "cloud_fraction"], scale=1113,
                mask_fn=lambda img: img.updateMask(img.select("cloud_fraction").lt(0.3)).select(["tropospheric_NO2_column_number_density"]).copyProperties(img, ["system:time_start"]),
                rename={"tropospheric_NO2_column_number_density": "no2"}),
    "so2": dict(collection="COPERNICUS/S5P/OFFL/L3_SO2", bands=["SO2_column_number_density", "cloud_fraction"], scale=1113,
                mask_fn=lambda img: img.updateMask(img.select("cloud_fraction").lt(0.3)).select(["SO2_column_number_density"]).copyProperties(img, ["system:time_start"]),
                rename={"SO2_column_number_density": "so2"}),
    "co": dict(collection="COPERNICUS/S5P/OFFL/L3_CO", bands=["CO_column_number_density"], scale=1113,
               mask_fn=None, rename={"CO_column_number_density": "co"}),
    "o3": dict(collection="COPERNICUS/S5P/OFFL/L3_O3", bands=["O3_column_number_density", "cloud_fraction"], scale=1113,
               mask_fn=lambda img: img.updateMask(img.select("cloud_fraction").lt(0.3)).select(["O3_column_number_density"]).copyProperties(img, ["system:time_start"]),
               rename={"O3_column_number_density": "o3"}),
    "hcho": dict(collection="COPERNICUS/S5P/OFFL/L3_HCHO", bands=["tropospheric_HCHO_column_number_density", "cloud_fraction"], scale=1113,
                 mask_fn=lambda img: img.updateMask(img.select("cloud_fraction").lt(0.3)).select(["tropospheric_HCHO_column_number_density"]).copyProperties(img, ["system:time_start"]),
                 rename={"tropospheric_HCHO_column_number_density": "hcho"}),
}

BUFFER_M = 10_000


def _add_utc_minute_of_day(img):
    d = ee.Date(img.get("system:time_start"))
    minute_of_day = d.get("hour").multiply(60).add(d.get("minute"))
    return img.set("utc_minute_of_day", minute_of_day)


def _overpass_windowed_daily(ic, region, start, end, region_reducer, scale, n_bands):
    start_date = ee.Date(start)
    n = _n_periods(start, end, 1)
    periods = ee.List.sequence(0, n - 1)

    def make_period(i):
        p_start = start_date.advance(ee.Number(i), "day")
        p_end = p_start.advance(1, "day")
        sub = ic.filterDate(p_start, p_end).map(_add_utc_minute_of_day)
        windowed = sub.filter(ee.Filter.rangeContains("utc_minute_of_day", WINDOW_START_UTC_MIN, WINDOW_END_UTC_MIN))
        real_mean = windowed.mean().toFloat()
        fallback = ee.Image.constant([0] * n_bands).rename(ic.first().bandNames()).selfMask().toFloat()
        img = ee.Image(ee.Algorithms.If(windowed.size().gt(0), real_mean, fallback))
        return img.set({"period_start": p_start.format("YYYY-MM-dd"), "n_in_window": windowed.size()})

    imgs = ee.ImageCollection(periods.map(make_period))

    def reduce_fx(img):
        stats = img.reduceRegion(reducer=region_reducer, geometry=region, scale=scale, bestEffort=True, maxPixels=1e8)
        stats = stats.set("date", img.get("period_start")).set("n_in_window", img.get("n_in_window"))
        return ee.Feature(None, stats)

    return ee.FeatureCollection(imgs.map(reduce_fx))


def _year_chunks(start: str, end: str):
    """Split [start,end) into <=365-day chunks - a single interactive call
    spanning ~7 years hits GEE's 'User memory limit exceeded' (confirmed by
    direct test); one year (365 daily periods) confirmed safe (~51s/call)."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    chunks = []
    cur = s
    while cur < e:
        nxt = min(cur + pd.Timedelta(days=365), e)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return chunks


def extract_one_station_source(station_id, lat, lon, start, end, source, retries=3, call_delay: float = 0.0):
    cfg = SOURCES[source]
    out_dir = os.path.join(V3_ROOT, "data", "raw_overpass", source)
    os.makedirs(out_dir, exist_ok=True)
    safe_id = "".join(c if c.isalnum() else "_" for c in station_id)
    fname = os.path.join(out_dir, f"{safe_id}.csv")
    if os.path.exists(fname):
        return pd.read_csv(fname)

    region = ee.Geometry.Point(lon, lat).buffer(BUFFER_M)
    if cfg["mask_fn"]:
        out_bands = [b for b in cfg["bands"] if b in cfg["rename"]]
    else:
        out_bands = cfg["bands"]
    n_bands = len(out_bands)

    year_dfs = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        ic = ee.ImageCollection(cfg["collection"]).filterDate(chunk_start, chunk_end).filterBounds(region).select(cfg["bands"])
        if cfg["mask_fn"]:
            ic = ic.map(cfg["mask_fn"])
        for attempt in range(retries):
            if call_delay > 0:
                # PROACTIVE throttle before every call, not just reactive backoff
                # after a failure - same fix pattern as this project's existing
                # Open-Meteo BLH rate-limit incident (root doc, section 5.9).
                time.sleep(call_delay)
            try:
                fc = _overpass_windowed_daily(ic, region, chunk_start, chunk_end, region_reducer=ee.Reducer.mean(), scale=cfg["scale"], n_bands=n_bands)
                df = _fc_to_df(fc, rename=cfg["rename"])
                year_dfs.append(df)
                break
            except Exception as e:
                print(f"  [retry {attempt+1}] {station_id}/{source}/{chunk_start}: {e}")
                time.sleep(15)  # bumped from 5s - recovering from a quota/Restricted-Mode incident, not just a transient blip
        else:
            raise RuntimeError(f"Extraction failed for {station_id}/{source}/{chunk_start}")

    full = pd.concat(year_dfs, ignore_index=True)
    full["station_id"] = station_id
    full.to_csv(fname, index=False)
    return full


def extract_all(stations: pd.DataFrame, start: str, end: str, sources: list[str], n_workers: int = N_WORKERS, call_delay: float = 0.0):
    for source in sources:
        total = len(stations)
        done = 0

        def _run(row, source=source):
            return extract_one_station_source(row.station_id, row.lat, row.lon, start, end, source, call_delay=call_delay)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run, row): row.station_id for row in stations.itertuples(index=False)}
            for fut in as_completed(futures):
                sid = futures[fut]
                done += 1
                try:
                    df = fut.result()
                    if done % 10 == 0 or done == total:
                        print(f"[{source}] {done}/{total} done (last: {sid}, {len(df)} rows)")
                except Exception as e:
                    print(f"[{source}] FAILED {sid}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", default=list(SOURCES.keys()))
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--workers", type=int, default=2, help="reduced from the original 8 after the Restricted Mode incident - see PROJECT_DOCUMENTATION.md section 4.1.1")
    parser.add_argument("--call-delay", type=float, default=3.0, help="proactive seconds to sleep before every interactive call")
    parser.add_argument("--n-stations", type=int, default=None, help="limit to first N stations, for testing")
    parser.add_argument("--project", default=None, help="GCP/EE project ID - defaults to gee_auth.py's DEFAULT_PROJECT (isro-bah-2026). Pass a fresh project ID to bypass that project's Restricted Mode (see PROJECT_DOCUMENTATION.md section 4.1.1) - independent quota, confirmed via GEE's own docs.")
    args = parser.parse_args()

    if args.project:
        init_ee(project=args.project)
    else:
        init_ee()

    import json
    with open(os.path.join(V3_ROOT, "data", "clean101_report.json")) as f:
        report = json.load(f)
    station_ids = report["final_stations"]
    master = pd.read_csv(os.path.join(TRIAL02_ROOT, "data_pipeline", "legacy_v1_data", "processed", "cpcb_v2_station_master.csv"))
    stations = master[master.station_id.isin(station_ids)][["station_id", "lat", "lon"]].reset_index(drop=True)
    if args.n_stations:
        stations = stations.head(args.n_stations)

    print(f"Extracting {args.sources} for {len(stations)} stations, {args.start}..{args.end}, workers={args.workers}, call_delay={args.call_delay}s")
    t0 = time.time()
    extract_all(stations, args.start, args.end, args.sources, n_workers=args.workers, call_delay=args.call_delay)
    print(f"Done in {time.time()-t0:.0f}s")
