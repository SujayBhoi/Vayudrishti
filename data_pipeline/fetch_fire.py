"""v2: MODIS/VIIRS fire-pixel count weekly time series at all v2 CPCB
station points (Objective 1 feature), and the shared India-wide fire feed
used by Objective 2's attribution step.

Station-side uses the same proven-safe interactive per-station pattern as
the other v2 fetch scripts (see fetch_modis_aod.py's docstring for why).
"""

import argparse
import time

import ee

from fetch_cpcb import load_station_master
from gee_auth import init_ee
from interactive_point_extract import extract_all_stations

REGIONAL_BUFFER_M = 50_000


def mask_fire(img):
    return img.updateMask(img.select("T21").gte(0))


def fetch_station_fire(start, end, workers=8):
    stations = load_station_master()
    print(f"Extracting station fire counts for {len(stations)} stations, {start}..{end}")
    t0 = time.time()
    extract_all_stations(
        stations, start, end,
        collection="FIRMS",
        bands=["T21", "confidence"],
        buffer_m=REGIONAL_BUFFER_M, scale=1000,
        mask_fn=None,
        rename={"T21": "fire_t21", "confidence": "fire_confidence"},
        out_dir="data_pipeline/raw_v2/fire",
        n_workers=workers, label="FIRE",
    )
    print(f"Station fire extraction complete in {time.time()-t0:.0f}s")


def get_india_fire_collection(start, end):
    """India-wide fire detections for Objective 2 (used directly by that
    pipeline, not exported here)."""
    india = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "India")).geometry()
    mod = ee.ImageCollection("MODIS/061/MOD14A1").filterDate(start, end).filterBounds(india).select("FireMask")
    viirs = ee.ImageCollection("NASA/VIIRS/002/VNP14A1").filterDate(start, end).filterBounds(india).select("FireMask")
    return mod.merge(viirs).map(lambda img: img.updateMask(img.gte(7)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    init_ee()
    fetch_station_fire(args.start, args.end, workers=args.workers)
