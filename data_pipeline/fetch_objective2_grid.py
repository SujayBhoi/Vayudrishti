"""Objective 2: India-wide HCHO raster + wind/fire grid, dense multi-year
coverage - the national replacement for v1's Punjab-Haryana-Delhi-UP
corridor-only version (legacy_v1_src/hotspot_grid_extract_continuous.py).

Rewritten after the AOD point-extraction incident (see CALIBRATION.md /
conversation record): HCHO is a granule/orbit-heavy source with the same
~0.09 EECU-hr-per-date compositing cost confirmed in Checkpoint 0's
calibration, REGARDLESS of whether it's extracted as scattered points or a
full raster - so it gets the same safe treatment as Objective 1's spatial
grid: one Export.image.toDrive raster per representative date (proven
working, no projection issues, cost known precisely), not a national
point-batch export. This is also the more natural fit for detect_hotspots.py
anyway (percentile threshold + DBSCAN on a real spatial field, not a coarse
point sample).

Wind (ERA5) and fire stay as national point/grid batch exports - both
confirmed cheap at national scope in Checkpoint 0's calibration (ERA5 native
~9km: 10.5 EECU-sec for a full-India date), so the original batched
approach is safe for those specifically.

Date selection is NOT hardcoded here - see select_dates() and the budget
note below; the actual date count is set from the real remaining EECU
budget once the Objective 1 station extraction's actual cost is known,
not assumed in advance.
"""

import argparse
import os

import ee
import pandas as pd

from batch_utils import build_periodic_point_table, export_table
from build_india_grid import build_grid
from gee_auth import init_ee

GRID_PATH = os.path.join(os.path.dirname(__file__), "raw_v2", "grids", "india_grid_0.25deg.csv")
HCHO_RASTER_DIR = "hcho_grid"


def _evenly_spaced(candidates: list, budget: int) -> list:
    """Pick `budget` items evenly spaced across `candidates` (not just a
    fixed-step slice from the front, which biases toward earlier items when
    len(candidates) isn't a clean multiple of budget)."""
    if budget >= len(candidates):
        return candidates
    if budget <= 0:
        return []
    idx = [round(i * (len(candidates) - 1) / (budget - 1)) for i in range(budget)] if budget > 1 else [0]
    return [candidates[i] for i in sorted(set(idx))]


def select_dates(n_dates: int, years=range(2018, 2026)) -> list[str]:
    """Weight toward Oct-Dec (peak stubble season) with broader contrast
    coverage across other months, per the plan. BUG FIX: the first version
    of this function only generated one candidate date per (year, month) -
    24 Oct-Dec candidates and 72 other-month candidates total, 96 max
    regardless of n_dates - so a 330-date request silently produced only 96.
    This version generates real DAILY candidates within Oct-Dec (true dense
    coverage of the actual candidate pool) and evenly-spaced picks from both
    pools sized to the requested budget."""
    oct_dec_budget = int(n_dates * 0.6)
    other_budget = n_dates - oct_dec_budget

    oct_dec_candidates = []
    for y in years:
        for m in [10, 11, 12]:
            n_days = pd.Timestamp(year=y, month=m, day=1).days_in_month
            for day in range(1, n_days + 1):
                oct_dec_candidates.append(pd.Timestamp(year=y, month=m, day=day))
    oct_dec_sel = _evenly_spaced(oct_dec_candidates, oct_dec_budget)

    other_candidates = []
    for y in years:
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            for day in (5, 15, 25):
                other_candidates.append(pd.Timestamp(year=y, month=m, day=day))
    other_sel = _evenly_spaced(other_candidates, other_budget)

    all_dates = sorted(oct_dec_sel + other_sel)
    return [d.strftime("%Y-%m-%d") for d in all_dates]


def _hcho_band_for_date(india_geom, date_str: str) -> ee.Image:
    """FIX: dates before S5P HCHO's actual launch (2018-04-30) - or any date
    with zero valid images after cloud filtering - make ic.mean() return a
    bandless image, which crashes ee.Image.cat() downstream with "Image has
    no bands" (this broke topup batch 0, which spanned 2018-01-05..
    2018-10-08 and included several pre-launch dates). v1's original
    hotspot_grid_extract.py guarded against exactly this with an
    ee.Algorithms.If fallback - replicated here."""
    d = pd.Timestamp(date_str)
    safe_name = date_str.replace("-", "_")
    ic = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_HCHO")
        .filterDate(d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        .select(["tropospheric_HCHO_column_number_density", "cloud_fraction"])
        .map(lambda img: img.updateMask(img.select("cloud_fraction").lt(0.3)))
        .select(["tropospheric_HCHO_column_number_density"])
    )
    fallback = ee.Image.constant(0).selfMask().rename("tropospheric_HCHO_column_number_density").reproject(crs="EPSG:4326", scale=1113)
    img = ee.Image(ee.Algorithms.If(ic.size().gt(0), ic.mean(), fallback))
    return img.clip(india_geom).rename(f"hcho_{safe_name}")


def export_hcho_raster_batch(india_geom, date_strs: list[str], batch_tag: str, folder="ps3_obj2_hcho"):
    """One multi-band GeoTIFF per call (one band per date) instead of one
    task per date - same per-date EECU cost, far fewer tasks to queue
    through this project's apparent single-concurrent-batch-task limit."""
    bands = [_hcho_band_for_date(india_geom, d) for d in date_strs]
    combined = ee.Image.cat(bands)
    task = ee.batch.Export.image.toDrive(
        image=combined, description=f"v2_obj2_hcho_batch_{batch_tag}", folder=folder,
        region=india_geom, scale=1113, maxPixels=1e9, fileFormat="GeoTIFF",
    )
    task.start()
    return task


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def fetch_wind_grid(points_fc, start, end):
    bands = ["u_component_of_wind_10m", "v_component_of_wind_10m"]
    ic = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end).select(bands)
    fc = build_periodic_point_table(
        ic, points_fc, start, end, 7,
        combine_reducer=lambda sub: sub.mean(),
        region_reducer=ee.Reducer.mean(),
        scale=9000,
    )
    return export_table(fc, description=f"v2_obj2_wind_grid_{start}_{end}")


def fetch_fire_grid(points_fc, start, end):
    from fetch_fire import get_india_fire_collection
    ic = get_india_fire_collection(start, end).select(["FireMask"])
    fc = build_periodic_point_table(
        ic, points_fc.map(lambda f: f.buffer(50_000)), start, end, 7,
        combine_reducer=lambda sub: sub.mosaic(),
        region_reducer=ee.Reducer.count(),
        scale=1000,
    )
    return export_table(fc, description=f"v2_obj2_fire_grid_{start}_{end}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-hcho-dates", type=int, required=True, help="set from actual remaining EECU budget, not a default guess")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--do-wind-fire", action="store_true")
    args = parser.parse_args()

    init_ee()
    india = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "India")).geometry()

    dates = select_dates(args.n_hcho_dates)
    print(f"Selected {len(dates)} HCHO dates (Oct-Dec weighted): {dates[:5]}...{dates[-5:]}")
    for i, batch in enumerate(chunk(dates, 15)):
        t = export_hcho_raster_batch(india, batch, batch_tag=f"{i:03d}")
        print(f"  queued HCHO batch {i} ({len(batch)} dates, {batch[0]}..{batch[-1]}): {t.id}")

    if args.do_wind_fire:
        grid = build_grid(0.25)
        os.makedirs(os.path.dirname(GRID_PATH), exist_ok=True)
        grid.to_csv(GRID_PATH, index=False)
        points_fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(float(r.lon), float(r.lat)), {"grid_id": r.grid_id}) for r in grid.itertuples()])
        t_wind = fetch_wind_grid(points_fc, args.start, args.end)
        print("wind grid task:", t_wind.id)
        t_fire = fetch_fire_grid(points_fc, args.start, args.end)
        print("fire grid task:", t_fire.id)
