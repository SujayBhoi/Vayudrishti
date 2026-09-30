"""Objective 1: India-wide feature grid for spatial AQI prediction - pulls
AOD + 5 TROPOMI gases + ERA5 + fire as multi-band rasters at native
per-source resolution, for a representative date set sized against the
Checkpoint 0 EECU calibration (60 EECU-hr reserved).

Uses the same proven pattern as Objective 2's HCHO batch export
(fetch_objective2_grid.py): one multi-band GeoTIFF per date-batch, each
source exported at its own native resolution (not forced onto one uniform
grid - see CALIBRATION.md for why: forcing ERA5 to 1km cost 24x more than
its native ~9km for zero real precision gain). Each source gets its own
export task (different native resolution/crs per source), batched across
dates the same way as the HCHO grid to keep task count manageable.
"""

import argparse
import os

import ee
import pandas as pd

from fetch_objective2_grid import _evenly_spaced, chunk
from gee_auth import init_ee

# FIX: matches the correction already made in fetch_tropomi.py after the
# real error was observed there - "qa_value" doesn't exist as a band on
# ANY of these OFFL L3 collections (verified from actual GEE error output,
# not assumed). NO2/SO2/O3 use cloud_fraction like HCHO; CO has neither
# cloud_fraction nor qa_value (its SWIR retrieval is far less cloud-
# sensitive, standard to skip QA masking for it). This config wasn't kept in
# sync with that fix and caused 23 failed batch tasks here before being caught.
SOURCES = {
    "aod": dict(collection="MODIS/061/MCD19A2_GRANULES", bands=["Optical_Depth_047", "Optical_Depth_055"], aux_band=None, scale=1000, mask=lambda img: img.updateMask(img.select("Optical_Depth_047").gte(0)).multiply(0.001)),
    "no2": dict(collection="COPERNICUS/S5P/OFFL/L3_NO2", bands=["tropospheric_NO2_column_number_density"], aux_band="cloud_fraction", scale=1113, mask=lambda img: img.updateMask(img.select("cloud_fraction").lt(0.3))),
    "so2": dict(collection="COPERNICUS/S5P/OFFL/L3_SO2", bands=["SO2_column_number_density"], aux_band="cloud_fraction", scale=1113, mask=lambda img: img.updateMask(img.select("cloud_fraction").lt(0.3))),
    "co": dict(collection="COPERNICUS/S5P/OFFL/L3_CO", bands=["CO_column_number_density"], aux_band=None, scale=1113, mask=None),
    "o3": dict(collection="COPERNICUS/S5P/OFFL/L3_O3", bands=["O3_column_number_density"], aux_band="cloud_fraction", scale=1113, mask=lambda img: img.updateMask(img.select("cloud_fraction").lt(0.3))),
    "hcho": dict(collection="COPERNICUS/S5P/OFFL/L3_HCHO", bands=["tropospheric_HCHO_column_number_density"], aux_band="cloud_fraction", scale=1113, mask=lambda img: img.updateMask(img.select("cloud_fraction").lt(0.3))),
    "era5": dict(collection="ECMWF/ERA5_LAND/DAILY_AGGR", bands=["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m", "v_component_of_wind_10m"], aux_band=None, scale=9000, mask=None),
}


def select_representative_dates(n_dates: int, years=range(2018, 2026)) -> list[str]:
    """Spread across all years/seasons - unlike Objective 2, this isn't
    Oct-Dec weighted (that's a stubble-burning-specific choice); Objective
    1's spatial AQI map should represent typical conditions year-round.

    BUG FIX (same class of bug as Objective 2's original select_dates): the
    original version here only generated one candidate per (year, month) -
    8 years x 12 months = 96 candidates max, so any n_dates request above 96
    silently produced only 96 dates. This generates real daily candidates
    across the full range so n_dates up to len(candidates) is honored."""
    candidates = []
    for y in years:
        start = pd.Timestamp(year=y, month=1, day=1)
        end = pd.Timestamp(year=y, month=12, day=31)
        candidates.extend(pd.date_range(start, end, freq="D"))
    return [d.strftime("%Y-%m-%d") for d in _evenly_spaced(candidates, n_dates)]


def _band_for_date(cfg, india_geom, date_str, band_name, scale):
    # FIX: same "no images that day" guard added to Objective 2's HCHO
    # export after it crashed with "Image has no bands" - any date can have
    # zero valid images after filtering (not just pre-launch dates), which
    # otherwise breaks ee.Image.cat() when this band gets combined with others.
    d = pd.Timestamp(date_str)
    ic = ee.ImageCollection(cfg["collection"]).filterDate(d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    select_bands = list(cfg["bands"])
    if cfg["aux_band"] and cfg["aux_band"] not in select_bands:
        select_bands.append(cfg["aux_band"])
    ic = ic.select(select_bands)
    if cfg["mask"]:
        ic = ic.map(cfg["mask"])
    ic = ic.select(cfg["bands"])
    # .toFloat() matters: ee.Image.constant(0) defaults to Byte type, while
    # ic.mean() is Float64. Combining 10 dates via ee.Image.cat() into one
    # multi-band export fails ("inconsistent types: Byte and Float64") the
    # moment even one date in the batch needs the fallback - hit this on the
    # "fixed" NO2 batch after the qa_value fix, same underlying image-
    # construction pattern as Objective 2's HCHO fix but exposed here because
    # multiple dates get combined into a single multi-band image per export.
    fallback = ee.Image.constant(0).selfMask().rename(band_name).reproject(crs="EPSG:4326", scale=scale).toFloat()
    img = ee.Image(ee.Algorithms.If(ic.size().gt(0), ic.select(band_name).mean().toFloat(), fallback))
    safe_date = date_str.replace("-", "_")
    return img.clip(india_geom).rename(f"{band_name}_{safe_date}")


def export_source_batch(source_name, cfg, india_geom, date_strs, batch_tag):
    bands = []
    for d in date_strs:
        for b in cfg["bands"]:
            bands.append(_band_for_date(cfg, india_geom, d, b, cfg["scale"]))
    combined = ee.Image.cat(bands)
    task = ee.batch.Export.image.toDrive(
        image=combined, description=f"v2_obj1_grid_{source_name}_{batch_tag}",
        folder=f"ps3_obj1_grid_{source_name}", region=india_geom,
        scale=cfg["scale"], maxPixels=1e9, fileFormat="GeoTIFF",
    )
    task.start()
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-dates", type=int, required=True)
    parser.add_argument("--sources", nargs="*", default=list(SOURCES.keys()))
    parser.add_argument("--dates-per-batch", type=int, default=10)
    args = parser.parse_args()

    init_ee()
    india = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "India")).geometry()
    dates = select_representative_dates(args.n_dates)
    print(f"Selected {len(dates)} dates: {dates[:3]}...{dates[-3:]}")

    for source_name in args.sources:
        cfg = SOURCES[source_name]
        for i, batch in enumerate(chunk(dates, args.dates_per_batch)):
            t = export_source_batch(source_name, cfg, india, batch, batch_tag=f"{i:03d}")
            print(f"  {source_name} batch {i} ({len(batch)} dates): {t.id}")
