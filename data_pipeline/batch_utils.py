"""Shared helpers for multi-point GEE time-series extraction.

v1 (legacy_v1_src/gee_extract.py) looped one ee.Geometry.Point at a time,
calling .reduceRegion() + .getInfo() per (city, variable) — fine for 13
cities, but far too slow sequentially for 417+ stations.

v2 batches all station points into a single ee.FeatureCollection and uses
.reduceRegions() (plural) so every point is reduced in one server-side call
per image/period. Combined with weekly bucketing (same reasoning as v1 — see
that file's docstring on the getInfo 5000-element cap) and a single
Export.table.toDrive task per variable, this turns "417 stations x N years"
into one batch task instead of thousands of interactive calls, which is both
far faster and matches GEE's guidance to prefer fewer, larger batch tasks
over many small ones.
"""

from __future__ import annotations

import ee
import pandas as pd


def n_periods(start: str, end: str, period_days: int) -> int:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    return int(-(-days // period_days))  # ceil


def stations_to_fc(stations: pd.DataFrame, id_col: str = "station_id") -> ee.FeatureCollection:
    """stations must have columns [id_col, 'lat', 'lon']."""
    feats = [
        ee.Feature(ee.Geometry.Point(float(row.lon), float(row.lat)), {"station_id": str(getattr(row, id_col))})
        for row in stations.itertuples(index=False)
    ]
    return ee.FeatureCollection(feats)


def build_periodic_point_table(
    ic: ee.ImageCollection,
    points_fc: ee.FeatureCollection,
    start: str,
    end: str,
    period_days: int,
    combine_reducer,
    region_reducer: ee.Reducer,
    scale: int,
) -> ee.FeatureCollection:
    """Bucket ic into periods, combine each bucket into one image, then
    reduceRegions once per bucket across ALL points simultaneously.
    Returns a flattened FeatureCollection with one feature per (point, period).
    """
    start_date = ee.Date(start)
    n = n_periods(start, end, period_days)
    periods = ee.List.sequence(0, n - 1)

    def make_period(i):
        p_start = start_date.advance(ee.Number(i).multiply(period_days), "day")
        p_end = p_start.advance(period_days, "day")
        sub = ic.filterDate(p_start, p_end)
        img = combine_reducer(sub)
        reduced = img.reduceRegions(collection=points_fc, reducer=region_reducer, scale=scale)
        return reduced.map(lambda f: f.set("period_start", p_start.format("YYYY-MM-dd")))

    return ee.FeatureCollection(periods.map(make_period)).flatten()


def build_scoped_periodic_point_table(
    ic: ee.ImageCollection,
    stations: "pd.DataFrame",
    buffer_m: int,
    start: str,
    end: str,
    period_days: int,
    combine_reducer,
    region_reducer: ee.Reducer,
    scale: int,
    id_col: str = "station_id",
) -> ee.FeatureCollection:
    """Per-station-scoped version of build_periodic_point_table.

    INCIDENT (Checkpoint 1): the unscoped version above builds one composite
    image per period across the WHOLE points_fc extent before reducing to
    points. For granule/orbit-heavy sources (MODIS AOD, TROPOMI) with points
    scattered nationwide, that composite step silently forces full-country
    compositing every period - it burned 10.2 EECU-hr in about a minute
    before being caught and cancelled. v1's original per-city loop
    (legacy_v1_src/gee_extract.py) never had this problem because it called
    .filterBounds(region) with a small per-city buffer BEFORE compositing,
    letting GEE prune to only nearby granules.

    This function restores that scoping while still producing one batch
    export task (not one task per station): map over stations OUTERMOST, and
    for each station independently filterBounds to its own small buffer
    before doing the periodic composite+reduce, matching v1's proven-cheap
    cost profile per station. Flatten all stations' results together.
    """
    start_date = ee.Date(start)
    n = n_periods(start, end, period_days)
    periods = ee.List.sequence(0, n - 1)

    def per_station(row):
        geom = ee.Geometry.Point(float(row.lon), float(row.lat)).buffer(buffer_m)
        station_id = str(getattr(row, id_col))
        scoped_ic = ic.filterBounds(geom)

        def make_period(i):
            p_start = start_date.advance(ee.Number(i).multiply(period_days), "day")
            p_end = p_start.advance(period_days, "day")
            sub = scoped_ic.filterDate(p_start, p_end)
            # .reproject() BEFORE reduceRegion is required, not just an
            # explicit crs= on reduceRegion itself: sources with a native
            # non-EPSG:4326 tile grid (e.g. MODIS sinusoidal, SR-ORG:6974)
            # otherwise fail table exports with "Unable to compute
            # intersection of geometries in projections X and Y" - this hit
            # every AOD test above (single-station too, so it's not a
            # cross-station tile-boundary issue, just a batch-export-time
            # projection reconciliation requirement). Materializing the
            # composite in EPSG:4326 first avoids the on-the-fly reprojection
            # that fails during export.
            img = combine_reducer(sub).reproject(crs="EPSG:4326", scale=scale)
            stats = img.reduceRegion(reducer=region_reducer, geometry=geom, scale=scale, maxPixels=1e8)
            return ee.Feature(None, stats.set("date", p_start.format("YYYY-MM-dd")).set("station_id", station_id))

        return ee.FeatureCollection(periods.map(make_period))

    per_station_fcs = [per_station(row) for row in stations.itertuples(index=False)]
    return ee.FeatureCollection(per_station_fcs).flatten()


def export_table(fc: ee.FeatureCollection, description: str, folder: str = "ps3_v2_exports") -> ee.batch.Task:
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=folder,
        fileFormat="CSV",
    )
    task.start()
    return task
