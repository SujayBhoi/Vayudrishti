"""v2: Sentinel-5P TROPOMI multi-pollutant weekly time series at all v2 CPCB
station points - NO2, SO2, CO, O3, HCHO (v1 only had HCHO).

Uses the same proven-safe interactive per-station pattern as
fetch_modis_aod.py (see that file's docstring for why: the batched national
Export.table.toDrive approach forces expensive per-period compositing across
the whole country regardless of how few points are ultimately sampled -
these are all multi-orbit-pass daily products, same cost risk as AOD, so
they get the same safe treatment even though the specific projection crash
seen with AOD's sinusoidal grid wasn't independently confirmed for TROPOMI).

QA filtering: HCHO keeps v1's cloud_fraction<0.3 approach (already
validated); NO2/SO2/CO/O3 use qa_value>0.5, the standard Sentinel-5P L3
recommendation for general tropospheric-column screening.
"""

import argparse
import time

from fetch_cpcb import load_station_master
from gee_auth import init_ee
from interactive_point_extract import extract_all_stations

BUFFER_M = 10_000

# FIX (found when these first ran): the assumed "qa_value" band does not
# exist on any of these actual OFFL L3 collections' band lists (verified via
# the real error output, which lists each collection's true available
# bands). NO2/SO2/O3 do have "cloud_fraction" (same QA approach as HCHO);
# CO has neither cloud_fraction nor qa_value at all (CO's SWIR retrieval is
# far less cloud-sensitive than the others, so this is standard practice,
# not a gap) - CO gets no QA mask.
GASES = {
    "no2": dict(collection="COPERNICUS/S5P/OFFL/L3_NO2", band="tropospheric_NO2_column_number_density", qa_band="cloud_fraction", qa_max=0.3, out_col="no2"),
    "so2": dict(collection="COPERNICUS/S5P/OFFL/L3_SO2", band="SO2_column_number_density", qa_band="cloud_fraction", qa_max=0.3, out_col="so2"),
    "co": dict(collection="COPERNICUS/S5P/OFFL/L3_CO", band="CO_column_number_density", qa_band=None, out_col="co"),
    "o3": dict(collection="COPERNICUS/S5P/OFFL/L3_O3", band="O3_column_number_density", qa_band="cloud_fraction", qa_max=0.3, out_col="o3"),
    "hcho": dict(collection="COPERNICUS/S5P/OFFL/L3_HCHO", band="tropospheric_HCHO_column_number_density", qa_band="cloud_fraction", qa_max=0.3, out_col="hcho"),
}


def make_mask_fn(cfg):
    if cfg.get("qa_band") is None:
        return None

    def mask_fn(img):
        return img.updateMask(img.select(cfg["qa_band"]).lt(cfg["qa_max"])).select([cfg["band"]])
    return mask_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gases", nargs="*", default=list(GASES.keys()))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    init_ee()
    stations = load_station_master()

    for gas in args.gases:
        cfg = GASES[gas]
        print(f"Extracting {gas.upper()} for {len(stations)} stations, {args.start}..{args.end}")
        t0 = time.time()
        bands = [cfg["band"]] + ([cfg["qa_band"]] if cfg["qa_band"] else [])
        extract_all_stations(
            stations, args.start, args.end,
            collection=cfg["collection"],
            bands=bands,
            buffer_m=BUFFER_M, scale=1113,
            mask_fn=make_mask_fn(cfg),
            rename={cfg["band"]: cfg["out_col"]},
            out_dir=f"data_pipeline/raw_v2/{gas}",
            n_workers=args.workers, label=gas.upper(),
        )
        print(f"{gas.upper()} extraction complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
