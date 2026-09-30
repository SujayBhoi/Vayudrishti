"""Driver: assembles the final v2 model_dataset.csv from all per-station raw
extraction outputs (AOD, 5 gases, ERA5, fire, BLH) + CPCB ground truth,
mirroring v1's build_satellite_met_dataset.py + merge_final_dataset.py but
generalized to 412 stations and 7 satellite variables instead of 13 cities
and 2.

Each source directory has one CSV per station (safe_id.csv) with columns
[date, <value columns>, station_id]. This script loads all of them per
source, concatenates, then hands off to features/merge_sources.py.
"""

import glob
import os

import pandas as pd

RAW_ROOT = os.path.join(os.path.dirname(__file__), "..", "data_pipeline")
NESTED_ROOT = os.path.join(RAW_ROOT, "data_pipeline", "raw_v2")  # every fetch_*.py that used a cwd-relative
# "data_pipeline/raw_v2/..." out_dir string nested one level deeper when run from inside data_pipeline/ -
# confirmed for aod, hcho, no2, so2, co, o3, era5, fire. Only fetch_openmeteo_blh.py used a __file__-relative
# path (not cwd-relative), so blh alone is NOT nested.

SOURCE_DIRS = {
    "aod": os.path.join(NESTED_ROOT, "aod"),
    "no2": os.path.join(NESTED_ROOT, "no2"),
    "so2": os.path.join(NESTED_ROOT, "so2"),
    "co": os.path.join(NESTED_ROOT, "co"),
    "o3": os.path.join(NESTED_ROOT, "o3"),
    "hcho": os.path.join(NESTED_ROOT, "hcho"),
    "era5": os.path.join(NESTED_ROOT, "era5"),
    "fire": os.path.join(NESTED_ROOT, "fire"),
    "blh": os.path.join(RAW_ROOT, "raw_v2", "blh"),
}


def load_source(name: str) -> pd.DataFrame:
    d = SOURCE_DIRS[name]
    files = glob.glob(os.path.join(d, "*.csv"))
    if not files:
        print(f"  [warn] no files found for {name} in {d}")
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    rename_map = {"period_start": "date"}
    df = df.rename(columns=rename_map)
    # Normalize to plain "YYYY-MM-DD" strings - GEE-derived sources already
    # write dates this way, but cpcb_process_v2's compute_weekly_aqi produces
    # real Timestamps, and pandas refuses to merge str vs datetime64 key
    # columns outright (a real ValueError, not a silent no-match), so every
    # source needs the same representation before merge_sources.merge_all.
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def report_coverage():
    print("Source file counts (target 412 each):")
    for name, d in SOURCE_DIRS.items():
        n = len(glob.glob(os.path.join(d, "*.csv"))) if os.path.isdir(d) else 0
        print(f"  {name}: {n}")


def build(out_path: str) -> pd.DataFrame:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
    from cpcb_process_v2 import compute_weekly_aqi
    from fetch_cpcb import load_station_master
    from merge_sources import merge_all

    stations = load_station_master()
    print(f"Station master: {len(stations)} stations")

    aod = load_source("aod")
    gases = {g: load_source(g) for g in ["no2", "so2", "co", "o3", "hcho"]}
    era5 = load_source("era5")
    fire = load_source("fire")
    blh = load_source("blh")
    cpcb = compute_weekly_aqi(list(stations["station_id"]))
    cpcb["date"] = pd.to_datetime(cpcb["date"]).dt.strftime("%Y-%m-%d")

    for name, df in [("aod", aod), *gases.items(), ("era5", era5), ("fire", fire), ("blh", blh), ("cpcb", cpcb)]:
        print(f"  {name}: {len(df)} rows" + (f", cols={list(df.columns)}" if not df.empty else " (EMPTY)"))

    merged = merge_all(aod, gases, era5, fire, blh, cpcb)
    merged = merged.merge(stations[["station_id", "city", "state"]], on="station_id", how="left")
    print(f"\nFinal merged dataset: {merged.shape}")
    merged.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return merged


if __name__ == "__main__":
    report_coverage()
