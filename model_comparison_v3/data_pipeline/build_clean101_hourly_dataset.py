"""Build the clean, overpass-matched 101-station daily AQI ground-truth table.

Source: two hourly CPCB files (official source, found in Downloads):
  - Cleaned_Data_2019_2024_common_stations_combined.csv (122 stations, hourly)
  - Cleaned_Data_Final_2025.parquet (413 stations, hourly, contains one
    mistagged 2023 fragment for "Sector 30, Faridabad - HSPCB" - dropped)

Cleaning (approved): discard file's own AQI (recompute via the project's
existing CPCB-2014 breakpoint formula, cpcb_process_v2.py), discard/rejoin
Latitude/Longitude from cpcb_v2_station_master.csv by station name (file's
own lat/lon is a single bogus fixed value for every row), drop AT/RH/WS/RF
(sparse ground-sensor met - ERA5 is used instead, joined separately).

Station filter: >=80% hourly completeness in BOTH the 2019-2024 window
(52,608 expected hours) and the 2025 window (8,760 expected hours), computed
against real expected-hour counts, not each station's own reporting span.
The one 17.3%-complete outlier is excluded explicitly either way.

Windowed ground truth: real confirmed satellite overpass times (TROPOMI
~13:08 IST, MODIS/Aqua ~13:10-13:45 IST, verified via direct GEE
system:time_start queries) -> combined anchor ~13:18 IST -> window
11:18-15:18 IST. Ground truth AQI for a station-day = mean of hourly
(breakpoint-recomputed) AQI readings whose timestamp falls in that window.
Days with zero in-window readings are left as gaps (NaN), not interpolated.

Output: model_comparison_v3/data/clean101_ground_truth_daily.csv
        model_comparison_v3/data/clean101_report.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
TRIAL02_ROOT = os.path.dirname(V3_ROOT)
DATA_OUT_DIR = os.path.join(V3_ROOT, "data")
os.makedirs(DATA_OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(TRIAL02_ROOT, "data_pipeline"))
from cpcb_process_v2 import sub_index  # noqa: E402

CSV_2019_2024 = r"C:\Users\mohit\Downloads\Cleaned_Data_2019_2024_common_stations_combined.csv"
PARQUET_2025 = r"C:\Users\mohit\Downloads\Cleaned_Data_Final_2025.parquet"
MASTER_LIST = os.path.join(TRIAL02_ROOT, "data_pipeline", "legacy_v1_data", "processed", "cpcb_v2_station_master.csv")

# hourly-file column -> BREAKPOINTS key (NH3 not present in this source, skipped - fine, max() over available pollutants)
POLLUTANT_COL_MAP = {
    "PM2.5 (\xb5g/m\xb3)": "PM2_5",
    "PM10 (\xb5g/m\xb3)": "PM10",
    "NO2 (\xb5g/m\xb3)": "NO2",
    "SO2 (\xb5g/m\xb3)": "SO2",
    "CO (mg/m\xb3)": "CO",
    "Ozone (\xb5g/m\xb3)": "Ozone",
}

WINDOW_START_MIN = 11 * 60 + 18  # 11:18 IST
WINDOW_END_MIN = 15 * 60 + 18    # 15:18 IST

COMPLETENESS_THRESHOLD = 0.8
EXPLICIT_OUTLIER = "Gobind Pura, Yamuna Nagar - HSPCB"


def recompute_aqi(df: pd.DataFrame) -> pd.Series:
    sub_indices = []
    for col, bp_key in POLLUTANT_COL_MAP.items():
        s = df[col].apply(lambda c, k=bp_key: sub_index(k, c))
        sub_indices.append(s.rename(bp_key))
    sub_df = pd.concat(sub_indices, axis=1)
    return sub_df.max(axis=1, skipna=True)


def load_sources():
    print("Loading 2019-2024 hourly CSV ...")
    df19 = pd.read_csv(CSV_2019_2024, parse_dates=["Timestamp"])
    print(f"  {df19.shape}")

    print("Loading 2025 hourly parquet ...")
    df25 = pd.read_parquet(PARQUET_2025)
    df25["Timestamp"] = pd.to_datetime(df25["Timestamp"])
    n_before = len(df25)
    stray_mask = df25["Timestamp"].dt.year != 2025
    n_stray = int(stray_mask.sum())
    df25 = df25[~stray_mask].copy()
    print(f"  dropped {n_stray} mistagged non-2025 rows (stray fragment), {n_before} -> {len(df25)}")

    return df19, df25


def clean(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["aqi_recomputed"] = recompute_aqi(df)
    aqi_over_500 = int((df["AQI"] > 500).sum())

    keep_cols = ["Timestamp", "Station", "City", "State", "aqi_recomputed"]
    df = df[keep_cols].rename(columns={"Timestamp": "timestamp", "Station": "station_id"})

    n_before = len(df)
    df = df.merge(master[["station_id", "lat", "lon"]], on="station_id", how="inner")
    n_after = len(df)

    return df, {"aqi_over_500_dropped_col": aqi_over_500, "rows_before_master_join": n_before, "rows_after_master_join": n_after}


def completeness(df: pd.DataFrame, expected_hours: int) -> pd.Series:
    counts = df.groupby("station_id")["timestamp"].nunique()
    return counts / expected_hours


def windowed_daily_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    minute_of_day = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    in_window = (minute_of_day >= WINDOW_START_MIN) & (minute_of_day <= WINDOW_END_MIN)
    win = df[in_window].copy()
    win["date"] = win["timestamp"].dt.normalize()
    daily = (
        win.groupby(["station_id", "date"])
        .agg(aqi=("aqi_recomputed", "mean"), lat=("lat", "first"), lon=("lon", "first"),
             city=("city", "first") if "city" in win.columns else ("station_id", "first"),
             n_readings_in_window=("aqi_recomputed", "count"))
        .reset_index()
    )
    return daily


def main():
    master = pd.read_csv(MASTER_LIST)
    df19_raw, df25_raw = load_sources()

    df19, stats19 = clean(df19_raw, master)
    df25, stats25 = clean(df25_raw, master)
    df19 = df19.rename(columns={"City": "city", "State": "state"})
    df25 = df25.rename(columns={"City": "city", "State": "state"})

    n_exp_19 = len(pd.date_range("2019-01-01", "2024-12-31 23:00:00", freq="h"))
    n_exp_25 = len(pd.date_range("2025-01-01", "2025-12-31 23:00:00", freq="h"))

    common = (set(df19.station_id.unique()) & set(df25.station_id.unique())) & set(master.station_id)
    comp19 = completeness(df19[df19.station_id.isin(common)], n_exp_19).reindex(common)
    comp25 = completeness(df25[df25.station_id.isin(common)], n_exp_25).reindex(common)

    # stations present in the 2019-2024 file but with zero 2025 data at all -
    # never enter `common`, so wouldn't otherwise appear in station_detail
    no_2025_at_all = sorted(set(df19.station_id.unique()) - set(df25.station_id.unique()))
    no_2025_detail = []
    loc0 = master.set_index("station_id")[["city", "state"]]
    for sid in no_2025_at_all:
        no_2025_detail.append({
            "station_id": sid,
            "city": loc0.loc[sid, "city"] if sid in loc0.index else None,
            "state": loc0.loc[sid, "state"] if sid in loc0.index else None,
            "exclusion_reason": "no 2025 data available at all (present in the 2019-2024 file, absent from the 2025 file)",
        })

    passes = (comp19 >= COMPLETENESS_THRESHOLD) & (comp25 >= COMPLETENESS_THRESHOLD)
    if EXPLICIT_OUTLIER in passes.index:
        passes[EXPLICIT_OUTLIER] = False
    final_stations = sorted(passes[passes].index.tolist())

    # per-station location + completeness, for the full common scope (kept AND dropped) - traceability
    loc = master.set_index("station_id")[["city", "state"]]
    station_detail = []
    for sid in sorted(common):
        row = {
            "station_id": sid,
            "city": loc.loc[sid, "city"] if sid in loc.index else None,
            "state": loc.loc[sid, "state"] if sid in loc.index else None,
            "completeness_2019_2024": round(float(comp19.get(sid, float("nan"))), 4),
            "completeness_2025": round(float(comp25.get(sid, float("nan"))), 4),
            "kept": bool(passes.get(sid, False)),
        }
        if not row["kept"]:
            if sid == EXPLICIT_OUTLIER:
                row["exclusion_reason"] = "explicitly excluded outlier (2025 completeness 17.3%, far below threshold)"
            elif comp25.get(sid, 0) < COMPLETENESS_THRESHOLD and comp19.get(sid, 0) >= COMPLETENESS_THRESHOLD:
                row["exclusion_reason"] = f"completeness {comp25.get(sid,0):.1%} in 2025, below the {COMPLETENESS_THRESHOLD:.0%} threshold (2019-2024 was fine)"
            elif comp19.get(sid, 0) < COMPLETENESS_THRESHOLD and comp25.get(sid, 0) >= COMPLETENESS_THRESHOLD:
                row["exclusion_reason"] = f"completeness {comp19.get(sid,0):.1%} in 2019-2024, below the {COMPLETENESS_THRESHOLD:.0%} threshold (2025 was fine)"
            else:
                row["exclusion_reason"] = f"completeness below {COMPLETENESS_THRESHOLD:.0%} threshold in both windows"
        station_detail.append(row)

    print(f"\nStarting common scope: {len(common)} stations")
    print(f"Final clean101 stations (>= {COMPLETENESS_THRESHOLD:.0%} both windows): {len(final_stations)}")

    df19_f = df19[df19.station_id.isin(final_stations)]
    df25_f = df25[df25.station_id.isin(final_stations)]

    daily19 = windowed_daily_ground_truth(df19_f)
    daily25 = windowed_daily_ground_truth(df25_f)
    daily = pd.concat([daily19, daily25], ignore_index=True).sort_values(["station_id", "date"])

    out_path = os.path.join(DATA_OUT_DIR, "clean101_ground_truth_daily.csv")
    daily.to_csv(out_path, index=False)
    print(f"\nSaved windowed daily ground truth: {daily.shape} -> {out_path}")
    print(f"Date range: {daily.date.min()} to {daily.date.max()}")
    print(f"Days with zero in-window readings (gaps): checked via n_readings_in_window==0 count: "
          f"{(daily.n_readings_in_window == 0).sum()}")

    report = {
        "csv_2019_2024_cleaning": stats19,
        "parquet_2025_cleaning": stats25,
        "n_expected_hours_2019_2024": n_exp_19,
        "n_expected_hours_2025": n_exp_25,
        "n_common_starting_stations": len(common),
        "n_final_clean101_stations": len(final_stations),
        "final_stations": final_stations,
        "dropped_stations": sorted(set(common) - set(final_stations)),
        "completeness_threshold": COMPLETENESS_THRESHOLD,
        "explicit_outlier_excluded": EXPLICIT_OUTLIER,
        "windowed_ground_truth_rows": int(len(daily)),
        "windowed_ground_truth_date_range": [str(daily.date.min()), str(daily.date.max())],
        "window_ist": "11:18-15:18",
        "station_detail": station_detail,
        "no_2025_data_stations": no_2025_detail,
    }
    report_path = os.path.join(DATA_OUT_DIR, "clean101_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved report -> {report_path}")

    assert len(final_stations) == 101, f"Expected 101 stations, got {len(final_stations)} - investigate before proceeding"
    print("\nSanity check passed: 101 stations confirmed.")


if __name__ == "__main__":
    main()
