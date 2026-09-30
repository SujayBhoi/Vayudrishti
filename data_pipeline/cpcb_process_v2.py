"""v2 CPCB AQI computation - same official CPCB 2014 National AQI breakpoint
methodology as v1 (legacy_v1_src/cpcb_process.py), reused verbatim, but
applied directly to cpcb_cleaned_daily_FINAL.csv's already-daily,
already-per-station rows instead of raw hourly station files. Since v2
trains per-station (not per-city), there's no cross-station averaging step
at all here - each station's own daily concentrations map straight to its
own daily AQI, then resample to weekly to match the satellite cadence.
"""

import os

import numpy as np
import pandas as pd

RAW_CSV_PATH = r"C:\Users\mohit\Downloads\cpcb_cleaned_daily_FINAL.csv"

# Same breakpoint tables as v1, column names adapted to this source's headers
# (PM2_5 not "PM2.5 (ug/m3)", Ozone not "Ozone (ug/m3)", etc - units assumed
# consistent with CPCB standard reporting, matching v1's assumption).
BREAKPOINTS = {
    "PM2_5": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300), (120, 250, 300, 400), (250, 380, 400, 500)],
    "PM10": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300), (350, 430, 300, 400), (430, 510, 400, 500)],
    "NO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200), (180, 280, 200, 300), (280, 400, 300, 400), (400, 800, 400, 500)],
    "SO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 380, 100, 200), (380, 800, 200, 300), (800, 1600, 300, 400), (1600, 2100, 400, 500)],
    "Ozone": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 168, 100, 200), (168, 208, 200, 300), (208, 748, 300, 400), (748, 1000, 400, 500)],
    "CO": [(0, 1.0, 0, 50), (1.0, 2.0, 50, 100), (2.0, 10, 100, 200), (10, 17, 200, 300), (17, 34, 300, 400), (34, 50, 400, 500)],
    "NH3": [(0, 200, 0, 50), (200, 400, 50, 100), (400, 800, 100, 200), (800, 1200, 200, 300), (1200, 1800, 300, 400), (1800, 2400, 400, 500)],
}


def sub_index(pollutant, conc):
    if pd.isna(conc):
        return np.nan
    table = BREAKPOINTS[pollutant]
    for blo, bhi, ilo, ihi in table:
        if blo <= conc <= bhi:
            return (ihi - ilo) / (bhi - blo) * (conc - blo) + ilo
    if conc > table[-1][1]:
        return 500.0
    return np.nan


def compute_weekly_aqi(station_ids: list[str] | None = None) -> pd.DataFrame:
    cols = ["station_id", "date"] + list(BREAKPOINTS.keys())
    df = pd.read_csv(RAW_CSV_PATH, usecols=cols, parse_dates=["date"])
    if station_ids is not None:
        df = df[df["station_id"].isin(station_ids)]

    pollutants = list(BREAKPOINTS.keys())
    for p in pollutants:
        df[f"sub_{p}"] = df[p].apply(lambda c, p=p: sub_index(p, c))
    sub_cols = [f"sub_{p}" for p in pollutants]
    df["aqi"] = df[sub_cols].max(axis=1, skipna=True)
    df["n_pollutants"] = df[sub_cols].notna().sum(axis=1)
    df = df[df["n_pollutants"] >= 1]

    anchor = pd.Timestamp("2018-01-01")
    df["week_start"] = anchor + pd.to_timedelta((df["date"] - anchor).dt.days // 7 * 7, unit="D")
    weekly = df.groupby(["station_id", "week_start"])["aqi"].mean().reset_index()
    weekly = weekly.rename(columns={"week_start": "date"})
    return weekly


if __name__ == "__main__":
    weekly = compute_weekly_aqi()
    print(f"Computed weekly AQI: {weekly.shape}")
    print(f"Stations: {weekly['station_id'].nunique()}, date range: {weekly['date'].min()} to {weekly['date'].max()}")
    out_path = os.path.join(os.path.dirname(__file__), "legacy_v1_data", "processed", "cpcb_aqi_weekly_v2.csv")
    weekly.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
