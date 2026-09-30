"""CPCB ground-station ingestion for Objective 1 v2 (expanded station set).

Source: cpcb_cleaned_daily_FINAL.csv (536 stations, daily, 2018-01-01 to
2025-12-31, pollutants + met vars + lat/lon per station). This supersedes
the 13-city subset used in v1 (which came from a separate 454-station raw
hourly Kaggle archive processed by legacy_v1_src/cpcb_process.py).

Station filtering: keep stations with >=50% row-level coverage across the
major pollutants (PM2.5/PM10/NO2/SO2/CO/Ozone) used to compute the CPCB 2014
National AQI. This threshold was picked from the actual completeness
distribution of the source file (243 stations >=80%, 417 >=50%, 470 >=40%),
not an arbitrary guess: 50% keeps ~417 of 536 stations while excluding a
long tail of stations with 0-10% coverage that would just inject noise
during training. Adjustable via MIN_COVERAGE.
"""

from __future__ import annotations

import pandas as pd

RAW_CSV_PATH = r"C:\Users\mohit\Downloads\cpcb_cleaned_daily_FINAL.csv"
MAJOR_POLLUTANTS = ["PM2_5", "PM10", "NO2", "SO2", "CO", "Ozone"]
MIN_COVERAGE = 0.50


def load_station_master(min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Return one row per station: station_id, state, city, lat, lon, coverage."""
    df = pd.read_csv(RAW_CSV_PATH, usecols=["station_id", "state", "city", "date", "lat", "lon", *MAJOR_POLLUTANTS])
    has_data = df[MAJOR_POLLUTANTS].notna().any(axis=1)
    coverage = has_data.groupby(df["station_id"]).mean()
    meta = df.groupby("station_id").agg(
        state=("state", "first"),
        city=("city", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        n_rows=("date", "count"),
    )
    meta["coverage"] = coverage
    kept = meta[meta["coverage"] >= min_coverage].reset_index()
    # A handful of stations (5 as of the source CSV) have NaN lat/lon in every
    # row - can't be used for satellite point extraction regardless of
    # pollutant coverage, so drop them here rather than letting NaN leak into
    # GEE geometry construction downstream (which produces an invalid export
    # payload, not a clean error).
    n_before = len(kept)
    kept = kept.dropna(subset=["lat", "lon"])
    if len(kept) < n_before:
        print(f"Dropped {n_before - len(kept)} stations with missing lat/lon")
    return kept.sort_values("coverage", ascending=False)


def load_daily_pollutants(station_ids: list[str] | None = None) -> pd.DataFrame:
    """Full daily pollutant + met table, optionally filtered to given station_ids."""
    df = pd.read_csv(RAW_CSV_PATH, parse_dates=["date"])
    if station_ids is not None:
        df = df[df["station_id"].isin(station_ids)]
    return df


if __name__ == "__main__":
    stations = load_station_master()
    print(f"Kept {len(stations)} / 536 stations at >={MIN_COVERAGE:.0%} coverage")
    print(stations.head(10).to_string(index=False))
    out_path = "data_pipeline/legacy_v1_data/processed/cpcb_v2_station_master.csv"
    stations.to_csv(out_path, index=False)
    print(f"Saved station master to {out_path}")
