"""Join overpass-windowed ground truth + satellite + ERA5 into the final
daily clean101 modeling table, with daily-appropriate lag/rolling features.

Feature scope, explicit: AOD (2 bands) + 5 TROPOMI gases + ERA5-derived met
(temp_c, rh_pct, wind_speed, wind_dir_deg) - no fire/BLH in this comparison
(not part of the overpass-windowed extraction; the existing v2 feature set
has them, this daily comparison deliberately doesn't, to keep the new
extraction scoped to what the overpass-matching requirement asked for).

Inner join throughout (a prediction/training row is only meaningful if
ground truth AND every feature source actually has data for that
(station,date) - same fail-safe choice already used elsewhere in this
project, see PROJECT_DOCUMENTATION.md bug 5.14).

station_aqi_target_enc computed from TRAIN rows only (date < 2025-01-01) -
no leakage into the 2025 test year.
"""

import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
TRIAL02_ROOT = os.path.dirname(V3_ROOT)
DATA_DIR = os.path.join(V3_ROOT, "data")

sys.path.insert(0, os.path.join(TRIAL02_ROOT, "features"))
from merge_sources import derive_wind_met  # noqa: E402

SOURCES = ["aod", "no2", "so2", "co", "o3", "hcho"]
LAG_ROLL_VARS = ["aod_047", "aod_055", "no2", "so2", "co", "o3", "hcho", "temp_c", "rh_pct", "wind_speed"]
SPLIT_DATE = pd.Timestamp("2025-01-01")


def load_source(source: str) -> pd.DataFrame:
    src_dir = os.path.join(DATA_DIR, "raw_overpass", source)
    parts = []
    for fname in os.listdir(src_dir):
        df = pd.read_csv(os.path.join(src_dir, fname))
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    drop_cols = [c for c in full.columns if c == "n_in_window"]
    return full.drop(columns=drop_cols)


def load_era5() -> pd.DataFrame:
    src_dir = os.path.join(DATA_DIR, "raw_daily_era5")
    parts = [pd.read_csv(os.path.join(src_dir, f)) for f in os.listdir(src_dir)]
    full = pd.concat(parts, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    full = derive_wind_met(full)
    keep = ["station_id", "date", "temp_c", "rh_pct", "wind_speed", "wind_dir_deg"]
    return full[keep]


def add_lag_roll_features_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily-appropriate lags/rolls (lag1/lag3/lag7, roll3/roll7) - replaces
    engineer_features.py's weekly lag1/lag2/roll2/roll4 for this daily-cadence
    comparison (see PROJECT_DOCUMENTATION.md section 2.4, cadence consequence)."""
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
    grp = df.groupby("station_id")
    new_cols = {}
    for var in LAG_ROLL_VARS:
        if var not in df.columns:
            continue
        new_cols[f"{var}_lag1"] = grp[var].shift(1)
        new_cols[f"{var}_lag3"] = grp[var].shift(3)
        new_cols[f"{var}_lag7"] = grp[var].shift(7)
        new_cols[f"{var}_roll3"] = grp[var].transform(lambda s: s.rolling(3, min_periods=1).mean())
        new_cols[f"{var}_roll7"] = grp[var].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def feature_cols(df: pd.DataFrame) -> list[str]:
    base = [c for c in ["aod_047", "aod_055", "no2", "so2", "co", "o3", "hcho", "temp_c", "rh_pct", "wind_speed", "wind_dir_deg", "month", "doy_sin", "doy_cos"] if c in df.columns]
    lag_roll = [c for c in df.columns if any(c.startswith(v + "_") for v in LAG_ROLL_VARS) and (c.endswith(("_lag1", "_lag3", "_lag7", "_roll3", "_roll7")))]
    return base + lag_roll + ["station_aqi_target_enc"]


def main():
    gt = pd.read_csv(os.path.join(DATA_DIR, "clean101_ground_truth_daily.csv"), parse_dates=["date"])
    gt = gt.rename(columns={"aqi": "aqi"})[["station_id", "date", "aqi", "lat", "lon", "city"]]
    print(f"Ground truth: {gt.shape}")

    merged = gt.copy()
    for source in SOURCES:
        src_df = load_source(source)
        merged = merged.merge(src_df, on=["station_id", "date"], how="inner")
        print(f"After {source}: {merged.shape}")

    era5 = load_era5()
    merged = merged.merge(era5, on=["station_id", "date"], how="inner")
    print(f"After ERA5: {merged.shape}")

    merged["month"] = merged["date"].dt.month
    doy = merged["date"].dt.dayofyear
    merged["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    merged["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    merged = add_lag_roll_features_daily(merged)

    train_mask = merged["date"] < SPLIT_DATE
    station_target_map = merged.loc[train_mask].groupby("station_id")["aqi"].mean()
    merged["station_aqi_target_enc"] = merged["station_id"].map(station_target_map)
    merged["split"] = np.where(train_mask, "train", "test")

    out_path = os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nSaved: {merged.shape} -> {out_path}")
    print(f"Train rows: {(merged['split']=='train').sum()}, Test rows: {(merged['split']=='test').sum()}")
    print(f"Stations: {merged.station_id.nunique()}")
    print(f"Feature columns ({len(feature_cols(merged))}): {feature_cols(merged)}")

    station_target_map.to_csv(os.path.join(DATA_DIR, "station_target_map.csv"))


if __name__ == "__main__":
    main()
