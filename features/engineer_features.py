"""v2 feature engineering — generalizes v1's per-city lag/rolling/seasonal
features (legacy_v1_src/train_aqi_model.py) to per-STATION, which is also
the structural fix for the Kanpur/Pune negative-R2 issue.

Root cause confirmed by direct inspection of the raw CPCB archive: v1
computed "city AQI" as an unweighted daily mean across whatever stations
happened to be reporting that day. For Kanpur, one station (UP005) supplied
~82% of pre-cutoff rows at a high pollution baseline (PM2.5 mean 110.6),
while 3 more stations only came online in 2020-2021; post-cutoff, all 4
stations report at equal, regular cadence with a much lower, more balanced
mean (49.8-65.8). The "city" time series therefore has an artificial level
shift baked in that has nothing to do with real air-quality change - the
model learns a baseline anchored to one dominant early station and then
sees a structurally different mix at test time. Pune shows the same pattern
in the opposite direction (post-cutoff mean pulled UP by newer stations).

v2's fix isn't a reweighting patch - it's structural: each of Kanpur's 4 and
Pune's 8 stations becomes its own row/station_id, with its own continuous
per-station history and its own satellite feature extraction. There is no
city-level averaging left to shift under a station roster change. City
identity is kept only as a categorical feature (for spatial pooling of
signal), not as the unit of aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_ROLL_VARS = [
    "aod_047", "aod_055", "no2", "so2", "co", "o3", "hcho",
    "fire_count", "temp_c", "rh_pct", "wind_speed", "blh",
]

BASE_FEATURE_COLS = [
    "aod_047", "aod_055", "no2", "so2", "co", "o3", "hcho",
    "fire_count", "fire_max_conf", "temp_c", "rh_pct", "wind_speed", "wind_dir_deg", "blh",
    "lat", "lon", "month", "doy_sin", "doy_cos",
]


def add_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station lag-1/lag-2 and 2wk/4wk rolling means. Frame must be
    sorted by (station_id, date) before shift/rolling read top-to-bottom."""
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
    grp = df.groupby("station_id")
    new_cols = {}
    for var in LAG_ROLL_VARS:
        if var not in df.columns:
            continue
        new_cols[f"{var}_lag1"] = grp[var].shift(1)
        new_cols[f"{var}_lag2"] = grp[var].shift(2)
        new_cols[f"{var}_roll2"] = grp[var].transform(lambda s: s.rolling(2, min_periods=1).mean())
        new_cols[f"{var}_roll4"] = grp[var].transform(lambda s: s.rolling(4, min_periods=1).mean())
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def engineer_features(df: pd.DataFrame, train_cutoff=None, station_target_map=None):
    """station_target_map replaces v1's city_target_map: a train-only
    target-encoded per-station AQI baseline. Since v2 no longer averages
    across stations, this encoding is now genuinely station-specific rather
    than diluted by whichever stations happened to report that week."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_lag_roll_features(df)

    df["month"] = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if "fire_max_conf" in df.columns:
        df["fire_max_conf"] = df["fire_max_conf"].fillna(0)

    if station_target_map is None:
        if train_cutoff is None:
            raise ValueError("engineer_features needs train_cutoff (first call) or station_target_map (reuse)")
        train_mask_for_encoding = df["date"] <= train_cutoff
        station_target_map = df.loc[train_mask_for_encoding].groupby("station_id")["aqi"].mean()
    df["station_aqi_target_enc"] = df["station_id"].map(station_target_map)

    city_dummies = pd.get_dummies(df["city"], prefix="city") if "city" in df.columns else pd.DataFrame(index=df.index)

    return df, city_dummies, station_target_map


def feature_cols(df: pd.DataFrame) -> list[str]:
    lag_roll_cols = [
        c for c in df.columns
        if any(c.startswith(v + "_") for v in LAG_ROLL_VARS)
        and (c.endswith("_lag1") or c.endswith("_lag2") or c.endswith("_roll2") or c.endswith("_roll4"))
    ]
    present_base = [c for c in BASE_FEATURE_COLS if c in df.columns]
    return present_base + lag_roll_cols + ["station_aqi_target_enc"]
