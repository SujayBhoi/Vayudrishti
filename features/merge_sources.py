"""Merges per-source v2 station tables (AOD, 5 TROPOMI gases, ERA5, fire,
BLH, CPCB pollutants) into one training table, keyed on (station_id,
period_start). Mirrors v1's build_satellite_met_dataset.py +
merge_final_dataset.py, generalized from per-city to per-station and from
2 satellite vars (AOD/HCHO) to 7 (AOD x2 + 5 gases).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def derive_wind_met(df: pd.DataFrame) -> pd.DataFrame:
    """Same derivations as v1's build_satellite_met_dataset.py: wind
    speed/direction from u/v, temp in C, RH% via the Magnus formula."""
    out = df.copy()
    if {"u_component_of_wind_10m", "v_component_of_wind_10m"}.issubset(out.columns):
        out["wind_speed"] = np.hypot(out["u_component_of_wind_10m"], out["v_component_of_wind_10m"])
        out["wind_dir_deg"] = (np.degrees(np.arctan2(out["u_component_of_wind_10m"], out["v_component_of_wind_10m"])) + 360) % 360
    if "temperature_2m" in out.columns:
        out["temp_c"] = out["temperature_2m"] - 273.15
    if {"temperature_2m", "dewpoint_temperature_2m"}.issubset(out.columns):
        t = out["temperature_2m"] - 273.15
        td = out["dewpoint_temperature_2m"] - 273.15
        # Magnus formula RH%
        a, b = 17.625, 243.04
        out["rh_pct"] = 100 * np.exp((a * td) / (b + td)) / np.exp((a * t) / (b + t))
    return out


def merge_all(aod, gases: dict, era5, fire, blh, cpcb_aqi) -> pd.DataFrame:
    """All inputs indexed on (station_id, date) - every source's raw CSV
    already has a 'date' column (the extraction scripts write the GEE
    period_start property directly as 'date', and build_v2_dataset.py's
    load_source() also normalizes any leftover 'period_start' name), so no
    column needs renaming here; merging on 'period_start' was a bug from an
    earlier draft where source dataframes still had that column name."""
    df = aod.rename(columns={"Optical_Depth_047": "aod_047", "Optical_Depth_055": "aod_055"})
    for gas_name, gas_df in gases.items():
        df = df.merge(gas_df, on=["station_id", "date"], how="outer")
    df = df.merge(era5, on=["station_id", "date"], how="outer")
    df = derive_wind_met(df)
    df = df.merge(fire, on=["station_id", "date"], how="outer")
    df = df.merge(blh, on=["station_id", "date"], how="left")
    df = df.merge(cpcb_aqi, on=["station_id", "date"], how="inner")  # inner: need ground truth to train
    return df
