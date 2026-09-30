"""ADDITIONAL, standalone layer - pulls real PM2.5 (overpass-windowed,
same methodology as everywhere else in this project: 11:18-15:18 IST mean)
for ALL 101 stations across the FULL date range (2019-2024 train +
2025 test), for the two-stage gases->PM2.5->AQI reference model. Does not
touch the primary CNN-LSTM pipeline, Streamlit UI, or Objective 2 - purely
additive, new files only.
"""

import json
import os

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(V3_ROOT, "data")

PM25_COL = "PM2.5 (\xb5g/m\xb3)"
WINDOW_START_MIN = 11 * 60 + 18
WINDOW_END_MIN = 15 * 60 + 18


def windowed_daily_pm25(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    minute_of_day = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute
    in_window = (minute_of_day >= WINDOW_START_MIN) & (minute_of_day <= WINDOW_END_MIN)
    win = df[in_window].copy()
    win["date"] = win["Timestamp"].dt.normalize()
    return win.groupby(["Station", "date"])[PM25_COL].mean().rename("pm25").reset_index()


def main():
    with open(os.path.join(DATA_DIR, "clean101_report.json")) as f:
        report = json.load(f)
    stations = set(report["final_stations"])

    print("Loading 2019-2024 PM2.5 ...")
    df19 = pd.read_csv(r"C:\Users\mohit\Downloads\Cleaned_Data_2019_2024_common_stations_combined.csv", usecols=["Station", "Timestamp", PM25_COL])
    df19 = df19[df19.Station.isin(stations)]
    daily19 = windowed_daily_pm25(df19)
    print(f"  {daily19.shape}")

    print("Loading 2025 PM2.5 ...")
    df25 = pd.read_parquet(r"C:\Users\mohit\Downloads\Cleaned_Data_Final_2025.parquet", columns=["Station", "Timestamp", PM25_COL])
    df25 = df25[df25.Station.isin(stations)]
    df25 = df25[pd.to_datetime(df25["Timestamp"]).dt.year == 2025]
    daily25 = windowed_daily_pm25(df25)
    print(f"  {daily25.shape}")

    full = pd.concat([daily19, daily25], ignore_index=True).rename(columns={"Station": "station_id"})
    out_path = os.path.join(DATA_DIR, "pm25_reference_daily.csv")
    full.to_csv(out_path, index=False)
    print(f"\nSaved {full.shape} -> {out_path}")
    print(f"Stations: {full.station_id.nunique()}, date range: {full.date.min()} to {full.date.max()}")


if __name__ == "__main__":
    main()
