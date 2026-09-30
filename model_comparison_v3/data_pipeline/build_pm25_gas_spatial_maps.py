"""ADDITIONAL, standalone analysis layer - NOT part of the primary CNN-LSTM
AQI pipeline, does not modify or depend on it beyond reading the same
already-processed gas/weather dataset. Spatial maps comparing PM2.5 against
each gas/weather variable, station by station, for one representative day
in the 2025 test period.

PM2.5 itself is pulled fresh here (same overpass-window methodology already
used everywhere else: 11:18-15:18 IST) since it was intentionally NOT kept
in the main model_dataset_clean101_daily.csv (used only to compute the AQI
target, then dropped to avoid feature/target leakage - see
PROJECT_DOCUMENTATION.md section 7.4). This script's PM2.5 is real,
observed ground-truth, not a model prediction (no PM2.5 model exists yet -
that would be part #2 of this request, a separate, not-yet-built model).
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
TRIAL02_ROOT = os.path.dirname(V3_ROOT)
DATA_DIR = os.path.join(V3_ROOT, "data")
FIG_DIR = os.path.join(V3_ROOT, "figures", "pm25_analysis")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from india_boundary import draw_state_boundaries  # noqa: E402

PM25_COL = "PM2.5 (\xb5g/m\xb3)"
WINDOW_START_MIN = 11 * 60 + 18
WINDOW_END_MIN = 15 * 60 + 18
REPRESENTATIVE_DAY = "2025-11-26"  # full 101/101 station coverage confirmed, peak pollution season

VARS = {
    "no2": ("NO2", "TROPOMI column density"),
    "so2": ("SO2", "TROPOMI column density"),
    "o3": ("Ozone (O3)", "TROPOMI column density"),
    "co": ("CO", "TROPOMI column density"),
    "temp_c": ("Temperature", "deg C"),
    "wind_speed": ("Wind speed", "m/s"),
    "rh_pct": ("Relative humidity", "%"),
}

C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_MUTED = "#898781"
C_AXIS = "#c3c2b7"
from matplotlib.colors import LinearSegmentedColormap
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fde3d2", "#f8bd97", "#f2955c", "#eb6834", "#c94f1f", "#9c3a14"])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE, "savefig.facecolor": C_SURFACE,
})


def load_pm25_for_day(day: str, stations: set) -> pd.DataFrame:
    df = pd.read_parquet(r"C:\Users\mohit\Downloads\Cleaned_Data_Final_2025.parquet", columns=["Station", "Timestamp", PM25_COL])
    df = df[df.Station.isin(stations)]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    day_ts = pd.Timestamp(day)
    df = df[df["Timestamp"].dt.normalize() == day_ts]
    minute_of_day = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute
    in_window = (minute_of_day >= WINDOW_START_MIN) & (minute_of_day <= WINDOW_END_MIN)
    win = df[in_window]
    daily = win.groupby("Station")[PM25_COL].mean().rename("pm25").reset_index()
    return daily.rename(columns={"Station": "station_id"})


def make_pair_map(ax_left, ax_right, df, left_col, left_label, left_cmap, right_col, right_label, right_cmap, right_unit):
    for ax, col, label, cmap in [(ax_left, left_col, left_label, left_cmap), (ax_right, right_col, right_label, right_cmap)]:
        sc = ax.scatter(df["lon"], df["lat"], c=df[col], cmap=cmap, s=42, edgecolors="white", linewidths=0.4, zorder=4)
        draw_state_boundaries(ax, linewidth=0.5, alpha=0.6)
        ax.set_xlim(68, 97.5)
        ax.set_ylim(6.5, 37.5)
        ax.set_title(label, fontsize=11.5, fontweight="bold", color=C_INK, loc="left")
        for spine in ax.spines.values():
            spine.set_color(C_AXIS)
        ax.set_xticks([]); ax.set_yticks([])
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=8, color=C_MUTED)


def main():
    with open(os.path.join(DATA_DIR, "clean101_report.json")) as f:
        report = json.load(f)
    stations = set(report["final_stations"])

    pm25 = load_pm25_for_day(REPRESENTATIVE_DAY, stations)
    print(f"PM2.5 loaded for {REPRESENTATIVE_DAY}: {len(pm25)} stations")

    main_df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    day_df = main_df[main_df["date"] == pd.Timestamp(REPRESENTATIVE_DAY)]
    print(f"Main dataset rows for {REPRESENTATIVE_DAY}: {len(day_df)}")

    merged = pm25.merge(day_df, on="station_id", how="inner")
    print(f"Merged (PM2.5 + gases/weather + lat/lon): {len(merged)} stations")

    saved = []
    for i, (var, (label, unit)) in enumerate(VARS.items()):
        if var not in merged.columns:
            print(f"  skip {var}: not in dataset")
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5), dpi=200)
        make_pair_map(ax1, ax2, merged, "pm25", f"PM2.5 (\u00b5g/m\u00b3)", SEQ_ORANGE, var, f"{label} ({unit})", SEQ_BLUE, unit)
        corr = merged["pm25"].corr(merged[var])
        fig.suptitle(f"PM2.5 vs {label} \u2014 {REPRESENTATIVE_DAY}  (station-level Pearson r = {corr:.3f})",
                     fontsize=13.5, fontweight="bold", color=C_INK, x=0.02, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(FIG_DIR, f"pm25_vs_{var}_{REPRESENTATIVE_DAY}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        saved.append((var, corr, out))
        print(f"  saved {var}: r={corr:.3f} -> {out}")

    summary = pd.DataFrame(saved, columns=["variable", "pearson_r_vs_pm25", "file"])
    summary.to_csv(os.path.join(FIG_DIR, "pm25_correlation_summary.csv"), index=False)
    print(f"\nSaved {len(saved)} maps + summary to {FIG_DIR}")


if __name__ == "__main__":
    main()
