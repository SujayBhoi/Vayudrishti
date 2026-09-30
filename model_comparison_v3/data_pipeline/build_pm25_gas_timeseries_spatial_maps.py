"""ADDITIONAL, standalone analysis layer - time-series-spatial version of
build_pm25_gas_spatial_maps.py. Same PM2.5-vs-gas/weather spatial map idea,
but repeated across 4 seasonally-representative dates in the 2025 test
period (winter/pre-monsoon/monsoon/post-monsoon-stubble) in ONE composite
figure per variable, so the spatial pattern's evolution across the year is
visible at a glance rather than scattered across dozens of separate files.

Does not modify or depend on the primary CNN-LSTM pipeline, Streamlit UI,
or Objective 2 - reads only the same already-processed satellite gas/
weather dataset and pulls PM2.5 fresh from CPCB (see the single-day
version's docstring for why PM2.5 specifically has to come from CPCB -
no satellite instrument measures particulate mass).
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(V3_ROOT, "data")
FIG_DIR = os.path.join(V3_ROOT, "figures", "pm25_analysis")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from india_boundary import draw_state_boundaries  # noqa: E402

PM25_COL = "PM2.5 (\xb5g/m\xb3)"
WINDOW_START_MIN = 11 * 60 + 18
WINDOW_END_MIN = 15 * 60 + 18

# best-coverage day found per season (101/101 or 100/101 stations)
DATES = [
    ("2025-01-12", "Winter"),
    ("2025-04-03", "Pre-monsoon"),
    ("2025-07-01", "Monsoon"),
    ("2025-10-01", "Post-monsoon / stubble season"),
]

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
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fde3d2", "#f8bd97", "#f2955c", "#eb6834", "#c94f1f", "#9c3a14"])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE, "savefig.facecolor": C_SURFACE,
})


def load_pm25_for_day(day: str, stations: set, pm25_all: pd.DataFrame) -> pd.DataFrame:
    day_ts = pd.Timestamp(day)
    sub = pm25_all[pm25_all["date"] == day_ts]
    return sub[["Station", "pm25"]].rename(columns={"Station": "station_id"})


def draw_one_panel(ax, df, col, cmap):
    sc = ax.scatter(df["lon"], df["lat"], c=df[col], cmap=cmap, s=26, edgecolors="white", linewidths=0.3, zorder=4)
    draw_state_boundaries(ax, linewidth=0.4, alpha=0.55)
    ax.set_xlim(68, 97.5)
    ax.set_ylim(6.5, 37.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C_AXIS)
    return sc


def main():
    with open(os.path.join(DATA_DIR, "clean101_report.json")) as f:
        report = json.load(f)
    stations = set(report["final_stations"])

    print("Loading full-year PM2.5 (overpass-windowed) ...")
    df25 = pd.read_parquet(r"C:\Users\mohit\Downloads\Cleaned_Data_Final_2025.parquet", columns=["Station", "Timestamp", PM25_COL])
    df25 = df25[df25.Station.isin(stations)]
    df25["Timestamp"] = pd.to_datetime(df25["Timestamp"])
    minute_of_day = df25["Timestamp"].dt.hour * 60 + df25["Timestamp"].dt.minute
    in_window = (minute_of_day >= WINDOW_START_MIN) & (minute_of_day <= WINDOW_END_MIN)
    win = df25[in_window].copy()
    win["date"] = win["Timestamp"].dt.normalize()
    pm25_all = win.groupby(["Station", "date"])[PM25_COL].mean().rename("pm25").reset_index()

    main_df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])

    for var, (label, unit) in VARS.items():
        if var not in main_df.columns:
            print(f"skip {var}: not in dataset")
            continue

        fig, axes = plt.subplots(len(DATES), 2, figsize=(10.5, 4.1 * len(DATES)), dpi=180)
        corrs = []
        for row, (date_str, season) in enumerate(DATES):
            pm25_day = load_pm25_for_day(date_str, stations, pm25_all)
            day_df = main_df[main_df["date"] == pd.Timestamp(date_str)]
            merged = pm25_day.merge(day_df, on="station_id", how="inner")
            corr = merged["pm25"].corr(merged[var])
            corrs.append(corr)

            ax_l, ax_r = axes[row]
            sc1 = draw_one_panel(ax_l, merged, "pm25", SEQ_ORANGE)
            sc2 = draw_one_panel(ax_r, merged, var, SEQ_BLUE)
            ax_l.set_ylabel(f"{season}\n{date_str}", fontsize=10.5, fontweight="bold", color=C_INK, rotation=0, ha="right", va="center", labelpad=40)
            if row == 0:
                ax_l.set_title("PM2.5 (\u00b5g/m\u00b3)", fontsize=12, fontweight="bold", color=C_INK)
                ax_r.set_title(f"{label} ({unit})", fontsize=12, fontweight="bold", color=C_INK)
            ax_l.text(0.02, 0.02, f"r={corr:.2f}", transform=ax_l.transAxes, fontsize=9, color=C_INK_SECONDARY,
                       bbox=dict(boxstyle="round", facecolor=C_SURFACE, edgecolor=C_AXIS, alpha=0.85))
            plt.colorbar(sc1, ax=ax_l, fraction=0.045, pad=0.02).ax.tick_params(labelsize=7, color=C_MUTED)
            plt.colorbar(sc2, ax=ax_r, fraction=0.045, pad=0.02).ax.tick_params(labelsize=7, color=C_MUTED)

        fig.suptitle(f"PM2.5 vs {label} across 2025 seasons", fontsize=15, fontweight="bold", color=C_INK, x=0.02, ha="left", y=0.995)
        fig.tight_layout(rect=[0.03, 0, 1, 0.98])
        out = os.path.join(FIG_DIR, f"pm25_vs_{var}_timeseries_2025.png")
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"saved {var}: r by season = {[round(c,3) for c in corrs]} -> {out}")


if __name__ == "__main__":
    main()
