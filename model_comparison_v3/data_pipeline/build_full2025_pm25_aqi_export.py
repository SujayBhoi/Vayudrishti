"""ADDITIONAL deliverable - full 2025 test-period PM2.5 + AQI predictions
(not just January). Reads already-saved test_predictions.csv from the
primary XGBoost model and the two-stage gases->PM2.5->AQI model - no
retraining, no new extraction. Companion to build_jan2025_prediction_chart.py
(same pattern, full year instead of one month).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
FIG_DIR = os.path.join(V3_ROOT, "figures")
DATA_DIR = os.path.join(V3_ROOT, "data")
MODELS_DIR = os.path.join(V3_ROOT, "models", "objective1_aqi", "artifacts")

C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_ACTUAL = "#1a1a1a"
C_XGB = "#2a78d6"
C_TWOSTAGE = "#eb6834"
C_PM25 = "#7a4fb5"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE, "savefig.facecolor": C_SURFACE,
})


def main():
    xgb = pd.read_csv(os.path.join(MODELS_DIR, "xgboost", "test_predictions.csv"), parse_dates=["date"])
    two = pd.read_csv(os.path.join(MODELS_DIR, "two_stage_pm25_aqi", "test_predictions.csv"), parse_dates=["date"])

    xgb_daily = xgb.groupby("date").agg(actual=("aqi", "mean"), pred_xgb=("pred", "mean")).reset_index()
    two_daily = two.groupby("date").agg(pred_two=("pred", "mean"), pm25_actual=("pm25_actual", "mean"),
                                          pm25_pred=("pm25_pred", "mean")).reset_index()
    daily = xgb_daily.merge(two_daily, on="date", how="outer").sort_values("date")

    # ---- Chart 1: full-year AQI, actual vs both models ----
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=200)
    ax.plot(daily["date"], daily["actual"], color=C_ACTUAL, linewidth=1.8, label="Actual AQI (national mean)", zorder=5)
    ax.plot(daily["date"], daily["pred_xgb"], color=C_XGB, linewidth=1.3, linestyle="--", label="XGBoost predicted (primary)", zorder=4, alpha=0.9)
    ax.plot(daily["date"], daily["pred_two"], color=C_TWOSTAGE, linewidth=1.3, linestyle="--", label="Two-stage predicted (PM2.5->AQI)", zorder=3, alpha=0.9)
    ax.set_ylabel("AQI (national daily mean, 101 stations)")
    ax.set_title("Full 2025 test period: actual vs predicted AQI", fontsize=14, fontweight="bold", color=C_INK, loc="left", pad=14)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, "full2025_aqi_prediction_comparison.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved chart -> {out_png}")

    # ---- Chart 2: full-year PM2.5, actual vs predicted (Stage 1 output) ----
    fig2, ax2 = plt.subplots(figsize=(13, 5), dpi=200)
    ax2.plot(daily["date"], daily["pm25_actual"], color=C_ACTUAL, linewidth=1.6, label="Actual PM2.5 (CPCB, national mean)", zorder=4)
    ax2.plot(daily["date"], daily["pm25_pred"], color=C_PM25, linewidth=1.4, linestyle="--", label="Predicted PM2.5 (Stage 1: gases+weather)", zorder=3)
    ax2.set_ylabel("PM2.5 (µg/m³, national daily mean)")
    ax2.set_title("Full 2025 test period: satellite-predicted vs actual PM2.5", fontsize=14, fontweight="bold", color=C_INK, loc="left", pad=14)
    ax2.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.spines["left"].set_color(C_AXIS)
    ax2.spines["bottom"].set_color(C_AXIS)
    ax2.legend(frameon=False, loc="upper right", fontsize=10)
    fig2.tight_layout()
    out_png2 = os.path.join(FIG_DIR, "full2025_pm25_prediction_comparison.png")
    fig2.savefig(out_png2, dpi=200)
    plt.close(fig2)
    print(f"Saved chart -> {out_png2}")

    # ---- Full per-station, per-day export ----
    export = xgb[["station_id", "date", "aqi", "pred"]].rename(columns={"pred": "pred_xgboost_aqi"})
    export = export.merge(
        two[["station_id", "date", "pred", "pm25_actual", "pm25_pred"]].rename(columns={"pred": "pred_two_stage_aqi"}),
        on=["station_id", "date"], how="left",
    )
    out_csv = os.path.join(DATA_DIR, "full2025_predictions_export.csv")
    export.to_csv(out_csv, index=False)
    print(f"Saved export -> {out_csv} ({len(export)} rows, {export.station_id.nunique()} stations)")

    from sklearn.metrics import r2_score, mean_absolute_error
    two_valid = export.dropna(subset=["pred_two_stage_aqi"])
    pm25_valid = export.dropna(subset=["pm25_actual", "pm25_pred"])
    print("\nFull 2025 test-period accuracy (all stations, all test days):")
    print(f"  XGBoost AQI:       R2={r2_score(export.aqi, export.pred_xgboost_aqi):.3f}  MAE={mean_absolute_error(export.aqi, export.pred_xgboost_aqi):.2f}")
    print(f"  Two-stage AQI:     R2={r2_score(two_valid.aqi, two_valid.pred_two_stage_aqi):.3f}  MAE={mean_absolute_error(two_valid.aqi, two_valid.pred_two_stage_aqi):.2f}")
    print(f"  Stage 1 PM2.5:     R2={r2_score(pm25_valid.pm25_actual, pm25_valid.pm25_pred):.3f}  MAE={mean_absolute_error(pm25_valid.pm25_actual, pm25_valid.pm25_pred):.2f}")


if __name__ == "__main__":
    main()
