"""ADDITIONAL deliverable - January 2025 AQI prediction comparison: actual
vs XGBoost (primary direct model) vs the two-stage gases->PM2.5->AQI
reference model. Does not touch the primary pipeline, only reads its
already-saved test_predictions.csv files.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
FIG_DIR = os.path.join(V3_ROOT, "figures")
MODELS_DIR = os.path.join(os.path.dirname(THIS_DIR), "models", "objective1_aqi", "artifacts")

C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_ACTUAL = "#1a1a1a"
C_XGB = "#2a78d6"
C_TWOSTAGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE, "savefig.facecolor": C_SURFACE,
})


def load_daily(model_dir, label):
    df = pd.read_csv(os.path.join(MODELS_DIR, model_dir, "test_predictions.csv"), parse_dates=["date"])
    jan = df[(df.date >= "2025-01-01") & (df.date <= "2025-01-31")]
    daily = jan.groupby("date").agg(actual=("aqi", "mean"), pred=("pred", "mean")).reset_index()
    daily["model"] = label
    return jan, daily


def main():
    xgb_jan, xgb_daily = load_daily("xgboost", "XGBoost")
    two_jan, two_daily = load_daily("two_stage_pm25_aqi", "Two-stage (PM2.5->AQI)")

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    ax.plot(xgb_daily["date"], xgb_daily["actual"], color=C_ACTUAL, linewidth=2.2, label="Actual AQI (national mean)", zorder=5)
    ax.plot(xgb_daily["date"], xgb_daily["pred"], color=C_XGB, linewidth=1.8, linestyle="--", label="XGBoost predicted (primary)", zorder=4)
    ax.plot(two_daily["date"], two_daily["pred"], color=C_TWOSTAGE, linewidth=1.8, linestyle="--", label="Two-stage predicted (PM2.5→AQI)", zorder=3)
    ax.set_ylabel("AQI (national daily mean, 101 stations)")
    ax.set_title("January 2025 test period: actual vs predicted AQI", fontsize=14, fontweight="bold", color=C_INK, loc="left", pad=14)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, "jan2025_aqi_prediction_comparison.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved chart -> {out_png}")

    # per-station, per-day export for the mentor to review directly
    export = xgb_jan[["station_id", "date", "aqi", "pred"]].rename(columns={"pred": "pred_xgboost"})
    export = export.merge(two_jan[["station_id", "date", "pred", "pm25_pred"]].rename(columns={"pred": "pred_two_stage_aqi", "pm25_pred": "pred_pm25"}),
                           on=["station_id", "date"], how="left")
    out_csv = os.path.join(V3_ROOT, "data", "jan2025_predictions_export.csv")
    export.to_csv(out_csv, index=False)
    print(f"Saved export -> {out_csv} ({len(export)} rows, {export.station_id.nunique()} stations)")

    # quick accuracy summary for Jan 2025 specifically
    from sklearn.metrics import r2_score, mean_absolute_error
    print("\nJanuary 2025 test-period accuracy:")
    print(f"  XGBoost:    R2={r2_score(xgb_jan.aqi, xgb_jan.pred):.3f}  MAE={mean_absolute_error(xgb_jan.aqi, xgb_jan.pred):.2f}")
    print(f"  Two-stage:  R2={r2_score(two_jan.aqi, two_jan.pred):.3f}  MAE={mean_absolute_error(two_jan.aqi, two_jan.pred):.2f}")


if __name__ == "__main__":
    main()
