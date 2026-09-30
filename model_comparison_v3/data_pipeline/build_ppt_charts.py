"""ADDITIONAL, presentation-only charts - Objective 1's 5-model comparison
bar chart and a static spatial AQI map (both only existed as live Streamlit
components before). Reads only already-saved artifacts, no retraining.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
FIG_DIR = os.path.join(V3_ROOT, "figures", "ppt")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from india_boundary import draw_state_boundaries  # noqa: E402

C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"

MODELS = {
    "linear_regression": "Linear\nRegression", "random_forest": "Random\nForest",
    "xgboost": "XGBoost", "arima": "ARIMA", "cnn_lstm": "CNN-LSTM",
}
BAR_COLORS = {
    "linear_regression": "#c3c2b7", "random_forest": "#eb6834", "xgboost": "#1a7a4c",
    "arima": "#2a78d6", "cnn_lstm": "#898781",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE, "savefig.facecolor": C_SURFACE,
})


def chart_obj1_comparison():
    rows = []
    for key, label in MODELS.items():
        with open(os.path.join(V3_ROOT, "models", "objective1_aqi", "artifacts", key, "metrics.json")) as f:
            m = json.load(f)["test"]
        rows.append({"key": key, "label": label, "r2": m["r2"]})
    df = pd.DataFrame(rows).sort_values("r2", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    bars = ax.bar(df["label"], df["r2"], color=[BAR_COLORS[k] for k in df["key"]], width=0.6, zorder=3)
    for b, v in zip(bars, df["r2"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=12,
                 fontweight="bold", color=C_INK)
    ax.set_ylabel("Test R² (2025 held-out, 101 stations)", fontsize=11, color=C_INK_SECONDARY)
    ax.set_title("Objective 1 — Surface AQI: 5-model comparison", fontsize=15, fontweight="bold",
                  color=C_INK, loc="left", pad=14)
    ax.set_ylim(0, 0.6)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.tick_params(axis="x", labelsize=11, colors=C_INK)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "obj1_model_comparison.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def chart_obj1_spatial_map():
    df = pd.read_csv(os.path.join(V3_ROOT, "data", "all_models_spatial", "xgboost_spatial_latest.csv"))
    cmap = LinearSegmentedColormap.from_list(
        "aqi", ["#00b050", "#92d050", "#ffff00", "#ff9900", "#ff0000", "#7f0000"])

    fig, ax = plt.subplots(figsize=(7.5, 8), dpi=200)
    sc = ax.scatter(df["lon"], df["lat"], c=df["predicted_aqi"], cmap=cmap, vmin=0, vmax=400,
                     s=70, edgecolors="white", linewidths=0.6, zorder=4)
    draw_state_boundaries(ax, linewidth=0.5, alpha=0.6)
    ax.set_xlim(68, 97.5)
    ax.set_ylim(6.5, 37.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C_AXIS)
    ax.set_title(f"Objective 1 — Predicted AQI, {df['date'].iloc[0]}\n(XGBoost, 101 real CPCB stations)",
                  fontsize=13, fontweight="bold", color=C_INK, loc="left", pad=10)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Predicted AQI", fontsize=10, color=C_INK_SECONDARY)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "obj1_spatial_map.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    chart_obj1_comparison()
    chart_obj1_spatial_map()
