"""Presentation-quality standalone chart PNGs for Objective 2 (statistical
baseline vs ConvLSTM). Palette/form choices follow the dataviz skill's
reference palette (references/palette.md): categorical slot 1 (blue) and
slot 2 (orange) for the two-model comparison, fixed order throughout; one
axis per chart (no dual-axis - RMSE and R2 are always separate panels/
files); sequential blue ramp for the spatial magnitude map; thin marks,
recessive gridlines, direct labels.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from india_boundary import draw_state_boundaries  # noqa: E402

V3_ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(V3_ROOT, "data")
FIG_DIR = os.path.join(V3_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- palette (light mode, dataviz skill reference palette) ---
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_SERIES1_BASELINE = "#eb6834"   # categorical slot 2 (orange) - baseline
C_SERIES1_CONVLSTM = "#2a78d6"   # categorical slot 1 (blue) - ConvLSTM
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "axes.edgecolor": C_AXIS,
    "axes.labelcolor": C_INK_SECONDARY,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "text.color": C_INK,
    "figure.facecolor": C_SURFACE,
    "axes.facecolor": C_SURFACE,
    "savefig.facecolor": C_SURFACE,
})


def style_axes(ax, ygrid=True):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    if ygrid:
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def chart_regional_r2():
    with open(os.path.join(os.path.dirname(THIS_DIR), "models", "objective2_hcho", "artifacts", "evaluation_summary.json")) as f:
        summary = json.load(f)
    regions = ["corridor", "rest_of_india", "national_full"]
    labels = ["Corridor\n(Punjab-Haryana-Delhi-UP)", "Rest of India", "National\n(full grid)"]
    baseline_r2 = [summary["regional_breakdown"][r]["statistical_baseline"]["r2"] for r in regions]
    convlstm_r2 = [summary["regional_breakdown"][r]["convlstm"]["r2"] for r in regions]

    x = np.arange(len(regions))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    b1 = ax.bar(x - width / 2, baseline_r2, width, color=C_SERIES1_BASELINE, label="Statistical baseline (climatology)", zorder=3)
    b2 = ax.bar(x + width / 2, convlstm_r2, width, color=C_SERIES1_CONVLSTM, label="ConvLSTM", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4 if h >= 0 else -14),
                        ha="center", fontsize=9, color=C_INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Test R²")
    ax.set_title("Objective 2: HCHO prediction R² by region", fontsize=13, fontweight="bold", color=C_INK, loc="left", pad=14)
    ax.axhline(0, color=C_AXIS, linewidth=1)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "obj2_regional_r2.png"), dpi=200)
    plt.close(fig)


MODEL_ORDER = [
    ("statistical_baseline", "Statistical\nbaseline", "#eb6834"),   # slot 2 orange
    ("persistence", "Persistence\n(naive)", "#eda100"),             # slot 4 yellow
    ("linear_regression", "Linear\nRegression", "#4a3aa7"),         # slot 7 violet
    ("random_forest", "Random\nForest", "#1baf7a"),                 # slot 3 aqua
    ("convlstm", "ConvLSTM", "#2a78d6"),                            # slot 1 blue
]


def _load_all_model_metrics():
    base_dir = os.path.join(os.path.dirname(THIS_DIR), "models", "objective2_hcho", "artifacts")
    out = []
    for key, label, color in MODEL_ORDER:
        path = os.path.join(base_dir, key, "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        out.append((key, label, color, m))
    return out


def chart_metrics_comparison():
    all_models = _load_all_model_metrics()
    labels = [m[1] for m in all_models]
    colors = [m[2] for m in all_models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)

    ax = axes[0]
    r2_vals = [m[3]["test"]["r2"] for m in all_models]
    bars = ax.bar(labels, r2_vals, color=colors, width=0.6, zorder=3)
    for bar, v in zip(bars, r2_vals):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points",
                     xytext=(0, 4 if v >= 0 else -14), ha="center", fontsize=9.5, color=C_INK_SECONDARY)
    ax.set_title("Test R² (national, full grid)", fontsize=11, color=C_INK, loc="left")
    ax.axhline(0, color=C_AXIS, linewidth=1)
    ax.tick_params(axis="x", labelsize=8.5)
    style_axes(ax)

    ax2 = axes[1]
    rmse_vals = [m[3]["test"]["rmse"] * 1e5 for m in all_models]
    bars2 = ax2.bar(labels, rmse_vals, color=colors, width=0.6, zorder=3)
    for bar, v in zip(bars2, rmse_vals):
        ax2.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9.5, color=C_INK_SECONDARY)
    ax2.set_title("Test RMSE (×10⁻⁵ mol/m²)", fontsize=11, color=C_INK, loc="left")
    ax2.tick_params(axis="x", labelsize=8.5)
    style_axes(ax2)

    fig.suptitle("Objective 2: 5-model comparison", fontsize=13, fontweight="bold", color=C_INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "obj2_metrics_comparison.png"), dpi=200)
    plt.close(fig)


def chart_cv_stability():
    with open(os.path.join(os.path.dirname(THIS_DIR), "models", "objective2_hcho", "artifacts", "evaluation_summary.json")) as f:
        summary = json.load(f)
    cv = summary["rolling_origin_cv"]
    n_folds = cv["statistical_baseline"]["n_folds"]
    fold_labels = [f"Fold {i}" for i in range(n_folds)]
    base_r2 = [f["r2"] for f in cv["statistical_baseline"]["fold_details"]]
    conv_r2 = [f["r2"] for f in cv["convlstm"]["fold_details"]]

    x = np.arange(n_folds)
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    b1 = ax.bar(x - width / 2, base_r2, width, color=C_SERIES1_BASELINE, label="Statistical baseline", zorder=3)
    b2 = ax.bar(x + width / 2, conv_r2, width, color=C_SERIES1_CONVLSTM, label="ConvLSTM", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4 if h >= 0 else -14),
                        ha="center", fontsize=9, color=C_INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.axhline(0, color=C_AXIS, linewidth=1)
    ax.set_ylabel("Test R²")
    ax.set_title("Rolling-origin CV: ConvLSTM beats the baseline in every fold", fontsize=12.5, fontweight="bold", color=C_INK, loc="left", pad=14)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "obj2_cv_stability.png"), dpi=200)
    plt.close(fig)


def chart_channel_ablation():
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(THIS_DIR), "models", "objective2_hcho", "artifacts", "convlstm", "channel_ablation.csv"))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    bars = ax.barh(df["channel"], df["r2_drop_when_zeroed"], color=C_SERIES1_CONVLSTM, zorder=3)
    for bar, v in zip(bars, df["r2_drop_when_zeroed"]):
        ax.annotate(f"{v:.4f}", (v, bar.get_y() + bar.get_height() / 2), textcoords="offset points", xytext=(4, 0), va="center", fontsize=9.5, color=C_INK_SECONDARY)
    ax.set_xlabel("R² drop when this channel is zeroed out")
    ax.set_title("ConvLSTM channel importance (ablation)", fontsize=12.5, fontweight="bold", color=C_INK, loc="left", pad=12)
    ax.axvline(0, color=C_AXIS, linewidth=1)
    style_axes(ax, ygrid=False)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "obj2_channel_ablation.png"), dpi=200)
    plt.close(fig)


def chart_spatial_hcho_map():
    data = np.load(os.path.join(DATA_DIR, "spatiotemporal_grid.npz"))
    tensor, mask, lats, lons = data["tensor"], data["mask"], data["lats"], data["lons"]
    with open(os.path.join(DATA_DIR, "spatiotemporal_grid_manifest.json")) as f:
        manifest = json.load(f)
    last_month_idx = manifest["n_months"] - 1
    hcho = tensor[last_month_idx, 0].copy()
    hcho[~mask] = np.nan

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(6.5, 7), dpi=200)
    im = ax.imshow(hcho, origin="lower", extent=[lons.min(), lons.max(), lats.min(), lats.max()], cmap=cmap, aspect="auto")
    draw_state_boundaries(ax)
    ax.set_xlim(lons.min(), lons.max()); ax.set_ylim(lats.min(), lats.max())
    ax.set_title(f"National HCHO column density—{manifest['all_months'][last_month_idx]}", fontsize=12.5, fontweight="bold", color=C_INK, loc="left", pad=12)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("HCHO (mol/m²)", color=C_INK_SECONDARY)
    cbar.ax.yaxis.set_tick_params(color=C_MUTED)
    for spine in ax.spines.values():
        spine.set_color(C_AXIS)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "obj2_hcho_spatial_map.png"), dpi=200)
    plt.close(fig)


def main():
    chart_metrics_comparison()
    chart_regional_r2()
    chart_cv_stability()
    chart_channel_ablation()
    chart_spatial_hcho_map()
    print(f"Saved 5 charts to {FIG_DIR}")


if __name__ == "__main__":
    main()
