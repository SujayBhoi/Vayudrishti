"""Charts for the predicted-hotspot analysis (build_hotspot_prediction_analysis.py):
regional precision/recall, and a spatial map of where ConvLSTM's predicted
hotspots land vs where hotspots actually occurred, for one sample test month.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from india_boundary import draw_state_boundaries  # noqa: E402

V3_ROOT = os.path.dirname(THIS_DIR)
FIG_DIR = os.path.join(V3_ROOT, "figures")
ARTIFACTS_DIR = os.path.join(os.path.dirname(THIS_DIR), "models", "objective2_hcho", "artifacts")

C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_PRECISION = "#eb6834"
C_RECALL = "#2a78d6"
C_TP = "#1baf7a"   # correctly predicted hotspot - green
C_FN = "#e34948"   # missed hotspot - red
C_FP = "#eda100"   # false alarm - yellow
C_BG = "#dfe7f0"   # background india cell

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "axes.edgecolor": C_AXIS, "axes.labelcolor": C_INK_SECONDARY,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "text.color": C_INK,
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE, "savefig.facecolor": C_SURFACE,
})


def style_axes(ax, ygrid=True):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    if ygrid:
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def chart_regional_precision_recall():
    with open(os.path.join(ARTIFACTS_DIR, "hotspot_prediction_analysis.json")) as f:
        data = json.load(f)
    results = data["results"]
    regions = ["National (all test months pooled)", "Corridor (Punjab-Haryana-Delhi-UP)",
               "East (Odisha/WB/Assam/NE)", "South (Kerala/TN/Karnataka/AP)", "West (Gujarat/Maharashtra)"]
    labels = ["National", "Corridor\n(Punjab-Haryana)", "East", "South", "West"]
    precision = [results[r]["precision"] for r in regions]
    recall = [results[r]["recall"] for r in regions]

    x = np.arange(len(regions))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    b1 = ax.bar(x - width / 2, precision, width, color=C_PRECISION, label="Precision", zorder=3)
    b2 = ax.bar(x + width / 2, recall, width, color=C_RECALL, label="Recall", zorder=3)
    ax.axhline(0.05, color=C_MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.annotate("random-guess baseline (5%)", (len(regions) - 0.6, 0.052), fontsize=8.5, color=C_MUTED)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9, color=C_INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Where ConvLSTM finds real hotspots — by region", fontsize=13, fontweight="bold", color=C_INK, loc="left", pad=14)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "obj2_hotspot_regional_precision_recall.png"), dpi=200)
    plt.close(fig)


def chart_predicted_vs_actual_map():
    grids = np.load(os.path.join(ARTIFACTS_DIR, "hotspot_prediction_grids.npz"), allow_pickle=True)
    actual_hot, pred_hot, mask = grids["actual_hot"], grids["pred_hot"], grids["mask"]
    lats, lons = grids["lats"], grids["lons"]

    month_idx = actual_hot.shape[0] - 1  # last test month, matches the other spatial map (Dec 2025)
    a, p, m = actual_hot[month_idx], pred_hot[month_idx], mask

    # classify each cell: 0=out of India, 1=background(neither), 2=false positive, 3=false negative(missed), 4=true positive
    cls = np.zeros(a.shape, dtype=int)
    cls[m & ~a & ~p] = 1
    cls[m & ~a & p] = 2
    cls[m & a & ~p] = 3
    cls[m & a & p] = 4

    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ["#00000000", C_BG, C_FP, C_FN, C_TP]  # transparent, bg, fp, fn, tp
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(6.8, 7.4), dpi=200)
    ax.imshow(cls, origin="lower", extent=[lons.min(), lons.max(), lats.min(), lats.max()], cmap=cmap, norm=norm, aspect="auto")
    draw_state_boundaries(ax, color="#5a5a56", linewidth=0.5, alpha=0.6)
    ax.set_xlim(lons.min(), lons.max()); ax.set_ylim(lats.min(), lats.max())
    ax.set_title("Predicted vs actual HCHO hotspots — Dec 2025 (test month)", fontsize=12.5, fontweight="bold", color=C_INK, loc="left", pad=12)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    for spine in ax.spines.values():
        spine.set_color(C_AXIS)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=C_TP, label=f"Correctly predicted ({int((cls==4).sum())} cells)"),
        Patch(facecolor=C_FN, label=f"Missed hotspot ({int((cls==3).sum())} cells)"),
        Patch(facecolor=C_FP, label=f"False alarm ({int((cls==2).sum())} cells)"),
        Patch(facecolor=C_BG, label="Background (no hotspot either way)"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", frameon=True, facecolor=C_SURFACE, edgecolor=C_AXIS, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "obj2_hotspot_predicted_vs_actual_map.png"), dpi=200)
    plt.close(fig)


def main():
    chart_regional_precision_recall()
    chart_predicted_vs_actual_map()
    print(f"Saved 2 charts to {FIG_DIR}")


if __name__ == "__main__":
    main()
