"""Run this ONCE after cloning the repo, before running anything else.

6 files exceeded GitHub's 100MB per-file limit, so they're stored here as
.gz and must be decompressed in place first. This script finds every .gz
under this repo that has a matching pattern from the list below and
decompresses it next to itself (e.g. foo.csv.gz -> foo.csv), then leaves
the .gz alone (safe to re-run).
"""

import gzip
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))

TARGETS = [
    "models/objective2_hcho/artifacts/hcho_flagged.csv.gz",
    "model_comparison_v3/data/model_dataset_clean101_daily.csv.gz",
    "model_comparison_v3/data/pm25_aqi_grid_predictions.csv.gz",
    "model_comparison_v3/models/objective1_aqi/artifacts/random_forest/random_forest.pkl.gz",
    "model_comparison_v3/models/objective1_aqi/artifacts/two_stage_pm25_aqi/stage1_gases_to_pm25.pkl.gz",
    "model_comparison_v3/models/objective2_hcho/artifacts/month_grid_features.csv.gz",
]


def main():
    for rel in TARGETS:
        gz_path = os.path.join(ROOT, os.path.normpath(rel))
        out_path = gz_path[:-3]  # strip ".gz"
        if not os.path.exists(gz_path):
            print(f"SKIP (not found): {rel}")
            continue
        if os.path.exists(out_path):
            print(f"SKIP (already decompressed): {rel}")
            continue
        print(f"Decompressing {rel} ...")
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("\nDone. You can now run: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
