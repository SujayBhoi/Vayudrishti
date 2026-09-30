"""Reads the multi-band HCHO GeoTIFFs produced by fetch_objective2_grid.py
(one band per date, batched ~15 dates/file) out of the local Drive-synced
folder, and converts them into a long-format (grid_id/lat/lon/date/hcho)
table - the input contract detect_hotspots.py expects.

No Google Drive Desktop sync is installed on this machine, so files are
fetched via download_from_drive.py (Drive API, same OAuth creds as Earth
Engine) into data_pipeline/drive_downloads/ps3_obj2_hcho/ - run that first,
or point PS3_DRIVE_DIR at an actual Drive Desktop sync path if one exists.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import rasterio

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "drive_downloads", "ps3_obj2_hcho")
DRIVE_LOCAL_DIR = os.environ.get("PS3_DRIVE_DIR", DEFAULT_DIR)


def band_desc_to_date(desc: str) -> str:
    m = re.search(r"hcho_(\d{4})_(\d{2})_(\d{2})", desc)
    if not m:
        return desc
    y, mo, d = m.groups()
    return f"{y}-{mo}-{d}"


def load_all_hcho_long(tif_dir: str = DRIVE_LOCAL_DIR, downsample_factor: int = 1) -> pd.DataFrame:
    """downsample_factor > 1 subsamples pixels (e.g. every Nth row/col) to
    keep the resulting point table a manageable size for DBSCAN, since the
    native ~1.1km grid over all of India is several million pixels/band."""
    tif_paths = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFFs found in {tif_dir} - sync Drive locally or set PS3_DRIVE_DIR")

    frames = []
    for path in tif_paths:
        with rasterio.open(path) as src:
            transform = src.transform
            for band_idx in range(1, src.count + 1):
                desc = src.descriptions[band_idx - 1] or f"band_{band_idx}"
                date_str = band_desc_to_date(desc)
                arr = src.read(band_idx, out_shape=(
                    src.height // downsample_factor, src.width // downsample_factor
                ))
                rows, cols = np.where(~np.isnan(arr) & (arr != src.nodata if src.nodata is not None else True))
                if len(rows) == 0:
                    continue
                xs, ys = rasterio.transform.xy(transform * transform.scale(downsample_factor, downsample_factor), rows, cols)
                frames.append(pd.DataFrame({
                    "lon": xs, "lat": ys, "hcho": arr[rows, cols], "date": date_str,
                }))
    if not frames:
        return pd.DataFrame(columns=["lon", "lat", "hcho", "date"])
    out = pd.concat(frames, ignore_index=True)
    out["grid_id"] = [f"px{i:08d}" for i in range(len(out))]
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Native ~1.1km pixels; DBSCAN in detect_hotspots.py/hotspot_analysis.py
    # was designed around a ~0.25deg (~28km) grid, so factor=25 lands close
    # to that scale rather than loading hundreds of millions of raw pixels.
    parser.add_argument("--downsample-factor", type=int, default=25)
    args = parser.parse_args()

    df = load_all_hcho_long(downsample_factor=args.downsample_factor)
    print(f"Loaded {len(df)} HCHO pixel-date records across {df['date'].nunique()} dates")
    out_path = os.path.join(os.path.dirname(__file__), "legacy_v1_data", "processed", "hcho_grid_national.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
