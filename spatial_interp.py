"""Genuine IDW (inverse-distance-weighted) interpolation of station-level
model predictions onto a continuous grid, clipped to the real India
boundary - a real interpolation of real predictions at the 101 stations,
not a fabricated/independently-trained grid model (none of the 5 Objective
1 models were trained on gridded features, only station-level ones - see
PROJECT_DOCUMENTATION.md and this session's earlier scoping decision)."""

from __future__ import annotations

import io

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from shapely.geometry import shape
from shapely.prepared import prep

# Continuous graduated scale (not discrete bins) across the same CPCB
# category anchor colors, so severity is visually obvious at a glance.
AQI_CMAP = LinearSegmentedColormap.from_list(
    "aqi_continuous",
    [(0 / 500, "#00b050"), (50 / 500, "#92d050"), (100 / 500, "#ffff00"),
     (200 / 500, "#ff9900"), (300 / 500, "#ff0000"), (400 / 500, "#7f0000"), (500 / 500, "#4b0000")],
)
AQI_NORM = Normalize(vmin=0, vmax=500, clip=True)


def idw_interpolate(lats, lons, values, grid_lat, grid_lon, power=2.0, eps=1e-6):
    """Vectorized IDW: grid_lat/grid_lon are 2D meshgrid arrays."""
    lats = np.asarray(lats)[:, None, None]
    lons = np.asarray(lons)[:, None, None]
    values = np.asarray(values)[:, None, None]
    d = np.sqrt((grid_lat[None, :, :] - lats) ** 2 + (grid_lon[None, :, :] - lons) ** 2) + eps
    w = 1.0 / (d ** power)
    return (w * values).sum(axis=0) / w.sum(axis=0)


def boundary_mask(boundary_geojson: dict, grid_lat: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    polys = [shape(f["geometry"]) for f in boundary_geojson["features"]]
    prepared = [prep(p) for p in polys]
    from shapely.geometry import Point
    mask = np.zeros(grid_lat.shape, dtype=bool)
    flat_lat, flat_lon = grid_lat.ravel(), grid_lon.ravel()
    for i in range(len(flat_lat)):
        pt = Point(flat_lon[i], flat_lat[i])
        if any(p.contains(pt) for p in prepared):
            mask.flat[i] = True
    return mask


def build_grid(boundary_geojson: dict, resolution: int = 120):
    polys = [shape(f["geometry"]) for f in boundary_geojson["features"]]
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    lon_lin = np.linspace(minx, maxx, resolution)
    lat_lin = np.linspace(miny, maxy, resolution)
    grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)
    return grid_lat, grid_lon, (minx, miny, maxx, maxy)


def render_overlay_png(grid_lat, grid_lon, values, mask, bounds):
    rgba = AQI_CMAP(AQI_NORM(values))
    rgba[..., 3] = np.where(mask, 0.6, 0.0)

    fig, ax = plt.subplots(figsize=(grid_lat.shape[1] / 100, grid_lat.shape[0] / 100), dpi=100)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(rgba, origin="lower", extent=bounds, aspect="auto")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
