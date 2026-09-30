"""Loads the real India state/UT boundary shapefile (geoBoundaries-IND-ADM0
_simplified.geojson - despite the ADM0 name, contains all 36 state/UT ADM1
polygons) for overlay on matplotlib maps, without needing geopandas."""

import json

SHAPEFILE_PATH = r"C:\Users\mohit\Downloads\India_Shapefiles\geoBoundaries-IND-ADM0_simplified.geojson"


def load_state_boundaries():
    """Returns a list of rings (each a list of (lon, lat) tuples) across all
    states/UTs and all polygon parts - ready for matplotlib ax.plot()."""
    with open(SHAPEFILE_PATH, encoding="utf-8") as f:
        gj = json.load(f)
    rings = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                rings.append(ring)
    return rings


def draw_state_boundaries(ax, color="#3a3a38", linewidth=0.6, alpha=0.75, zorder=5):
    rings = load_state_boundaries()
    for ring in rings:
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        ax.plot(lons, lats, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, solid_capstyle="round")
    return rings
