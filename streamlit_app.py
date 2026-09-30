"""VayuDrishti Streamlit dashboard - replaces the earlier React frontend.

Objective 1 (AQI): all 5 trained models (CNN-LSTM default/primary per mentor
guidance, others selectable) shown on a spatially-interpolated heatmap over
the corrected India boundary (Survey-of-India-compliant, includes Aksai
Chin/Shaksgam/POK - see frontend/public/data/india_boundary.geojson), plus
the real test-set predictions (actual vs predicted, per station) so the
underlying evidence is visible, not just a headline number.

Objective 2 (HCHO): existing anomaly-threshold hotspots for now; the new
Getis-Ord Gi* / Anselin Local Moran's I comparison and wind+fire seasonal
analysis are built separately and wired in once ready.
"""

from __future__ import annotations

import base64
import json
import os
import sys

import folium
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from spatial_interp import build_grid, boundary_mask, idw_interpolate, render_overlay_png

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_comparison_v3", "data_pipeline"))
from india_boundary import draw_state_boundaries  # noqa: E402

# Same rule-based CPCB-category advisory text as backend/health_advisory.py
ADVISORY = {
    "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
    "Satisfactory": "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    "Moderate": "Sensitive groups (children, elderly, people with respiratory or heart conditions) should limit prolonged outdoor exertion.",
    "Poor": "Sensitive groups should avoid outdoor exertion. General public may experience breathing discomfort on prolonged exposure.",
    "Very Poor": "Sensitive groups should avoid all outdoor activity. General public should limit outdoor exertion.",
    "Severe": "Everyone should avoid outdoor activity. Sensitive groups should remain indoors and keep activity levels low.",
}
SAFEGUARDS = {
    "Good": ["No special precautions needed.", "Good time for outdoor exercise and ventilation."],
    "Satisfactory": ["Sensitive individuals can reduce prolonged outdoor exertion.", "Ventilate homes normally."],
    "Moderate": ["Children, elderly, and those with asthma/heart conditions should limit prolonged outdoor exertion.", "Consider a mask (N95) for sensitive groups during outdoor activity.", "Keep windows closed during peak traffic hours."],
    "Poor": ["Avoid outdoor exercise; shift workouts indoors.", "Use N95/N99 masks outdoors.", "Run air purifiers indoors if available.", "Sensitive groups should consult a doctor if symptoms (breathlessness, irritation) appear."],
    "Very Poor": ["Avoid outdoor activity entirely for sensitive groups; general public should minimize time outdoors.", "Keep windows/doors closed, use air purifiers.", "Wear a well-fitted N95/N99 mask if going outside is unavoidable.", "Schools may consider limiting outdoor activities."],
    "Severe": ["Stay indoors; avoid all outdoor physical activity.", "Use air purifiers; seal gaps under doors/windows.", "Seek medical attention promptly for breathing difficulty, chest pain, or persistent cough.", "Vulnerable groups (children, elderly, pregnant, pre-existing respiratory/cardiac conditions) at highest risk - extra caution advised."],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.join(BASE_DIR, "model_comparison_v3")
OBJ1_ARTIFACTS = os.path.join(V3_DIR, "models", "objective1_aqi", "artifacts")
SPATIAL_DIR = os.path.join(V3_DIR, "data", "all_models_spatial")
BOUNDARY_PATH = os.path.join(BASE_DIR, "frontend", "public", "data", "india_boundary.geojson")
OBJ2_ARTIFACTS_LEGACY = os.path.join(BASE_DIR, "models", "objective2_hcho", "artifacts")

FEATURE_IMPORTANCE_FILES = {
    "cnn_lstm": ("permutation_importance.csv", "feature", "r2_drop"),
    "xgboost": ("feature_importances.csv", "feature", "importance"),
    "random_forest": ("feature_importances.csv", "feature", "importance"),
    "linear_regression": ("standardized_coefficients.csv", "feature", "standardized_coef"),
}

MODELS = {
    "CNN-LSTM": "cnn_lstm",
    "XGBoost": "xgboost",
    "ARIMA": "arima",
    "Random Forest": "random_forest",
    "Linear Regression": "linear_regression",
}
MODEL_NOTES = {
    "CNN-LSTM": "Primary model for this project (mentor guidance). Deep temporal model over satellite/weather features + lag/rolling history.",
    "XGBoost": "Highest raw test R2 of the 5. Gradient-boosted trees over the same 65 features.",
    "ARIMA": "Deliberately univariate - each station's own AQI history only, no satellite/weather features. Its spatial pattern reflects per-station historical dynamics, not a spatial covariate.",
    "Random Forest": "Tree ensemble over the same 65 features as XGBoost/Linear Regression/CNN-LSTM.",
    "Linear Regression": "Simplest baseline - standardized linear fit over the same 65 features.",
}

AQI_COLORS = {
    "Good": "#00b050", "Satisfactory": "#92d050", "Moderate": "#ffff00",
    "Poor": "#ff9900", "Very Poor": "#ff0000", "Severe": "#7f0000",
}

st.set_page_config(page_title="VayuDrishti", layout="wide", page_icon="\U0001F32B")


@st.cache_data
def load_boundary():
    with open(BOUNDARY_PATH) as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    """Computed directly from each model's real test_predictions.csv (the
    same file driving the map/testing-data table below), not metrics.json -
    guarantees the reported numbers always match what's actually shown,
    even if metrics.json gets regenerated by a different run later."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rows = []
    for label, key in MODELS.items():
        path = os.path.join(OBJ1_ARTIFACTS, key, "test_predictions.csv")
        d = pd.read_csv(path)
        y_true, y_pred = d["aqi"], d["pred"]
        mape = float((( y_true - y_pred).abs() / y_true.clip(lower=1.0)).mean() * 100)
        rows.append({
            "Model": label,
            "R2": r2_score(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": mape,
        })
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


@st.cache_data
def load_spatial(model_key: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(SPATIAL_DIR, f"{model_key}_spatial_latest.csv"))


@st.cache_data
def load_all_test_predictions() -> pd.DataFrame:
    return pd.read_csv(os.path.join(SPATIAL_DIR, "all_models_test_predictions.csv"), parse_dates=["date"])


@st.cache_data
def load_feature_importance(model_key: str):
    if model_key not in FEATURE_IMPORTANCE_FILES:
        return None
    fname, feat_col, val_col = FEATURE_IMPORTANCE_FILES[model_key]
    path = os.path.join(OBJ1_ARTIFACTS, model_key, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["abs_val"] = df[val_col].abs()
    return df.sort_values("abs_val", ascending=False).head(8)[[feat_col, val_col]].rename(
        columns={feat_col: "feature", val_col: "importance"})


@st.cache_data
def _raw_features_full():
    return pd.read_csv(os.path.join(V3_DIR, "data", "model_dataset_clean101_daily.csv"), parse_dates=["date"])


def load_raw_features_row(station_id: str, date_str: str):
    df = _raw_features_full()
    row = df[(df["station_id"] == station_id) & (df["date"] == pd.Timestamp(date_str))]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def aqi_category_label(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def aqi_color(aqi: float) -> str:
    if aqi <= 50:
        return AQI_COLORS["Good"]
    if aqi <= 100:
        return AQI_COLORS["Satisfactory"]
    if aqi <= 200:
        return AQI_COLORS["Moderate"]
    if aqi <= 300:
        return AQI_COLORS["Poor"]
    if aqi <= 400:
        return AQI_COLORS["Very Poor"]
    return AQI_COLORS["Severe"]


@st.cache_data
def get_grid_and_mask(_boundary_geojson_str: str, resolution: int = 130):
    boundary_geojson = json.loads(_boundary_geojson_str)
    grid_lat, grid_lon, bounds = build_grid(boundary_geojson, resolution=resolution)
    mask = boundary_mask(boundary_geojson, grid_lat, grid_lon)
    return grid_lat, grid_lon, bounds, mask


@st.cache_data
def compute_idw_overlay(cache_key: str, spatial_df: pd.DataFrame, _boundary_geojson_str: str):
    grid_lat, grid_lon, bounds, mask = get_grid_and_mask(_boundary_geojson_str)
    values = idw_interpolate(spatial_df["lat"], spatial_df["lon"], spatial_df["predicted_aqi"], grid_lat, grid_lon)
    png_bytes = render_overlay_png(grid_lat, grid_lon, values, mask, bounds)
    return png_bytes, bounds


@st.cache_data
def compute_variable_grids(cache_key: str, day_df: pd.DataFrame, columns: tuple, _boundary_geojson_str: str):
    """Same IDW machinery as compute_idw_overlay, generalized to any set of
    columns at once - reused so every gas/PM2.5/AQI panel is interpolated
    from the SAME real 4,447-point grid, at the SAME resolution/mask,
    consistent with the continuous AQI map above."""
    grid_lat, grid_lon, bounds, mask = get_grid_and_mask(_boundary_geojson_str)
    out = {}
    for col in columns:
        values = idw_interpolate(day_df["lat"], day_df["lon"], day_df[col], grid_lat, grid_lon)
        out[col] = np.where(mask, values, np.nan)
    return out, bounds


@st.cache_data
def load_all_test_predictions_indexed():
    df = load_all_test_predictions().copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def spatial_df_for_date(model_label: str, date_str: str) -> pd.DataFrame:
    all_preds = load_all_test_predictions_indexed()
    sub = all_preds[(all_preds["model"] == model_label) & (all_preds["date"] == date_str)].copy()
    sub = sub.rename(columns={"pred": "predicted_aqi", "aqi": "actual_aqi"})
    sub["aqi_category"] = sub["predicted_aqi"].apply(aqi_category_label)
    return sub


def render_map(spatial_df: pd.DataFrame, boundary_geojson: dict, cache_key: str, value_col: str = "predicted_aqi", use_idw: bool = True):
    m = folium.Map(location=[22.9734, 80.6569], zoom_start=5, tiles="CartoDB positron")

    folium.GeoJson(
        boundary_geojson,
        style_function=lambda _: {"color": "#2b3a55", "weight": 1.6, "fillColor": "#dbe6f5", "fillOpacity": 0.05},
    ).add_to(m)

    if use_idw:
        png_bytes, bounds = compute_idw_overlay(cache_key, spatial_df, json.dumps(boundary_geojson))
        minx, miny, maxx, maxy = bounds
        data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        folium.raster_layers.ImageOverlay(
            image=data_uri, bounds=[[miny, minx], [maxy, maxx]], opacity=1.0, name="IDW interpolation",
        ).add_to(m)
    else:
        heat_data = [[r.lat, r.lon, max(getattr(r, value_col), 1.0)] for r in spatial_df.itertuples()]
        HeatMap(heat_data, radius=32, blur=24, min_opacity=0.35, max_zoom=6).add_to(m)

    for r in spatial_df.itertuples():
        # "bullet" marker: white outer ring + colored core, clearly distinct from the raster fill
        folium.CircleMarker(
            location=[r.lat, r.lon], radius=9, color="white", fill=True, fill_color="white",
            fill_opacity=1.0, weight=2,
        ).add_to(m)
        folium.CircleMarker(
            location=[r.lat, r.lon],
            radius=6,
            color="#1a2634",
            fill=True,
            fill_color=aqi_color(getattr(r, value_col)),
            fill_opacity=1.0,
            weight=1.2,
            popup=folium.Popup(
                f"<b>{r.city}</b><br>{r.station_id}<br>Predicted AQI: {getattr(r, value_col):.1f}<br>"
                f"Actual AQI: {r.actual_aqi:.1f}<br>Category: {r.aqi_category}<br>Date: {r.date}",
                max_width=250,
            ),
        ).add_to(m)

    return m


def objective1_view(boundary_geojson):
    st.subheader("Objective 1 · Surface AQI Prediction")

    metrics_df = load_metrics()
    model_label = st.selectbox("Model", list(MODELS.keys()), index=list(MODELS.keys()).index("CNN-LSTM"))
    model_key = MODELS[model_label]
    st.caption(MODEL_NOTES[model_label])

    all_preds_idx = load_all_test_predictions_indexed()
    model_dates = sorted(all_preds_idx[all_preds_idx["model"] == model_label]["date"].unique())
    selected_date = st.select_slider("Test date (real 2025 held-out data)", options=model_dates, value=model_dates[-1])

    row = metrics_df[metrics_df["Model"] == model_label].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test R²", f"{row.R2:.3f}")
    c2.metric("RMSE", f"{row.RMSE:.1f}")
    c3.metric("MAE", f"{row.MAE:.1f}")
    c4.metric("MAPE", f"{row.MAPE:.1f}%" if pd.notna(row.MAPE) else "n/a")

    map_col, table_col = st.columns([3, 2])
    with map_col:
        st.markdown(f"**Spatial prediction map — {selected_date}** (IDW interpolation of real station predictions for this exact date, clipped to the corrected India boundary)")
        spatial_df = spatial_df_for_date(model_label, selected_date)
        if spatial_df.empty:
            st.warning(f"No {model_label} predictions for {selected_date} (this station may have insufficient lookback history for that date) — showing latest available instead.")
            spatial_df = load_spatial(model_key)
        m = render_map(spatial_df, boundary_geojson, f"{model_key}_{selected_date}")
        st_folium(m, width=None, height=520, key=f"map_{model_key}_{selected_date}")
        st.caption("Continuous color gradient (green→yellow→orange→red→maroon) = IDW-interpolated surface between real station predictions for the selected date. White-ringed dots = actual CPCB station locations, clearly distinct from the interpolated fill — not a grid-trained model (none of the 5 models were trained on gridded features, only station-level ones).")

    with table_col:
        st.markdown("**5-model comparison** (test set)")
        highlight = metrics_df.style.apply(
            lambda s: ["background-color: #fff3cd" if v == model_label else "" for v in metrics_df["Model"]], axis=0
        )
        st.dataframe(
            metrics_df.style.format({"R2": "{:.3f}", "RMSE": "{:.1f}", "MAE": "{:.1f}", "MAPE": "{:.1f}"}),
            hide_index=True, width="stretch",
        )
        st.bar_chart(metrics_df.set_index("Model")["R2"])

    st.markdown(f"### Region view — {selected_date}")
    region_col1, region_col2 = st.columns([1, 2])
    with region_col1:
        cities = sorted(spatial_df["city"].dropna().unique())
        selected_city = st.selectbox("Select a city/region", cities, index=0)
        city_rows = spatial_df[spatial_df["city"] == selected_city]
        region_mean = city_rows["predicted_aqi"].mean()
        region_cat = aqi_category_label(region_mean)
        st.metric(f"{selected_city} — avg. predicted AQI ({selected_date})", f"{region_mean:.0f}", region_cat)
        st.caption(f"Averaged across {len(city_rows)} station(s) in {selected_city}: "
                   + ", ".join(f"{r.station_id.split(',')[0]} ({r.predicted_aqi:.0f})" for r in city_rows.itertuples()))

        st.markdown("**Per-station date-wise trend**")
        selected_station = st.selectbox("Select a CPCB station", sorted(city_rows["station_id"].unique()))
        station_series = all_preds_idx[(all_preds_idx["model"] == model_label) & (all_preds_idx["station_id"] == selected_station)].sort_values("date")
        st.line_chart(station_series.set_index("date")[["aqi", "pred"]].rename(columns={"aqi": "Actual", "pred": "Predicted"}))
        st.caption(f"Full real 2025 test-period trend for {selected_station.split(',')[0]} — scroll/zoom to inspect; selected date {selected_date} marked below.")
        row_at_date = station_series[station_series["date"] == selected_date]
        if len(row_at_date):
            st.caption(f"On {selected_date}: actual={row_at_date['aqi'].iloc[0]:.1f}, predicted={row_at_date['pred'].iloc[0]:.1f}")

    with region_col2:
        st.markdown(f"**Cause analysis — why {selected_station.split(',')[0]} is predicted this way on {selected_date}**")
        raw_row = load_raw_features_row(selected_station, selected_date)
        importance_df = load_feature_importance(model_key)
        if importance_df is not None and raw_row is not None:
            display = importance_df.copy()
            display["actual value on this date"] = display["feature"].map(lambda f: raw_row.get(f, float("nan")))
            st.dataframe(display.style.format({"importance": "{:.4f}", "actual value on this date": "{:.3g}"}), hide_index=True, width="stretch")
            st.caption(f"Top globally-important features for {model_label}, shown with this specific station's REAL feature "
                       f"values on {selected_date} (not a static example — changes with the date/station selected above)." +
                       (" Importance ranking itself is global (permutation importance from an earlier tuning iteration), not re-measured per-instance given time constraints." if model_key == "cnn_lstm" else ""))
        elif importance_df is not None:
            st.bar_chart(importance_df.set_index("feature")["importance"])
            st.caption(f"Top global feature-importance signals for {model_label} (raw per-date values unavailable for this row).")
        else:
            st.info(f"No feature-importance file available for {model_label} (ARIMA is deliberately univariate — see model note above).")

        st.markdown("**Safeguard measures** (by predicted category)")
        st.warning(ADVISORY.get(region_cat, "No advisory available."))
        for measure in SAFEGUARDS.get(region_cat, []):
            st.markdown(f"- {measure}")

    st.markdown("### Testing data — actual vs predicted (real 2025 test set)")
    all_preds = load_all_test_predictions()
    model_preds = all_preds[all_preds["model"] == model_label].copy()
    model_preds["error"] = (model_preds["pred"] - model_preds["aqi"]).round(1)
    model_preds["pred"] = model_preds["pred"].round(1)

    station_filter = st.multiselect("Filter by station (optional)", sorted(model_preds["station_id"].unique()))
    if station_filter:
        model_preds = model_preds[model_preds["station_id"].isin(station_filter)]

    st.dataframe(
        model_preds[["station_id", "city", "date", "aqi", "pred", "error"]]
        .rename(columns={"aqi": "actual", "pred": "predicted"})
        .sort_values("date", ascending=False),
        height=320, width="stretch",
    )
    st.caption(f"{len(model_preds):,} test rows shown for {model_label} — real held-out 2025 data, never used for training or hyperparameter selection.")


OBJ2_MODELS = {
    "ConvLSTM": "convlstm", "Random Forest": "random_forest", "Linear Regression": "linear_regression",
    "Statistical Baseline": "statistical_baseline", "Persistence (naive)": "persistence",
}


@st.cache_data
def load_obj2_metrics():
    rows = []
    for label, key in OBJ2_MODELS.items():
        path = os.path.join(V3_DIR, "models", "objective2_hcho", "artifacts", key, "metrics.json")
        with open(path) as f:
            d = json.load(f)
        t = d["test"]
        rows.append({"Model": label, "R2": t["r2"], "RMSE": t["rmse"], "MAE": t["mae"], "MAPE": t.get("mape")})
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


@st.cache_data
def load_spatial_stats_comparison():
    path = os.path.join(V3_DIR, "models", "objective2_hcho", "artifacts", "spatial_stats_comparison.csv")
    summary_path = os.path.join(V3_DIR, "models", "objective2_hcho", "artifacts", "spatial_stats_comparison_summary.json")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    with open(summary_path) as f:
        summary = json.load(f)
    return df, summary


@st.cache_data
def load_wind_fire():
    d1 = os.path.join(V3_DIR, "models", "objective2_hcho", "artifacts")
    ts = pd.read_csv(os.path.join(d1, "wind_fire_monthly_timeseries.csv"))
    seasonal = pd.read_csv(os.path.join(d1, "wind_fire_seasonal_climatology.csv"))
    with open(os.path.join(d1, "wind_fire_summary.json")) as f:
        summary = json.load(f)
    return ts, seasonal, summary


@st.cache_data
def load_month_grid():
    return pd.read_csv(os.path.join(V3_DIR, "models", "objective2_hcho", "artifacts", "month_grid_features.csv"))


# Same IMD 4-season convention as models/objective2_hcho/wind_fire_seasonal.py
SEASON_MONTH_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
}
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]


def hcho_color(v, vmin, vmax):
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    r, g, b, _ = mpl.colormaps["YlOrRd"](norm(v))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def add_wind_arrow(fg, lat, lon, wind_dir_deg, wind_speed, wspeed_max, popup=None):
    """Animated tail (flowing dashed line, real wind_u/wind_v direction) +
    a small CSS-rotated triangular arrowhead pointing along the real bearing
    (wind_dir_deg is degrees clockwise from north; CSS rotate() is also
    clockwise, so rotation=wind_dir_deg lines up directly - no plugin
    dependency, pure DivIcon, renders reliably with zero external JS).
    Magnitude (real wind_speed) drives THREE visual channels at once - line
    length, line thickness, and color depth - so speed differences are
    unmistakable at a glance, not just implied by one subtle cue."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    speed_frac = min(wind_speed / wspeed_max, 1.0) if wspeed_max > 0 else 0.0
    arrow_len = 0.12 + 0.38 * speed_frac
    line_weight = 1.5 + 3.5 * speed_frac
    rad = np.radians(wind_dir_deg)
    end_lat = lat + arrow_len * np.sin(rad)
    end_lon = lon + arrow_len * np.cos(rad)
    bc = mpl.colormaps["Blues"](mcolors.Normalize(-0.3, 1.0, clip=True)(speed_frac))
    color = f"#{int(bc[0]*255):02x}{int(bc[1]*255):02x}{int(bc[2]*255):02x}"
    folium.PolyLine([[lat, lon], [end_lat, end_lon]], color=color, weight=line_weight, opacity=0.95,
                     className="wind-flow-line").add_to(fg)
    arrow_html = (
        f'<div style="transform: rotate({wind_dir_deg:.0f}deg); width:0; height:0; '
        f'border-left:3.5px solid transparent; border-right:3.5px solid transparent; '
        f'border-bottom:7px solid {color}; filter: drop-shadow(0 0 1px #1a2634);"></div>'
    )
    folium.Marker(
        location=[end_lat, end_lon],
        icon=folium.DivIcon(html=arrow_html, icon_size=(8, 8), icon_anchor=(4, 4)),
        popup=popup,
    ).add_to(fg)


def add_color_legend(m, colors, vmin, vmax, caption):
    """branca linear color-scale legend, pinned to the map corner - shows
    the real numeric range each color band represents (not just relative
    color, an actual calibrated scale)."""
    from branca.colormap import LinearColormap
    cmap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=caption)
    cmap.add_to(m)


WIND_FLOW_CSS = """
<style>
.wind-flow-line { stroke-dasharray: 6 6; animation: vd-dashflow 0.9s linear infinite; }
@keyframes vd-dashflow { to { stroke-dashoffset: -24; } }
</style>
"""


def objective2_view(boundary_geojson):
    st.subheader("Objective 2 · HCHO Hotspot Detection")

    tabs = st.tabs(["Hotspot maps", "Monthly spatial view (HCHO+wind+fire)", "Wind + Fire seasonal analysis", "Model comparison"])

    with tabs[0]:
        st.markdown("#### Existing method: anomaly threshold (>p90 vs regional/seasonal baseline)")
        flagged_path = os.path.join(OBJ2_ARTIFACTS_LEGACY, "hcho_flagged.csv")
        flagged = pd.read_csv(flagged_path, parse_dates=["date"])
        hotspots = flagged[flagged["is_hotspot"] == True]  # noqa: E712
        latest_date = hotspots["date"].max()
        latest = hotspots[hotspots["date"] == latest_date]
        st.caption(f"{len(latest)} flagged points on {latest_date.date()}")

        m = folium.Map(location=[22.9734, 80.6569], zoom_start=4.3, tiles="CartoDB positron")
        folium.GeoJson(
            boundary_geojson,
            style_function=lambda _: {"color": "#2b3a55", "weight": 1.6, "fillColor": "#dbe6f5", "fillOpacity": 0.15},
        ).add_to(m)
        if len(latest):
            heat_data = [[r.lat, r.lon, float(r.hcho_anomaly)] for r in latest.itertuples()]
            HeatMap(heat_data, radius=28, blur=20, min_opacity=0.35, max_zoom=6).add_to(m)
            for r in latest.itertuples():
                folium.CircleMarker(
                    location=[r.lat, r.lon], radius=4, color="#ff6600", fill=True, fill_color="#ffcc00",
                    fill_opacity=0.8, weight=1,
                    popup=f"HCHO: {r.hcho:.2e}<br>Anomaly: {r.hcho_anomaly:.2e}<br>Date: {r.date.date()}",
                ).add_to(m)
        st_folium(m, width=None, height=420, key="obj2_map_existing")

        st.markdown("#### DBSCAN spatial clustering")
        clusters_path = os.path.join(OBJ2_ARTIFACTS_LEGACY, "hcho_clusters.csv")
        clusters = pd.read_csv(clusters_path, parse_dates=["date"])
        clusters_month = clusters[clusters["date"].dt.to_period("M").astype(str) == "2025-11"]
        st.caption(f"{len(clusters_month)} DBSCAN clusters found across India in 2025-11 "
                   f"(eps=15km, min_samples=3 — see models/objective2_hcho/cluster_fires.py). "
                   f"Full-India coverage confirmed: cluster latitudes span "
                   f"{clusters['lat'].min():.1f}°-{clusters['lat'].max():.1f}°N, not just the Punjab-Haryana corridor.")
        m_db = folium.Map(location=[22.9734, 80.6569], zoom_start=4.3, tiles="CartoDB positron")
        folium.GeoJson(
            boundary_geojson,
            style_function=lambda _: {"color": "#2b3a55", "weight": 1.6, "fillColor": "#dbe6f5", "fillOpacity": 0.15},
        ).add_to(m_db)
        for r in clusters_month.itertuples():
            radius = min(4 + r.n_points / 20, 16)
            folium.CircleMarker(location=[r.lat, r.lon], radius=radius, color="#8e24aa", fill=True,
                                 fill_color="#ce93d8", fill_opacity=0.7, weight=1.2,
                                 popup=f"Cluster {r.cluster_id}<br>{r.n_points} points<br>mean HCHO={r.mean_hcho:.2e}<br>{r.date.date()}").add_to(m_db)
        st_folium(m_db, width=None, height=420, key="obj2_map_dbscan")

        st.markdown("#### New: Getis-Ord Gi* and Anselin's Local Moran's I (LISA)")
        stats_df, stats_summary = load_spatial_stats_comparison()
        if stats_df is None:
            st.warning("Spatial statistics not yet computed.")
        else:
            st.caption(f"Applied to the national monthly HCHO grid for {stats_summary['month']} "
                       f"({stats_summary['n_grid_cells']} real India land cells) — the month with the most "
                       f"existing-method-flagged hotspots in the 2018-2025 record (stubble-burning season).")
            method_choice = st.radio("Statistic", ["Getis-Ord Gi*", "Anselin Local Moran's I (LISA)"], horizontal=True)

            m2 = folium.Map(location=[22.9734, 80.6569], zoom_start=4.3, tiles="CartoDB positron")
            folium.GeoJson(
                boundary_geojson,
                style_function=lambda _: {"color": "#2b3a55", "weight": 1.6, "fillColor": "#dbe6f5", "fillOpacity": 0.15},
            ).add_to(m2)
            if method_choice == "Getis-Ord Gi*":
                sig = stats_df[stats_df["getis_ord_hotspot"] | stats_df["getis_ord_coldspot"]]
                for r in sig.itertuples():
                    color = "#ff0000" if r.getis_ord_hotspot else "#0066ff"
                    folium.CircleMarker(location=[r.lat, r.lon], radius=5, color=color, fill=True, fill_color=color,
                                         fill_opacity=0.85, weight=1,
                                         popup=f"Gi* z={r.getis_ord_z:.2f}, p={r.getis_ord_p:.3f}").add_to(m2)
                st.caption(f"{stats_summary['getis_ord']['n_hotspot_cells']} statistically significant hotspot cells "
                           f"(z>1.96, p<0.05, red), {stats_summary['getis_ord']['n_coldspot_cells']} coldspot cells (blue) "
                           f"out of {stats_summary['n_grid_cells']} — Gi* is notably stricter/more localized than LISA below.")
            else:
                sig = stats_df[stats_df["lisa_hh_cluster"] | stats_df["lisa_ll_cluster"]]
                for r in sig.itertuples():
                    color = "#ff0000" if r.lisa_hh_cluster else "#0066ff"
                    folium.CircleMarker(location=[r.lat, r.lon], radius=5, color=color, fill=True, fill_color=color,
                                         fill_opacity=0.85, weight=1,
                                         popup=f"LISA I={r.lisa_I:.3f}, p={r.lisa_p:.3f}").add_to(m2)
                st.caption(f"{stats_summary['lisa']['n_hh_cluster_cells']} High-High cluster cells (red), "
                           f"{stats_summary['lisa']['n_ll_cluster_cells']} Low-Low cluster cells (blue) "
                           f"out of {stats_summary['n_grid_cells']} ({stats_summary['lisa']['pct_hh']}% of the country).")
            st_folium(m2, width=None, height=420, key=f"obj2_map_{method_choice}")

            st.markdown("### 3-method hotspot-detection comparison (2025-11)")
            comparison_rows = [
                {"Method": "Percentile threshold (existing)", "Detections": f"{stats_summary['existing_anomaly_threshold']['n_flagged_station_days']:,} station-days",
                 "Statistical rigor": "None — fixed >p90 cutoff", "Coverage": "Very liberal, flags most of N. India",
                 "Verdict": "Simple, fast, but no significance testing — high false-positive risk"},
                {"Method": "DBSCAN clustering", "Detections": f"{len(clusters_month)} clusters",
                 "Statistical rigor": "None — density-based grouping only", "Coverage": "Full India (confirmed by lat span above)",
                 "Verdict": "Good at finding contiguous shapes, but no p-value — can't say a cluster is statistically real vs. noise"},
                {"Method": "Getis-Ord Gi*", "Detections": f"{stats_summary['getis_ord']['n_hotspot_cells']} cells ({stats_summary['getis_ord']['pct_hotspot']}%)",
                 "Statistical rigor": "999-permutation significance test (p<0.05)", "Coverage": "Full national grid",
                 "Verdict": "Best method — the only one with real statistical confidence per cell, at the cost of being conservative"},
            ]
            st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")
            st.caption("**Getis-Ord Gi* is the recommended method** for genuinely defensible hotspot claims — it's the only "
                       "one of the three that distinguishes a statistically significant cluster from random noise (permutation "
                       "test, p<0.05). The percentile threshold and DBSCAN are useful for fast exploratory screening but "
                       "neither can support a claim like 'this region is a real hotspot, not chance' the way Gi* can.")
            st.caption(stats_summary["agreement"]["note"])

    with tabs[1]:
        st.markdown("#### Monthly spatial view — HCHO, fire, and wind together")
        grid = load_month_grid()
        months_available = sorted(grid["month"].unique())
        sel_month = st.select_slider("Month (spatiotemporal grid, 2018-2025)", options=months_available, value="2025-11")
        gm = grid[grid["month"] == sel_month]
        st.caption(f"{len(gm):,} real India grid cells for {sel_month}. Moving this slider changes the HCHO color layer, "
                   f"fire markers, and wind arrows below together.")

        layer_choice = st.multiselect("Layers", ["HCHO concentration", "Fire density", "Wind (animated)"],
                                       default=["HCHO concentration", "Fire density", "Wind (animated)"])

        m3 = folium.Map(location=[22.9734, 80.6569], zoom_start=4.3, tiles="CartoDB positron")
        folium.GeoJson(
            boundary_geojson,
            style_function=lambda _: {"color": "#2b3a55", "weight": 1.6, "fillColor": "#dbe6f5", "fillOpacity": 0.1},
        ).add_to(m3)

        # animated "flowing dash" CSS for wind lines - real SVG stroke-dashoffset
        # animation (marching-ants style), arrowhead marker gives real direction
        m3.get_root().html.add_child(folium.Element(WIND_FLOW_CSS))

        if "HCHO concentration" in layer_choice:
            hcho_fg = folium.FeatureGroup(name="HCHO concentration", show=True)
            vmin, vmax = gm["hcho_today"].quantile(0.02), gm["hcho_today"].quantile(0.98)
            for r in gm.itertuples():
                folium.CircleMarker(location=[r.lat, r.lon], radius=3, color=None, fill=True,
                                     fill_color=hcho_color(r.hcho_today, vmin, vmax), fill_opacity=0.7,
                                     popup=f"HCHO: {r.hcho_today:.2e}<br>7-month avg: {r.hcho_7month_avg:.2e}<br>"
                                           f"Fire count: {r.fire_count:.0f}<br>Wind: {r.wind_speed:.1f} m/s @ {r.wind_dir_deg:.0f}°").add_to(hcho_fg)
            hcho_fg.add_to(m3)
            add_color_legend(m3, ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"], vmin, vmax, "HCHO column density")

        if "Fire density" in layer_choice:
            fire_fg = folium.FeatureGroup(name="Fire density", show=True)
            fire_points = gm[gm["fire_count"] > 0]
            if len(fire_points):
                heat_data = [[r.lat, r.lon, float(r.fire_count)] for r in fire_points.itertuples()]
                HeatMap(heat_data, radius=22, blur=18, min_opacity=0.25, max_zoom=6,
                        gradient={0.2: "#ffeb3b", 0.5: "#ff9800", 0.8: "#f4511e", 1.0: "#b71c1c"}).add_to(fire_fg)
                add_color_legend(m3, ["#ffeb3b", "#ff9800", "#f4511e", "#b71c1c"],
                                  0, float(fire_points["fire_count"].max()), "Fire detections (count)")
            fire_fg.add_to(m3)

        if "Wind (animated)" in layer_choice:
            wind_fg = folium.FeatureGroup(name="Wind (animated)", show=True)
            wind_valid = gm.dropna(subset=["wind_speed", "wind_dir_deg"])
            wind_sample = wind_valid.iloc[::max(1, len(wind_valid) // 900)]
            wspeed_max = max(gm["wind_speed"].max(), 0.1)
            for r in wind_sample.itertuples():
                add_wind_arrow(wind_fg, r.lat, r.lon, r.wind_dir_deg, r.wind_speed, wspeed_max,
                                popup=f"Wind: {r.wind_speed:.1f} m/s @ {r.wind_dir_deg:.0f}°")
            wind_fg.add_to(m3)
            add_color_legend(m3, ["#deebf7", "#9ecae1", "#4292c6", "#08519c"], 0, wspeed_max, "Wind speed (m/s)")

        folium.LayerControl(collapsed=False).add_to(m3)
        st_folium(m3, width=None, height=560, key=f"obj2_month_map_{sel_month}")
        st.caption("HCHO dots = concentration (light→dark = low→high, 2nd-98th percentile). Fire = smooth density "
                   "heatmap (yellow→red = low→high fire-count intensity), not overlapping rings. Wind = animated "
                   "flowing dashed lines (real CSS motion along each real wind vector — darker/thicker/longer = "
                   "faster, all three scale with real wind_speed) with a small triangular arrowhead rotated to the "
                   "real bearing, drawn densely across the whole map (~900 of the real grid cells) so the flow "
                   "pattern is visible at a glance, not sparse. Color-scale legends (bottom-left) show the real "
                   "numeric range behind each color. Use the Layers control (top-right of map) or the multiselect above to toggle "
                   "each independently. All three layers rebuild from real per-month wind_u/wind_v/fire_count/hcho "
                   "values (no placeholders) when you move the slider.")

        st.markdown("**Reasoning for this month's top HCHO cells**")
        top_cells = gm.sort_values("hcho_today", ascending=False).head(5)
        for r in top_cells.itertuples():
            fire_note = f"elevated fire activity ({r.fire_count:.0f} detections)" if r.fire_count > gm["fire_count"].quantile(0.75) else "no unusual fire activity"
            wind_note = f"low wind ({r.wind_speed:.1f} m/s, trapping pollutants)" if r.wind_speed < gm["wind_speed"].quantile(0.25) else f"moderate/high wind ({r.wind_speed:.1f} m/s, dispersing pollutants)"
            trend_note = "rising vs its own 7-month average" if r.hcho_today > r.hcho_7month_avg else "below its own 7-month average"
            st.markdown(f"- **({r.lat:.2f}, {r.lon:.2f})**: HCHO={r.hcho_today:.2e}, {trend_note}, {fire_note}, {wind_note}")
        st.caption("Feature engineering used: today's HCHO, prior-month HCHO, 7-month rolling average, fire count, "
                   "wind speed/direction (u/v components), month, lat, lon — all real values from the spatiotemporal "
                   "grid tensor. MODIS AOD, temperature, humidity, and pressure were NOT extracted into this grid "
                   "(only hcho/fire_count/wind_u/wind_v were pulled for ConvLSTM) — adding them would need a new GEE "
                   "extraction, out of scope given remaining time; reasoning text above is rule-based on the features "
                   "that are genuinely available, not a trained model's SHAP output.")

    with tabs[2]:
        st.markdown("#### Wind + Fire, combined seasonally")
        ts, seasonal, wf_summary = load_wind_fire()
        st.caption("National monthly means across all 4,447 real India grid cells, 2018-2025, from the same "
                   "spatiotemporal tensor used to train ConvLSTM (channels: hcho, fire_count, wind_u, wind_v).")

        st.markdown("**Seasonal spatial map** — where fire and wind actually sit on the map, not just national averages")
        season_choice = st.radio("Season (IMD 4-season convention)", SEASON_ORDER, horizontal=True, index=SEASON_ORDER.index("Post-Monsoon"))
        grid_all = load_month_grid()
        season_map_series = grid_all["month_num"].map(SEASON_MONTH_MAP)
        gs = grid_all[season_map_series == season_choice]
        gs_cell = gs.groupby(["lat", "lon"], as_index=False).agg(
            fire_count=("fire_count", "mean"), wind_u=("wind_u", "mean"), wind_v=("wind_v", "mean"),
        ).dropna(subset=["wind_u", "wind_v"])
        gs_cell["wind_speed"] = np.hypot(gs_cell["wind_u"], gs_cell["wind_v"])
        gs_cell["wind_dir_deg"] = (np.degrees(np.arctan2(gs_cell["wind_u"], gs_cell["wind_v"])) + 360) % 360
        st.caption(f"{len(gs_cell):,} real India grid cells, averaged across every {season_choice} month "
                   f"2018-2025 (circular mean of real wind_u/wind_v, not a naive degree average).")

        m_season = folium.Map(location=[22.9734, 80.6569], zoom_start=4.3, tiles="CartoDB positron")
        folium.GeoJson(
            boundary_geojson,
            style_function=lambda _: {"color": "#2b3a55", "weight": 1.6, "fillColor": "#dbe6f5", "fillOpacity": 0.1},
        ).add_to(m_season)
        m_season.get_root().html.add_child(folium.Element(WIND_FLOW_CSS))

        fire_fg_s = folium.FeatureGroup(name="Fire density", show=True)
        fire_points_s = gs_cell[gs_cell["fire_count"] > 0]
        if len(fire_points_s):
            heat_data_s = [[r.lat, r.lon, float(r.fire_count)] for r in fire_points_s.itertuples()]
            HeatMap(heat_data_s, radius=22, blur=18, min_opacity=0.25, max_zoom=6,
                    gradient={0.2: "#ffeb3b", 0.5: "#ff9800", 0.8: "#f4511e", 1.0: "#b71c1c"}).add_to(fire_fg_s)
            add_color_legend(m_season, ["#ffeb3b", "#ff9800", "#f4511e", "#b71c1c"],
                              0, float(fire_points_s["fire_count"].max()), f"{season_choice} mean fire detections")
        fire_fg_s.add_to(m_season)

        wind_fg_s = folium.FeatureGroup(name="Wind (animated)", show=True)
        wind_sample_s = gs_cell.iloc[::max(1, len(gs_cell) // 900)]
        wspeed_max_s = max(gs_cell["wind_speed"].max(), 0.1)
        for r in wind_sample_s.itertuples():
            add_wind_arrow(wind_fg_s, r.lat, r.lon, r.wind_dir_deg, r.wind_speed, wspeed_max_s,
                            popup=f"{season_choice} mean wind: {r.wind_speed:.1f} m/s @ {r.wind_dir_deg:.0f}°")
        wind_fg_s.add_to(m_season)
        add_color_legend(m_season, ["#deebf7", "#9ecae1", "#4292c6", "#08519c"], 0, wspeed_max_s,
                          f"{season_choice} mean wind speed (m/s)")

        folium.LayerControl(collapsed=False).add_to(m_season)
        st_folium(m_season, width=None, height=520, key=f"obj2_seasonal_map_{season_choice}")
        st.caption("Fire = smooth density heatmap (yellow→red = low→high, 8-year seasonal average). Wind = animated "
                   "flowing arrows drawn across ~900 real grid cells (dense, whole-map coverage, not sparse); "
                   "length/thickness/color-depth all scale with real seasonal-mean wind_speed, so faster-wind "
                   "regions are unmistakable at a glance. Color-scale legends (bottom-left) show the real numeric "
                   "range behind each color. Compare across seasons: Post-Monsoon shows the Punjab-Haryana "
                   "stubble-burning fire concentration under low, stagnant wind — the classic winter-smog trapping "
                   "mechanism; Summer shows broader, more diffuse fire activity under stronger dispersing wind.")

        s1, s2, s3 = st.columns(3)
        s1.metric("Peak fire season", wf_summary["peak_fire_season"])
        s2.metric("Peak HCHO season", wf_summary["peak_hcho_season"])
        s3.metric("Lowest-wind season", wf_summary["lowest_wind_season"])

        st.markdown("**Seasonal climatology (8-year average, IMD 4-season convention)**")
        st.dataframe(seasonal.style.format({c: "{:.4g}" for c in seasonal.columns if c != "season"}), hide_index=True, width="stretch")
        chart_col1, chart_col2 = st.columns(2)
        chart_col1.bar_chart(seasonal.set_index("season")[["mean_fire_count_mean"]])
        chart_col1.caption("Mean fire count by season")
        chart_col2.bar_chart(seasonal.set_index("season")[["mean_wind_speed_mean"]])
        chart_col2.caption("Mean wind speed by season")

        st.markdown("**Monthly time series, full record**")
        st.line_chart(ts.set_index("month")[["mean_fire_count", "mean_wind_speed"]])
        st.caption(f"Correlations: fire-vs-HCHO r={wf_summary['correlations']['fire_vs_hcho']:.3f}, "
                   f"wind-vs-HCHO r={wf_summary['correlations']['wind_vs_hcho']:.3f}, "
                   f"fire-vs-wind r={wf_summary['correlations']['fire_vs_wind']:.3f}. "
                   "Fire and HCHO peak nationally in Summer (broader than just Punjab stubble-burning); "
                   "Post-Monsoon has the lowest wind speed — the stagnant-air condition that traps whatever "
                   "pollution is present, consistent with the known North-India winter-smog mechanism even "
                   "though national fire count itself peaks earlier in the year.")

    with tabs[3]:
        st.markdown("#### 5-model comparison (Objective 2, real test set)")
        obj2_metrics = load_obj2_metrics()
        st.dataframe(
            obj2_metrics.style.format({"R2": "{:.3f}", "RMSE": "{:.2e}", "MAE": "{:.2e}", "MAPE": "{:.1f}"}),
            hide_index=True, width="stretch",
        )
        st.bar_chart(obj2_metrics.set_index("Model")["R2"])
        st.caption("Random Forest currently edges out ConvLSTM on raw test R² — reported honestly, not smoothed over "
                   "(see PROJECT_DOCUMENTATION.md section 5 for the full discussion). Both comfortably beat the "
                   "naive persistence baseline (R²=-0.592, worse than predicting the mean).")


PM25_ANALYSIS_DIR = os.path.join(V3_DIR, "figures", "pm25_analysis")
JAN2025_CHART = os.path.join(V3_DIR, "figures", "jan2025_aqi_prediction_comparison.png")
JAN2025_EXPORT = os.path.join(V3_DIR, "data", "jan2025_predictions_export.csv")
FULL2025_AQI_CHART = os.path.join(V3_DIR, "figures", "full2025_aqi_prediction_comparison.png")
FULL2025_PM25_CHART = os.path.join(V3_DIR, "figures", "full2025_pm25_prediction_comparison.png")
FULL2025_EXPORT = os.path.join(V3_DIR, "data", "full2025_predictions_export.csv")
TWO_STAGE_ARTIFACTS = os.path.join(OBJ1_ARTIFACTS, "two_stage_pm25_aqi")
PM25_AQI_GRID_PATH = os.path.join(V3_DIR, "data", "pm25_aqi_grid_predictions.csv")
PM25_REFERENCE_PATH = os.path.join(V3_DIR, "data", "pm25_reference_daily.csv")

PM25_VARS = {
    "no2": "NO2", "so2": "SO2", "o3": "Ozone (O3)", "co": "CO",
    "temp_c": "Temperature", "wind_speed": "Wind speed", "rh_pct": "Relative humidity",
}

# Same order/definition as train_two_stage_pm25_aqi.py's STAGE1_FEATURES -
# the real matrix columns Stage 1 (gases+weather -> PM2.5) was trained on.
# aod/no2/so2/co/o3/hcho = satellite (MODIS MAIAC + Sentinel-5P TROPOMI),
# temp_c/rh_pct/wind_speed/wind_dir_deg = ERA5 weather, reused unmodified.
STAGE1_FEATURES = ["aod_047", "aod_055", "no2", "so2", "co", "o3", "hcho",
                    "temp_c", "rh_pct", "wind_speed", "wind_dir_deg", "month", "doy_sin", "doy_cos"]

# Every satellite gas input + the calculated PM2.5/AQI, each its own
# continuous national map for the selected day - column, display label,
# unit, and a distinct sequential colormap per variable so panels are
# visually distinguishable at a glance, not just by their titles.
GAS_MAP_PANELS = [
    ("aod_047", "AOD (0.47µm)", "MODIS MAIAC", "YlOrBr"),
    ("no2", "NO2", "TROPOMI column density", "Purples"),
    ("so2", "SO2", "TROPOMI column density", "Oranges"),
    ("co", "CO", "TROPOMI column density", "Greys"),
    ("o3", "Ozone (O3)", "TROPOMI column density", "BuGn"),
    ("hcho", "HCHO", "TROPOMI column density", "RdPu"),
    ("predicted_pm25", "Predicted PM2.5", "µg/m³ (calculated)", "YlOrRd"),
    ("predicted_aqi", "Predicted AQI", "calculated from PM2.5", "RdYlGn_r"),
]


# Same 6 satellite gas/aerosol inputs as GAS_MAP_PANELS, reused for the
# single-station trace (metrics + per-gas time series) - column, short
# label, unit label.
GAS_TRACE_VARS = [
    ("aod_047", "AOD", ""),
    ("no2", "NO2", "mol/m²"),
    ("so2", "SO2", "mol/m²"),
    ("o3", "O3", "mol/m²"),
    ("co", "CO", "mol/m²"),
    ("hcho", "HCHO", "mol/m²"),
]


def render_gas_maps_grid(day_df: pd.DataFrame, date_str: str, boundary_geojson_str: str):
    cols = tuple(c for c, *_ in GAS_MAP_PANELS)
    grids, bounds = compute_variable_grids(f"gasgrid_{date_str}", day_df, cols, boundary_geojson_str)
    minx, miny, maxx, maxy = bounds

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), dpi=150)
    for ax, (col, label, unit, cmap) in zip(axes.flat, GAS_MAP_PANELS):
        im = ax.imshow(grids[col], origin="lower", extent=bounds, cmap=cmap, aspect="auto")
        draw_state_boundaries(ax, linewidth=0.4, alpha=0.6)
        ax.set_xlim(68, 97.5)
        ax.set_ylim(6.5, 37.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=12, fontweight="bold", color="#12233D")
        ax.set_xlabel(unit, fontsize=8.5, color="#5B6B85")
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"All satellite inputs + calculated PM2.5/AQI — {date_str} (continuous, national grid)",
                 fontsize=14, fontweight="bold", color="#12233D", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


@st.cache_data
def load_two_stage_metrics():
    with open(os.path.join(TWO_STAGE_ARTIFACTS, "metrics.json")) as f:
        return json.load(f)


@st.cache_data
def load_station_performance_matrix():
    """Real per-station test-set metrics (RMSE/MAE/R2/MAPE, full 2025 test
    period, not just January) - already computed and saved by each model's
    own eval_utils.per_station_metrics() at training time, reused here
    as-is rather than recomputed from the export CSVs."""
    xgb = pd.read_csv(os.path.join(OBJ1_ARTIFACTS, "xgboost", "per_station_metrics.csv"))
    two = pd.read_csv(os.path.join(TWO_STAGE_ARTIFACTS, "per_station_metrics.csv"))
    coords = load_station_coords()
    m = xgb.merge(two, on="station_id", suffixes=("_xgb", "_two"))
    m = m.merge(coords, on="station_id", how="left")
    return m[["station_id", "city", "r2_xgb", "rmse_xgb", "mae_xgb", "r2_two", "rmse_two", "mae_two", "n_xgb"]].rename(
        columns={"n_xgb": "n_test_days"})


@st.cache_resource
def load_stage1_model():
    stage1 = joblib.load(os.path.join(TWO_STAGE_ARTIFACTS, "stage1_gases_to_pm25.pkl"))
    imputer1 = joblib.load(os.path.join(TWO_STAGE_ARTIFACTS, "stage1_imputer.pkl"))
    return stage1, imputer1


@st.cache_data
def load_pm25_correlation_summary():
    return pd.read_csv(os.path.join(PM25_ANALYSIS_DIR, "pm25_correlation_summary.csv"))


@st.cache_data
def load_pm25_aqi_grid():
    """Continuous national grid: Objective 1's already-downloaded satellite
    grid (AOD + 5 gases + ERA5, 4447 points) run through the two-stage
    model's Stage 1 (gases+weather -> PM2.5), then converted to AQI via the
    real CPCB PM2.5 breakpoint formula - see build_pm25_aqi_grid.py."""
    return pd.read_csv(PM25_AQI_GRID_PATH)


@st.cache_data
def load_pm25_ground_truth():
    return pd.read_csv(PM25_REFERENCE_PATH, parse_dates=["date"])


@st.cache_data
def load_station_coords():
    return load_spatial("xgboost")[["station_id", "lat", "lon", "city"]].drop_duplicates("station_id")


def render_pm25_grid_map(grid_df: pd.DataFrame, date_str: str, boundary_geojson: dict):
    """Continuous IDW surface from the dense 4447-point national grid
    (reuses the exact same compute_idw_overlay/render_overlay_png path as
    the Objective 1 station map - just fed ~44x denser input points, which
    is what makes the surface look genuinely continuous instead of patchy
    station-interpolation) + small real-CPCB-station markers on top for
    ground-truth comparison (kept visually distinct from the 101-station
    Objective 1 map: smaller, no popup clutter, muted outline)."""
    m = folium.Map(location=[22.9734, 80.6569], zoom_start=5, tiles="CartoDB positron")
    folium.GeoJson(
        boundary_geojson,
        style_function=lambda _: {"color": "#2b3a55", "weight": 1.6, "fillColor": "#dbe6f5", "fillOpacity": 0.05},
    ).add_to(m)

    png_bytes, bounds = compute_idw_overlay(f"pm25grid_{date_str}", grid_df, json.dumps(boundary_geojson))
    minx, miny, maxx, maxy = bounds
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    folium.raster_layers.ImageOverlay(
        image=data_uri, bounds=[[miny, minx], [maxy, maxx]], opacity=1.0, name="Satellite PM2.5-derived AQI",
    ).add_to(m)

    ground_truth = load_pm25_ground_truth()
    coords = load_station_coords()
    gt_day = ground_truth[ground_truth["date"] == pd.Timestamp(date_str)].merge(coords, on="station_id", how="inner")
    all_preds_idx = load_all_test_predictions_indexed()
    actual_aqi_day = all_preds_idx[(all_preds_idx["model"] == "XGBoost") & (all_preds_idx["date"] == date_str)][["station_id", "aqi"]]
    gt_day = gt_day.merge(actual_aqi_day, on="station_id", how="left")

    for r in gt_day.itertuples():
        aqi_txt = f"{r.aqi:.0f}" if pd.notna(getattr(r, "aqi", float("nan"))) else "n/a"
        folium.CircleMarker(
            location=[r.lat, r.lon], radius=4, color="#1a2634", weight=1, fill=True,
            fill_color="#3b82f6", fill_opacity=0.9,
            popup=folium.Popup(f"<b>{r.city}</b><br>{r.station_id}<br>Ground CPCB PM2.5: {r.pm25:.1f} µg/m³<br>"
                                f"Ground actual AQI: {aqi_txt}<br>Date: {date_str}", max_width=230),
        ).add_to(m)

    return m


def pm25_deepdive_view(boundary_geojson):
    st.subheader("PM2.5 → AQI Deep-Dive (additional analysis)")
    st.caption("Standalone reference analysis requested alongside the primary 5-model AQI pipeline above — "
               "does not replace CNN-LSTM/XGBoost as the primary approach. Explores the gases→PM2.5→AQI "
               "relationship directly and gives a second, independent AQI estimate for comparison.")

    tabs = st.tabs([
        "Gas ↔ PM2.5 spatial maps",
        "Continuous spatial AQI-via-PM2.5 map",
        "Two-stage model (gases→PM2.5→AQI)",
        "January 2025 prediction comparison",
        "Full 2025 prediction comparison",
        "Single-station data trace (matrix → PM2.5)",
    ])

    with tabs[0]:
        st.markdown("#### How each satellite gas/weather variable relates to ground PM2.5")
        st.caption("PM2.5 has no satellite instrument (TROPOMI/MODIS measure gas columns and optical depth, "
                   "not particulate mass) — so PM2.5 here comes from CPCB ground stations, windowed to the "
                   "same 11:18-15:18 IST satellite-overpass period used everywhere else in this project. "
                   "The gas/weather side is 100% satellite/ERA5-derived, not CPCB.")

        corr_df = load_pm25_correlation_summary()
        view_mode = st.radio("View", ["Single day (2025-11-26)", "4-season time series"], horizontal=True)
        var_label = st.selectbox("Variable", list(PM25_VARS.values()))
        var_key = [k for k, v in PM25_VARS.items() if v == var_label][0]

        if view_mode.startswith("Single"):
            img_path = os.path.join(PM25_ANALYSIS_DIR, f"pm25_vs_{var_key}_2025-11-26.png")
            r = corr_df[corr_df["variable"] == var_key]["pearson_r_vs_pm25"].iloc[0]
            st.caption(f"Pearson r (PM2.5 vs {var_label}, 2025-11-26, national): {r:.3f}")
        else:
            img_path = os.path.join(PM25_ANALYSIS_DIR, f"pm25_vs_{var_key}_timeseries_2025.png")
            st.caption("Same variable across 4 seasonally-representative 2025 dates (winter / pre-monsoon / "
                       "monsoon / post-monsoon-stubble) — correlation strength/sign shown per-row on the chart itself.")

        if os.path.exists(img_path):
            st.image(img_path, width="stretch")
        else:
            st.warning(f"Chart not found: {img_path}")

        st.markdown("**All-variable correlation summary (2025-11-26, national)**")
        st.dataframe(corr_df[["variable", "pearson_r_vs_pm25"]].style.format({"pearson_r_vs_pm25": "{:.3f}"}),
                     hide_index=True, width="stretch")

    with tabs[1]:
        st.markdown("#### Satellite-derived AQI, continuous across all of India")
        st.caption("Not station dots — a true continuous surface. Reuses Objective 1's already-downloaded national "
                   "0.25° grid (AOD + all 5 TROPOMI gases + ERA5 weather, 4,447 points covering all of India, not "
                   "just where CPCB stations sit), runs the two-stage model's Stage 1 (gases+weather → PM2.5) across "
                   "every grid point, then converts predicted PM2.5 → AQI with the same CPCB breakpoint formula "
                   "used throughout this project. Rendered with the exact same IDW-interpolation code as the "
                   "Objective 1 map above — just fed ~44× more input points (4,447 vs. 101 stations), which is what "
                   "makes this surface look genuinely continuous rather than patchy.")

        grid_all = load_pm25_aqi_grid()
        grid_dates = sorted(grid_all["date"].unique())
        grid_dates_2025 = [d for d in grid_dates if d.startswith("2025")]
        sel_grid_date = st.select_slider(
            "Date (satellite grid — real overpass composite dates, sparser than the daily station slider above)",
            options=grid_dates_2025, value=grid_dates_2025[-1],
        )
        st.caption(f"{len(grid_dates_2025)} real satellite-composite dates available across 2025 (grid dates aren't "
                   f"daily like the station-level test set — each is a real date with usable, cloud-filtered "
                   f"satellite coverage across the national grid).")

        grid_day = grid_all[grid_all["date"] == sel_grid_date]
        c1, c2, c3 = st.columns(3)
        c1.metric("National mean predicted PM2.5", f"{grid_day['predicted_pm25'].mean():.1f} µg/m³")
        c2.metric("National mean predicted AQI", f"{grid_day['predicted_aqi'].mean():.0f}")
        c3.metric("Grid points this date", f"{len(grid_day):,}")

        m_grid = render_pm25_grid_map(grid_day, sel_grid_date, boundary_geojson)
        st_folium(m_grid, width=None, height=540, key=f"pm25_grid_map_{sel_grid_date}")
        st.caption("Continuous color fill = satellite-derived predicted AQI (via predicted PM2.5), IDW-interpolated "
                   "across the full 4,447-point national grid. Small blue dots = real CPCB ground stations — click "
                   "one to compare its real ground PM2.5/AQI reading against the satellite-derived surface underneath.")

        st.markdown("#### Every gas that went into that PM2.5 calculation, mapped separately")
        st.caption("Same day, same national grid, same IDW rendering as the map above — but each satellite input "
                   "(AOD, NO2, SO2, CO, O3, HCHO) gets its own continuous map instead of being collapsed into one "
                   "number, so you can see which gas is actually driving the calculated PM2.5/AQI in each region on "
                   f"{sel_grid_date}.")
        gas_missing = [c for c, *_ in GAS_MAP_PANELS if c not in grid_day.columns]
        if gas_missing:
            st.warning(f"Grid file is missing columns {gas_missing} — re-run build_pm25_aqi_grid.py to regenerate "
                       f"pm25_aqi_grid_predictions.csv with the raw gas columns included.")
        else:
            fig = render_gas_maps_grid(grid_day, sel_grid_date, json.dumps(boundary_geojson))
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption("8 panels, all IDW-interpolated from the same 4,447-point national grid for this exact date: "
                       "the 6 satellite gas/aerosol inputs (each its own colormap so panels are visually distinct), "
                       "the calculated PM2.5 (Stage 1 model output), and the resulting calculated AQI (CPCB "
                       "breakpoint formula applied to that PM2.5).")

    with tabs[2]:
        st.markdown("#### Two-stage reference model")
        st.caption("Stage 1: satellite gases (NO2/SO2/CO/O3/AOD/HCHO) + ERA5 weather → predicted PM2.5 (Random Forest). "
                   "Stage 2: predicted PM2.5 → AQI (Linear Regression). Stage 2 trains on Stage 1's OWN train-period "
                   "predictions, not real PM2.5, matching real deployment where true PM2.5 is never known in advance. "
                   "Same 2019-2024 train / 2025 test split as every other model in this project.")

        m = load_two_stage_metrics()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Stage 1 — gases+weather → PM2.5**")
            s1 = m["stage1_pm25"]["test"]
            st.metric("Test R²", f"{s1['r2']:.3f}")
            st.metric("Test RMSE", f"{s1['rmse']:.1f} µg/m³")
            st.metric("Test MAE", f"{s1['mae']:.1f} µg/m³")
        with c2:
            st.markdown("**Stage 2 / end-to-end — → AQI**")
            s2 = m["end_to_end"]["test"]
            st.metric("Test R²", f"{s2['r2']:.3f}")
            st.metric("Test RMSE", f"{s2['rmse']:.1f}")
            st.metric("Test MAE", f"{s2['mae']:.1f}")
        st.caption("For comparison, the primary XGBoost direct gases→AQI model scores test R²=0.519 — the two-stage "
                   "route loses accuracy at each hand-off (gases→PM2.5 is itself only R²=0.386), consistent with it "
                   "being a reference/comparison model rather than the recommended approach.")

        st.markdown("**Stage 1 input features**")
        st.write(", ".join(m["stage1_features"]))

    with tabs[3]:
        st.markdown("#### January 2025 test period: actual vs. predicted AQI")
        st.caption("Direct comparison for the specific month requested — primary XGBoost model vs. the two-stage "
                   "reference model, both evaluated on real held-out January 2025 data, national daily mean across "
                   "101 stations.")
        if os.path.exists(JAN2025_CHART):
            st.image(JAN2025_CHART, width="stretch")
        else:
            st.warning(f"Chart not found: {JAN2025_CHART}")

        if os.path.exists(JAN2025_EXPORT):
            export_df = pd.read_csv(JAN2025_EXPORT)
            from sklearn.metrics import r2_score, mean_absolute_error
            xgb_r2 = r2_score(export_df["aqi"], export_df["pred_xgboost"])
            xgb_mae = mean_absolute_error(export_df["aqi"], export_df["pred_xgboost"])
            two_valid = export_df.dropna(subset=["pred_two_stage_aqi"])
            two_r2 = r2_score(two_valid["aqi"], two_valid["pred_two_stage_aqi"])
            two_mae = mean_absolute_error(two_valid["aqi"], two_valid["pred_two_stage_aqi"])
            c1, c2 = st.columns(2)
            c1.metric("XGBoost — Jan 2025 R²", f"{xgb_r2:.3f}", f"MAE {xgb_mae:.1f}")
            c2.metric("Two-stage — Jan 2025 R²", f"{two_r2:.3f}", f"MAE {two_mae:.1f}")

            st.markdown("**Per-station, per-day predictions (January 2025)**")
            station_filter = st.multiselect("Filter by station (optional)", sorted(export_df["station_id"].unique()), key="jan2025_station_filter")
            show_df = export_df[export_df["station_id"].isin(station_filter)] if station_filter else export_df
            st.dataframe(show_df.sort_values("date", ascending=False), height=320, width="stretch")
            st.caption(f"{len(export_df):,} station-days shown. Full export: {os.path.basename(JAN2025_EXPORT)}.")
        else:
            st.warning(f"Export not found: {JAN2025_EXPORT}")

    with tabs[4]:
        st.markdown("#### Full 2025 test period: actual vs. predicted AQI and PM2.5")
        st.caption("Same comparison as the January-only tab, extended to the full real 2025 held-out test period "
                   "(Jan 1 - Dec 31), 101 stations. Same already-saved test_predictions.csv files, no retraining — "
                   "these are the exact numbers behind this project's headline test-set metrics.")
        if os.path.exists(FULL2025_AQI_CHART):
            st.image(FULL2025_AQI_CHART, width="stretch")
        else:
            st.warning(f"Chart not found: {FULL2025_AQI_CHART}")
        if os.path.exists(FULL2025_PM25_CHART):
            st.image(FULL2025_PM25_CHART, width="stretch")
        else:
            st.warning(f"Chart not found: {FULL2025_PM25_CHART}")

        if os.path.exists(FULL2025_EXPORT):
            full_df = pd.read_csv(FULL2025_EXPORT)
            from sklearn.metrics import r2_score, mean_absolute_error
            xgb_r2 = r2_score(full_df["aqi"], full_df["pred_xgboost_aqi"])
            xgb_mae = mean_absolute_error(full_df["aqi"], full_df["pred_xgboost_aqi"])
            two_valid = full_df.dropna(subset=["pred_two_stage_aqi"])
            two_r2 = r2_score(two_valid["aqi"], two_valid["pred_two_stage_aqi"])
            two_mae = mean_absolute_error(two_valid["aqi"], two_valid["pred_two_stage_aqi"])
            pm25_valid = full_df.dropna(subset=["pm25_actual", "pm25_pred"])
            pm25_r2 = r2_score(pm25_valid["pm25_actual"], pm25_valid["pm25_pred"])
            pm25_mae = mean_absolute_error(pm25_valid["pm25_actual"], pm25_valid["pm25_pred"])
            c1, c2, c3 = st.columns(3)
            c1.metric("XGBoost AQI — full 2025 R²", f"{xgb_r2:.3f}", f"MAE {xgb_mae:.1f}")
            c2.metric("Two-stage AQI — full 2025 R²", f"{two_r2:.3f}", f"MAE {two_mae:.1f}")
            c3.metric("Stage 1 PM2.5 — full 2025 R²", f"{pm25_r2:.3f}", f"MAE {pm25_mae:.1f} µg/m³")

            st.markdown("**Per-station, per-day predictions (full 2025)**")
            station_filter = st.multiselect("Filter by station (optional)", sorted(full_df["station_id"].unique()), key="full2025_station_filter")
            show_df = full_df[full_df["station_id"].isin(station_filter)] if station_filter else full_df
            st.dataframe(show_df.sort_values("date", ascending=False), height=320, width="stretch")
            st.caption(f"{len(full_df):,} station-days shown. Full export: {os.path.basename(FULL2025_EXPORT)}.")

            st.markdown("**Station-wise performance matrix**")
            st.caption("One row per station — real per-station RMSE/MAE/R² for both models on the full 2025 test "
                       "period (already computed and saved at training time, not re-derived here), so you can see "
                       "exactly which stations each model handles well vs. poorly, not just the national average.")
            perf = load_station_performance_matrix()
            sort_choice = st.selectbox(
                "Sort by", ["XGBoost R² (best first)", "XGBoost R² (worst first)", "Two-stage R² (best first)", "Station name"],
                key="station_perf_sort",
            )
            if sort_choice == "XGBoost R² (best first)":
                perf = perf.sort_values("r2_xgb", ascending=False)
            elif sort_choice == "XGBoost R² (worst first)":
                perf = perf.sort_values("r2_xgb", ascending=True)
            elif sort_choice == "Two-stage R² (best first)":
                perf = perf.sort_values("r2_two", ascending=False)
            else:
                perf = perf.sort_values("station_id")
            st.dataframe(
                perf.rename(columns={
                    "station_id": "Station", "city": "City", "n_test_days": "Test days",
                    "r2_xgb": "XGBoost R²", "rmse_xgb": "XGBoost RMSE", "mae_xgb": "XGBoost MAE",
                    "r2_two": "Two-stage R²", "rmse_two": "Two-stage RMSE", "mae_two": "Two-stage MAE",
                }).style.format({
                    "XGBoost R²": "{:.3f}", "XGBoost RMSE": "{:.1f}", "XGBoost MAE": "{:.1f}",
                    "Two-stage R²": "{:.3f}", "Two-stage RMSE": "{:.1f}", "Two-stage MAE": "{:.1f}",
                }).background_gradient(subset=["XGBoost R²", "Two-stage R²"], cmap="RdYlGn", vmin=-0.5, vmax=1.0),
                height=340, width="stretch", hide_index=True,
            )
            st.caption(f"{len(perf)} stations. Green = higher R² (better fit for that station), red = lower/negative "
                       f"R² (model struggles at that station) — same 101-station set, same 2025 test split as every "
                       f"other metric in this project.")
        else:
            st.warning(f"Export not found: {FULL2025_EXPORT}")

    with tabs[5]:
        st.markdown("#### Single-station data trace: real matrix in, real PM2.5 out")
        st.caption("Picks one CPCB station and traces the pipeline end to end: real satellite/weather values for "
                   "that station (AOD from MODIS MAIAC, NO2/SO2/CO/O3/HCHO from Sentinel-5P TROPOMI - the satellite "
                   "'gas obtained via the spatial route', not CPCB - plus ERA5 weather) assembled into the exact "
                   "feature matrix Stage 1 was trained on, run live through the already-trained Stage 1 model "
                   "(gases+weather → PM2.5), and compared against the real CPCB ground-truth PM2.5 for that station.")

        all_features = _raw_features_full()
        station_list = sorted(all_features["station_id"].unique())
        sel_station = st.selectbox("Station", station_list, key="trace_station")

        station_df = all_features[all_features["station_id"] == sel_station].sort_values("date").copy()
        missing_cols = [c for c in STAGE1_FEATURES if c not in station_df.columns]
        if missing_cols:
            st.error(f"Missing required matrix columns for this station: {missing_cols}")
        else:
            stage1, imputer1 = load_stage1_model()
            X_station = imputer1.transform(station_df[STAGE1_FEATURES])
            station_df["predicted_pm25_live"] = stage1.predict(X_station)

            gt = load_pm25_ground_truth()
            gt_station = gt[gt["station_id"] == sel_station][["date", "pm25"]].rename(columns={"pm25": "actual_pm25"})
            station_df = station_df.merge(gt_station, on="date", how="left")

            st.markdown("**1. Real feature matrix** (satellite + weather values, one row per date — this is the "
                         "literal numeric matrix fed to the model, not a schematic)")
            display_cols = ["date", "split"] + STAGE1_FEATURES
            display_cols = [c for c in display_cols if c in station_df.columns]
            st.dataframe(station_df[display_cols].sort_values("date", ascending=False), height=260, width="stretch")
            st.caption(f"{len(station_df):,} dates for {sel_station.split(',')[0]}, columns = the exact "
                       f"{len(STAGE1_FEATURES)} Stage 1 features (AOD x2, 5 TROPOMI gases, 4 ERA5-derived weather "
                       f"terms, month + seasonal encoding).")

            st.markdown("**2. One row, walked through the model live**")
            trace_date = st.selectbox("Date", station_df["date"].dt.strftime("%Y-%m-%d").tolist(), key="trace_date")
            row = station_df[station_df["date"] == pd.Timestamp(trace_date)].iloc[0]

            st.markdown("PM2.5 (calculated by the model — the value everything else here feeds into)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted PM2.5 (live model run)", f"{row['predicted_pm25_live']:.1f} µg/m³")
            c2.metric("Actual PM2.5 (CPCB ground truth)", f"{row['actual_pm25']:.1f} µg/m³" if pd.notna(row["actual_pm25"]) else "n/a")
            c3.metric("Split", str(row.get("split", "n/a")))
            st.caption("Matrix row -> `imputer1.transform()` -> `stage1.predict()` -> the PM2.5 value above, computed "
                       "live in this app right now from the saved model artifacts (not read from a pre-computed file) "
                       "so the pipeline is directly verifiable date by date.")

            st.markdown(f"Other satellite gases, same station, same date ({trace_date}) — real inputs, not predictions")
            gcols = st.columns(len(GAS_TRACE_VARS))
            for gc, (col, label, unit) in zip(gcols, GAS_TRACE_VARS):
                val = row.get(col, float("nan"))
                gc.metric(label, f"{val:.3g} {unit}" if pd.notna(val) else "n/a")

            st.markdown("**3. Time series — real values, this station, full history**")
            st.markdown("PM2.5 (actual vs. live-predicted)")
            ts_df = station_df[["date", "actual_pm25", "predicted_pm25_live", "split"]].set_index("date")
            st.line_chart(ts_df[["actual_pm25", "predicted_pm25_live"]].rename(
                columns={"actual_pm25": "Actual PM2.5 (CPCB)", "predicted_pm25_live": "Predicted PM2.5 (Stage 1, live)"}))
            st.caption("Full 2019-2025 real timeline for this one station. Predicted line is computed live across "
                       "every date shown (train AND test), not just the saved 2025 test-period predictions - the "
                       "model was only trained on 2019-2024 rows (split=train), so its 2025 (split=test) predictions "
                       "are the genuinely honest out-of-sample check; train-period predictions are shown too for "
                       "visual fit reference but are not evidence of generalization.")

            st.markdown("Each satellite gas, real values (no prediction involved — these are the direct TROPOMI/MODIS inputs)")
            gas_ts = station_df.set_index("date")
            gcol_pairs = [GAS_TRACE_VARS[i:i + 3] for i in range(0, len(GAS_TRACE_VARS), 3)]
            for row_group in gcol_pairs:
                cols3 = st.columns(len(row_group))
                for c, (col, label, unit) in zip(cols3, row_group):
                    with c:
                        st.caption(f"{label} ({unit})")
                        st.line_chart(gas_ts[[col]].rename(columns={col: label}), height=180)
            st.caption("Same station, same date range, straight from the real satellite grid feature table used to "
                       "train Stage 1 - no model involved in these gas lines, they're the raw measured inputs.")


def main():
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;'>"
        "<div style='width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,#3b82f6,#ff6b35);"
        "display:flex;align-items:center;justify-content:center;color:white;font-weight:700;'>VD</div>"
        "<div><h2 style='margin:0;'>VayuDrishti</h2>"
        "<span style='color:#7a8699;font-size:13px;'>Satellite AQI &amp; HCHO · India</span></div>"
        "</div>", unsafe_allow_html=True,
    )
    st.write("")

    boundary_geojson = load_boundary()
    objective = st.radio(
        "Objective",
        ["Objective 1 · AQI", "Objective 2 · HCHO Hotspots", "PM2.5 Deep-Dive (additional)"],
        horizontal=True,
    )

    if objective.startswith("Objective 1"):
        objective1_view(boundary_geojson)
    elif objective.startswith("Objective 2"):
        objective2_view(boundary_geojson)
    else:
        pm25_deepdive_view(boundary_geojson)


if __name__ == "__main__":
    main()
