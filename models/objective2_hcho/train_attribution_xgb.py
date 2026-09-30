"""Objective 2, step 2: ATTRIBUTION — explain why already-flagged hotspots
(from detect_hotspots.py) are elevated, fire-driven or otherwise.

This fixes the teammate's notebook on two specific points found during review:
  1. The notebook trained XGBoost to predict raw HCHO magnitude for every
     pixel (test R2=0.20, weak) and called the output "attribution_score" —
     that's just the model reconstructing the input's own scale, not an
     explanation of anomalousness. Here the model is applied only to points
     detect_hotspots.py already flagged as statistically anomalous; the
     question being answered is "why is THIS flagged point elevated"
     (fire-driven vs other), not "predict HCHO everywhere."
  2. The notebook's target was the pixel's own raw HCHO value. Here too we
     keep predicting HCHO (SHAP needs a real regression target to attribute
     to features meaningfully) but restrict training+scoring to the flagged
     hotspot population specifically, so SHAP values describe what drives
     elevation among hotspots, not general background variation.

Features unchanged from the notebook (see cluster_fires.py docstring for why
they were kept): fire_dist_km, fire_count_nearby, wind_alignment, u, v.
"""

from __future__ import annotations

import joblib
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURE_COLS = ["fire_dist_km", "fire_count_nearby", "wind_alignment", "u", "v"]
TARGET_COL = "hcho_anomaly"  # deviation from regional baseline (compute_baseline.py) - not raw HCHO, see that
# module's docstring for why: raw-value regression among already-flagged points showed ~0 R2 across 4
# independent checks, because baseline geographic/seasonal variation swamped fire's marginal contribution.


def _hyperparams_for_size(n_train: int) -> dict:
    """The original notebook-derived hyperparams (300 trees, depth 5) were
    tuned implicitly for a much larger pixel-level dataset. Applied to the
    much smaller flagged-hotspot population (thousands, not millions of
    rows) they overfit badly: the corridor+season subset (~1100 train rows)
    showed test R2=-0.009 despite real ~0.1 raw feature correlations -
    XGBoost was memorizing noise, not learning the (weak but real)
    relationship. Scale capacity down for small n rather than using one
    fixed config for every sample size."""
    if n_train < 3000:
        return dict(n_estimators=60, max_depth=3, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=10.0, min_child_weight=5)
    return dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)


def train(hotspots_with_features: pd.DataFrame):
    # dropna changes row count vs the input - callers must use the returned
    # `df` (not the original input) to build any row-aligned metadata/output,
    # or risk a length mismatch (caught when run_pipeline.py's explain_all
    # step tried to align against the pre-dropna frame).
    df = hotspots_with_features.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = _hyperparams_for_size(len(X_train))
    model = XGBRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    return model, shap_values, test_r2, X, df


def save(model, path: str):
    joblib.dump(model, path)
