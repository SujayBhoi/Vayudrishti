# ISRO BAH 2026 — PS3 Surface AQI & HCHO Hotspots — Full Project Documentation

**Written:** 2026-08-05, mid-session, because the user's context window was
running low. This file is meant to let a completely fresh session (or a
person with zero prior context) understand everything that has happened,
why, and what's next. If you are resuming cold: read this whole file before
touching any code.

**Problem statement (PS3):** Development of Surface AQI & Identification of
HCHO Hotspots over India using Satellite Data, for the ISRO Bharatiya Antariksh
Hackathon 2026. Two objectives: (1) predict ground-level Air Quality Index
anywhere in India from satellite data, filling the gap where CPCB monitoring
stations are sparse; (2) detect HCHO (formaldehyde) hotspots nationally and
explain what's driving them (fire-attribution, e.g. stubble burning).

---

## 1. Architecture and folder structure

Rebuilt from scratch into `Trial -02` (this directory), reusing/extending a
prior prototype that lives untouched at the sibling directory `Trial/`
(referred to throughout as "v1" or "legacy"). **`Trial/` is never modified**
— it is the permanent, always-available fallback.

```
Trial -02/
├── PROJECT_STATUS.md          - shorter running log, updated at milestones
├── PROJECT_DOCUMENTATION.md   - this file, the comprehensive one
├── DATA_SOURCES.md            - copied from v1, dataset citations
├── data_pipeline/             - all ingestion scripts, one per source
│   ├── legacy_v1_src/         - v1's original scripts, copied in as reference (untouched)
│   ├── legacy_v1_data/        - v1's small raw files + processed datasets (copied in)
│   ├── raw_v2/                - v2 station-level extraction outputs (per-station CSVs)
│   ├── data_pipeline/         - NESTED PATH ARTIFACT, see section 5 bug log - some
│   │                             raw_v2/{source} outputs actually land ONE LEVEL
│   │                             DEEPER here due to a cwd-relative path bug
│   ├── drive_downloads/       - files pulled from Google Drive (GEE batch export outputs)
│   ├── gee_auth.py            - shared Earth Engine init
│   ├── fetch_cpcb.py          - CPCB v2 station master list builder
│   ├── cpcb_process_v2.py     - CPCB v2 weekly AQI computation (breakpoint formula)
│   ├── interactive_point_extract.py  - THE proven-safe per-station GEE extraction
│   │                             pattern (see bug log) - use this for any new
│   │                             granule-heavy source, not a national batch
│   ├── fetch_modis_aod.py, fetch_tropomi.py, fetch_era5.py, fetch_fire.py,
│   │   fetch_openmeteo_blh.py - v2 per-station extraction, all 412 stations
│   ├── batch_utils.py         - shared GEE batch-export helpers (used for
│   │                             Objective 2's grids and Objective 1's spatial grid)
│   ├── fetch_objective1_grid.py - Objective 1's India-wide spatial AQI grid extraction
│   ├── fetch_objective2_grid.py - Objective 2's India-wide HCHO/wind/fire grid extraction
│   ├── build_india_grid.py    - generates the 0.25deg national point grid (4448 points)
│   ├── load_hcho_rasters.py   - reads HCHO GeoTIFFs -> long pixel-date table
│   ├── load_objective2_grids.py - reads wind/fire grid CSVs, joins lat/lon
│   ├── load_objective1_grid.py - NEW (this session): reads Objective 1's per-source
│   │                             grid GeoTIFFs, point-samples at the national grid
│   ├── download_from_drive.py - Drive API downloader (no Drive Desktop sync on this machine)
│   └── run_all_v2_ingestion.py - convenience driver (mostly superseded by manual launches)
├── features/
│   ├── merge_sources.py       - joins all v2 satellite/met sources + CPCB on (station_id, date)
│   ├── engineer_features.py   - lag/rolling/seasonal features, per-station (v2's structural fix)
│   └── build_v2_dataset.py    - driver: loads all 9 sources, merges, saves model_dataset_v2.csv
├── models/
│   ├── objective1_aqi/
│   │   ├── artifacts/baseline_v1/  - v1's EXACT trained model, copied in UNCHANGED
│   │   ├── artifacts/v2/           - v2's trained model + metrics
│   │   ├── CALIBRATION.md          - the original EECU cost calibration writeup
│   │   ├── train_extended_xgb.py   - v2 training (includes cold-start-station fix)
│   │   ├── verify_baseline_v1.py   - confirms the copied v1 model still reproduces R²=0.671
│   │   ├── predict_grid.py         - applies a trained model across the spatial grid
│   │   ├── build_spatial_predictions.py - NEW (this session): full pipeline driver -
│   │   │                             loads grid features, fills the blh/target-enc gaps,
│   │   │                             runs v2, produces the primary spatial AQI map
│   │   ├── artifacts/location_agnostic/ - NEW: secondary model dropping station/city
│   │   │                             identity entirely (R²=0.573 vs v2's 0.660)
│   │   ├── train_location_agnostic.py - NEW: trains the above
│   │   └── train_cnn_lstm.py       - Checkpoint 5 skeleton, NOT yet run
│   └── objective2_hcho/
│       ├── artifacts/               - detection + attribution outputs
│       ├── detect_hotspots.py       - percentile threshold + DBSCAN (the "detect" step)
│       ├── cluster_fires.py         - fire clustering + attribution feature computation
│       ├── compute_baseline.py      - regional HCHO baseline / anomaly computation
│       ├── train_attribution_xgb.py - attribution model training + SHAP
│       ├── explain_hotspots.py      - SHAP -> plain-language text
│       └── run_pipeline.py          - THE end-to-end orchestrator (detect->attribute->explain)
├── backend/                    - FastAPI app, 5 endpoints, reads precomputed artifacts only
├── frontend/                   - Vite+React+Leaflet+Recharts app, builds cleanly
└── notebooks/                  - exploratory only, not used in production path
```

**Why this layered structure:** data acquisition (slow, network-bound, GEE
quota-bound) is decoupled from feature engineering, which is decoupled from
model training, which is decoupled from serving (backend/frontend). This
means a data pull failure doesn't require re-deriving features, a model
retrain doesn't require re-pulling data, etc. — and it demonstrates real
software engineering rather than one notebook.

---

## 2. Every dataset used

### Objective 1, v1 (13 cities) — the guaranteed baseline
- **Ground truth:** CPCB National AQI (2014 methodology, computed from
  pollutant breakpoints), 13 major Indian cities, weekly, **2018-01-01 to
  2023-03-27**. Sourced from a Kaggle-hosted CPCB CCR dataset (hourly,
  processed by `legacy_v1_src/cpcb_process.py`).
- **Satellite/met features:** MODIS AOD (MCD19A2, 1km, bands
  Optical_Depth_047/055), Sentinel-5P TROPOMI HCHO only (not the other 4
  gases), MODIS/VIIRS fire (FIRMS), ERA5-Land meteorology, Open-Meteo
  boundary layer height (GEE's ERA5 mirror lacks BLH).
- **Result:** 13 cities × weekly, pooled test R²=0.671.

### Objective 1, v2 (412 stations) — the national extension
- **Ground truth:** `cpcb_cleaned_daily_FINAL.csv` (found in the user's
  Downloads folder, pre-cleaned, NOT the raw Kaggle hourly archive) — 536
  unique stations, daily, **2018-01-01 to 2025-12-31**, with lat/lon already
  attached. Filtered to stations with ≥50% pollutant-data completeness → 417
  stations → 412 after dropping 5 with missing/NaN coordinates.
- **Satellite/met features:** same as v1 PLUS the 4 additional TROPOMI gases
  (NO2, SO2, CO, O3) — now a genuine multi-pollutant database as the problem
  statement requires. All extracted per-station via
  `interactive_point_extract.py` (see bug log for why not a national batch).
- **Merged training table:** `data_pipeline/legacy_v1_data/processed/model_dataset_v2.csv`
  — 97,593 station-weeks, 412 stations, 23 columns.
- **Result:** pooled test R²=0.660 after the cold-start fix (see section 3).

### Objective 2 — India-wide HCHO, fire, wind grids
- **HCHO:** Sentinel-5P TROPOMI (`COPERNICUS/S5P/OFFL/L3_HCHO`), full-India
  raster exports (not point samples), native ~1113m grid, cloud_fraction<0.3
  filter. **330 dates requested, 351 actually extracted** (weighted 60%
  Oct-Dec across 2018-2025 for stubble-burning season, 40% other months for
  contrast), downsampled to ~27km resolution (factor=25) when loaded locally
  for DBSCAN tractability. Final loaded table: 1,150,296 pixel-date records.
- **Fire:** merged MODIS (`MODIS/061/MOD14A1`) + VIIRS
  (`NASA/VIIRS/002/VNP14A1`) FireMask≥7, aggregated to the 0.25° national
  grid (4448 points) as a weekly fire-pixel COUNT within a 50km buffer — not
  individual fire-pixel locations (see section 4 for why this matters).
  Covers all 418 weeks 2018-2026. 1.85GB file (bloated, see bug log).
- **Wind:** ERA5-Land u/v 10m wind components, same 0.25° national grid, all
  418 weeks. 268MB file.
- **National grid:** `data_pipeline/build_india_grid.py`, 0.25° spacing
  (~28km, matches ERA5 native resolution), 4448 points after clipping to the
  actual India boundary (LSIB dataset).

### Objective 1's spatial AQI grid (the "predict anywhere" capability)
- India-wide, AOD + 5 TROPOMI gases + ERA5, each source at its OWN native
  resolution (not forced onto one uniform grid — see CALIBRATION.md), for
  ~250 representative dates spread across all years/seasons (not
  Oct-Dec-weighted like Objective 2 — this should represent typical
  conditions year-round). **Still extracting as of this writing** — see
  section 8.

---

## 3. Objective 1 — full history

### v1 baseline
Trained via `legacy_v1_src/train_aqi_model.py`: XGBoost, chronological 80/20
holdout, per-city lag-1/lag-2 + 2wk/4wk rolling means, seasonal encoding
(sin/cos of day-of-year), city dummy variables, train-only target-encoded
city baseline, sample-weighted (recency-weighted, half-life 365 days) to
handle a known drift issue. RandomizedSearchCV, 60 iterations, time-based CV
folds. Random Forest trained alongside as an imputation-based comparison
baseline (XGBoost handles NaN natively; RF needed imputation).

**Metrics:** pooled test R²=0.671, RMSE=42.25, MAE=28.80. Per-city spread was
wide: Kolkata 0.836 down to Kanpur -0.433, Pune -1.171 (negative R² —
diagnosed and fixed structurally in v2, see below).

This exact model was copied UNCHANGED into
`models/objective1_aqi/artifacts/baseline_v1/` and re-verified
(`verify_baseline_v1.py`) to reproduce R²=0.6714 exactly from its persisted
predictions — confirms the copy didn't corrupt anything. **This is the
permanent, always-safe fallback for the demo.**

### The Kanpur/Pune negative-R² issue — root cause and fix
**Diagnosis** (done by reading the raw Kaggle CPCB hourly archive directly,
not guessing): v1 computed "city AQI" as an **unweighted daily mean across
whatever stations happened to be reporting that day**
(`legacy_v1_src/cpcb_process.py`: `combined.groupby("date")["aqi"].mean()`).

For Kanpur: one station (UP005) supplied 82% of pre-cutoff (training-period)
rows at a high pollution baseline (PM2.5 mean ≈110.6 µg/m³). Three more
stations only came online in 2020-2021 with far fewer pre-cutoff rows. Post-
cutoff (test period), all 4 stations report at equal, regular cadence with a
much lower, more balanced mean (49.8-65.8 µg/m³ per-station). The "Kanpur"
time series therefore has an **artificial level shift baked in that has
nothing to do with real air-quality change** — the model learned a baseline
anchored to one dominant early station, then saw a structurally different
mix at test time. Pune showed the identical pattern in the opposite
direction (post-cutoff mean pulled UP by newer, more polluted stations).

**Fix (structural, not a patch):** v2 trains **per-station**, not per-city-
average. Each of Kanpur's 4 and Pune's 8 stations becomes its own row/
station_id with its own continuous history. There is no city-level averaging
left to shift under a station-roster change. Implemented in
`features/engineer_features.py` (station_aqi_target_enc replaces v1's
city_aqi_target_enc).

**Result confirmed the fix worked:** in v2, 6 of 11 Kanpur/Pune stations now
have POSITIVE R² (0.72, 0.44, 0.43, 0.39, 0.24, 0.09) — a real improvement
over v1's catastrophic pooled -0.43/-1.17. A few individual stations are
still negative, but that's now a station-specific data-quality issue, not
the citywide-aggregation artifact.

### v2 training — two passes, with a bug found and fixed in between

**First pass:** pooled test R²=0.599 — WORSE than v1 despite far more data
and stations. Investigated via per-station metrics: mean per-station R² was
actually **-0.31** (negative!) despite pooled R² looking better — a long
tail of remote/less-monitored stations (Kohima -24.68, Srinagar -16.17, Sri
Vijaya Puram -15.41) dragged it down.

**Diagnosis:** those worst stations had **ZERO rows before the chronological
train/test cutoff** — verified directly (Kohima: 0 train rows, 53 test rows;
Srinagar: 0 train, 52 test; Sri Vijaya Puram: 0 train, 44 test). These are
stations that only started reporting to CPCB within the final ~14 months of
the 8-year window. Two compounding effects: (1) `station_aqi_target_enc` is
undefined (NaN) for every one of their rows since it's built only from
training-period data: (2) the model never saw the location at all during
training — this is a cold-start/out-of-distribution extrapolation task, not
"forecast the near future for a known place," which is what the R² metric
is meant to measure. Because R² is one pooled number across all test rows,
a handful of catastrophic predictions at 9 stations dragged the ENTIRE
metric down.

**Fix:** exclude stations with zero pre-cutoff training rows entirely (both
train and test — they contribute nothing to train anyway). Implemented as
`drop_zero_training_stations()` in `train_extended_xgb.py`. 9 of 412
stations excluded.

**Second pass (final):** pooled test R²=**0.660**, RMSE=49.39, MAE=33.85, on
403 stations (412 minus 9). Mean per-station R² improved from -0.31 to
+0.027; worst single-station outlier improved from -24.68 to -7.88.

**Promotion decision:** v1 (0.671) still edges out v2 (0.660) on the strict
"only promote if it genuinely beats v1" rule, so **v1 stays primary**. But
this is presented as a strong result: v2 achieves ~98% of v1's accuracy
while covering **30x more stations nationally**. The remaining gap is
diagnosed as genuine — the remaining underperforming v2 stations (Gaya,
Motihari, Muzaffarpur, Mumbai, etc.) mostly have 150-350 weeks of training
history, ruling out the cold-start explanation for them; it's real
heterogeneity/noise from covering many more, more diverse locations, not a
further fixable bug. Filtering more stations to chase a higher number would
cross into cherry-picking rather than a legitimate fix.

### Spatial AQI grid feature-availability gap, and the resulting model choice
While building the actual "predict anywhere" prediction pipeline
(2026-08-05, this session), found that neither v1 nor v2's trained model can
run purely off what `fetch_objective1_grid.py` extracts: `station_aqi_target_enc`/
`city_aqi_target_enc` (v1's single strongest feature, 15.2% importance; v2's
station_aqi_target_enc + city dummies together ~15%) have no defined value
at a grid point that isn't one of the 13/403 trained locations - not a
missing-data problem, a genuinely undefined one. Separately, `blh` was never
grid-extracted at all (not in `fetch_objective1_grid.py`'s SOURCES dict),
and v1 additionally needs `fire_count`/`fire_max_conf`, also never
extracted.

**Decision (with the user):** two-track approach, same "guaranteed baseline
+ attempted stronger version" pattern used throughout this project.
1. **Primary map:** use v2 (not v1) - v2's feature set doesn't need fire at
   all, so it only has the `blh` gap, not both. `station_aqi_target_enc` is
   filled via a regional 2°×2° average of trained stations' target-encoding
   around each grid point (same regional-baseline idea as Objective 2's HCHO
   anomaly baseline), widening the box for the few grid points with no
   station within 2°. `blh` is filled via a day-of-year climatology (mean
   BLH across all 403 stations/years, grouped by day-of-year) since BLH is
   primarily seasonal, far less location-sensitive than a pollution
   baseline. City dummies stay 0 everywhere. Implemented in
   `data_pipeline/load_objective1_grid.py` (grid GeoTIFF -> point-sampled
   feature table) and `models/objective1_aqi/build_spatial_predictions.py`
   (the full assembly + prediction driver).
2. **Secondary, location-agnostic model** (`models/objective1_aqi/
   train_location_agnostic.py`): drops station/city identity entirely,
   trained only on satellite/met + lat/lon + seasonal features, so it's
   architecturally valid at any point without a proxy. Same chronological-
   holdout methodology as v2 (same 9-zero-training-station exclusion, same
   XGBoost search space) for a directly comparable number.
   **Result: pooled test R²=0.573** (vs v2's 0.660 with station identity,
   v1's 0.671) - RMSE=55.39, MAE=40.17, on the same 78,006/19,187
   train/test split. Verified not a degenerate/collapsed predictor (the
   5.11 lesson applied here too): feature importance spreads sensibly
   across real physical variables (doy_cos 18.6%, temp_c 7.2%, rh_pct,
   no2, aod_047 - no single dominant/trivial split), and per-station test
   R² varies genuinely (median 0.25, 192/316 stations positive, up to
   0.886, long negative tail from stations with unusually strong individual
   character that only a per-station baseline can capture) - not clustered
   suspiciously near zero the way a constant predictor's would be.
   **Verdict: validates reasonably well** - a real ~13% relative accuracy
   cost for losing station identity, not severe enough to be incredible.
   Per the user's decision, presented alongside the primary map (not a
   replacement) as the explicit, honest answer to "how does this generalize
   to areas with no nearby station at all."

### Spatial AQI grid — status
The actual "predict AQI at any point in India" capability. Calibration test
(5 dates, AOD+ERA5) succeeded: AOD 0.044 EECU-hr/date, ERA5 0.001 EECU-hr/date
— cheap enough to support ~250 dates within the reserved 60 EECU-hr budget.
Full extraction launched, hit two bugs along the way (see section 5), fixed,
**still running as of this writing** (see section 8 for exact current state).
`predict_grid.py` (applies a trained model to grid features) is written but
has not yet been run against real grid data since the extraction isn't done.

---

## 4. Objective 2 — full history

### Detection — DONE, validated, working well
`models/objective2_hcho/detect_hotspots.py`: percentile threshold (p90,
computed against the full national multi-year HCHO distribution, not a
self-referential already-selected window) + DBSCAN spatial clustering
(eps=0.4deg≈44km, min_samples=3) on the flagged points. Uses ONLY the HCHO
field — no fire/wind data touches this step at all, matching the
"statistical threshold or clustering to identify hotspots" requirement
literally.

**Validated on the full 351-date dataset:** 115,030 hotspot pixel-dates
flagged (threshold=0.000328775 mol/m²), 9,244 DBSCAN clusters, genuinely
spread across India (Delhi, Assam, Odisha, Maharashtra, Kerala confirmed in
early testing — not just the old Punjab-Haryana corridor). **This is the
strong, presentable, working half of Objective 2.**

### Attribution — architecture built correctly, but shows a genuine null result
`models/objective2_hcho/cluster_fires.py` computes, per flagged hotspot
point: `fire_dist_km` (distance to nearest fire-active grid cell),
`fire_count_nearby` (count of fire-active cells within 50km),
`wind_alignment` (cosine similarity between the wind vector and the
fire→point vector), plus raw `u`/`v`. These 5 features feed
`train_attribution_xgb.py`, which trains XGBoost to predict HCHO (originally
raw value, later redesigned to a baseline-relative anomaly — see below) and
produces SHAP values via `explain_hotspots.py`.

**Why NOT individual fire pixels:** the fire data is a 0.25°-grid-cell
WEEKLY COUNT (matching v1's original, more mature `hotspot_grid_extract.py`
approach — fire_density via focal sum on a grid), not raw fire-pixel
detections like the teammate's original notebook used. This is a
scientifically defensible adaptation (it's what v1 already validated), not
a shortcut.

**The two-model design (explicit decision, see section 6):**
- **PRIMARY = "corridor_season"**: trained only on hotspots in the Punjab-
  Haryana-Delhi-western-UP corridor (lat 27-32°N, lon 73-81°E) during Oct-Dec
  (the actual stubble-burning season/region).
- **SECONDARY = "national_all_season"**: trained on every flagged hotspot,
  everywhere, all year.
- Detection itself stays fully national/year-round regardless — only the
  ATTRIBUTION step is split.

**The null-result finding, in full:**
1. First full-pipeline run (with a real date-matching bug present, see
   section 5): R²≈0 for both models, on the 84-dates-so-far partial sample.
2. After fixing date-matching: still R²≈0. Investigated the FEATURE VALUES
   directly (not just the model score) — confirmed real, sensible variance
   (fire_dist_km 17-59km, fire_count_nearby 0-12, wind_alignment -0.996 to
   +0.9998) — so the features themselves are computed correctly, this isn't
   a broken calculation.
3. Checked raw correlation (not model-dependent): overall r≈-0.03 to 0.04;
   restricted to Oct-Dec + corridor, r≈-0.09 to 0.10 — weak but real, and the
   direction is physically sensible (closer/more fires → higher HCHO).
4. Tried heavier XGBoost, then realized 300 trees/depth 5 was overfitting a
   ~1,100-row corridor subset; added regularization — but overshot into a
   model that collapsed to predicting a CONSTANT (verified: prediction std
   ≈ 0) for both raw and anomaly targets, which is WHY two different targets
   produced numerically near-identical R² (a genuine red flag caught and
   explained, not a coincidence left unexplained).
5. Found the right middle ground (moderate regularization) — genuinely
   varying predictions confirmed, but R² still ≈0, now confirmed via 5-fold
   CV with 3 different hyperparameter configs, all converging to R²≈-0.008.
6. Cross-checked with plain linear regression (much lower variance, better
   suited to weak signals at small n): 5-fold CV mean R²≈-0.0025 (raw
   target). Confirms this isn't an XGBoost-specific artifact.
7. **Redesigned the target as a baseline-relative anomaly**
   (`compute_baseline.py`: HCHO minus the mean HCHO for that point's own
   2°×2° region across all available dates) — the idea being that raw HCHO
   has large latitude/humidity/vegetation-driven baseline variation that
   swamps fire's marginal contribution. This is the scientifically correct
   fix for the target formulation. Result: linear CV R² improved from
   -0.0025 to +0.0067 — a small improvement, but still noisy/inconsistent
   sign across folds. Diagnosed WHY the improvement was small: in the
   corridor+season subset, baseline geographic variation only explains ~31%
   of total HCHO variance there (std 2.6e-5 vs total std 8.4e-5) — so most
   of the remaining variance still isn't fire.
8. **The critical test: reran everything on the FULL 351-date dataset**
   (11.5x more data than the partial 84-date sample used for all the above:
   15,982 corridor points vs 1,386). Result: corridor R²=-0.001, national
   R²=0.002 — barely moved from the small-sample estimate. **This proves the
   null result is real, not a sample-size problem** — if a learnable
   relationship existed at this feature/resolution, 11.5x more data would
   have revealed it.

**Conclusion (agreed with the user):** genuine, weak-but-real result, not a
total null. Most likely explanations for the weakness: (a) weekly temporal
aggregation is coarse relative to plume-transport dynamics; (b) grid-cell
fire COUNTS are a fairly crude proxy for actual fire intensity/duration/
plume strength. The raw correlation direction is consistently right, just
not strong enough to be a demo-headline number on its own.

**Lag-window follow-up test (requested by user, done):** tested 0/1/2-day
lag between fire activity and HCHO measurement (instead of same-week
nearest-match) to check the plume-transport-lag theory specifically. Using
5-fold CV + linear regression (see methodology note below) on the corridor+
season subset (n=15,982): lag=0d R²=0.0175, lag=1d R²=0.0153, lag=2d
R²=0.0165 — **no meaningful difference across lags**, all within fold-to-
fold noise. Raw correlation ticks up marginally with lag (0.087→0.096→0.097)
but not enough to matter. **Conclusion: lag is not the primary explanation
for the weak signal; stopped here per the user's time-boxed instruction.**

**Methodology note (important, found while running the lag test):** the R²
numbers above (0.015-0.0175, weakly POSITIVE) come from 5-fold
cross-validated plain LinearRegression. The earlier-reported "-0.001"
finding came from a single 80/20 train/test split with XGBoost — a noisier,
less reliable estimate for a weak-signal problem like this. **The more
accurate characterization is "weakly positive, ~1.5-2% of variance
explained," not "zero."** Same qualitative conclusion either way (too weak
to be a confident standalone attribution model), but worth using the more
careful number if asked for a specific figure.

**Demo framing, locked in with the user:** lead with Detection (the strong,
validated capability — 115K hotspots, 9,244 clusters, real national spread).
Frame Attribution honestly as "consistent directional signal, not yet
statistically reliable at weekly/25km resolution" rather than a confident
causal claim. This is defensible and demonstrates rigor rather than looking
broken.

**Requested next step (not yet done as of this writing):** test whether a
1-2 day LAG between fire activity and HCHO measurement (instead of same-week
nearest-match) addresses the plume-transport theory. User explicitly said:
try this since it's cheap with data already in hand, but don't sink
significant further time into it if it doesn't help quickly — the current
honest framing is already solid enough to present regardless of outcome.

---

## 5. Bug/incident log — every one, with root cause and fix

This project hit a genuinely large number of real bugs. Listed chronologically.

### 5.1 AOD national-batch compositing cost incident (the most serious one)
**What happened:** the first attempt at v2's station-level point extraction
used one combined GEE batch export (`Export.table.toDrive`) covering all 412
scattered stations nationally at once. Within about a minute of running,
`ee.data.getTaskStatus()` showed **10.2 EECU-hours** consumed (6.8% of the
entire 150 EECU-hr/month budget) for a task that was still not done. Caught
and cancelled immediately; the other 7 queued (not-yet-started) tasks
cancelled at zero cost.

**Root cause:** MODIS AOD (`MCD19A2_GRANULES`) is stored as many overlapping
per-day swath granule tiles. Building a periodic (weekly) COMPOSITE image
(`ic.mean()`) requires GEE to combine ALL granules touching the geographic
extent of the computation BEFORE reducing to points. When station points are
scattered across the whole country, that extent is effectively the whole
country — so "cheap point extraction" silently became "full national
compositing, 417 times" (once per week), regardless of how few points were
ultimately sampled. v1's original per-city loop never hit this because it
called `.filterBounds(small_local_region)` BEFORE compositing, which lets
GEE prune to only nearby granules — the batched version dropped that
per-point geographic scoping when combining stations into one call.

**Fix:** reverted to v1's proven pattern — INTERACTIVE per-station calls
(not a national batch), each scoped to a small buffer via `filterBounds`,
looped over all 412 stations with light thread-pool concurrency (8 workers).
Implemented in `data_pipeline/interactive_point_extract.py`. Confirmed safe
and fast at small scale (10 stations, 3 months, 29 seconds) before running
at full scale. **This is now the standard pattern for any new granule-heavy
source extraction in this project — do not build a new national batch
export for point data.**

A follow-up attempt at a "per-station-scoped BATCH export" (map stations
outermost in the GEE graph, still one combined task) was also tried as a
middle ground, but hit a persistent, unrelated projection error (5.2 below)
and was abandoned in favor of the fully interactive approach.

### 5.2 MODIS sinusoidal projection error in batch-export context
**What happened:** while testing the per-station-scoped batch alternative
(5.1), every attempt failed with `"Unable to compute intersection of
geometries in projections SR-ORG:6974 and EPSG:4326"` — even for a SINGLE
station, single week. Tried explicit `crs=` on reduceRegion, tried
`.reproject()` before reduceRegion (worked interactively for one manual
test), tried `.mosaic()` instead of `.mean()` — all failed the same way in
actual BATCH EXPORT context specifically (never reproduced interactively).

**Root cause:** not fully resolved (abandoned rather than root-caused,
because 5.1's fix — going fully interactive — sidesteps it entirely and IS
proven safe). Likely something specific to how `Export.table.toDrive`
reconciles MODIS's native sinusoidal tile grid during table export,
possibly related to empty-collection edge cases on weeks where no granule
touches a tiny station buffer.

**Fix:** abandoned the batch-scoped approach for AOD/TROPOMI entirely,
switched fully to 5.1's interactive pattern. Cost: 5 small failed diagnostic
tasks, all under 40 EECU-seconds combined (negligible).

### 5.3 `qa_value` band assumption — wrong, found TWICE
**What happened:** assumed all Sentinel-5P TROPOMI OFFL L3 products expose a
`qa_value` band for quality filtering. First hit as a real failure in
`fetch_tropomi.py` (station-level v2 extraction) — ALL 4 gas extractions
(NO2, SO2, CO, O3) failed on every station.

**Root cause (verified from actual GEE error output, not assumed):** NO2,
SO2, O3 collections expose `cloud_fraction` but NOT `qa_value`. CO exposes
NEITHER (its SWIR retrieval is far less cloud-sensitive than the others —
standard practice to skip QA masking for it, not a gap).

**Fix (first instance):** NO2/SO2/O3 → `cloud_fraction<0.3` (same pattern as
HCHO); CO → no QA mask at all. Fixed in `fetch_tropomi.py`, tested at small
scale (3 stations), then relaunched at full scale successfully.

**Found again, un-synced:** the SAME wrong `qa_value` assumption was baked
into `fetch_objective1_grid.py`'s SOURCES config (written separately, before
the fix above was made, and never updated to match). Caused 23 failed batch
tasks in the Objective 1 spatial grid extraction before being caught.
**Lesson: when the same collection/config pattern appears in more than one
file, a fix in one place must be propagated to the other — it wasn't, and
that gap cost real queue time twice.**

**Fix (second instance):** same correction applied to
`fetch_objective1_grid.py`'s SOURCES dict. ~100 broken-config pending tasks
cancelled, relaunched with the fix.

### 5.4 Empty-image / "zero bands" crash on dates with no valid data
**What happened:** an Objective 2 HCHO batch export failed with `"Can't get
band number 0. Image has no bands."` A later Objective 1 grid HCHO topup
batch hit the same error.

**Root cause:** any date can have zero valid satellite images after date-
range + cloud-fraction filtering (either genuinely before a product's
operational launch, or just an all-cloudy day over India with no usable
pass). `ImageCollection.mean()` on an empty collection returns a bandless
image, which crashes downstream when combined with other (real) bands via
`ee.Image.cat()`.

**Fix:** `ee.Algorithms.If(collection.size().gt(0), realMean, fallbackImage)`
— a masked constant-0 fallback image with an explicit
`.reproject(crs=..., scale=...)` (an unprojected fallback silently breaks
downstream sampling even when the guard itself works) — matching what v1's
ORIGINAL `hotspot_grid_extract.py` already did for exactly this reason (a
pattern that existed in the reused code but wasn't initially carried over
into the new scripts). Applied in `fetch_objective2_grid.py` and
`fetch_objective1_grid.py`.

**Follow-up bug in the fix itself (5.4b):** the fallback constant image
(`ee.Image.constant(0)`) defaults to Byte type, while real data reductions
are Float64. Exporting multiple dates' bands together via `ee.Image.cat()`
fails with `"inconsistent types: Byte and Float64"` the moment even ONE date
in a multi-date batch needs the fallback. Fixed by explicitly `.toFloat()`
on both the fallback AND the real-data branch, ensuring type consistency
regardless of which branch any given date takes. (Caught in
`fetch_objective1_grid.py`; the Objective 2 version got lucky and never hit
a batch mixing both branches, but carries the same latent risk — worth
checking if it's ever reused for a new run.)

### 5.5 Date-matching bug in Objective 2 attribution (silently wrong join)
**What happened:** attribution features are built by joining each flagged
HCHO date against the "same-period" fire/wind grid data. The first
implementation used an EXACT STRING match (`fire_df["date"] == date`).

**Root cause:** HCHO dates are monthly (the 15th of each month, per the
representative-date selection); fire/wind grids are WEEKLY buckets anchored
at 2018-01-01. An exact match between these two only aligns by pure
coincidence (verified: 13 of 84 dates matched, essentially random luck based
on whether a given month's 15th happened to land on a 7-day-multiple-from-
Jan-1 boundary). Every other date's hotspots were silently attributed
against a essentially-arbitrary, unrelated week's fire/wind conditions
(or dropped if that exact date never appeared in the fire/wind data at all).
This was the actual, quieter cause of the "R²≈0" finding before it was
re-confirmed as a genuine null result via other means (section 4).

**Fix:** match each HCHO date to its NEAREST available fire/wind week (via
`np.argmin(np.abs(fire_dates - pd.Timestamp(date)))`), not an exact string.
Verified after the fix: 84/84 (later 336/336 on the full dataset) matched,
max gap 3 days — genuinely the same real-world period every time.

### 5.6 `dropna` row-count mismatch between training and metadata
**What happened:** `train_attribution_xgb.py`'s `train()` drops rows with
NaN features/target before fitting. The caller (`run_pipeline.py`) tried to
attach SHAP explanations back onto the ORIGINAL (pre-dropna) metadata frame,
causing a `ValueError: Length of values (3172) does not match length of
index (3183)`.

**Fix:** `train()` now returns the dropna'd, `reset_index(drop=True)`'d
dataframe alongside the model/X/SHAP values, and the caller uses THAT
returned frame (not the original input) for any row-aligned metadata.

### 5.7 Cwd-relative "nested path" bug (recurring, cosmetic but confusing)
**What happened:** several `fetch_*.py` scripts define their output
directory as a plain string like `"data_pipeline/raw_v2/aod"`. This is
CWD-RELATIVE, not relative to the script's own location. Most launch
commands `cd` into `data_pipeline/` first, so the actual path ends up
nested one level deeper: `data_pipeline/data_pipeline/raw_v2/aod/`.

**Confirmed nested:** aod, hcho, no2, so2, co, o3, era5, fire (all of
`fetch_modis_aod.py`, `fetch_tropomi.py`, `fetch_era5.py`, `fetch_fire.py`).
**NOT nested:** blh (`fetch_openmeteo_blh.py` uses
`os.path.dirname(__file__)` — a `__file__`-relative path, immune to this
particular bug by accident of a different coding style, not a fix).

**Impact:** caused repeated confusion when manually checking progress
(checking the wrong/expected flat path showed 0 files when the process was
actually working fine at the nested path) and a genuinely wrong monitor
script early on (watching the flat path for 4 sources, which would NEVER
have fired complete). Never caused actual data loss or corruption — files
were always written correctly, just not where a naive path guess expected.

**Fix:** not changed in the fetch scripts themselves (would require
re-verifying every already-completed extraction's actual location).
Instead, `features/build_v2_dataset.py`'s `SOURCE_DIRS` dict has the
CORRECT (nested-aware) paths hardcoded with an explanatory comment. **Any
new code reading these outputs should check that dict's paths, not assume.**

### 5.8 Bloated fire-grid export (1.85GB where ~50MB would do)
**What happened:** `fetch_objective2_grid.py`'s fire-grid export applies a
50km buffer to each of the 4448 grid points BEFORE `reduceRegions` (needed
so the region reduction genuinely counts fires within 50km, not just at the
exact point). But this means the exported table's `.geo` column carries the
FULL buffered POLYGON geometry for every single row — repeated ~1.86 million
times (4448 points × 418 weeks). Not wrong, just extremely wasteful.

**Fix:** not re-exported (would cost more EECU + time for a non-functional
issue). `data_pipeline/load_objective2_grids.py` works around it by
`usecols=`-selecting only the needed columns (dropping `.geo` entirely) and
joining `grid_id` → lat/lon from the local `india_grid_0.25deg.csv` instead
of parsing the polygon.

### 5.9 Open-Meteo rate-limit crash (BLH extraction)
**What happened:** `fetch_openmeteo_blh.py` crashed the entire 412-station
loop after 225 successful stations, hitting a `429 Too Many Requests` that
the original fixed-5-second retry couldn't clear.

**Fix:** two changes — (1) PROACTIVE throttling (a small delay, 1.5s, before
EVERY request, not just reactive backoff after a failure already happened);
(2) made the main loop resilient to any single station's failure (catch,
log, continue; retry failed ones once more at the end after a cooldown)
rather than letting one bad station crash the whole run. Confirmed working:
resumed cleanly from the 225 already-cached stations, no further crashes.

### 5.10 OAuth token expiry mid-download (Drive API)
**What happened:** `download_from_drive.py` fetched a Drive API access
token ONCE at the start of a folder download. A 27-file, multi-gigabyte
download (spanning over an hour of wall-clock across session updates) ran
long enough for the token to expire, causing a `401 Unauthorized` partway
through (crashed after 15-17 of 27 files).

**Fix:** refresh the OAuth token before EVERY file download (cheap —
google-auth is a no-op if the token still has a healthy remaining lifetime),
not once per folder. Also added: atomic writes (download to a `.part`
temp file, `os.path.replace()` only on success — so a crash mid-download
never leaves a file that LOOKS complete but isn't), and per-file exception
handling (one failed file logs and continues rather than aborting the whole
folder). Confirmed: resumed cleanly, all 27 files downloaded successfully.

### 5.11 Over-regularized attribution model collapsed to a constant predictor
**What happened:** part of the Objective 2 attribution investigation
(section 4) — after seeing R²≈-0.009 for a heavily-regularized XGBoost, a
second target formulation (anomaly instead of raw HCHO) was tried and
produced an numerically ALMOST IDENTICAL R² (-0.00877 vs -0.00913) —
suspicious, since two genuinely different targets shouldn't score nearly
identically by chance.

**Root cause, found by directly inspecting predictions (not just the R²
number):** `preds.std()` was ~0 for BOTH targets — the model, over-
regularized (`reg_lambda=10, min_child_weight=5, max_depth=3, only 60
trees`), had fully collapsed to predicting a constant (≈ the training mean)
regardless of input features, for either target. R² for a constant
predictor depends only on the gap between train-mean and test-mean relative
to test variance — which happened to be similar for both targets since one
is just a (non-constant, but small-relative-to-total-variance) shift of the
other.

**Fix:** reduced regularization to a genuinely learning-capable middle
ground; re-verified predictions now had real variance across configs.
**Lesson applied:** don't trust a metric number alone when something looks
suspiciously coincidental — check the actual predictions.

### 5.12 Two separate "only ~96 dates instead of requested N" bugs
**What happened:** BOTH `fetch_objective2_grid.py`'s `select_dates()` and
(separately, un-synced) `fetch_objective1_grid.py`'s
`select_representative_dates()` generated only ONE candidate date per
(year, month) combination — 8 years × 12 months = 96 candidates max,
regardless of what `n_dates` was requested. A request for 330 dates silently
produced 96; a request for 250 dates also silently produced 96.

**Root cause:** both functions built their candidate date list as one
`day=15` timestamp per month, rather than real daily candidates across the
full range.

**Fix (applied twice, same pattern, in each file separately since they're
independent modules):** generate real daily candidates (`pd.date_range`)
across the full year range, then evenly-subsample down to the requested
count via `_evenly_spaced()`. Verified in both cases: requesting 330 (or
250) dates now actually returns that many. Both times, the fix was applied
via a "topup" launch (compute the corrected full date list, subtract
already-submitted dates, launch only the difference) rather than
re-launching everything from scratch, to avoid wasting already-completed
work.

### 5.13 Killing a "duplicate" background process killed the only process
**What happened (2026-08-05, this session):** while parallelizing the
Objective 1 grid file downloads from Drive, `Get-CimInstance Win32_Process`
consistently showed **two distinct PIDs** running the identical
`python.exe ... download_from_drive.py`/`concurrent_download.py` command
line for every single background launch (6 separate launches -> 12 PIDs;
1 combined launch -> 2 PIDs). Read as a real duplicate-process race (a
genuine risk if true: two independent processes could both pass the
"file doesn't exist yet" check for the same not-yet-downloaded file and
race-write the same `.part` path), one of the two PIDs was killed via
`Stop-Process`. The entire background job immediately died - including the
PID that was deliberately left running - and the harness reported the task
as FAILED (exit 127).

**Root cause:** the two listed PIDs were not independent duplicate
invocations - they're both facets of the SAME logical background job as
this harness's Windows/Git-Bash backgrounding mechanism represents it
(confirmed empirically: killing either one tears down both). The earlier
6-separate-launches version had already been downloading correctly for
~45 minutes with no file corruption (every completed `.tif` verified
readable via `rasterio.open()` + a real pixel read afterward) despite
showing the same 2-PIDs-per-launch pattern throughout - direct evidence
there was never a real race, just a process-listing artifact.

**Fix:** stopped treating a second same-command PID as a duplicate to kill.
Re-launched the concurrent downloader once, cleanly, and left both listed
PIDs alone. Cost: one clean restart of the download (thread-pool downloader
re-lists already-present files and skips them via the existing
`os.path.exists` guard, so nothing already downloaded was re-fetched or
lost - just the one in-flight file at the moment of the kill, cleaned up
via the standard `.part`-file deletion already used for interrupted
downloads).

**Lesson applied:** verify a "duplicate" is real (e.g. check whether killing
one changes the other's liveness) before killing anything in an environment
whose process-tracking semantics haven't been established yet - don't
assume a process listing tool's output maps 1:1 to "independent, race-
capable processes" without confirming that mapping first.

### 5.14 Outer merge across partially-downloaded sources doubled row count
**What happened (2026-08-05, this session):** running
`build_spatial_predictions.py` as a sanity check against partially-downloaded
grid data produced 2,170,136 predictions from only 1,085,068 loaded feature
rows - exactly 2x, an immediate red flag (same "don't trust a suspicious
round-number coincidence" discipline as 5.11).

**Root cause:** `load_objective1_grid.py`'s `build_grid_feature_table()`
merged each of the 7 sources onto the accumulating table with
`how="outer"` on `["grid_id","date"]`. Because each source had a
DIFFERENT, independently-partial set of downloaded dates at the time (AOD
was much further ahead than the other 5), an outer merge created a brand
new row for every (grid_id, date) combination that existed in a
later-merged source but not in the first-merged one (AOD) - and since
lat/lon only travels with the first source, every such row got
`lat=NaN, lon=NaN`. Verified directly: `grid_points` after dedup had 8894
rows instead of 4447 - every single grid_id had exactly one real-lat/lon
row and one all-NaN row.

**Fix:** changed to `how="inner"` - a prediction row is only meaningful if
every source actually has data for that (grid_id, date) pair. This is
correct for the partial-download case AND is the right fail-safe choice
for the eventual complete-data case too, in case any future re-run's
per-source date lists don't perfectly align for some other reason.
Re-ran after the fix: 142,304 rows in, 142,304 predictions out - clean 1:1,
confirmed correct.

---

## 6. Key decisions locked in, and why

1. **Free tier only, no billing.** Explicit user decision when the EECU
   budget tradeoff was first discussed (Checkpoint 0 calibration showed the
   free 150 EECU-hr/month Community tier vs. paid unlimited). All scope
   decisions since respect this ceiling.
2. **330 HCHO dates for Objective 2** (not the original ~860-date plan) —
   chosen after real per-date cost came in at ~0.038 EECU-hr/date (batched
   multi-band export), cheaper than the initial single-date calibration
   estimate of ~0.09 EECU-hr/date. Weighted 60% Oct-Dec (peak stubble
   season) / 40% other months for contrast.
3. **Per-station `filterBounds`-scoped interactive extraction** for all
   granule-heavy sources (AOD, all 5 TROPOMI gases) — the direct fix for the
   5.1 incident. NOT a batched national point extraction, regardless of how
   tempting the efficiency looks on paper — it actively costs MORE for these
   specific sources. ERA5/fire ARE safe to batch nationally (simple,
   already-gridded daily products, no compositing cost) — confirmed by
   calibration.
4. **Anomaly-target redesign for attribution** (section 4, section 5-adjacent)
   — predicting HCHO deviation from a 2°×2° regional baseline instead of raw
   value, to isolate fire's marginal contribution from baseline geographic/
   seasonal variation. The scientifically correct fix for the target
   formulation, even though it only partially improved the result (the
   deeper issue turned out to be temporal/spatial resolution, not target
   formulation alone).
5. **Two-model attribution structure** (corridor+season PRIMARY, national
   all-season SECONDARY, kept separate) rather than one blended national
   model or silently narrowing the "national" detection scope back down to
   just the corridor. Detection stays fully national regardless.
6. **v1 never modified in place.** `Trial/` (sibling dir) stays completely
   untouched throughout, as an independent, always-available fallback.
7. **9 zero-training-history stations excluded from v2's AQI evaluation**
   (cold-start artifact, not real signal) — a structural fix, not
   cherry-picking, since those stations contributed nothing to training
   either way.
8. **Demo framing: Detection-first for Objective 2.** Lead with the strong,
   validated 115K-hotspot/9,244-cluster national detection result;
   Attribution framed honestly as "consistent directional signal, not yet
   statistically reliable at weekly/25km resolution" — agreed with the user
   after the full-dataset null result was confirmed robust.

---

## 7. EECU budget status

**As of 2026-08-05 (this session, re-checked live): ~69.27 of 150 EECU-hr used
(46.2%).** Healthy remaining margin (~80.7 EECU-hr). The 5.1 incident (10.2
EECU-hr) remains the single largest cost event; everything since has been
measured and controlled.
Check live via:
```python
import ee
ee.Initialize(project='isro-bah-2026')
ops = list(ee.data.listOperations())
total = sum(float(op.get('metadata',{}).get('batchEecuUsageSeconds',0) or 0) for op in ops)
print(f'{total/3600:.2f} of 150 EECU-hr')
```

---

## 8. Currently running / queued in the background

1. **Objective 1 spatial grid extraction** — re-audited fresh via
   `ee.batch.Task.list()` (2026-08-05, this session), not trusting the old
   snapshot above. Actual live state at that point:
   - AOD: 11 plain + 25 topup (100->250-date upgrade), all 36 COMPLETED.
   - ERA5: 11 plain + 25 topup, all 36 COMPLETED.
   - HCHO: plain (10 batches, n=100) 7 COMPLETED/3 FAILED; topup (25
     batches, n=250) 22 COMPLETED/3 FAILED. The 6 failures were all the
     "Image with no bands" empty-collection crash, submitted before the
     empty-image guard fix landed.
   - NO2/SO2/CO/O3 (each identical pattern): "plain" (10 batches, n=100,
     pre-qa_value-fix) all 10 FAILED — but fully superseded by "fixed"
     (25 batches, n=250, corrected cloud_fraction/no-mask config): 24/25
     COMPLETED, 1 FAILED per source on the Byte/Float64 mismatch (5.4b).
     A further "topup" set (25 batches each, same n=250 date range as
     "fixed") exists but is entirely CANCELLED for all 4 sources —
     apparently a redundant resubmission caught and cancelled before
     running, superseded by "fixed" already mostly succeeding. Left
     cancelled/not relaunched (see action taken below) since relaunching
     it would just re-export dates "fixed" already covers, burning EECU
     for no new coverage. 2 stray one-off "_test" tasks (no2_test, co_test)
     also FAILED on Byte/Float64 — diagnostic-only, not part of systematic
     date coverage, not relaunched.

   **Action taken this session:** confirmed via direct interactive `getInfo()`
   calls against live GEE (not just reading the code) that both fixes now
   hold — an empty-collection HCHO date no longer crashes, and a multi-date
   NO2/SO2/CO/O3 image combining a real date with a fallback date no longer
   throws the Byte/Float64 type error. Reconstructed each failed batch's
   exact date range deterministically (`select_representative_dates` +
   `chunk` are pure functions of `n_dates`; 25 batches @ 10/batch implies
   n=250 was used for "fixed"/"topup", 10 batches @10/batch implies n=100 for
   "plain" — consistent with the 100->250-date upgrade AOD/ERA5 also went
   through). Relaunched exactly the 10 genuinely-still-broken batches (not a
   full re-run, to avoid re-submitting already-completed work):
   no2_fixed001, so2_fixed002, co_fixed001, o3_fixed002 (Byte/Float64),
   hcho_plain000/001/004, hcho_topup000/001/002 (empty-image guard). All 10
   confirmed queued (RUNNING/READY) via `getTaskStatus()` immediately after
   submission. EECU budget checked immediately before launch: 69.27/150
   EECU-hr (46.2%) — healthy margin for this small (10-batch) relaunch.
   **Update (2026-08-05, later same session):** 7 of the 10 relaunched
   batches confirmed COMPLETED (no2/so2/co/o3 fixed retries all done; HCHO
   plain000/001/004 retries all done). Remaining 3 (hcho_topup000/001/002
   retries) still draining - 1 RUNNING, 2 READY at last check. Once these
   land, the grid extraction is fully done at n≈242 representative dates
   (see below - not literally 250, `_evenly_spaced`'s rounding dedup).

   **Also this session:** downloading all 7 sources' completed GeoTIFFs from
   Drive to build the actual prediction pipeline (previously nothing had
   been downloaded for Objective 1's grid at all). ERA5 (100% complete) was
   downloaded and loader-tested first: `data_pipeline/load_objective1_grid.py`
   correctly samples each source's rasters at the 4447-point national grid;
   confirmed 242 unique dates (not 250) in the "topup"/"fixed" batches -
   `select_representative_dates(250)`'s `_evenly_spaced()` dedups rounded
   indices down to 242 unique daily dates, which is what was actually
   submitted originally, not a new bug. Downloads for the other 6 sources
   were slow serially (AOD alone ~3GB across 25 files); parallelized to a
   single 6-worker thread-pool downloader across all sources at once (see
   5.13 for an incident hit while doing this - a process-kill mistake, no
   data loss, lesson recorded). Concurrent downloading confirmed genuinely
   faster once running cleanly.

   Also built the actual prediction pipeline this session -
   `models/objective1_aqi/build_spatial_predictions.py` - and discovered
   (see the new "Spatial AQI grid feature-availability gap" subsection
   above, under section 3) that neither v1 nor v2 can run purely off what
   the grid extraction provides: `blh` was never grid-extracted at all, and
   v1 additionally needs fire data, also never extracted. Resolved by using
   v2 (avoids the fire gap) with a regional 2°×2° target-encoding proxy and
   a BLH day-of-year climatology - see that subsection for full detail.
   Also trained and validated a secondary location-agnostic model
   (R²=0.573) per the user's explicit decision to present both.

   **Next action needed:** let the queue + downloads finish (no hard
   deadline), then run `build_spatial_predictions.py` for real and inspect
   the output distribution before calling the map done.
2. **Objective 2 attribution lag-window test — DONE.** No meaningful
   improvement from 1-2 day lag (see section 4 for full numbers). Confirmed
   the weak signal isn't a lag/timing artifact. Also surfaced a methodology
   correction: the honest number is "weakly positive, ~1.5-2% variance
   explained" (5-fold CV linear regression), not "R²≈0" (which came from a
   noisier single-split XGBoost estimate). Demo framing conclusion unchanged
   either way — this is a supporting-detail-level result, not a headline one.
3. Objective 2's HCHO/fire/wind data pipeline is COMPLETE — no longer
   running, not a bottleneck. Downloaded, processed, and the full analysis
   has been run successfully.

---

## 9. Exact next steps, in priority order

1. **[IN PROGRESS, nearly done] Objective 1 grid extraction's remaining
   failures were re-flushed this session (2026-08-05)** — see section 8.
   7 of 10 relaunched batches COMPLETED; the last 3 (HCHO topup) still
   draining. If any still fail, re-diagnose from the fresh error message
   rather than assuming it's the same bug again.
2. **[IN PROGRESS] Building the actual spatial AQI map.** Discovered
   `predict_grid.py` alone wasn't enough - no loader existed for the grid
   GeoTIFFs, and neither trained model's full feature set was actually
   grid-extracted (see section 3's "feature-availability gap" subsection).
   Built `data_pipeline/load_objective1_grid.py` (GeoTIFF -> point-sampled
   feature table, tested against ERA5's complete data) and
   `models/objective1_aqi/build_spatial_predictions.py` (full pipeline:
   v2 model + regional target-enc proxy + BLH climatology). Also built and
   validated the secondary location-agnostic model (R²=0.573,
   `train_location_agnostic.py`) per the user's explicit two-track decision.
   **What's left:** downloads for the other 6 sources (aod/no2/so2/co/o3/
   hcho) are still catching up via a parallelized downloader; once done,
   run `build_spatial_predictions.py` for real and sanity-check the output
   distribution before calling this done.
3. Wire the backend (`backend/store.py` already expects the right file
   locations, but hasn't been tested against final v2/Objective 2 artifacts
   end-to-end) and test all 5 endpoints for real.
4. Test the frontend in an actual browser against a running backend (built
   and compiles cleanly, but never opened in a browser against live data).
5. Only after all of the above is solid: attempt CNN-LSTM (Checkpoint 5) —
   explicitly lowest priority, skeleton written, torch installed, not
   started. Fall back to v1 XGBoost as primary if it doesn't pan out
   quickly, per the original working principle set at the very start of
   this project.

---

## 10. Known limitations / honest caveats to state to judges

1. **Objective 1's v2 (412-station) model does not beat v1 (13-city)** on
   pooled test R² (0.660 vs 0.671) — presented honestly as "near-equivalent
   accuracy with 30x more national coverage," not hidden or as a "v2 wins."
2. **Objective 2's fire-attribution is a validated null result at current
   resolution** (weekly, 0.25°/~28km grid) — the DIRECTION of the fire→HCHO
   relationship is physically consistent and real (confirmed via raw
   correlation, ~0.09-0.10 in the corridor+season subset), but not strong
   enough for reliable point-level prediction. This is being stated
   honestly rather than papered over with an inflated/cherry-picked number.
3. **Spatial AQI grid resolution is source-dependent, not uniform** — AOD/
   fire at native ~1km, TROPOMI gases at ~1.1km, ERA5 at native ~9km. This
   is a deliberate, EECU-cost-informed choice (forcing ERA5 to 1km measured
   24x more expensive for zero real precision gain), not an oversight.
4. **HCHO hotspot dates are a curated representative sample (351 dates),
   not full daily coverage** across 2018-2026 — a deliberate scope decision
   given the free-tier EECU budget, weighted toward the actual
   stubble-burning season for the highest-value coverage.
5. **CNN-LSTM (the problem statement's suggested deep-learning approach) has
   not been attempted yet** as of this writing — XGBoost is the guaranteed,
   working primary model regardless of whether CNN-LSTM is attempted later.
6. **The v2 CPCB ground-truth source** (`cpcb_cleaned_daily_FINAL.csv`) is a
   pre-cleaned file the user already had, not pulled fresh from an official
   CPCB bulk API — cite it as such if asked (v1's citation trail is in
   `DATA_SOURCES.md`; v2's needs the same treatment if not already done).

---

## 11. Extra context for the user's own understanding

This session ran continuously for a very long stretch and covered an
unusually large amount of ground for what might look, from the outside,
like "just running some scripts." A few things worth knowing about HOW this
went, not just what the end state is:

- **Almost every genuine bug in this log was caught by verifying results
  directly (reading actual error messages, checking actual prediction
  variance, computing actual EECU costs) rather than trusting a summary
  number.** The two most consequential examples: the 5.1 AOD cost incident
  was caught within about a minute by directly polling `getTaskStatus()`
  rather than assuming a batch job was behaving as designed; the 5.11
  "identical R²" coincidence was caught by checking `preds.std()` directly
  rather than accepting a plausible-looking-but-wrong number.
- **The Objective 2 attribution null result took a genuinely long
  investigation (sections 4 and 5.11) before being confirmed as real rather
  than a bug.** That time was well spent — a wrong "it's broken, I'll just
  tune harder" conclusion would have either wasted much more time chasing
  an unfixable problem, or worse, produced a falsely-confident model that
  would have misled the demo. The final honest framing (agreed with the
  user) is scientifically stronger than a forced positive number would have
  been.
- **Several bugs were the same root issue appearing twice** because a fix
  made in one script wasn't propagated to a near-duplicate script written
  separately (the qa_value bug, the date-selection bug). If more scripts get
  added that duplicate logic from `fetch_tropomi.py` or
  `fetch_objective2_grid.py`, check those files' bug-fix comments first
  rather than re-deriving the same config from scratch.
- **The EECU budget has been actively, carefully managed throughout** — not
  just spent freely. Every new large extraction was tested at small scale
  first before committing to a full run (with the one exception being the
  5.1 incident itself, which is exactly WHY that discipline was adopted for
  everything after it).

---

## 12. Objective 2 — Worked Example

Everything below is ONE real hotspot, traced through all three pipeline
stages (detect → cluster → attribute), pulled directly from this project's
own saved output files (`hcho_flagged.csv`, `hcho_clusters.csv`, the fire/
wind grids, and the trained `attribution_xgb_corridor_season.pkl` model) —
no numbers here are invented or illustrative. The point sits in Punjab,
very close to Ludhiana (within roughly 7km by straight-line distance), and
was flagged on **2024-11-07** — squarely in stubble-burning season.

### Step 1: Detection

The pipeline flags a point as a hotspot using nothing but its own HCHO
value against a threshold computed once, nationally, across the full
multi-year dataset (the 90th percentile, `threshold=0.000328775 mol/m²` —
see section 4). No fire or wind data is involved at this stage at all.

Our point, at **lat=30.8896°N, lon=75.7817°E**, recorded a tropospheric
HCHO column density of **0.000771 mol/m²** on 2024-11-07 — **2.3x the
national threshold** — so it's flagged `is_hotspot=True`. That same day,
**533 points were flagged as hotspots nationwide**; this is one of them.

### Step 2: Clustering

Every date's flagged points are run through DBSCAN separately (one
clustering pass per date, not across dates), with `eps=0.4°` (~44km) and
`min_samples=3`. Our point lands in **cluster #1 for 2024-11-07**, which
DBSCAN grouped from **140 of that day's 533 flagged points (26% of the
day's national total)**.

That cluster's actual geographic footprint is much bigger than "hotspot"
might suggest — it spans **lat 27.14°N–31.89°N, lon 73.53°E–78.28°E**,
roughly **500km along the Punjab-Haryana-Delhi-western UP corridor**. This
isn't a bug or a loose parameter: DBSCAN only requires each point to be
within 44km of *some other point already in the cluster*, not within 44km
of a center. When stubble burning drives up HCHO continuously across a
whole agricultural belt, that reads as one long chain of nearby points,
which DBSCAN correctly links into a single cluster rather than many small,
artificially separate ones. It's an honest reflection of what widespread
seasonal burning actually looks like from orbit, not a tight local plume.
(Also worth being precise about: a "cluster" here has no time dimension
beyond one date — the same physical location can be part of a completely
different cluster on a different flagged date, since clustering restarts
fresh every day.)

Within this cluster, HCHO values ranged from the flagging threshold up to
our point's 0.000771 mol/m² (the cluster's peak), against a fairly tight
regional baseline (`hcho_baseline` mean 0.000183, std only 0.000019 across
all 140 points) — meaning the elevation above baseline
(`hcho_anomaly`, mean 0.000267, up to our point's **0.000577**) is real and
consistent, not baseline noise.

### Step 3: Attribution

This is where the pipeline tries to explain *why* HCHO was elevated here,
using fire and wind data. Fire/wind grids are weekly; our HCHO date
(2024-11-07) was matched to the nearest available week, **2024-11-04**
(a 3-day gap — well within the documented max, see bug 5.5), which had
**1,931 fire-active grid cells nationally** and full wind coverage.

For our exact point, the computed attribution features (via
`cluster_fires.py`'s `build_attribution_features()`, matching the trained
model's inputs exactly) were:

| Feature | Value |
|---|---|
| `fire_dist_km` (distance to nearest fire-active cell) | **16.98 km** |
| `fire_count_nearby` (fire-active cells within 50km) | **12** |
| `wind_alignment` (cosine similarity, wind vs. fire→point direction) | **-0.44** |
| `u`, `v` (wind components) | -0.389, -0.144 (blowing toward the southwest) |
| `hcho_baseline` (2°×2° regional) | 0.000194 mol/m² |
| `hcho_anomaly` (observed) | **0.000577 mol/m²** |

There was real, substantial fire activity nearby — a fire-active cell just
17km away, and 12 within the 50km radius — consistent with active stubble
burning in the immediate area. The negative `wind_alignment` means the wind
wasn't blowing from *that specific nearest fire* toward this point (with 12
fires nearby, the nearest one isn't necessarily the one actually feeding
this location's plume — a real limitation of using "nearest fire" as the
feature rather than a proper plume model).

Feeding these exact features into the trained **corridor_season** XGBoost
model gives a **predicted `hcho_anomaly` of 0.000248 mol/m²** — less than
half the actual observed 0.000577. And the model's own SHAP-based
explanation (generated by `explain_hotspots.py`, saved verbatim in
`hotspot_explanations_corridor_season.csv` for this exact point) reads:

> *"distance to nearest fire lowered the HCHO level (impact +0); number of
> nearby fires lowered the HCHO level (impact +0); wind alignment from fire
> toward this location lowered the HCHO level (impact +0)"*

All three of the model's top features carry essentially **zero SHAP
impact** for this specific point — the model doesn't see anything about
*this point's particular* fire distance, fire count, or wind alignment as
distinguishing it from a typical corridor-season hotspot in its training
data.

**Why this one example matters:** it's the single-point, concrete face of
the aggregate finding in section 4. This is about as favorable a case as
the attribution model gets — real, nearby, plentiful fire activity, during
the actual stubble-burning season, in the actual stubble-burning corridor —
and the model still (a) can't confidently attribute the elevation to any
specific fire-related feature, and (b) underestimates the magnitude by more
than half. That's exactly what a genuinely weak-but-real signal
(~1.5-2% of variance explained, section 4) looks like up close: the
*direction* is physically sensible (fires were there, HCHO was elevated),
but there isn't enough signal in these 5 features at this weekly/25km
resolution for the model to turn "fires were nearby" into an accurate,
confident, point-specific prediction.
