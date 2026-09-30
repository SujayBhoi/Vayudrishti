# Model Comparison Build (v3) — Overpass-Matched Objective 1 + Objective 2 ConvLSTM

**Status as of this writing:** Phase 0 in progress. Clean101 ground-truth
dataset built and verified. Overpass-windowed satellite extraction running.
This document is updated at every phase, not written once at the end — see
"Running status" at the bottom for the current, authoritative state.

This build is physically isolated in `Trial -02/model_comparison_v3/` and
never modifies the existing `models/objective1_aqi/`, `models/objective2_hcho/`,
`backend/`, `frontend/`, or the root-level `PROJECT_DOCUMENTATION.md`/
`PROJECT_STATUS.md`. It reuses the existing `venv` (GPU-enabled,
`torch==2.11.0+cu128`) and existing data (`cpcb_v2_station_master.csv`,
the whole `data_pipeline/`/`features/` tree) in place.

---

## 1. Mentor requirements driving this build

1. **MODIS MAIAC** (MCD19A2) confirmed as the AOD product — already what
   this project uses (`MODIS/061/MCD19A2_GRANULES`), no change needed there.
2. **Overpass-time matching**: ground CPCB readings should be matched to the
   real satellite overpass time, not a full-day/full-week average.
3. **Multi-model comparison**: 5 architectures for Objective 1 (Linear
   Regression, ARIMA/SARIMA, Random Forest, XGBoost, CNN-LSTM), full
   evaluation suite (metrics, timing, residuals, SHAP/importance,
   per-station breakdown, cross-validation, written verdict). 2 architectures
   for Objective 2 (statistical baseline, ConvLSTM spatio-temporal), same
   evaluation depth, full national grid (all 4,447 points).
4. **2019-2024 / 2025 split**: train on 2019-2024, test on 2025 — a genuine
   held-out future year, not a random or quantile split.
5. **This project's own scope**: determine the winning Objective 1
   architecture on a clean, apples-to-apples comparison. Scaling the winner
   to the full 403-station national set is explicitly out of scope here
   (handled separately, elsewhere).

---

## 2. Data investigation story, in order

### 2.1 Starting point: two "daily" CPCB files
Two files initially provided (`Cleaned_Data_2019_2024_daily_common_stations.csv`,
`Cleaned_Data_2025_aligned_common_stations.csv.xls` — despite the `.xls`
name, plain CSV), 121-122 common stations, single official source. Inspected
directly: genuinely daily (no time-of-day component at all), which meant
overpass-time matching was initially ruled infeasible with this source (no
sub-daily granularity to align to a specific overpass time).

**Cleaning issues found and fixed** (same rules applied later to the hourly
files too):
- `Latitude`/`Longitude` columns were corrupted — every single station
  shared one identical bogus coordinate pair (16.5151, 80.5197, near
  Krishna district, Andhra Pradesh) regardless of the station's real
  location (Jaipur, Chennai, Delhi, etc.). Fixed by discarding these
  columns entirely and rejoining real coordinates from the project's
  existing, correctly-geocoded `cpcb_v2_station_master.csv` by exact
  station name (100% match confirmed).
- The files' own `AQI` column exceeded the official CPCB 0-500 scale on
  1.42%/0.93% of rows. Fixed by discarding it and recomputing AQI from raw
  pollutant concentrations using this project's own already-validated
  breakpoint formula (`data_pipeline/cpcb_process_v2.py`).
- Ground-sensor `AT`/`RH`/`WS`/`RF` columns were 14-50% NaN — dropped in
  favor of the project's existing ERA5-derived meteorology.
- No negatives, no duplicate rows, no station-name whitespace issues.

### 2.2 Discovery: genuinely hourly CPCB data already existed
After the mentor confirmed overpass-time matching was a hard requirement
(not optional), investigation resumed. Two more files were found already
present in Downloads (not assumed — inspected directly byte-for-byte):
- `Cleaned_Data_2019_2024_common_stations_combined.csv` — genuinely hourly
  (`00:00:00, 01:00:00, 02:00:00...` verified across all 24 hours), 122
  stations, 2019-2024, 5,995,567 rows.
- `Cleaned_Data_Final_2025.parquet` — genuinely hourly, 413 stations. Named
  "2025" but actually contained one mistagged fragment: a single station
  (`Sector 30, Faridabad - HSPCB`) with 7,704 rows wrongly dated 2023 — a
  real bug in the source file, not genuine 2023 coverage. Dropped; the file
  is treated as 2025-only (3,182,575 real rows).

This resolved the overpass-matching blocker: with real hourly ground truth,
CPCB readings can be averaged within a window around the actual satellite
pass time instead of over a full day.

### 2.3 Completeness analysis, at each stage
Completeness was computed against **real expected-count denominators**
(actual calendar days/hours in each window), never against a station's own
reporting span — an "own span" measure is misleading (a station active for
only 50 weeks can show 100% completeness against its own short span while
covering almost none of the actual study period).

- Daily files, 2019-2024 window (2,192 expected days): min completeness
  83.7%, essentially all 121/122 common stations already well past 80%.
- Hourly CSV, 2019-2024 window (52,608 expected hours): mean 93.4%.
- Hourly parquet, 2025 window (8,760 expected hours): mean 87.1%, wider
  spread, min 17.3% — a real outlier.
- **Final rule applied**: >=80% hourly completeness in BOTH the 2019-2024
  window AND the 2025 window, computed on the 121-station starting scope
  (common to both hourly files and the station master list), with the
  17.3%-complete outlier excluded explicitly regardless of the general
  threshold. Result: **101 of 121 stations kept, 20 dropped** (see section 3
  for the full list).

### 2.4 Overpass-time research
Real satellite pass times were checked directly via GEE (`system:time_start`
on actual images at Delhi and Chennai points), not assumed:
- Sentinel-5P TROPOMI: consistent ~13:08 IST daily pass.
- MODIS (`MCD19A2_GRANULES`): candidates clustering at 13:10-13:45 IST.
  **Correction, verified later via each granule's `SATELLITE` property (`T`=
  Terra, `A`=Aqua):** this is genuinely a Terra+Aqua mix, not Aqua-only, as
  first assumed. A sample day at a Delhi point showed roughly even Terra/
  Aqua split overall (51 T / 49 A candidates touching the point across the
  whole day), and **within the actual 11:18-15:18 IST window specifically,
  both platforms appear together** (12 Aqua, 7 Terra on that sample - e.g.
  Terra granules landing at 13:45 IST, alongside Aqua ones at the same
  time). This makes physical sense: Terra/Aqua's *idealized* equator-
  crossing times (~10:30/~13:30 local solar time) describe the crossing on
  the satellite's own ground track, not the actual observation time at an
  arbitrary point off that track, which shifts with scan geometry. The
  window filter is satellite-agnostic by design (checks clock time only,
  not platform) - this is treated as correct behavior (more real data,
  still genuinely time-constrained), not a bug, but the earlier "MODIS/
  Aqua" framing in this doc was imprecise and is corrected here.
- A raw `filterBounds` query at one station returns far more "touching"
  images per day than real usable overpasses (100/day for MODIS, 14/day for
  TROPOMI, at one point) — wide swaths from adjacent orbits clip India from
  multiple angles. This means picking "closest timestamp" alone isn't
  enough; a time-window filter plus per-candidate pixel validity (handled
  automatically by GEE's `.mean()` reducer skipping masked pixels) is
  needed.
- **Combined anchor**: mean of the two real anchors (~13:08, ~13:28) =
  ~13:18 IST. **Window: 11:18-15:18 IST** (anchor +/- 2 hours), equivalently
  05:48-09:48 UTC (IST has no DST, fixed +5:30 offset).
- Ground truth AQI for a station-day = mean of all hourly (breakpoint-
  recomputed) AQI readings whose timestamp falls in that window.
- **Cadence consequence**: true overpass-matching is inherently a *daily*
  concept (one overpass, one windowed average, per day) — this build
  therefore runs the 5-model comparison at **daily** cadence, not the
  weekly cadence used elsewhere in this project (the existing
  `model_dataset_v2.csv` stays weekly, untouched). ARIMA/SARIMA's seasonal
  period becomes m=7 (not m=52); CNN-LSTM's lookback window is redefined in
  days.

### 2.5 ERA5 granularity check
Verified directly by reading an actual extracted file
(`data_pipeline/data_pipeline/raw_v2/era5/*.csv`): the existing ERA5
extraction is **weekly** (rows exactly 7 days apart), even though the
underlying GEE collection (`ECMWF/ERA5_LAND/DAILY_AGGR`) is itself daily.
This build re-extracts ERA5 fresh at daily granularity (`period_days=1`)
rather than reusing the existing weekly files — cheap and low-risk, since
ERA5-Land is a simple gridded product (no compositing-cost risk like the
AOD/TROPOMI granule sources).

### 2.6 Satellite extraction: how it's actually processed, and real timing

`fetch_overpass_windowed_satellite.py` extends the existing
`interactive_point_extract.py` machinery: same proven-safe per-station
`filterBounds` pattern, same collections/bands/QA masks as
`fetch_modis_aod.py`/`fetch_tropomi.py` (AOD via `MCD19A2_GRANULES`; NO2,
SO2, O3, HCHO via `cloud_fraction<0.3`; CO with no QA mask, matching the
already-documented reason: CO's SWIR retrieval is far less cloud-sensitive).
The new part is `period_days=1` (daily, not weekly) plus a server-side
overpass-window filter applied *inside* each day's period, before the
mean/reduce step: every candidate image in that day is tagged with its
UTC-minute-of-day (from `system:time_start`), filtered to the
05:48-09:48 UTC band (== 11:18-15:18 IST), then averaged — with an
empty-window guard (a day with zero in-window images gets a masked
fallback, the same pattern already used elsewhere in this project for
empty-collection edge cases, including the Byte/Float64 type-consistency
fix for the fallback image).

**A real scale limit was hit and fixed during testing, not assumed away:**
a single interactive call spanning the full ~7-year range (2019-2026, ~2,557
daily periods) failed with `"User memory limit exceeded"` after ~142s.
Tested progressively smaller ranges; **one year (365 daily periods) is
stable** (confirmed: 1 station, AOD, full year, 51s, real varying output).
The extractor now chunks each station-source extraction into <=365-day
year-chunks internally, running each chunk as its own interactive call and
concatenating the results before caching one combined CSV per station-source
(matching the existing project's per-station-file resumability pattern - a
crash mid-run only costs re-doing that one station-source's chunks, nothing
else).

**Correctness spot-check** (1 station, AOD, one month): real, physically
plausible varying AOD values (0.25-1.64, consistent with Jaipur winter
pollution), 17-22 in-window candidate images per day out of ~100 raw
candidates touching the point per full day - proportionally consistent with
the ~2-hour window covering roughly 1/6 of the day (~100/6 ~ 17), not a
degenerate or constant output.

**Real measured timing, used to project the full run:**
- 1 station, 1 source, 1 year: ~51s (single-threaded).
- 2 stations x 2 sources x 7 years (28 year-calls), 4 workers: 430s total
  (~15.4s per call-slot with concurrency).
- Full run: 101 stations x 6 sources x 7 year-chunks = 4,242 interactive
  calls, at 8 workers. Projected on the order of several hours of
  wall-clock time (consistent with the earlier ~10+ hour ballpark given
  before this was scoped, now grounded in real per-call measurements rather
  than an estimate).
- **Currently running in the background** (started this session, full
  101-station x 6-source x 2019-2026 range, 8 workers, logging to
  `model_comparison_v3/data/overpass_extraction.log`). Already-completed
  station-source files from the validation tests (2 stations x 2 sources)
  are skipped/reused via the existing disk-cache-per-file pattern, not
  redone.

---

## 3. Stations excluded from the final 101-station set

20 of the 121 starting-scope stations were dropped for completeness, plus
1 station never entered the starting scope at all (no 2025 data of any
kind). All numbers below are the exact figures the build script computed
(`model_comparison_v3/data/clean101_report.json`), not estimates.

### 3.1 Systematic pattern: Haryana (HSPCB) 2025 collapse — 12 stations

These stations were fine in 2019-2024 (86-98% complete) but collapsed to
17-31% complete in 2025 specifically. This reads as a systematic 2025
reporting/connectivity issue at HSPCB (Haryana Pollution Control Board), not
scattered random noise — worth stating as an observed data-quality pattern
in the demo, not something to chase as a bug.

| Station | Location | 2019-2024 | 2025 | Reason |
|---|---|---|---|---|
| Gobind Pura, Yamuna Nagar - HSPCB | Yamuna Nagar, Haryana | 93.4% | 17.3% | Explicitly excluded outlier |
| Murthal, Sonipat - HSPCB | Sonipat, Haryana | 88.5% | 19.3% | 2025 below 80% threshold |
| Sector-18, Panipat - HSPCB | Panipat, Haryana | 90.3% | 22.4% | 2025 below 80% threshold |
| Municipal Corporation Office, Dharuhera - HSPCB | Dharuhera, Haryana | 92.6% | 23.0% | 2025 below 80% threshold |
| Nathu Colony, Ballabgarh - HSPCB | Ballabgarh, Haryana | 96.0% | 23.8% | 2025 below 80% threshold |
| Shyam Nagar, Palwal - HSPCB | Palwal, Haryana | 94.8% | 24.0% | 2025 below 80% threshold |
| Patti Mehar, Ambala - HSPCB | Ambala, Haryana | 95.8% | 24.5% | 2025 below 80% threshold |
| Shastri Nagar, Narnaul - HSPCB | Narnaul, Haryana | 93.6% | 24.5% | 2025 below 80% threshold |
| Sector-6, Panchkula - HSPCB | Panchkula, Haryana | 95.3% | 29.7% | 2025 below 80% threshold |
| MD University, Rohtak - HSPCB | Rohtak, Haryana | 98.2% | 29.9% | 2025 below 80% threshold |
| Arya Nagar, Bahadurgarh - HSPCB | Bahadurgarh, Haryana | 95.9% | 31.2% | 2025 below 80% threshold |
| F-Block, Sirsa - HSPCB | Sirsa, Haryana | 95.3% | 34.2% | 2025 below 80% threshold |

All 12 are HSPCB, all show the same 2019-2024-fine / 2025-collapsed shape.

### 3.2 Other 2025-side drops (not HSPCB) — 6 stations

| Station | Location | 2019-2024 | 2025 | Reason |
|---|---|---|---|---|
| Tata Stadium, Jorapokhar - JSPCB | Jorapokhar, Jharkhand | 46.1% | 33.7% | Below threshold in both windows |
| Worli, Mumbai - MPCB | Mumbai, Maharashtra | 86.9% | 34.4% | 2025 below 80% threshold |
| Vasai West, Mumbai - MPCB | Mumbai, Maharashtra | 81.2% | 67.0% | 2025 below 80% threshold |
| Ratanpura, Rupnagar - Ambuja Cements | Rupnagar, Punjab | 95.2% | 69.1% | 2025 below 80% threshold |
| Alandur Bus Depot, Chennai - CPCB | Chennai, Tamil Nadu | 94.9% | 75.0% | 2025 below 80% threshold |
| Sion, Mumbai - MPCB | Mumbai, Maharashtra | 89.8% | 79.7% | 2025 below 80% threshold |

### 3.3 2019-2024-side drops (2025 was actually fine) — 2 stations

| Station | Location | 2019-2024 | 2025 | Reason |
|---|---|---|---|---|
| SIDCO Kurichi, Coimbatore - TNPCB | Coimbatore, Tamil Nadu | 56.8% | 95.2% | 2019-2024 below 80% threshold |
| Jai Bhim Nagar, Meerut - UPPCB | Meerut, Uttar Pradesh | 75.2% | 96.8% | 2019-2024 below 80% threshold |

### 3.4 Never in scope at all — 1 station

| Station | Location | Reason |
|---|---|---|
| Sector- 16A, Faridabad - HSPCB | Faridabad, Haryana | Present in the 2019-2024 file, completely absent from the 2025 file — no 2025 data of any kind, so it never enters the common starting scope in the first place. |

**Total excluded: 20 (from the 121-station starting scope) + 1 (never in
scope) = 21. Final clean101 set: 101 stations.**

---

## 4. Decisions made and why

| Decision | Why |
|---|---|
| Recompute AQI from raw pollutants, don't trust file AQI columns | Both hourly and daily source files' own AQI columns exceed the official 0-500 CPCB scale on ~1-2.5% of rows; recomputing via the project's existing breakpoint formula also makes the target directly comparable to v1/v2 |
| Rejoin lat/lon from `cpcb_v2_station_master.csv`, don't trust file coordinates | Every source file's own lat/lon was corrupted (single bogus fixed value for every station) |
| Drop ground-sensor AT/RH/WS/RF, use ERA5 instead | 14-50% NaN in the ground-sensor columns; ERA5 is already complete and validated elsewhere in this project |
| >=80% hourly completeness threshold, both windows | Closely matches an independent ~121-station estimate; strict enough to be genuinely clean, loose enough to leave a usable station count for per-station modeling |
| 11:18-15:18 IST window (not exactly a single fixed "1:30pm") | Real confirmed overpass times (TROPOMI ~13:08, MODIS ~13:10-13:45, genuinely a Terra+Aqua mix per each granule's `SATELLITE` property - see section 2.4) cluster near but not exactly at 13:30 - a +/-2h window around the combined anchor covers both real anchors without arbitrarily picking one satellite over the other |
| Daily cadence for this comparison (not weekly, unlike the rest of the project) | Overpass-matching is inherently a daily concept; keeping this track weekly would re-introduce the multi-day averaging overpass-matching exists to avoid |
| Fresh ERA5 extraction at period_days=1 | Existing ERA5 extraction is weekly-composited (verified directly), even though the source collection is natively daily - cheap to re-extract, no compositing-cost risk |
| Year-chunked satellite extraction (not one call per station spanning 2019-2025) | A single interactive GEE call spanning ~7 years of daily periods hits "User memory limit exceeded" (confirmed by direct test); one year (365 daily periods) is stable, tested at ~51s/call |

---

## 4.1 Bug/incident log

Following this project's own established convention (every real bug logged
with root cause and fix, not just silently patched) for this build's own
issues:

### 4.1.1 GEE noncommercial-tier "Restricted Mode" during the satellite extraction

**What happened:** the full overpass-windowed satellite extraction (101
stations x 6 sources, 8 concurrent workers) ran cleanly for ~29 minutes
(58/606 files done, matching the earlier-measured ~1.22 files/min rate),
then began failing. Log showed 104 `"Too Many Requests: Exceeded Earth
Engine concurrency limit... Your project is in Restricted Mode"` warnings,
and **13 stations permanently failed** (exhausted all 3 retries) within the
AOD source alone. A direct, standalone `ee.Initialize()` check (nothing to
do with the extraction script) also failed with the same 429/Restricted
Mode error - confirming this is a project-wide API block, not a bug in the
extraction logic.

**Root cause:** the GEE project (`isro-bah-2026`) exceeded its noncommercial
tier's compute quota under sustained 8-way concurrent interactive load. This
is a different quota mechanism than the previously-tracked 150 EECU-hr/month
BATCH budget (see the existing root `PROJECT_DOCUMENTATION.md` section 7) -
an attempt to check that batch budget via the documented method
(`ee.data.listOperations()`) was itself rejected by the same restriction, so
the exact relationship between the two budgets isn't fully confirmed, but
the practical effect (API calls rejected outright) is the same either way.
8 concurrent workers was fine for the ERA5 extraction (101/101 stations, 5
minutes, zero errors) but too aggressive for the heavier AOD/TROPOMI
granule-level sources sustained over many minutes.

**Fix:** stopped the background extraction immediately (per this project's
own 5.13 lesson: a background job shows as two OS PIDs but is one logical
process - killing one confirmed both terminate, not a race). Will resume
at **2 concurrent workers** (down from 8) plus a **proactive fixed delay
before every interactive call** (not just reactive backoff after a
failure) - the same fix pattern already proven for this project's Open-Meteo
BLH rate-limit incident (existing root doc, section 5.9). Already-completed
station-source files are safe and will be skipped on resume (existing
per-file disk-cache pattern); the 13 permanently-failed stations have no
output file, so a resume run will retry them naturally - no special-cased
retry logic needed.

**Correction (found via GEE's own documentation, not assumed):** the
original plan was to wait out the restriction via a periodic background
watcher. Checking the actual docs
(https://developers.google.com/earth-engine/guides/noncommercial_tiers)
showed this was wrong - restricted mode "persists until your monthly quota
resets," and quotas "reset on the first of each calendar month at midnight
Pacific Time." From 2026-08-06, that means `isro-bah-2026` would stay
restricted until **2026-09-01** - not hours, as first estimated. The watcher
was stopped (it was polling a project that, per the docs, could not recover
within its check window).

**Revised fix**: the same docs confirm compute quota is "allocated to a
Cloud project and can't be transferred," and that an account can "have
multiple noncommercial projects" with independently-tracked quota - so a
**new GCP project, freshly registered for Earth Engine noncommercial use**,
gets its own independent quota, unaffected by `isro-bah-2026`'s restriction.
This is a config change, not a rewrite: `gee_auth.py`'s `init_ee()` already
accepted a `project` parameter; `fetch_overpass_windowed_satellite.py` and
`wait_and_resume_extraction.py` now expose it as a `--project` CLI flag.
Already-completed station-source files are untouched by a project switch -
the disk cache is keyed purely by filename, independent of which project
produced it.

**Status: waiting on a new project ID from the user** (they're creating and
registering one). Extraction resumes the moment it's provided, via
`fetch_overpass_windowed_satellite.py --project <new-id> --workers 2
--call-delay 3.0`.

---

## 5. Objective 2 — 5-model comparison: statistical baseline, persistence, Linear Regression, Random Forest, ConvLSTM (DONE)

Full national scope throughout (all 4,447 grid points, same as the existing
HCHO detection pipeline) - confirmed explicitly before building, see section
4. No new GEE extraction needed - HCHO/fire/wind grids were already fully
downloaded from the existing Objective 2 work.

**Grid build** (`build_spatiotemporal_grid.py`): the 4,447-point 0.25deg
lattice reconstructed as a dense masked 111 (lat) x 116 (lon) array. The
~1.15M raw HCHO pixel rows spatially joined onto this lattice via
`scipy.spatial.cKDTree` nearest-neighbor (the source file's own `grid_id` is
a meaningless per-row counter, confirmed earlier in this project). Binned
into regular monthly buckets (the 351 irregular HCHO dates -> monthly
averages), fire (sum) and wind (mean) aggregated to the same months from
their already-regular weekly grids (reusing the existing
`load_objective2_grids.py` loaders, not reimplemented). Result: **96
distinct year-months** (2018-01 to 2025-12), 2 flagged as having zero HCHO
coverage on any India cell (2018-05, 2021-07) - surfaced explicitly, not
silently zero-filled. Chronological split: 79 train months (2018-01 to
2024-07), 17 test months (2024-08 to 2025-12).

Five models compared, all on the exact same 6-month lookback window and
79/17 train/test month split (the tabular models flatten that window
per-cell instead of using ConvLSTM's spatial convolution/recurrence - see
`tabular_features.py`):

**Statistical baseline** (`train_statistical_baseline.py`): per-grid-cell
historical climatology, training-months-only. Test R2=0.039 - low but
genuinely positive, exactly what a "no learning" reference should show (it
captures real geographic baseline variation, zero temporal variance).

**Persistence** (`train_persistence.py`): predict next month = that
cell's own last-observed month, the classic naive time-series baseline
(distinct from climatology - this uses the most RECENT value, not the
historical mean). No training. Test R2=**-0.592**, meaningfully worse than
climatology - the honest read is that this monthly-binned HCHO tensor
doesn't have strong enough month-to-month continuity per cell (irregular
underlying satellite-pixel coverage, see section 2.4/2.5 elsewhere in this
doc) for "last month" to be a good predictor of "this month" - climatology
(a stable per-cell average) is the better naive reference for this data,
not recency.

**Linear Regression** (`train_linear_regression.py`, Objective 2 version):
per-cell, 24 features (6 months x 4 channels), no spatial context at all.
Test R2=0.013 - positive but the weakest of the "real" models, confirming
a plain linear fit on raw per-cell history isn't enough on its own.

**Random Forest** (`train_random_forest.py`, Objective 2 version): same 24
per-cell features as Linear Regression, 200 trees, depth 14. Test
R2=**0.155** - training took 122s (CPU, sklearn) vs ConvLSTM's 14s (GPU).
**This is the strongest single model on this exact test set**, ahead of
ConvLSTM - see honest discussion in the verdict below.

**ConvLSTM** (`train_convlstm.py`): custom ConvLSTMCell (Shi et al. 2015
formulation - PyTorch has no built-in layer), 1 layer, 16 hidden channels,
6-month lookback, GPU-trained (RTX 5060, **14 seconds** for 60 epochs -
confirms the earlier GPU time estimate was conservative for this grid size).
Test R2=0.116 - meaningfully beats the statistical baseline and both
simpler tabular models, confirming the spatio-temporal learning
(neighboring cells + fire/wind channels + time history) adds real value
over a pure historical-average or plain-linear reference - just not quite
as much value as Random Forest's non-linear per-cell fit, on this
particular test set (see verdict).

**Full comparison** (`obj2_metrics_comparison.png`):

| Model | Test R2 | Test RMSE (x10^-5 mol/m2) | Train time |
|---|---|---|---|
| Persistence (naive) | -0.592 | 12.44 | instant |
| Linear Regression | 0.013 | 9.86 | 0.17s |
| Statistical baseline | 0.039 | 9.73 | 0.007s |
| ConvLSTM | 0.116 | 9.33 | 14s (GPU) |
| **Random Forest** | **0.155** | **9.12** | 122s (CPU) |

**Regional breakdown** (`run_evaluation.py`, corridor = Punjab-Haryana-
Delhi-western-UP, lat 27-32N/lon 73-81E, same definition as the existing
attribution pipeline):

| Region | Cells | Baseline R2 | ConvLSTM R2 |
|---|---|---|---|
| Corridor | 543 | 0.015 | 0.093 |
| Rest of India | 3,904 | 0.038 | 0.115 |
| National (full) | 4,447 | 0.039 | 0.116 |

ConvLSTM's advantage over the statistical baseline holds in every region,
slightly larger outside the corridor - consistent with the corridor having
sparser/noisier fire-season dynamics that are harder for any model to
learn from monthly-binned data. **Scope note**: this regional breakdown and
the CV below were built before the persistence/Linear Regression/Random
Forest models existed and still only compare baseline vs ConvLSTM -
extending them to all 5 models (particularly checking whether Random
Forest's edge over ConvLSTM holds regionally and across CV folds too) is
flagged as a natural next step, not yet done.

**Rolling-origin cross-validation** (3 folds, retraining both models fresh
on progressively earlier 12-month test windows - confirms the result isn't
a lucky single split):

| Model | Fold 0 (months 84-96) | Fold 1 (72-84) | Fold 2 (60-72) | Mean +/- std |
|---|---|---|---|---|
| Statistical baseline | R2=0.032 | R2=-0.078 | R2=-0.044 | -0.030 +/- 0.046 |
| ConvLSTM | R2=0.113 | R2=0.077 | R2=0.066 | **0.085 +/- 0.020** |

**ConvLSTM beats the statistical baseline in every single fold** - the
baseline even goes negative twice (worse than predicting the training
mean), while ConvLSTM stays positive and comparatively stable (std 0.020 vs
0.046). This is a robust, cross-validated result, not a coincidence of the
main train/test split.

**Channel-ablation** (substitute for feature importance - SHAP's tree
explainer doesn't apply to a ConvLSTM; zero out one input channel at
inference, measure the R2 drop): saved to
`models/objective2_hcho/artifacts/convlstm/channel_ablation.csv`. This also
adds empirical texture to the project's existing fire-attribution
null-result finding from the original `models/objective2_hcho/` work.

**Objective 2 verdict, updated honestly for all 5 models**: two real,
useful findings, not one:

1. **Both "real" ML models (Random Forest, ConvLSTM) clearly beat the
   naive references** (persistence, statistical baseline, plain Linear
   Regression) - confirming that learning from recent HCHO/fire/wind
   history genuinely adds value over guessing a historical average or a
   last-known value.
2. **Random Forest (R2=0.155) edges out ConvLSTM (R2=0.116) on this exact
   test set** - reported as-is, not smoothed over. This is a real,
   plausible result, not a bug: with only 73 independent training months,
   ConvLSTM has comparatively little data to learn a spatial+temporal
   architecture from, while Random Forest's per-cell, non-linear fit on
   24 lag features needs far less data to do well - a well-known pattern
   in ML generally (tree ensembles often beat deep learning on small/
   tabular-shaped problems). ConvLSTM is trained in 14s on GPU vs Random
   Forest's 122s on CPU, so there's a real practicality tradeoff too, just
   in the opposite direction of what's often assumed (the deep model was
   actually cheaper to train here).

Both remain low-R2 in absolute terms (HCHO at monthly/25km resolution is
a genuinely hard prediction target, consistent with the existing project's
already-documented weak-signal findings for HCHO attribution) - the
honest framing is "both Random Forest and ConvLSTM demonstrably learn real
structure a naive baseline can't; on this specific test set, the simpler
tree ensemble currently edges out the deep model," not a claim that either
"solves" HCHO forecasting, and not a predetermined preference for the
more sophisticated architecture. Given ConvLSTM's genuine spatial
awareness (which the tabular models entirely lack) and comparable
performance, it remains the more architecturally interesting model to
present for the demo narrative - but Random Forest's current numeric edge
is real and should be stated plainly if asked, not hidden.

---

## 6. Running status

**Done:**
- [x] Environment: statsmodels 0.14.6, pmdarima 2.1.1 installed; `requirements.txt` created.
- [x] `build_clean101_hourly_dataset.py` — clean, windowed, 101-station daily ground-truth table. Output: `data/clean101_ground_truth_daily.csv` (247,809 rows), `data/clean101_report.json`.
- [x] `fetch_daily_era5.py` — full daily ERA5 extraction, all 101 stations, completed in 296s.
- [x] `build_spatiotemporal_grid.py` — full national spatio-temporal tensor (96 months x 4 channels x 111 x 116), no new GEE extraction needed.
- [x] Objective 2 complete end-to-end: **5 models** trained and compared (persistence, statistical baseline, Linear Regression, Random Forest, ConvLSTM) - see section 5 for full results. Random Forest currently edges out ConvLSTM on raw test R2 (0.155 vs 0.116), reported honestly. Regional breakdown + 3-fold CV still only cover baseline vs ConvLSTM (built before the 3 new models existed) - extending them is flagged as a next step, not yet done. Hotspot-prediction analysis (section 7.6) also only covers ConvLSTM so far.

**In progress:**
- [ ] `fetch_overpass_windowed_satellite.py` — overpass-windowed daily extraction for AOD + 5 TROPOMI gases, 101 stations x 6 sources x 7 year-chunks. Validated at small scale (see section 2.6), running against `isro-bah-2026-v2` (see section 4.1.1) at the hardened settings (2 workers, 3s call delay) — this is the long pole for Objective 1.

**Not yet started (blocked on the satellite extraction above):**
- [ ] `build_clean101_daily_dataset.py` — written and ready, joins ground truth + satellite + ERA5, daily-appropriate lag/rolling features. Not yet run (waiting on satellite data).
- [ ] All 5 Objective 1 models (Linear Regression, Random Forest, XGBoost, ARIMA/SARIMA, CNN-LSTM) — all 5 training scripts written and ready (`eval_utils.py` shared module done), not yet run.
- [ ] Objective 1 evaluation suite (metrics table, timing, residuals, SHAP/importance, per-station breakdown, CV, verdict).
- [ ] Presentation materials (PPTX, one-page PDF, standalone charts) for both objectives.

---

## 7. Quick briefing & FAQ (for a zero-context reader)

A condensed, plain-language companion to the detailed narrative above —
written so anyone (a mentor, a teammate walking in cold) can get oriented
fast and have real answers ready. Sections 1-6 above are the full technical
record; this section is the fast path through the same material. Snapshot
as of 2026-08-06, evening — extraction progress numbers below will be
stale by the time this is read; see section 6 for the authoritative
running status.

### 7.1 Running the Objective 2 charts

One script regenerates all five PNGs from whatever's currently in
`artifacts/` — nothing needs retraining to redraw them, no GPU or internet
needed (it only reads local result files):

```
cd "Trial -02"
.\venv\Scripts\python.exe model_comparison_v3\data_pipeline\build_objective2_charts.py
```

| File | Shows |
|---|---|
| `obj2_metrics_comparison.png` | Headline scorecard — test R² and RMSE, baseline vs ConvLSTM |
| `obj2_regional_r2.png` | Same comparison split by corridor / rest-of-India / national |
| `obj2_cv_stability.png` | 3-fold CV — proves the win isn't a lucky split |
| `obj2_channel_ablation.png` | What the model actually leans on (stand-in for feature importance) |
| `obj2_hcho_spatial_map.png` | One month's actual national HCHO map on the real 111x116 grid |

Both spatial maps (`obj2_hcho_spatial_map.png` and
`obj2_hotspot_predicted_vs_actual_map.png`) now overlay real state/UT
boundaries from the user-provided shapefile
(`C:\Users\mohit\Downloads\India_Shapefiles\geoBoundaries-IND-ADM0_simplified.geojson`
— despite the "ADM0" filename, it contains all 36 state/UT ADM1 polygons,
not just the country outline). Loaded directly via `json` + matplotlib in
`data_pipeline/india_boundary.py` (no geopandas dependency needed/available
in this venv). This also revealed the Andaman & Nicobar and Lakshadweep
islands, previously indistinguishable in the plain pixel-grid rendering.

### 7.2 What Objective 2 does, in one sentence

Predicts next month's national HCHO map from the last six months of data,
and checks whether that's genuinely better than just guessing "the
historical average for this spot." **Five models**, same data, same
6-month lookback, same 79/17 train/test split: **persistence** (naive,
last-observed-value), **statistical baseline** (per-cell historical mean),
**Linear Regression** and **Random Forest** (per-cell, 24 lag features, no
spatial context), and **ConvLSTM** (learns spatial+temporal patterns from
HCHO + fire + wind together, with real neighbor/spatial awareness the
other four lack).

**Result, reported honestly**: the two "real" ML models clearly beat the
naive references, but the ranking has a genuine surprise - **Random Forest
(test R2=0.155) currently edges out ConvLSTM (test R2=0.116)** on this
exact test set. Both comfortably beat the statistical baseline (0.039),
Linear Regression (0.013), and persistence (-0.592, worse than useless).
Likely reason: only 73 training months is little data for a deep spatial
model to learn from, while a tree ensemble's per-cell non-linear fit needs
less data to do well - a well-known general ML pattern, not a bug here.
Full table and discussion in section 5.

**Parameters worth knowing cold:**

| Parameter | Value |
|---|---|
| Grid coverage | Full national — all 4,447 points |
| Grid shape | 111 x 116 (lat x lon), 0.25 deg (~28km) spacing |
| Input channels | 4 — HCHO, fire count, wind-u, wind-v |
| Time span | 96 months, Jan 2018 - Dec 2025 |
| Train / test split | 79 / 17 months, chronological |
| Empty months (flagged, not faked) | 2 (2018-05, 2021-07) |
| Lookback window (all models) | 6 months |
| ConvLSTM hidden size | 16 channels |
| Random Forest | 200 trees, max depth 14 |
| Training time | ConvLSTM 13.9s (GPU) / Random Forest 122s (CPU) / Linear Regression 0.17s / baseline 0.007s / persistence instant |

**Why 0.25 deg (~28km) grid spacing specifically?** Verified directly from
`data_pipeline/build_india_grid.py`'s own CLI default, not assumed - its
comment reads: `"grid spacing in degrees; ~0.25deg ~= 28km, matches ERA5
native"`. The underlying logic: there's no point building a national grid
finer than the coarsest real input feeding it, since upsampling doesn't
create genuine information. This is consistent with a documented decision
elsewhere in the project (root `PROJECT_DOCUMENTATION.md`, limitation 3):
forcing the AQI spatial grid's ERA5 features to 1km was measured at **24x
more EECU cost for zero real precision gain**, since the source data isn't
actually that precise regardless of export resolution. Practically, 0.25deg
is the sweet spot: fine enough to show real regional structure (e.g.
separating the Punjab-Haryana corridor from the rest of India), coarse
enough to keep the whole grid (4,447 points) computationally tractable for
DBSCAN clustering and, now, ConvLSTM training. (One honest nuance: the
wind source actually used, `ECMWF/ERA5_LAND`, is natively ~9km, finer than
0.25deg - so the comment reflects the project's stated intent/consistent
philosophy more than a precise description of every single input today.)

### 7.2a What each Objective 2 chart is for

The four metric charts (`obj2_metrics_comparison`, `obj2_regional_r2`,
`obj2_cv_stability`, `obj2_channel_ablation`) each answer one specific
skeptical question a judge would ask:

| Chart | Question it answers |
|---|---|
| Metrics comparison | Did the model actually beat the simple baseline? |
| Regional R2 | Does the win hold up geographically, or only in one region? |
| CV stability | Is this a lucky train/test split? |
| Channel ablation | What is the model actually learning from? |

**The 5th chart, `obj2_hcho_spatial_map.png`, is different in kind.** The
first four are performance charts (bars, comparisons); this one is the
actual output artifact - a real map of India colored by HCHO concentration
for one month (currently December 2025, the most recent month in the
data). It is drawing the **real, observed** HCHO tensor built in
`build_spatiotemporal_grid.py`, at the true 111x116 national grid shape -
not a model prediction. Its job isn't to prove the model is good, it's to
prove the whole national-grid pipeline is real: that India's outline, the
resolution, and the coverage all come out looking like an actual country,
not an abstract dataset. See section 7.6 below for a predicted (not just
observed) version of this map, built to directly answer where the model
thinks hotspots are forming, region by region.

### 7.3 The EECU incident, condensed

EECU-hour is Google Earth Engine's free-tier compute budget (150/month).
Mid-extraction (8 parallel requests), the project hit that quota and
Google put it into **Restricted Mode** — a project-wide throttle. First
assumption (wrong): a short rate limit, worth waiting out with a periodic
check. Correction, verified via Google's own docs: Restricted Mode lasts
**until the calendar month resets**, not hours — the original project
would've stayed blocked until Sept 1. Real fix: quota is tied to the
specific Cloud project, not the account, so a **brand-new project
(`isro-bah-2026-v2`) gets its own fresh 150-hour budget**. Needed two
one-time permission grants (Service Usage Consumer, then Earth Engine
Resource Writer) before it would accept requests. Validated on 10 stations
first (zero errors), then resumed the full run at a deliberately slower,
safer pace: 2 parallel requests instead of 8, plus a 3-second pause before
every request. Full incident writeup with root cause: section 4.1.1.

**If asked:** no data was lost or re-fetched (every download is cached to
disk and skipped if it already exists) and no money was involved (free
tier throughout).

### 7.4 FAQ

**Why compare against naive baselines (persistence, statistical average) at all?**
Because a model that doesn't beat "just guess the historical average" or
"predict last month's value" isn't adding value. Both Random Forest and
ConvLSTM clear that bar convincingly; persistence and plain Linear
Regression barely do or don't.

**Which model actually won — ConvLSTM or Random Forest?**
Random Forest, on raw test R² (0.155 vs ConvLSTM's 0.116) — reported as-is,
not smoothed over. Likely because 73 training months is little data for a
spatial deep model, while a tree ensemble needs less data to fit
non-linear per-cell patterns well. ConvLSTM is the only model with genuine
spatial (neighbor-cell) awareness and trains far faster (14s GPU vs 122s
CPU), so it's still the more architecturally interesting model to present
— but Random Forest's numeric edge is real and should be stated plainly if
asked.

**Are these R² values (0.04-0.16 range) actually good?**
Low in absolute terms across every model — HCHO at monthly/28km resolution
is a genuinely hard target, consistent with this project's earlier finding
that fire-to-HCHO attribution is a weak signal at this resolution. Honest
framing: the two real ML models demonstrably learn structure the naive
baselines can't; none of them claim to "solve" HCHO forecasting.

**How do you know the win isn't a lucky split?**
3-fold rolling-origin CV, retrained fresh each fold. ConvLSTM beat the
baseline in every single fold; the baseline went negative (worse than a
flat guess) in 2 of 3.

**Is this the whole country, or a few cities?**
Objective 2 is the full national grid (4,447 points). Objective 1's clean
comparison uses 101 specific high-quality CPCB stations, deliberately
narrowed from a larger set for data-quality reasons.

**Why only 101 stations?**
Started from 121 with official single-source hourly ground truth, kept
those with >=80% completeness in both the 2019-2024 training window and
the 2025 test window. 12 of the 20 dropped were Haryana stations fine
through 2024 but collapsed in 2025 specifically — a real, flagged pattern
(section 3.1), not cherry-picking.

**Why "overpass-time matching" instead of daily averages?**
A mentor requirement. Ground pollution swings a lot across a day in ways a
satellite passing once at midday never sees; averaging in the wrong hours
smears the comparison. Ground data is matched to the real satellite pass
window (11:18am-3:18pm IST, confirmed from real satellite timestamps, not
assumed — section 2.4).

**Did the quota issue cost real money or lose data?**
No and no — free tier throughout, and the disk-cache-per-file pattern
means switching projects mid-extraction lost nothing already downloaded.

**If the chart script is re-run, will numbers change?**
No — it only reads already-saved results and redraws them. Re-running
training itself could shift ConvLSTM's numbers slightly (GPU training has
some randomness); the chart step alone is a pure re-draw.

### 7.6 Predicted hotspot map — does ConvLSTM know WHERE, region by region?

Built in response to a direct question: can the model show where HCHO
hotspots will form, the way the existing (pre-v3) detection pipeline
already showed the Punjab-Haryana corridor plus national spread (Delhi,
Assam, Odisha, Maharashtra, Kerala)? Scripts:
`build_hotspot_prediction_analysis.py`, `build_hotspot_prediction_charts.py`.

**First attempt failed, and that failure is itself the honest finding.**
Applying the existing, already-validated national hotspot threshold
(0.0003287748113506 mol/m2, from `models/objective2_hcho/artifacts/
summary.pkl`) directly to ConvLSTM's predictions flags **zero cells,
everywhere, in all 17 test months** — verified directly, not assumed.
Root cause, also verified directly: ConvLSTM's predicted values range
6.6e-5 to 2.87e-4 and never reach the 3.29e-4 threshold; predicted std
(2.6e-5) is ~4x smaller than actual std (9.9e-5). This is the well-known
"regression to the mean" behavior of a plain-MSE-trained model — minimizing
average squared error structurally discourages ever guessing an extreme
value, and the threshold itself was calibrated on raw daily pixels, not
this tensor's monthly-averaged values (monthly averaging softens extremes
even in the ACTUAL data).

**This does not mean the model has no spatial skill.** Per-month Spearman
rank correlation (predicted vs actual HCHO, real India cells) is
**positive in every one of the 17 test months** (mean 0.282, range
0.12-0.46) — the model reliably knows WHERE HCHO will be relatively
higher, it just can't be read off with an absolute magnitude threshold
calibrated for a spikier data source.

**Fix used**: a RELATIVE threshold instead of the absolute one — each
month's "predicted hotspots" = that month's own top 5% of predicted
values (`TOP_FRACTION = 0.05`), compared against the SAME absolute
threshold applied to the real data (so "actual hotspot" still means
genuinely exceeds the real-world threshold - only the predicted side
became relative).

**Regional precision/recall results** (`obj2_hotspot_regional_precision_recall.png`),
random-guess baseline for recall = 5% (matches the top-5% pick size):

| Region | Precision | Recall | vs random baseline |
|---|---|---|---|
| National | 0.11 | 0.12 | ~2.5x |
| Corridor (Punjab-Haryana-Delhi-UP) | 0.11 | **0.22** | ~4.5x — strongest region |
| East (Odisha/WB/Assam/NE) | 0.10 | 0.13 | ~2.6x |
| South (Kerala/TN/Karnataka/AP) | 0.11 | 0.05 | ~1x — essentially no skill |
| West (Gujarat/Maharashtra) | 0.20 | 0.07 | ~1.4x recall, high precision |

**Honest read:** the model's hotspot-finding skill is genuinely uneven
across India, not uniform — strongest in the corridor (matches domain
expectation: stubble burning is seasonal and spatially predictable, the
same signal the existing detection/attribution work already validated as
real), moderate in the East, and essentially indistinguishable from random
guessing in the South. Likely explanation: southern HCHO elevation is less
fire-driven (more humidity/biogenic-VOC influenced), a process the model's
input channels (HCHO history, fire count, wind) aren't well-suited to
anticipate. This is presented as a real, uneven capability, not smoothed
into one misleading national number.

**Spatial visualization** (`obj2_hotspot_predicted_vs_actual_map.png`, Dec
2025 test month): correctly-predicted, missed, and false-alarm cells
plotted on the real national grid. Visually confirms the story above -
almost all activity (both correct predictions and false alarms) clusters
in the corridor; the handful of missed hotspots are scattered more widely,
including isolated points in the South and East that the model didn't
anticipate.

**Region definitions**: approximate lat/lon bounding boxes (Corridor:
27-32N/73-81E; East: 19-28N/85-97E; South: 8-19N/74-85E; West:
15-24N/68-77E), assigned mutually exclusively in that priority order -
these are rough regional groupings for this analysis, not official state
boundaries.

---

## 8. Objective 1 — final 5-model results (all real, full-dataset)

All 5 models complete, same 101 stations, same 35,848-row 2025 test set,
same 65 features, chronological 2019-2024/2025 split:

| Model | Test R2 | Test RMSE | Train time |
|---|---|---|---|
| **XGBoost** | **0.519** | 63.02 | 821s (CPU) |
| ARIMA/SARIMA | 0.510 | 63.59 | 874s (6-worker parallelized, univariate - no satellite features) |
| Random Forest | 0.466 | 66.41 | 473s (CPU) |
| Linear Regression | 0.377 | 71.74 | ~1s |
| CNN-LSTM | 0.224 | 79.55 | GPU, 30 epochs, 667s. See caveat below - this number moved between runs |

**CNN-LSTM result instability - flagged honestly, not smoothed over**:
`train_cnn_lstm.py` does not set a random seed, so repeated runs on
identical data produce different results (different weight init, different
minibatch shuffling order). The first full-dataset attempt crashed with a
CUDA OOM during unbatched inference (fixed via `predict_batched()`); the
retry that followed completed successfully but printed test R2=0.367 at
one check, then - it turned out a SECOND retry had also been launched and
completed, silently overwriting the metrics.json with a different result,
**test R2=0.224 (verified as the final, current, on-disk value - complete
log, full 30-epoch training curve, no truncation)**. Both numbers are real
completed runs of the same script on the same data; the discrepancy is
run-to-run variance, not a bug in the metric computation itself. **If this
matters for the presentation, set a fixed `torch.manual_seed()` (and
`np.random.seed()`, since `build_sequences`' NaN-dropping order and the
StandardScaler fit are otherwise deterministic, only the model init/
minibatch order is not) before re-running for a stable, reproducible
number** - not yet done, flagged as a known gap.

**Notable finding, stated honestly**: ARIMA/SARIMA - using ONLY each
station's own AQI history, zero satellite or weather features - nearly
matches XGBoost and clearly beats both Random Forest and Linear Regression
(which DO get the full feature set), and clearly beats CNN-LSTM. This says
AQI has strong day-to-day autocorrelation on its own; the satellite-feature
tree models add real but comparatively modest value on top of that
baseline, while CNN-LSTM underperforms here - plausibly a data-size
mismatch (a deep sequence model with this many parameters may need more
than ~212K training rows across 101 stations to reliably beat simpler
models), compounded by the run-to-run instability noted above. XGBoost's
edge over ARIMA (0.519 vs 0.510) is real but narrow.

**Not yet built**: the consolidated evaluation suite (comparison chart,
residual analysis for the top 2-3, SHAP/importance comparison across
models, per-station breakdown for the winner, rolling-origin CV for the
top 2, written verdict) - individual per-model artifacts exist
(`per_station_metrics.csv`, importance/coefficient files, SHAP sample for
XGBoost) but haven't been assembled into the comparison layer yet.

---

## 9. Additional analysis — PM2.5 vs gas/weather spatial maps (separate from primary pipeline)

**Explicitly requested as an ADDITIONAL layer, not a replacement** - does
not modify, depend on, or risk the primary CNN-LSTM AQI pipeline (section
8), the Streamlit UI work, or Objective 2. Built entirely as new, isolated
files.

**What it shows**: for one representative day in the 2025 test period
(`2025-11-26`, confirmed full 101/101 station PM2.5 coverage, peak
pollution season for good visual dynamic range), a side-by-side spatial
map of PM2.5 against each of 7 gas/weather variables (NO2, SO2, O3, CO,
temperature, wind speed, relative humidity), one PNG per variable, plus a
station-level Pearson correlation for each.

**PM2.5 itself is pulled fresh** (`build_pm25_gas_spatial_maps.py`) using
the same overpass-window methodology as everywhere else in this project
(11:18-15:18 IST mean) - it was intentionally NOT kept in the main
`model_dataset_clean101_daily.csv` (used only to help compute the AQI
target, then dropped to avoid feature/target leakage - see section 7.4
FAQ). This is real, observed ground-truth PM2.5, not a model prediction -
no PM2.5 prediction model has been built (that would be the separate,
lower-priority two-stage model the user asked about; not attempted given
the deadline had already passed by the time this was requested - confirm
with the user before building it).

**Results, station-level Pearson r vs PM2.5 on 2025-11-26** (n=101 stations):

| Variable | r | Direction makes physical sense? |
|---|---|---|
| NO2 | 0.442 | Yes - shared combustion/traffic sources |
| Temperature | -0.451 | Yes - cold weather traps pollution near the surface |
| Wind speed | -0.320 | Yes - wind disperses pollution |
| Relative humidity | -0.162 | Plausible |
| Ozone (O3) | 0.202 | Weak positive |
| CO | 0.071 | Weak positive |
| SO2 | -0.013 | Near zero - SO2 sources (industrial) differ from PM2.5's typical drivers |

Output: `model_comparison_v3/figures/pm25_analysis/pm25_vs_{variable}_2025-11-26.png`
(7 files) + `pm25_correlation_summary.csv`.

**Time-series version** (`build_pm25_gas_timeseries_spatial_maps.py`):
same idea, extended across the year - one composite figure per variable,
each stacking the PM2.5-vs-variable map pair across 4 seasonally-
representative, full/near-full-coverage 2025 dates (winter 2025-01-12,
pre-monsoon 2025-04-03, monsoon 2025-07-01, post-monsoon/stubble
2025-10-01), so the spatial pattern's evolution across the year is visible
in one file instead of scattered across dozens. Confirmed: both PM2.5 and
the gas variables are satellite/CPCB-sourced exactly as in the single-day
version (gases from TROPOMI via `raw_overpass/`, verified directly from
`build_clean101_daily_dataset.py`'s `load_source()` - not CPCB).

Correlation with PM2.5 clearly shifts by season for several variables -
e.g. NO2: winter r=0.18, pre-monsoon r=0.09, monsoon r=0.54, post-monsoon
r=0.49 (much stronger correlation once monsoon rains clear background haze,
making combustion-source co-location more visible); temperature and
relative humidity both flip sign between winter and monsoon. Wind speed is
consistently negative across all 4 seasons (r=-0.60 to -0.08) - the only
variable with a stable-direction relationship year-round.

Output: `model_comparison_v3/figures/pm25_analysis/pm25_vs_{variable}_timeseries_2025.png`
(7 files, 4-season composite each).

### 9.1 Two-stage reference model: gases+weather -> PM2.5 -> AQI

**ADDITIONAL reference model, explicitly NOT primary** - `train_two_stage_
pm25_aqi.py`. Requested as an alternative approach for comparison against
the direct gases-to-AQI models in section 8. Does not modify or replace
CNN-LSTM/XGBoost/etc.

**Stage 1**: Random Forest, 14 physically-grounded features (AOD x2, 5
TROPOMI gases, temp/RH/wind/season - deliberately excludes
`station_aqi_target_enc` and lag/rolling features to keep this a clean,
simple same-day gases->PM2.5 mapping) -> predicts real PM2.5 (pulled fresh
via `build_pm25_reference_full.py`, same overpass-window methodology,
full 2019-2024/2025 range, 101 stations, 2,364 rows dropped for missing
PM2.5 target).

**Stage 2**: plain Linear Regression, predicted PM2.5 (single feature) ->
real AQI. Trained on Stage 1's OWN predictions for the training period
(not real training PM2.5) - this matches actual deployment conditions,
where real PM2.5 is never available, only Stage 1's estimate of it - so
Stage 2 learns to correct for Stage 1's typical error pattern rather than
assuming a perfect input it will never receive.

**Results** (same 2025 test period, same eval_utils metrics as every other
model here):

| Stage | Target | Test R2 |
|---|---|---|
| Stage 1 | PM2.5 (real) | 0.386 |
| Stage 2 / end-to-end | AQI (real) | 0.367 |

**Comparison against the direct models (section 8)**:

| Model | Test R2 (AQI) |
|---|---|
| XGBoost (direct) | 0.519 |
| ARIMA (direct) | 0.510 |
| Random Forest (direct) | 0.466 |
| Linear Regression (direct) | 0.377 |
| **Two-stage (gases->PM2.5->AQI)** | **0.367** |
| CNN-LSTM (direct) | 0.224 |

**Honest verdict**: the two-stage approach lands mid-pack - clearly beats
CNN-LSTM, roughly matches direct Linear Regression, but is meaningfully
behind XGBoost/ARIMA/Random Forest. Two real information losses compound:
Stage 1 only explains ~39% of real PM2.5 variance, and Stage 2 then sees
only that one (imperfect) number, discarding all the other pollutant/
weather signal the direct models use directly. A legitimate, defensible
reference approach, not a replacement for the direct models.

Artifacts: `models/objective1_aqi/artifacts/two_stage_pm25_aqi/` (both
stage models, combined metrics.json, per-station breakdown, test
predictions including both `pm25_pred` and final `pred` AQI columns).

### 9.2 January 2025 prediction comparison

**ADDITIONAL, standalone deliverable** - `data_pipeline/build_jan2025_
prediction_chart.py`. Filters both XGBoost's and the two-stage model's
already-saved `test_predictions.csv` to January 2025, computes national
daily-mean AQI (actual vs. both predicted), and exports a combined
per-station/per-day CSV for direct mentor review. Reads only already-saved
artifacts - does not retrain or touch either pipeline.

**January-specific accuracy** (differs from the full-2025-test-period
numbers above because January 2025 is a single, smog-heavy month, not the
full test distribution):

| Model | Jan 2025 R2 | Jan 2025 MAE |
|---|---|---|
| XGBoost (direct) | 0.425 | 56.8 |
| Two-stage (gases->PM2.5->AQI) | 0.160 | 67.8 |

Output: `figures/jan2025_aqi_prediction_comparison.png` (line chart, actual
vs. both models' predictions, national daily mean) and
`data/jan2025_predictions_export.csv` (3,039 station-days, 101 stations).

### 9.3 Live in the Streamlit app

All of section 9 (spatial maps, two-stage model metrics, January 2025
comparison) is now surfaced in the existing `streamlit_app.py` dashboard
(the pre-existing app at the `Trial -02` root, port 8501) under a new third
top-level radio option, **"PM2.5 Deep-Dive (additional)"**, alongside the
pre-existing "Objective 1 - AQI" and "Objective 2 - HCHO Hotspots" options.

This was added as a **new, self-contained `pm25_deepdive_view()` function**
plus one new branch in the top-level `main()` radio dispatch - the existing
`objective1_view()`/`objective2_view()` functions (map coloring, date
slider, per-station scrolling, feature-importance cause analysis, all 5
Objective 1 models, all Objective 2 tabs including Getis-Ord/LISA/DBSCAN/
wind-fire) were not modified. Verified live in-browser after the change:
both existing objectives still render correctly (confirmed CNN-LSTM test
R2=0.224 still matches, IDW map/slider/per-station trend all load), and the
new tab's 3 sub-tabs (spatial maps, two-stage metrics, January 2025
comparison) all render their real data with no console errors.

New tab contains 5 sub-tabs (grew from 3 to 5 in a follow-up round, see 9.4-9.5):
1. **Gas <-> PM2.5 spatial maps** - toggle between the single-day
   (2025-11-26) and 4-season time-series versions, per-variable Pearson r,
   full correlation summary table.
2. **Continuous spatial AQI-via-PM2.5 map** - see 9.4.
3. **Two-stage model** - Stage 1/Stage 2/end-to-end test metrics, feature
   list, one-line comparison against the primary XGBoost model.
4. **January 2025 prediction comparison** - the chart above, live-computed
   R2/MAE metrics, and a filterable per-station/per-day table of the export
   CSV.
5. **Full 2025 prediction comparison** - see 9.5.

### 9.4 Continuous national spatial map: satellite-predicted AQI via PM2.5

**Follow-up request**: the single-day/time-series maps in 9.0 only plot the
101 discrete CPCB station points - a true continuous, India-wide surface
was requested instead, reusing Objective 1's existing spatial-grid
infrastructure rather than building anything new.

**Confirmed directly reusable, no new GEE extraction** (checked before
writing any code):
- `models/objective1_aqi/artifacts/spatial_grid/` already has AOD + all 5
  TROPOMI gases + ERA5 sampled at the national 4,447-point 0.25 deg grid
  (`data_pipeline/load_objective1_grid.py`), 57 real dates (31 in 2025).
- `features/merge_sources.derive_wind_met()` - the exact same function
  already used to build `temp_c`/`rh_pct`/`wind_speed`/`wind_dir_deg` for
  the two-stage model's own station-level training data - so grid-inference
  features are derived identically to training, zero train/inference skew.
- `spatial_interp.py`'s `idw_interpolate`/`render_overlay_png` (the
  Streamlit app's existing Objective 1 map renderer) are fully generic -
  feeding them the ~4,447 grid points instead of 101 stations produces a
  visibly denser, more continuous surface with **zero changes to that
  rendering code**.

New script: `data_pipeline/build_pm25_aqi_grid.py` - loads the grid feature
table, derives wind/met, runs the two-stage model's already-trained Stage 1
(gases+weather -> PM2.5, no retraining) across all 4,447 x 57 rows, then
converts predicted PM2.5 -> AQI via `cpcb_process_v2.sub_index("PM2_5",
...)` directly (the real CPCB breakpoint formula, not Stage 2's learned
linear approximation - per the explicit request to use the breakpoint
formula). Output: `data/pm25_aqi_grid_predictions.csv` (253,479 rows).
Predicted PM2.5 range: 15.8-514.5 ug/m3, national daily mean tracks known
seasonality (higher Nov-Jan/Aug spike, lower Jun-Aug monsoon).

Wired into Streamlit as a new sub-tab with its own date slider (31 real
2025 grid dates - sparser than the daily station slider, since the grid
only has real satellite composite dates, not synthetic daily interpolation)
showing: national mean predicted PM2.5/AQI, the continuous IDW-interpolated
color surface, and small real-CPCB-station markers overlaid on top (click
for ground-truth PM2.5/AQI at that point) for visual comparison against the
surface underneath.

### 9.5 Full 2025 PM2.5/AQI export (not just January)

**Follow-up request**: extend the January-only comparison (9.2) to the
full 2025 test period. Trivial extension - `two_stage_pm25_aqi/test_
predictions.csv` and `xgboost/test_predictions.csv` already cover the
entire 2025 test period (Jan 1 - Dec 31); the January script had simply
filtered to one month. New script `build_full2025_pm25_aqi_export.py`
removes that filter: two charts (full-year AQI actual-vs-both-models,
full-year PM2.5 actual-vs-Stage1-predicted) plus
`data/full2025_predictions_export.csv` (35,848 station-days, 101
stations). Full-year accuracy numbers are identical to the project's
existing headline test-set metrics (XGBoost R2=0.519, two-stage
R2=0.367, Stage 1 PM2.5 R2=0.386) - confirms internal consistency, no
new modeling was needed for this part.

### 9.6 Wind visualization follow-up fixes (Objective 2 monthly + seasonal maps)

**Follow-up requests, both applied to `add_wind_arrow()` (shared helper,
used by both the Monthly spatial view and the new Wind + Fire seasonal
map)**:
- Arrowheads: switched from `folium.RegularPolygonMarker` (silently failed
  - depends on an external `leaflet-dvf` CDN plugin that never loaded in
  this environment, confirmed via DOM inspection: zero rendered triangles,
  `hasDvf: false` in the served HTML) to a pure CSS-rotated `DivIcon`
  triangle - zero external dependencies, renders reliably, made
  deliberately small (7px) per the follow-up "arrowhead should be a bit
  smaller" request.
- Magnitude now drives 3 visual channels at once (length, line thickness,
  color depth), not just color - real wind_speed differences are visually
  obvious, not subtle.
- Sample density raised from ~150 to ~900-1100 arrows per map (was too
  sparse to read at a glance) - confirmed via DOM inspection
  (`leaflet-interactive` path counts: 1107 on the seasonal map, ~900 wind
  lines + boundary polygons on the monthly map).
- Added real numeric color-scale legends (`branca.colormap.LinearColormap`)
  for every color-coded layer (HCHO, fire density, wind speed) on both
  maps - confirmed rendering with real calibrated values, e.g. "0.00003 ->
  0.00033 HCHO column density", "0 -> 617 Fire detections" (monthly view,
  2025-11), "Post-Monsoon mean wind speed (m/s)" (seasonal map).

### 9.7 Objective 1 single-station data trace (matrix -> PM2.5)

**Follow-up request**: for one CPCB station, verify end-to-end that the
satellite/weather data (AOD from MODIS MAIAC, NO2/SO2/CO/O3/HCHO from
Sentinel-5P TROPOMI - the "PM2.5 gas obtained via the spatial route", ERA5
weather) is genuinely "carried in a matrix" and that a real value comes out
the other end, plus a PM2.5 time series for that station. New sub-tab
("Single-station data trace") in the Streamlit PM2.5 Deep-Dive section:

1. **Real feature matrix** - station-selectable, shows the literal
   `model_dataset_clean101_daily.csv` rows (date x the 14 real
   `STAGE1_FEATURES` columns) for the chosen station, not a schematic.
2. **Live model trace** - for a chosen date, runs that exact matrix row
   through `imputer1.transform()` -> `stage1.predict()` **live in the app**
   (loaded fresh from the saved `.pkl` artifacts via `st.cache_resource`,
   not read from a pre-computed predictions file) and shows the resulting
   predicted PM2.5 next to the real CPCB ground-truth PM2.5 for that date -
   directly verifiable date by date. Confirmed live: Adarsh Nagar, 2025-XX
   example run showed predicted=61.0 ug/m3 vs actual=46.4 ug/m3 (split=train).
3. **Time series** - full 2019-2025 real PM2.5 for that one station (actual
   CPCB vs Stage-1-predicted, predicted computed live across every date
   shown, train and test both) - train-period predictions shown for visual
   fit reference only; the 2025 (test-split) predictions are the genuine
   out-of-sample evidence.

Note: the user's request was cut off mid-sentence ("...and we need") -
this covers everything stated before the cutoff; asked the user what the
rest of the sentence was rather than guessing further scope.
