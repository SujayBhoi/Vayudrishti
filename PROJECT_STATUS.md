# PROJECT STATUS SNAPSHOT

**Last updated:** 2026-08-04, ~23:15 (during live session)
**Purpose:** Resume point if context window fills / session restarts. Read this first.

---

## Checkpoint status

### Objective 1 (Surface AQI) — CORE MODELING DONE, GRID EXTRACTION JUST LAUNCHED
- v1 baseline (13 cities): **preserved untouched**, re-verified byte-for-byte
  reproduces R²=0.671 after copy into `models/objective1_aqi/artifacts/baseline_v1/`.
  This is the guaranteed-working fallback, always available.
- v2 (412 stations, multi-gas): all 9 raw sources (AOD, NO2, SO2, CO, O3, HCHO,
  ERA5, fire, BLH) fully extracted (412/412 stations each). Merged dataset built
  (97,593 station-weeks). Trained twice:
  - First pass: pooled test R²=0.599 (below v1)
  - After fixing a real bug (9 stations had zero pre-cutoff training rows —
    cold-start evaluation artifact): R²=**0.660**, RMSE=49.39, MAE=33.85
  - **v1 stays primary** per the "only promote if it genuinely beats v1" rule
    (0.660 < 0.671), but v2 is a strong secondary result: near-v1 accuracy
    while covering 30x more stations nationally.
- Spatial grid (the "predict AQI anywhere in India" capability): calibration
  test succeeded (AOD 0.044 EECU-hr/date, ERA5 0.001 EECU-hr/date measured).
  **Full ~250-date grid run was just launched** (see "Currently running" below)
  covering AOD + 5 gases + ERA5, multi-band batched per source.

### Objective 2 (HCHO Hotspots) — DETECTION DONE, ATTRIBUTION MID-ITERATION
- Detection (percentile threshold + DBSCAN): **fully working**, validated on
  real national data. Genuinely spread across India (not just Punjab-Haryana).
- HCHO data: **all 330 dates now extracted** (batch queue just finished: 27
  succeeded, 1 failed-then-fixed-and-retried-successfully). Only 84/330 dates
  have been downloaded+analyzed so far — **need to download the rest and rerun**.
- Attribution (fire/wind → HCHO explanation): architecture is two separate
  models (see "Decisions locked in"). On the partial 84-date sample, CV R² is
  not yet reliably positive (see "Key numbers"). This is diagnosed as likely a
  sample-size problem, not a broken pipeline — rerunning on the full 330-date
  set (now available) is the next validation step, explicitly requested by user.
- Wind + fire grids for attribution: **fully downloaded already** (1.85GB fire,
  268MB wind, both cover all 418 weeks 2018-2026) — not a bottleneck.

### Backend / Frontend — SCAFFOLDED, NOT YET WIRED TO FINAL DATA
- Backend: FastAPI app with all 5 endpoints implemented (`/predict/aqi`,
  `/aqi/history`, `/hotspots`, `/hotspots/{id}/explain`), reads from
  precomputed artifacts only (no live calls). Loads cleanly, not yet tested
  against final v2/Objective 2 artifacts end-to-end.
- Frontend: Vite+React+Leaflet+Recharts app scaffolded, builds cleanly
  (`npm run build` succeeds). Map, detail panel, history chart, hotspot
  explain, legend, date picker all coded. Not yet tested in a browser against
  a running backend.
- Node.js installed on this machine (via winget, with user approval) during
  this session.

### CNN-LSTM (Checkpoint 5) — NOT STARTED
Skeleton written (`models/objective1_aqi/train_cnn_lstm.py`), torch installed.
Explicitly lowest priority, attempt only after everything else is solid.

---

## Key numbers

**EECU budget:** 29.13 of 150 EECU-hr used (19.4%) as of last check. Free tier
only, no billing — locked decision. Healthy remaining margin (~121 EECU-hr).

**Objective 1 AQI model comparison:**
| | v1 (13 cities) | v2 (412 stations) |
|---|---|---|
| Test R² | **0.671** (primary) | 0.660 |
| Test RMSE | 42.25 | 49.39 |
| Test MAE | 28.80 | 33.85 |
| Stations | 13 | 403 (412 minus 9 cold-start-excluded) |

**Objective 2 attribution — FINAL result on FULL 351-date dataset (all HCHO
data downloaded, processed, and rerun):**
- Detection: 115,030 hotspot pixel-dates flagged, 9,244 DBSCAN clusters
- corridor_season (primary) model: test R²=-0.001 on 15,982 points
- national_all_season (secondary) model: test R²=0.002 on 114,697 points
- **This is a ROBUST NULL RESULT, not a sample-size artifact.** The 84-date
  partial sample showed R²≈-0.009 (corridor); with 11.5x more data (15,982 vs
  1,386 points) the estimate barely moved (-0.001). If a real, learnable
  fire→HCHO relationship existed at this feature/resolution, this much more
  data would have revealed it. It didn't. Conclusion: at 0.25° weekly grid
  resolution, `fire_dist_km`/`fire_count_nearby`/`wind_alignment` (grid-cell-
  count proxy, not raw fire pixels) do not reliably predict HCHO value or
  anomaly, even though the raw correlation direction is consistently right
  (~0.09-0.10 in corridor+season, weaker nationally). Likely explanations:
  weekly aggregation is too coarse for plume transport dynamics (fire today
  can elevate HCHO downwind 1-2 days later, not necessarily within the same
  matched week), and/or grid-cell fire counts are too crude a proxy for
  actual fire intensity/duration. **This has NOT been discussed with the user
  yet as of this snapshot — flag as the first thing to report on resume.**
- SHAP explanations ARE being generated (pipeline runs end-to-end
  successfully) but with near-zero-R² models, the explanations describe a
  model that has learned almost nothing - honest framing for the demo needed
  (see original recommendation: present detection as the strong, validated
  capability; attribution as "consistent-direction, not yet statistically
  reliable at this resolution" rather than a confident causal claim).

**HCHO date coverage:** 330 dates requested, 351 actually extracted (topup
selection produced slightly more than exactly 330, harmless), **all downloaded
and fully analyzed** — this data pipeline is now complete and no longer a
bottleneck.

---

## Decisions locked in this session

1. **Free tier only, no billing** — user explicit decision when EECU budget
   was discussed. All scope decisions respect the 150 EECU-hr/month ceiling.
2. **330 HCHO dates** for Objective 2 (not the original ~860 plan), Oct-Dec
   weighted (60%) + other-month contrast (40%), chosen after real per-date
   cost came in at ~0.038 EECU-hr/date (cheaper than initial estimate).
3. **Two-model attribution approach**: PRIMARY = trained only on Oct-Dec +
   Punjab-Haryana-Delhi-UP corridor (the actual validated fire season/region).
   SECONDARY = trained on all nationally-flagged hotspots, all seasons, kept
   separate and honestly labeled rather than blended into one misleading number.
   Detection itself stays fully national/year-round regardless.
4. **Anomaly-target redesign**: attribution model predicts HCHO deviation from
   a regional (2°×2° bin) baseline, not raw HCHO value — isolates fire's
   marginal contribution from baseline geographic/seasonal variation. Implemented
   in `models/objective2_hcho/compute_baseline.py`.
5. **Per-station `filterBounds`-scoped extraction** for all granule-heavy
   sources (AOD, all 5 TROPOMI gases) — NOT batched national point extraction.
   This was a real incident: the original batched design burned 10.2 EECU-hr
   in under a minute because compositing granules across scattered national
   points forces near-full-country compositing every period. Fixed via
   `data_pipeline/interactive_point_extract.py` (proven-safe, matches v1's
   original per-city pattern, just scaled to 412 stations). ERA5/fire are
   simple gridded products and were safe to batch nationally either way.
6. **v1 never modified in place** — `Trial/` (sibling dir) stays untouched as
   an independent, always-available fallback throughout.
7. **9 zero-training-history stations excluded** from v2 evaluation (cold-start
   artifact, not real signal) — see `train_extended_xgb.py`'s
   `drop_zero_training_stations()`.

---

## Known non-blocking issues / gotchas for whoever resumes

- **Nested-path bug**: several `fetch_*.py` scripts use a cwd-relative
  `"data_pipeline/raw_v2/..."` output path. When run from *inside*
  `data_pipeline/` (which most launch commands do), this nests one level
  deeper: actual path is `data_pipeline/data_pipeline/raw_v2/{source}/`.
  Confirmed for: aod, hcho, no2, so2, co, o3, era5, fire. NOT nested: blh
  (uses `__file__`-relative path, not cwd-relative). `features/build_v2_dataset.py`
  already has the correct paths hardcoded — check `SOURCE_DIRS` there before
  writing new path-dependent code.
- **Fire grid export is bloated** (1.85GB for what should be ~50MB) because a
  50km buffer was applied to points before `reduceRegions`, so the full
  buffered polygon gets written to the `.geo` column of every row. Not wrong,
  just wasteful — `data_pipeline/load_objective2_grids.py` already works
  around this by dropping `.geo` and joining `grid_id`→lat/lon from the local
  grid CSV instead.
- **GEE quota tier not confirmed programmatically** — Cloud Quotas/Monitoring
  APIs are disabled on this project and enabling them needs Console access we
  don't have from here. We're assuming Community tier (150 EECU-hr/month,
  the documented default) and it's been accurate so far (measured costs match
  expectations), but it's an assumption, not a verified fact.
- **This project runs GEE batch tasks close to one-at-a-time** — expect long
  queue waits for anything beyond a couple of batch tasks. Interactive calls
  (the `interactive_point_extract.py` pattern) don't have this limitation.

---

## Currently running / queued in the background (as of last update)

1. **Objective 1 spatial grid extraction** — IN PROGRESS, ~185 total batch
   tasks now queued (down from ~345 after cleanup - see below). AOD, ERA5,
   HCHO sources are fine and progressing (mix of SUCCEEDED/PENDING). NO2/SO2/
   CO/O3 hit the exact same "qa_value band doesn't exist" bug already fixed
   once in `fetch_tropomi.py` but not kept in sync in
   `fetch_objective1_grid.py` - 23 tasks failed before this was caught. Fixed
   (NO2/SO2/O3 now use cloud_fraction<0.3, CO uses no QA mask, matching the
   validated Objective 2 config) + added the same "no images that day" empty-
   collection guard HCHO needed. Cancelled 100 broken-config pending tasks,
   relaunched 100 fixed ones for the same 250-date coverage. Given the
   single-concurrent-task queue behavior, this will take a long time to fully
   drain (likely spans a day+ of wall-clock) - no hard deadline so that's fine.
2. Objective 2's HCHO pipeline is COMPLETE - see updated numbers above. No
   longer running/queued, this bottleneck is resolved.

## Objective 2 attribution finding — NEEDS TO BE DISCUSSED WITH USER

The full-dataset rerun (351 dates, 115K flagged hotspots) confirms the null
result from the partial sample is real, not a sample-size problem (see "Key
numbers" above for the full explanation). **This was reported to the user in
the same turn PROJECT_STATUS.md was last written, but their response hasn't
been seen yet as of this snapshot** - if resuming cold, check the actual
conversation for how the user wants to handle this for the demo (options
discussed: honest-caveat framing keeping the current two-model structure,
try different features/finer temporal matching, or lean on Objective 2's
DETECTION step - which works well - as the headline capability and describe
attribution more qualitatively/directionally rather than via R²).

---

## Exact next steps

1. **Download the remaining HCHO batches** (246 of 330 dates not yet pulled
   locally) via `data_pipeline/download_from_drive.py --folder ps3_obj2_hcho`,
   then `data_pipeline/load_hcho_rasters.py --downsample-factor 25` to rebuild
   `hcho_grid_national.csv` with the full dataset.
2. **Rerun `models/objective2_hcho/run_pipeline.py`'s `run()`** on the full
   330-date HCHO data + existing fire/wind grids. Report updated CV/test R²
   for both the corridor_season (primary) and national_all_season (secondary)
   attribution models, plus fresh sample SHAP explanations, back to the user.
3. **Monitor the Objective 1 grid extraction** to completion, then build
   `models/objective1_aqi/predict_grid.py`'s actual grid predictions from it.
4. Once both objectives have final artifacts: wire the backend (`backend/store.py`
   already expects the right file locations) and test all 5 endpoints for real,
   then test the frontend against a running backend in an actual browser.
5. Only after all of the above is solid: attempt CNN-LSTM (Checkpoint 5),
   lowest priority per the original working principle.

---

## Files/scripts that matter most if resuming cold

- `models/objective1_aqi/train_extended_xgb.py` — v2 AQI training, includes the
  cold-start-station fix
- `models/objective2_hcho/run_pipeline.py` — full detect→attribute→explain
  orchestration, includes the two-model split and date-matching fix
- `models/objective2_hcho/compute_baseline.py` — the anomaly-target redesign
- `data_pipeline/interactive_point_extract.py` — the proven-safe per-station
  extraction pattern (use this, not a national batch, for any new granule-heavy
  source extraction)
- `data_pipeline/fetch_objective1_grid.py` — spatial grid extraction (currently running)
- `data_pipeline/load_objective2_grids.py` — fire/wind grid loader (handles the
  bloated-geometry workaround)
- `models/objective1_aqi/CALIBRATION.md` — the original EECU calibration
  writeup from Checkpoint 0
