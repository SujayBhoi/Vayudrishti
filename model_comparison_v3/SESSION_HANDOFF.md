# SESSION HANDOFF — read this first in a new chat

**Updated:** 2026-08-07, 06:51 IST. **User's stated deadline (6:00 AM) has
now passed** — as of this update, current status with the user on the
presentation is unknown; ask directly rather than assume anything.
**All 5 Objective 1 models are now DONE with real, full-dataset results**
(table below) — this was completed despite a session interruption around
01:12-06:50 IST that killed background processes without warning (see
"What just happened" below, kept for reference). Read this file fully
before doing anything else.

## ALL 5 OBJECTIVE 1 MODELS COMPLETE — final results

Same 101 stations, same 35,848-row test set (2025), same features, chronological 2019-2024/2025 split:

| Model | Test R² | Test RMSE | Notes |
|---|---|---|---|
| **XGBoost** | **0.519** | 63.02 | Best. 821s train (CPU) |
| ARIMA/SARIMA | 0.510 | 63.59 | Zero satellite features, pure AQI autocorrelation - notable finding | 
| Random Forest | 0.466 | 66.41 | 473s train |
| Linear Regression | 0.377 | 71.74 | ~1s train |
| CNN-LSTM | 0.367 | — | GPU, 30 epochs. Hit a CUDA OOM on first attempt (unbatched inference on the full 211,961-row train set) - fixed via `predict_batched()`, already on disk, re-run succeeded |

All metrics.json files are real (verified `test.n` matches the full test
set, not the earlier stale dry-run numbers). **What's NOT done yet: the
consolidated Objective 1 evaluation suite** (comparison chart, residuals,
SHAP/importance comparison, per-station breakdown, CV, written verdict) —
see the existing "Objective 1 — what's left" section below, still accurate.

## What just happened (historical - kept so the pattern is recognized if it recurs)

## What just happened (so the same confusion doesn't repeat)

A background CNN-LSTM training process was killed when the previous
Claude Code process/session was torn down (mechanism unclear — possibly a
UI stop, a monitor timeout, or the app exiting). Background processes
launched via the Bash tool's `run_in_background` are real OS processes,
but they are NOT guaranteed to survive the harness session ending —
treat that as a real risk, not a one-off fluke. **After any new
long-running background launch, verify it's still alive periodically
(PowerShell process check below) rather than assuming a "still running"
status is durable.**

## Exact current state — verified against real file timestamps, not assumed

**Satellite extraction: 100% COMPLETE, 606/606, zero errors.** All 6
sources (AOD, NO2, SO2, CO, O3, HCHO) x 101 stations. This is done, stable,
on disk, will not need to be touched again.

**Full joined dataset: built and on disk.**
`model_comparison_v3/data/model_dataset_clean101_daily.csv` — 247,809
rows, 101 stations, 65 features, matches ground truth exactly (clean inner
join, nothing dropped). Train 211,961 / test 35,848 rows. No need to
rebuild unless the source data changes.

**Objective 1 models — 3 of 5 have REAL full-dataset results, 2 are
mid-run (just relaunched):**

| Model | Status | Test R² | Notes |
|---|---|---|---|
| Linear Regression | ✅ DONE (full data) | 0.377 | artifacts/linear_regression/metrics.json |
| Random Forest | ✅ DONE (full data) | 0.466 | artifacts/random_forest/metrics.json, 473s train |
| XGBoost | ✅ DONE (full data) | 0.519 | artifacts/xgboost/metrics.json, 821s train — current best |
| CNN-LSTM | 🔄 RELAUNCHED, in progress | — | previous run was killed mid-training; relaunched at 01:12 IST as background ID (see check command below). Old artifacts/cnn_lstm/metrics.json is STALE (17-station dry run, test n=3816 — do not trust it) |
| ARIMA/SARIMA | 🔄 RELAUNCHED, in progress | — | parallelized (6 workers), old artifacts/arima/metrics.json is STALE (17-station dry run — do not trust it) |

**How to tell if the stale artifacts have been overwritten with real ones:**
check `test.n` in the metrics.json — the real full-dataset value is
**35,848** for CNN-LSTM (or slightly less if sequence-window dropping
trims a few rows) and results should reference 101 stations for ARIMA
(`n_stations_fit` close to 101, not 17).

**Check if CNN-LSTM / ARIMA are still alive:**
```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Select-Object ProcessId, CommandLine
```
If neither `train_cnn_lstm.py` nor `train_arima.py` shows up and their
metrics.json files still show the stale numbers above (test n=3816 /
n_stations_fit=17), **they died again — just re-run them**, from
`Trial -02`:
```bash
.\venv\Scripts\python.exe model_comparison_v3\models\objective1_aqi\train_cnn_lstm.py
.\venv\Scripts\python.exe model_comparison_v3\models\objective1_aqi\train_arima.py
```
CNN-LSTM (GPU) should take a few minutes. ARIMA (parallelized, 6 CPU
workers) took 174s for 17 stations in testing — expect roughly
10-20 min for all 101, run it in the background (`run_in_background`
equivalent) since it exceeds a typical foreground timeout.

## Objective 1 — what's left after the last 2 models finish

**The evaluation suite has NOT been built yet** — only individual training
scripts exist so far. Needed (mirror the Objective 2 pattern already built
in `models/objective2_hcho/run_evaluation.py` and
`data_pipeline/build_objective2_charts.py` for consistency):
1. One consolidated metrics table across all 5 models (RMSE/MAE/R²/MAPE,
   train+test) + a comparison bar chart (reuse the same palette/style as
   `build_objective2_charts.py`).
2. Timing table (already captured per-model in each metrics.json —
   `train_seconds`, `inference_seconds_total` — just needs assembling).
3. Residual analysis (predicted vs actual, residuals vs predicted) for the
   top 2-3 models once real numbers are in — XGBoost is currently the
   leader (R²=0.519), Random Forest second (0.466).
4. Feature importance/SHAP comparison — XGBoost already saves
   `shap_values_sample.npy` and `feature_importances.csv`; Random Forest
   saves `feature_importances.csv`; Linear Regression saves
   `standardized_coefficients.csv`; ARIMA is N/A (univariate, documented
   as such); CNN-LSTM needs permutation importance (code for this already
   exists inside `train_cnn_lstm.py`, saves
   `permutation_importance.csv`).
5. Per-station breakdown for the winning model (`per_station_metrics.csv`
   already saved per-model by each script).
6. Rolling-origin cross-validation for the top 2 models — NOT built yet,
   needs new code (mirror `models/objective2_hcho/run_evaluation.py`'s
   `rolling_origin_cv()` pattern, adapted to per-station tabular data
   instead of the grid tensor).
7. Written verdict once all of the above is in.

## Objective 2 — DONE, but user said changes are needed (STILL not specified)

5 models compared (persistence, statistical baseline, Linear Regression,
Random Forest, ConvLSTM) — Random Forest currently wins (test R²=0.155)
over ConvLSTM (0.116), reported honestly. Full results in
`PROJECT_DOCUMENTATION.md` section 5. Charts in
`model_comparison_v3/figures/obj2_*.png`, using real state boundaries.

**The user has said TWICE across this session that Objective 2 changes are
needed, but never specified what.** Ask directly, first thing, rather than
guess or assume it's resolved.

## Other open items, not yet resolved

- India boundary should show J&K including POK per Survey-of-India
  convention. Current shapefile does NOT have this (verified: no Ladakh
  feature, J&K clipped at LOC). User was going to look for a proper
  Survey-of-India/Bhuvan/datameet source — ask if they found one.
- `D:\PS3\S5P data\` has real per-orbit Sentinel-5P L2 files but only a
  3-file/gas sample — not usable, mentioned in case more time-pressure
  relief is needed and a fuller archive turns up.
- Full presentation materials (PPTX, one-page PDF) — not started.

## Key file locations

- Full technical record: `model_comparison_v3/PROJECT_DOCUMENTATION.md`
  (section 7 = fast-scan briefing/FAQ, section 4.1 = GEE incident history,
  section 5 = Objective 2 full results)
- All data: `model_comparison_v3/data/`
- All model code + artifacts: `model_comparison_v3/models/`
- All charts: `model_comparison_v3/figures/`
- Isolated in `Trial -02/model_comparison_v3/` — never touches existing
  `models/objective1_aqi/`, `models/objective2_hcho/`, `backend/`,
  `frontend/`, or root-level docs.
