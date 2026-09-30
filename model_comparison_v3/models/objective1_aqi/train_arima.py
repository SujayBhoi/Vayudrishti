"""Per-station SARIMA on the clean101 daily dataset - the classic
pure-autoregressive statistical baseline. Deliberately NOT given the
satellite features (unlike the other 4 models) - this asymmetry is stated
explicitly in the writeup as a fairness caveat: ARIMA/SARIMA's role here is
"what can you predict from AQI's own history alone", not a 6th ML model
competing on the same inputs.

Daily cadence -> seasonal period m=7 (weekly cycle within daily data), not
the m=52 that would apply to weekly data. Walk-forward one-step-ahead
evaluation across the 2025 test days (append the true value after each
step, forecast the next - the standard, fair way to evaluate a time-series
model over a held-out span), via statsmodels' SARIMAX.append().

Any per-station fit failure falls back to a simpler order (or is skipped
and logged) - doesn't block other stations, matching this project's
established fallback discipline.
"""

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

N_WORKERS = 6  # per-station SARIMA fits are independent - parallelized across
# processes (not threads - statsmodels MLE fitting is CPU-bound, threading
# wouldn't help past the GIL). Measured 417s/11 stations (~38s/station)
# sequential - at 101 stations that's ~64 min serial, unacceptable given the
# deadline this was built under; 6 workers targets ~11 min instead.

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DATA_DIR = os.path.join(V3_ROOT, "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts", "arima")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from eval_utils import evaluate, per_station_metrics, timer, check_not_degenerate  # noqa: E402

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 7)
FALLBACK_ORDER = (1, 0, 0)
FALLBACK_SEASONAL_ORDER = (0, 0, 0, 0)
MIN_TRAIN_ROWS_FOR_SEASONAL = 30  # need enough history for a real m=7 seasonal fit


def fit_and_forecast_one_station(train_series: pd.Series, test_series: pd.Series):
    """Fit on train, then walk-forward one-step-ahead across test: forecast
    the next value, then append the TRUE observed value before forecasting
    the next one (standard walk-forward evaluation, not a single long-horizon
    forecast from the end of train)."""
    order, seasonal_order = (ORDER, SEASONAL_ORDER) if len(train_series) >= MIN_TRAIN_ROWS_FOR_SEASONAL else (FALLBACK_ORDER, FALLBACK_SEASONAL_ORDER)
    try:
        model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        used_fallback = False
    except Exception:
        try:
            model = SARIMAX(train_series, order=FALLBACK_ORDER, seasonal_order=FALLBACK_SEASONAL_ORDER, enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            used_fallback = True
        except Exception as e:
            return None, None, str(e)

    preds = []
    current_fit = fit
    for true_val in test_series.values:
        one_step = current_fit.forecast(steps=1)
        preds.append(float(one_step.iloc[0]))
        current_fit = current_fit.append([true_val], refit=False)
    return np.array(preds), used_fallback, None


def _worker(args):
    sid, train_series, test_series = args
    preds, used_fallback, err = fit_and_forecast_one_station(train_series, test_series)
    return sid, preds, used_fallback, err, test_series.index, test_series.values


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset_clean101_daily.csv"), parse_dates=["date"])
    df = df.sort_values(["station_id", "date"])

    jobs = []
    failures = []
    for sid, g in df.groupby("station_id"):
        g = g.set_index("date")
        train_series = g.loc[g["split"] == "train", "aqi"]
        test_g = g[g["split"] == "test"]
        test_series = test_g["aqi"]
        if len(train_series) < 10 or len(test_series) == 0:
            failures.append({"station_id": sid, "reason": "insufficient train/test rows"})
            continue
        jobs.append((sid, train_series, test_series))

    all_preds = []
    fallback_count = 0
    with timer() as t_train_all:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_worker, job): job[0] for job in jobs}
            done = 0
            for fut in as_completed(futures):
                sid = futures[fut]
                done += 1
                sid, preds, used_fallback, err, test_dates, test_vals = fut.result()
                if preds is None:
                    failures.append({"station_id": sid, "reason": err})
                    continue
                if used_fallback:
                    fallback_count += 1
                out = pd.DataFrame({"station_id": sid, "date": test_dates, "aqi": test_vals, "pred": preds})
                all_preds.append(out)
                if done % 10 == 0 or done == len(jobs):
                    print(f"[arima] {done}/{len(jobs)} stations done")

    results = pd.concat(all_preds, ignore_index=True)
    check_not_degenerate(results["pred"], "ARIMA/SARIMA")

    metrics = {
        "test": evaluate(results["aqi"], results["pred"]),
        "train_seconds_total": t_train_all["seconds"],
        "n_stations_fit": int(df.station_id.nunique() - len(failures)),
        "n_stations_failed": len(failures),
        "n_stations_used_fallback_order": fallback_count,
        "failures": failures,
        "order": list(ORDER),
        "seasonal_order": list(SEASONAL_ORDER),
        "note": "deliberately univariate (AQI history only, no satellite features) - see module docstring",
    }
    print(json.dumps(metrics, indent=2, default=str))

    per_station = per_station_metrics(results, "aqi", "pred")

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    per_station.to_csv(os.path.join(ARTIFACTS_DIR, "per_station_metrics.csv"), index=False)
    results.to_csv(os.path.join(ARTIFACTS_DIR, "test_predictions.csv"), index=False)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
