"""v2: Open-Meteo boundary layer height (BLH) at all v2 CPCB station points.

Same approach as v1 (legacy_v1_src/openmeteo_extract.py) — no GEE ERA5 mirror
has BLH, so it's pulled separately and resampled hourly->weekly. This is a
plain REST API, not GEE, so no EECU cost and no batching trick needed; it's
just a loop over stations with disk caching (identical pattern to v1, only
the station list is bigger: 417 vs 13).
"""

import os
import time

import pandas as pd
import requests

from fetch_cpcb import load_station_master

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw_v2", "blh")
os.makedirs(RAW_DIR, exist_ok=True)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


REQUEST_DELAY_S = 1.5  # proactive throttle between EVERY request


def extract_blh(station_id, lat, lon, start, end, retries=4):
    safe_id = "".join(c if c.isalnum() else "_" for c in station_id)
    fname = os.path.join(RAW_DIR, f"{safe_id}.csv")
    if os.path.exists(fname):
        return pd.read_csv(fname)

    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY_S)
            # Proactive throttle replaces pure reactive backoff: the earlier
            # version only slowed down AFTER a 429, but exponential backoff
            # (up to 320s x 6 retries) let the run appear "stuck" for a very
            # long time once rate-limited repeatedly, with no visibility
            # into why. A small delay before every request (not just after
            # failures) stays under Open-Meteo's limit in the first place.
            r = requests.get(
                ARCHIVE_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start,
                    "end_date": end,
                    "hourly": "boundary_layer_height",
                    "timezone": "UTC",
                },
                timeout=60,
            )
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  [blh 429 backoff {wait}s] {station_id}", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            j = r.json()
            df = pd.DataFrame({"datetime": j["hourly"]["time"], "blh": j["hourly"]["boundary_layer_height"]})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"] = df["datetime"].dt.date
            daily = df.groupby("date")["blh"].mean().reset_index()
            daily["date"] = pd.to_datetime(daily["date"])
            weekly = daily.set_index("date")["blh"].resample("7D", origin=pd.Timestamp(start)).mean().reset_index()
            weekly["date"] = weekly["date"].dt.strftime("%Y-%m-%d")
            weekly["station_id"] = station_id
            weekly.to_csv(fname, index=False)
            return weekly
        except Exception as e:
            print(f"  [blh retry {attempt+1}] {station_id}: {e}")
            time.sleep(5)
    raise RuntimeError(f"BLH extraction failed for {station_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    stations = load_station_master()
    print(f"Fetching BLH for {len(stations)} stations...")
    failed = []
    for i, row in enumerate(stations.itertuples(index=False)):
        try:
            extract_blh(row.station_id, row.lat, row.lon, args.start, args.end)
        except Exception as e:
            print(f"  FAILED (skipping, will retry at end): {row.station_id}: {e}")
            failed.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(stations)} done")

    if failed:
        print(f"\nRetrying {len(failed)} failed stations after a cooldown...")
        time.sleep(30)
        still_failed = []
        for row in failed:
            try:
                extract_blh(row.station_id, row.lat, row.lon, args.start, args.end)
            except Exception as e:
                still_failed.append((row.station_id, str(e)))
        if still_failed:
            print(f"Still failed after retry: {still_failed}")

    print("BLH extraction complete.")
