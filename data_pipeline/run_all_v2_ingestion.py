"""Driver: kicks off every v2 point-extraction batch export task (all 417
stations, 2018-01-01 to 2026-01-01) in one go. Per the quota analysis, this
is the "cheap" half of the plan's GEE work, so it runs at full density with
no station/date cap. Tasks queue on GEE's side (this project appears to run
one batch task at a time) and complete over the following hours - this
script only *starts* them, it doesn't wait.

Run once; each fetch_*.py script's underlying export tasks are independent,
so if one needs re-running it can be re-invoked on its own.
"""

import subprocess
import sys

START = "2018-01-01"
END = "2026-01-01"

SCRIPTS = [
    ["fetch_modis_aod.py"],
    ["fetch_tropomi.py"],
    ["fetch_era5.py"],
    ["fetch_fire.py"],
]

if __name__ == "__main__":
    for script in SCRIPTS:
        cmd = [sys.executable, *script, "--start", START, "--end", END]
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print("\nAll v2 GEE export tasks started (queued server-side; check status via ee.data.getTaskStatus).")
    print("BLH (fetch_openmeteo_blh.py) is a separate plain REST loop, not a GEE task - run it independently.")
