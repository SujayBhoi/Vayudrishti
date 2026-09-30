"""Waits for the GEE Restricted Mode block to lift (checking periodically,
not aggressively re-hitting the API while restricted - see
PROJECT_DOCUMENTATION.md section 4.1.1), then automatically resumes the
satellite extraction at the reduced-concurrency, throttled settings.
"""

import os
import subprocess
import sys
import time

CHECK_INTERVAL_SECONDS = 300  # 5 minutes - not aggressive
MAX_CHECKS = 48  # give up after ~4 hours of checking, don't loop forever silently

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(THIS_DIR)
TRIAL02_ROOT = os.path.dirname(V3_ROOT)
sys.path.insert(0, os.path.join(TRIAL02_ROOT, "data_pipeline"))


def restriction_lifted(project: str) -> bool:
    try:
        import ee
        ee.Initialize(project=project)
        # a trivial, cheap real call to confirm the API actually responds, not just Initialize()
        ee.Number(1).getInfo()
        return True
    except Exception as e:
        print(f"[check] still restricted: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="isro-bah-2026", help="Project to watch/resume against. Pass a fresh project ID to watch a new, non-restricted project instead.")
    args = parser.parse_args()

    for i in range(MAX_CHECKS):
        print(f"[check {i+1}/{MAX_CHECKS}] testing GEE access on project={args.project} ...")
        if restriction_lifted(args.project):
            print(f"GEE access confirmed restored on {args.project}. Resuming extraction at reduced concurrency (2 workers, 3s call delay).")
            script = os.path.join(THIS_DIR, "fetch_overpass_windowed_satellite.py")
            subprocess.run([sys.executable, script, "--start", "2019-01-01", "--end", "2026-01-01", "--workers", "2", "--call-delay", "3.0", "--project", args.project], check=False)
            return
        time.sleep(CHECK_INTERVAL_SECONDS)
    print("Gave up after MAX_CHECKS - GEE still restricted. Not retrying further automatically.")


if __name__ == "__main__":
    main()
