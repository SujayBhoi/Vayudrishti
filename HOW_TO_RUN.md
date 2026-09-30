# How to run this on your own laptop

This is the VayuDrishti satellite AQI / HCHO hotspot dashboard. Follow
these steps in order — the whole thing takes about 5-10 minutes.

## 0. What you need first

- **Python 3.11, 3.12, or 3.13** installed. Check with:
  ```bash
  python --version
  ```
  If that fails, or shows Python 2.x, get Python from
  [python.org/downloads](https://www.python.org/downloads/) (any 3.11+
  build works) and re-open your terminal afterward.
- **~1.5 GB free disk space** (the project ships compressed; a few files
  expand when you decompress them in step 3).
- You do **not** need a GPU, CUDA, or Google Earth Engine access to run
  the dashboard — those are only needed if you want to retrain the models
  from scratch, which this guide doesn't cover.

## 1. Get the project onto your machine

Unzip the project folder (or `git clone` it) anywhere you like, then open
a terminal **inside that folder** — the one containing `streamlit_app.py`.

Windows (PowerShell or Command Prompt):
```bash
cd path\to\Final Project
```
Mac/Linux:
```bash
cd path/to/Final\ Project
```

## 2. Create a virtual environment and install packages

This keeps the project's Python packages separate from anything else on
your machine.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-app.txt
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-app.txt
```

You'll know it worked if your terminal prompt now starts with `(venv)`.
This installs only what's needed to *run* the dashboard (Streamlit, the
map library, pandas, etc.) — about 10 packages, a couple of minutes on a
normal connection.

*(There's also a `requirements.txt` with everything needed to re-run the
full satellite-data pipeline and retrain models — you don't need that
just to view the dashboard, and it pulls in some heavier packages that
are slower to install on Windows. Ignore it unless you specifically want
to regenerate the data.)*

## 3. Decompress the large data files (one-time)

Six files were too large for normal file transfer/GitHub in their raw
form, so they're shipped `.gz`-compressed. Run this once:

```bash
python decompress_data.py
```

You should see 6 lines like `Decompressing ... ` followed by `Done. You
can now run: streamlit run streamlit_app.py`. This step needs ~700MB of
extra disk space (the decompressed files sit alongside the compressed
originals — safe to delete the `.gz` files afterward if you want the
space back, though there's no need to).

## 4. Run it

```bash
streamlit run streamlit_app.py
```

Your browser should open automatically to `http://localhost:8501`. If it
doesn't, open that URL yourself. You'll see a page titled **VayuDrishti**
with three sections at the top: **Objective 1 · AQI**, **Objective 2 ·
HCHO Hotspots**, and **PM2.5 Deep-Dive**.

To stop the app later, go back to the terminal and press `Ctrl+C`.

## Troubleshooting

- **`streamlit: command not found` / `'streamlit' is not recognized`** —
  your virtual environment isn't activated. Re-run the `venv\Scripts\activate`
  (Windows) or `source venv/bin/activate` (Mac/Linux) line from step 2 —
  you need to do this every time you open a new terminal, not just once.
- **`ModuleNotFoundError: No module named 'X'`** — you likely ran
  `streamlit run` before `pip install -r requirements-app.txt` finished,
  or in a terminal where the venv isn't active. Re-check step 2.
- **A page loads but shows a red error about a missing `.csv` or `.pkl`
  file** — step 3 (`decompress_data.py`) wasn't run, or didn't finish.
  Re-run it and check it prints all 6 "Decompressing..." lines with no
  errors.
- **Port 8501 already in use** — something else is already running there
  (maybe a previous copy of this app). Run
  `streamlit run streamlit_app.py --server.port 8502` instead and open
  `http://localhost:8502`.
- **It's slow the first time you open a tab/map** — normal; some views
  compute an interpolated map on first load and Streamlit caches it after
  that. Subsequent visits to the same view are fast.

## What you're looking at

- **Objective 1 · AQI** — 5 machine-learning models predicting ground-level
  Air Quality Index from satellite data, compared on real 2025 held-out
  test data, plus an interactive map and per-station drill-down.
- **Objective 2 · HCHO Hotspots** — detecting formaldehyde (a
  biomass/stubble-burning proxy) hotspots across India's national grid,
  5 methods compared, plus wind+fire seasonal analysis.
- **PM2.5 Deep-Dive** — an additional analysis deriving PM2.5 (and from
  it, AQI) directly from satellite gas columns, including a continuous
  national map and a single-station "data trace" that shows the real
  numbers flowing through the model live.

The full technical writeup — methodology, every modeling decision, and
all results including the honest negative findings — is in
[`model_comparison_v3/PROJECT_DOCUMENTATION.md`](model_comparison_v3/PROJECT_DOCUMENTATION.md).
