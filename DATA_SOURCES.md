# Data Sources

## Ground-station AQI (CPCB)

**Source:** CPCB CAAQM (Continuous Ambient Air Quality Monitoring) data, originally
published on CPCB's official Central Control Room (CCR) portal
(https://app.cpcbccr.com/ccr/), aggregated and redistributed as:

> Abhishek S. Jha, *"Time Series Air Quality Data of India (2010-2023)"*, Kaggle,
> https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023,
> CC-BY-NC-SA-4.0.

Raw file: `data/raw/cpcb_kaggle/time-series-air-quality-data-of-india-2010-2023.zip`
(454 station CSVs + `stations_info.csv`, hourly PM2.5/PM10/NO2/SO2/CO/O3/NH3 per
station). We compute AQI ourselves from these raw pollutant concentrations using the
official Indian National AQI (CPCB 2014) breakpoint methodology — see
`src/cpcb_process.py`.

### Why not pulled live from an official API

Two official routes were evaluated and both are genuinely unable to serve bulk
historical data at the scale this project needs (13 cities x 5+ years):

1. **data.gov.in CPCB API** (resource `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`) —
   tested directly. Every record returned carries `last_update` equal to the current
   date; the schema has no date-range parameter. This is architecturally a live
   snapshot feed, not a historical archive, regardless of API key or registration.
2. **CPCB CCR portal Advance Search** (`app.cpcbccr.com/ccr`) — captcha-gated,
   interactive-only, and cap the request at 1 week of data per station per pollutant.
   No documented bulk multi-station/multi-year export exists. The only people known
   to pull larger volumes did so by reverse-engineering internal endpoints and
   scraping at throttled rates — which would trade "verifiably official" provenance
   for automation, the opposite of what this project needs.

Given that, the Kaggle-compiled CCR dataset is the most defensible option: it is
CPCB-sourced (not synthetic or third-party estimated), publicly citable, and covers
our required window with real per-station hourly measurements.

## Satellite / meteorological data (Google Earth Engine)

All pulled live via `earthengine-api`, not downloaded manually:

| Variable | GEE Collection ID |
|---|---|
| MODIS AOD (1km) | `MODIS/061/MCD19A2_GRANULES` |
| Sentinel-5P TROPOMI HCHO | `COPERNICUS/S5P/OFFL/L3_HCHO` |
| MODIS active fire | `FIRMS` (MODIS-based; VIIRS fire collections in GEE only cover Sept 2023 onward, so are not usable for our 2018-2023 window) |
| ERA5-Land meteorology | `ECMWF/ERA5_LAND/DAILY_AGGR` |

**Boundary layer height** is not present in any GEE-mirrored ERA5 product, so it is
pulled separately from the [Open-Meteo Historical Weather API](https://open-meteo.com/)
(free, no key, ERA5-based reanalysis).

See `src/gee_extract.py` and `src/openmeteo_extract.py`.
