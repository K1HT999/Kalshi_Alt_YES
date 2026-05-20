# Data dependencies

The scripts in this repo expect three external data sources to live under
`data/` and `reports/` at the repository root. None of them ship with the
repo — they are either publicly downloadable (ASOS, HRRR) or must be
collected by the user (Kalshi snapshots).


## 1. Iowa Mesonet ASOS observations (KORD)

**Where**: <https://mesonet.agron.iastate.edu/request/download.phtml>

5-minute METAR observations from the official KORD weather station. Fields used:

- `valid` (UTC timestamp, ISO-8601)
- `tmpf` (temperature, °F)
- `dwpf` (dewpoint, °F)
- `relh` (relative humidity, %)
- `sknt` (wind speed, knots)
- `mslp` (sea-level pressure, mb)

Expected layout:
```
data/observed_weather/iem_asos/KORD_<YYYYMM>.json
```

Sample row:
```json
{"valid": "2025-01-15T18:55:00Z", "tmpf": 32.0, "dwpf": 18.0,
 "relh": 56.0, "sknt": 9, "mslp": 1019.2}
```

## 2. NOAA HRRR point forecasts (Chicago)

**Where**: <https://rapidrefresh.noaa.gov/hrrr/>

Hourly run, 18-hour or 48-hour forecasts of 2-m air temperature. Extracted
at Chicago coordinates and serialised one row per (run_time, forecast_hour)
pair.

Expected layout:
```
data/backfill_chicago/chicago_hrrr_points.jsonl
```

Each line:
```json
{"variable": "max_temp_f",
 "status": "ok",
 "run_time": "2025-01-15T12:00:00Z",
 "valid_time": "2025-01-15T22:00:00Z",
 "value": 38.4}
```

You can build this file yourself from raw HRRR GRIB output with a tool such as
[`herbie`](https://herbie.readthedocs.io/) or
[`xarray + cfgrib`](https://github.com/ecmwf/cfgrib). Point extraction is
straightforward; the only catch is to write one record per `(run_time,
forecast_hour)` rather than one per HRRR run.

### 2b. HRRR extras (Mode B only)

For Mode B (regime-modulated σ) you also need nine additional HRRR variables
at the same `(run_time, forecast_hour)` granularity:

| Variable | Field name in extras JSONL |
|---|---|
| 10-m wind speed | `wind_speed_10m_m_s` (also `wind_u_10m`, `wind_v_10m`) |
| Surface wind gust | `wind_gust_m_s` |
| Total cloud cover | `cloud_cover_pct` (also low/mid/high bands) |
| Surface pressure | `surface_pressure_pa` |
| Precipitation rate | `precip_rate_kg_m2_s` |
| Accumulated precipitation | `precip_in` |
| Downward shortwave radiation | `downward_shortwave_w_m2` |
| Planetary boundary-layer height | `pbl_height_m` |
| 925-mb wind / temp / dewpoint | `wind_u_925mb`, `wind_v_925mb`, `temp_925mb_f`, `dewpoint_925mb_f` |



## 3. Kalshi KXHIGHCHI snapshot history

**Where**: collected via the Kalshi public API. The series is
`KXHIGHCHI` (Chicago daily high temperature). A 60-minute polling cadence
across the contract universe is sufficient.

Expected layout:
```
data/backfill_chicago/chicago_60m.jsonl
```

Each line is one (ticker, snapshot_time) observation:

```json
{
  "as_of":               "2026-01-30T16:00:00+00:00",
  "ticker":              "KXHIGHCHI-26JAN31-T29",
  "event_ticker":        "KXHIGHCHI-26JAN31",
  "status":              "finalized",
  "close_time":          "2026-02-01T05:59:00Z",
  "yes_bid":             0.01,
  "yes_ask":             0.02,
  "volume":              23.0,
  "threshold_value":     29.0,
  "threshold_direction": "greater"
}
```

This snapshot file is not included. Building it requires running a polling
client against the Kalshi API for the duration of interest. The repo's
strategy scripts will pick up the file the moment it exists at the path
above.

## 4. Observed-high panel

A simple CSV joining each calendar date to that day's observed daily high
temperature (the settlement value for KXHIGHCHI).


Required columns:
- `observed_weather_valid_date` (ISO date)
- `observed_max_temp_f`

This file is also produced by the LongShot research workspace internal
pipeline and is not redistributed here, but recreating it from public
NWS / KORD daily summary data is straightforward.

---

