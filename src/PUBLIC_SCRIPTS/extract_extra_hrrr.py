"""
Extract Chicago point values for wind / precipitation / barometric (and a few
bonus) HRRR variables from the local GRIB cache.

Why this script exists
----------------------
The existing `chicago_hrrr_points.jsonl` only contains `max_temp_f` (a `TMP_2m`
extraction). The GRIB byte-range cache under `data/hrrr_cache/` already holds
every other field we want — but the prior extraction pipeline only requested
Chicago points for temperature. This script walks the cache, extracts the
missing variables at Chicago coordinates, and writes a sidecar JSONL with the
same schema as the original.

Usage
-----
    py research/extract_extra_hrrr.py                       # full cache
    py research/extract_extra_hrrr.py --max-dates 5         # quick smoke test
    py research/extract_extra_hrrr.py --output some.jsonl   # custom path

The default output is `data/backfill_chicago/chicago_hrrr_points_extras.jsonl`.
The script is idempotent; existing rows are skipped on re-runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_LS   = _HERE.parent

CACHE_DIR     = _LS / "data" / "hrrr_cache"
DEFAULT_OUT   = _LS / "data" / "backfill_chicago" / "chicago_hrrr_points_extras.jsonl"

# Chicago point of interest (matches the lat/lon used in chicago_hrrr_points.jsonl)
CHI_LAT = 41.9742
CHI_LON = -87.9073

# Fields to extract.  Maps GRIB field filename prefix → output variable
# name (mirrors the conventions in `kalshi_longshot_strategy/hrrr.py`).
FIELD_MAP: dict[str, dict[str, Any]] = {
    "UGRD_10m":        {"variable": "wind_u_10m",         "units": "m/s",     "transform": None},
    "VGRD_10m":        {"variable": "wind_v_10m",         "units": "m/s",     "transform": None},
    "WIND_10m":        {"variable": "wind_speed_10m_m_s", "units": "m/s",     "transform": None},
    "GUST_surface":    {"variable": "wind_gust_m_s",      "units": "m/s",     "transform": None},
    "PRATE_surface":   {"variable": "precip_rate_kg_m2_s","units": "kg/m^2/s","transform": None},
    "APCP_surface":    {"variable": "precip_in",          "units": "in",      "transform": None},
    "PRES_surface":    {"variable": "surface_pressure_pa","units": "Pa",      "transform": None},
    "TCDC_atmosphere": {"variable": "cloud_cover_pct",    "units": "%",       "transform": None},
    "LCDC_low":        {"variable": "low_cloud_cover_pct","units": "%",       "transform": None},
    "MCDC_middle":     {"variable": "mid_cloud_cover_pct","units": "%",       "transform": None},
    "HCDC_high":       {"variable": "high_cloud_cover_pct","units": "%",      "transform": None},
    "DSWRF_surface":   {"variable": "downward_shortwave_w_m2","units": "W/m^2","transform": None},
    "HPBL_surface":    {"variable": "pbl_height_m",       "units": "m",       "transform": None},
    "DPT_2m":          {"variable": "dewpoint_f",         "units": "F",       "transform": "k_to_f"},
    "TMP_925mb":       {"variable": "temp_925mb_f",       "units": "F",       "transform": "k_to_f"},
    "UGRD_925mb":      {"variable": "wind_u_925mb",       "units": "m/s",     "transform": None},
    "VGRD_925mb":      {"variable": "wind_v_925mb",       "units": "m/s",     "transform": None},
    # TMP_2m intentionally omitted — already in chicago_hrrr_points.jsonl
}

BOX_RADIUS = 2   # grid cells around the point; matches the prior extraction


def k_to_f(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


TRANSFORMS = {
    "k_to_f": k_to_f,
}


# ---------------------------------------------------------------------------
# GRIB point extraction
# ---------------------------------------------------------------------------

def _open_grib(grib_path: Path):
    """Open a GRIB file via xarray + cfgrib. Imports are local so the rest of
    the script (manifest scanning, caching) works without cfgrib installed."""
    import xarray as xr
    return xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def extract_point(grib_path: Path, lat: float, lon: float,
                  box_radius: int = BOX_RADIUS) -> dict[str, Any] | None:
    """Return a point payload (value, box stats, grid coords) at the given
    (lat, lon).  Returns None on any failure."""
    try:
        ds = _open_grib(grib_path)
    except Exception as exc:
        return {"status": "error", "error": f"open_dataset: {exc}"}

    try:
        data_vars = list(ds.data_vars)
        if not data_vars:
            return {"status": "error", "error": "no data variables"}
        da = ds[data_vars[0]].squeeze()
        lat_grid = np.asarray(ds["latitude"].values, dtype=float)
        lon_grid = np.asarray(ds["longitude"].values, dtype=float)
        values   = np.squeeze(np.asarray(da.values, dtype=float))
        raw_units = str(da.attrs.get("GRIB_units") or da.attrs.get("units") or "")
        data_var_name = str(da.name)

        # HRRR longitudes are stored in [0, 360); normalize the target
        lon_target = lon if lon >= 0 else lon + 360.0

        # Nearest grid cell (great-circle distance approximation in deg)
        dlat = lat_grid - lat
        dlon = lon_grid - lon_target
        dist2 = dlat * dlat + dlon * dlon
        idx = np.unravel_index(int(np.argmin(dist2)), lat_grid.shape)

        y, x = int(idx[0]), int(idx[1])
        raw_value = float(values[y, x])

        # Box stats
        y_lo = max(0, y - box_radius)
        y_hi = min(values.shape[0], y + box_radius + 1)
        x_lo = max(0, x - box_radius)
        x_hi = min(values.shape[1], x + box_radius + 1)
        box  = values[y_lo:y_hi, x_lo:x_hi].ravel()
        box  = box[~np.isnan(box)]
        if box.size == 0:
            return {"status": "error", "error": "empty box"}

        return {
            "status":        "ok",
            "raw_value":     raw_value,
            "raw_units":     raw_units,
            "data_var":      data_var_name,
            "grid_lat":      float(lat_grid[y, x]),
            "grid_lon":      float(lon_grid[y, x]),
            "grid_y":        y,
            "grid_x":        x,
            "box_radius_grid_cells": box_radius,
            "box_grid_count": int(box.size),
            "box_raw_mean":  float(box.mean()),
            "box_raw_min":   float(box.min()),
            "box_raw_max":   float(box.max()),
        }
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Cache walking
# ---------------------------------------------------------------------------

def iter_grib_files(cache_dir: Path, max_dates: int | None = None):
    """Yield (date_str, run_utc_iso, forecast_hour, field_name, grib_path)
       for every raw GRIB in the cache that matches a FIELD_MAP key."""
    import pandas as pd
    dates = sorted(d.name for d in cache_dir.iterdir()
                   if d.is_dir() and d.name.isdigit() and len(d.name) == 8)
    if max_dates:
        dates = dates[:max_dates]

    for date_str in dates:
        date_dir = cache_dir / date_str
        for hh_dir in sorted(date_dir.iterdir()):
            if not hh_dir.is_dir() or not hh_dir.name.isdigit():
                continue
            run_hh = int(hh_dir.name)
            run_utc = pd.Timestamp(
                f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {run_hh:02d}:00:00",
                tz="UTC",
            ).isoformat()
            for fh_dir in sorted(hh_dir.iterdir()):
                if not fh_dir.is_dir() or not fh_dir.name.startswith("f"):
                    continue
                forecast_hour = int(fh_dir.name[1:])
                for grib_file in fh_dir.glob("*.grib2"):
                    # Field name is everything before the last "_<digits>.grib2"
                    stem  = grib_file.stem  # e.g. UGRD_10m_77
                    field = stem.rsplit("_", 1)[0]
                    if field not in FIELD_MAP:
                        continue
                    yield date_str, run_utc, forecast_hour, field, grib_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir",    type=str, default=str(CACHE_DIR))
    parser.add_argument("--output",       type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--max-dates",    type=int, default=None,
                        help="Cap to first N dates for smoke testing.")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Hard cap on number of GRIBs processed.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Idempotency: read existing JSONL to find (run_utc, fh, field) tuples already done
    seen: set[tuple] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add((r.get("run_time"), int(r.get("forecast_hour", -1)), r.get("field_name")))
                except Exception:
                    continue
    if seen:
        print(f"  Resume: {len(seen)} (run, fh, field) tuples already in {out_path.name}",
              flush=True)

    import pandas as pd
    n_total = n_done = n_err = n_skip = 0
    n_per_field = {}
    import time
    t0 = time.time()

    with out_path.open("a", encoding="utf-8") as out:
        for date_str, run_iso, fh, field, grib in iter_grib_files(cache_dir, args.max_dates):
            if (run_iso, fh, field) in seen:
                n_skip += 1
                continue
            n_total += 1
            point = extract_point(grib, CHI_LAT, CHI_LON)
            if point is None or point.get("status") != "ok":
                n_err += 1
                continue

            spec = FIELD_MAP[field]
            transform = TRANSFORMS.get(spec["transform"]) if spec["transform"] else None
            raw_v = point["raw_value"]
            value = transform(raw_v) if transform else raw_v
            box_raw_mean = point["box_raw_mean"]
            box_raw_min  = point["box_raw_min"]
            box_raw_max  = point["box_raw_max"]
            box_mean = transform(box_raw_mean) if transform else box_raw_mean
            box_min  = transform(box_raw_min)  if transform else box_raw_min
            box_max  = transform(box_raw_max)  if transform else box_raw_max

            # valid_time = run_time + forecast_hour
            valid_iso = (pd.Timestamp(run_iso) + pd.Timedelta(hours=fh)).isoformat()

            row = {
                "value":          value,
                "units":          spec["units"],
                "variable":       spec["variable"],
                "raw_value":      raw_v,
                "raw_units":      point["raw_units"],
                "grid_lat":       point["grid_lat"],
                "grid_lon":       point["grid_lon"],
                "grid_y":         point["grid_y"],
                "grid_x":         point["grid_x"],
                "box_radius_grid_cells": point["box_radius_grid_cells"],
                "box_grid_count": point["box_grid_count"],
                "box_mean":       box_mean,
                "box_min":        box_min,
                "box_max":        box_max,
                "box_raw_mean":   box_raw_mean,
                "box_raw_min":    box_raw_min,
                "box_raw_max":    box_raw_max,
                "data_var":       point["data_var"],
                "status":         "ok",
                "city":           "chicago",
                "city_alias":     "chicago",
                "lat":            CHI_LAT,
                "lon":            CHI_LON,
                "run_time":       run_iso,
                "forecast_hour":  fh,
                "valid_time":     valid_iso,
                "field_name":     field,
            }
            out.write(json.dumps(row) + "\n")
            n_done += 1
            n_per_field[field] = n_per_field.get(field, 0) + 1
            seen.add((run_iso, fh, field))

            if n_done % 500 == 0:
                elapsed = time.time() - t0
                print(f"  done {n_done:>6}  err {n_err:>4}  skip {n_skip:>5}  elapsed {elapsed:>5.0f}s", flush=True)

            if args.limit and n_done >= args.limit:
                break

    elapsed = time.time() - t0
    print(f"\nExtraction complete in {elapsed:.0f}s")
    print(f"  Attempted : {n_total}")
    print(f"  Wrote     : {n_done}")
    print(f"  Skipped   : {n_skip} (already in output)")
    print(f"  Errors    : {n_err}")
    print(f"  Per field :")
    for k, v in sorted(n_per_field.items()):
        print(f"    {k:>22}: {v:>5}")
    print(f"  Output    : {out_path}")


if __name__ == "__main__":
    main()
