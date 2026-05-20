"""
Download HRRR byte ranges from NOAA AWS for 17 wind / precip / baro / bonus
variables, decode in memory (no GRIB cache on disk), and extract Chicago point
values directly into chicago_hrrr_points_extras.jsonl.

Strategy
--------
For each (run_time, forecast_hour) already in chicago_hrrr_points.jsonl:
  1.  Fetch the .idx file from S3
  2.  Identify byte ranges for our 17 target fields
  3.  Range-request each (using HTTP threading)
  4.  Decode each GRIB byte buffer with xarray + cfgrib in memory
  5.  Extract Chicago point, write to JSONL
  6.  Discard the GRIB bytes (no disk cache)

Idempotent + resumable: skips (run, fh, field) tuples already present.

Usage
-----
    py research/sync_extra_hrrr.py
    py research/sync_extra_hrrr.py --max-runs 5         # smoke test
    py research/sync_extra_hrrr.py --workers 16          # threads
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests

_HERE = Path(__file__).resolve().parent
_LS   = _HERE.parent

HRRR_AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HRRR_PRODUCT  = "wrfsfcf"

ORIG_POINTS = _LS / "data" / "backfill_chicago" / "chicago_hrrr_points.jsonl"
EXTRAS_PATH = _LS / "data" / "backfill_chicago" / "chicago_hrrr_points_extras.jsonl"

CHI_LAT = 41.9742
CHI_LON = -87.9073
BOX_RADIUS = 2

# Field regexes (mirrors kalshi_longshot_strategy/hrrr.py).  Each maps an
# IDX-line regex pattern to (output variable, units, transform).
FIELD_SPECS: dict[str, dict[str, Any]] = {
    "UGRD_10m":        {"pat": r":UGRD:10 m above ground:",     "variable": "wind_u_10m",          "units": "m/s",       "transform": None},
    "VGRD_10m":        {"pat": r":VGRD:10 m above ground:",     "variable": "wind_v_10m",          "units": "m/s",       "transform": None},
    "WIND_10m":        {"pat": r":WIND:10 m above ground:",     "variable": "wind_speed_10m_m_s",  "units": "m/s",       "transform": None},
    "GUST_surface":    {"pat": r":GUST:surface:",                "variable": "wind_gust_m_s",       "units": "m/s",       "transform": None},
    "PRATE_surface":   {"pat": r":PRATE:surface:",               "variable": "precip_rate_kg_m2_s", "units": "kg/m^2/s",  "transform": None},
    "APCP_surface":    {"pat": r":APCP:surface:",                "variable": "precip_in",           "units": "in",        "transform": None},
    "PRES_surface":    {"pat": r":PRES:surface:",                "variable": "surface_pressure_pa", "units": "Pa",        "transform": None},
    "TCDC_atmosphere": {"pat": r":TCDC:(?:entire atmosphere|entire atmospheric column):", "variable": "cloud_cover_pct", "units": "%", "transform": None},
    "LCDC_low":        {"pat": r":LCDC:low cloud layer:",        "variable": "low_cloud_cover_pct", "units": "%",         "transform": None},
    "MCDC_middle":     {"pat": r":MCDC:middle cloud layer:",     "variable": "mid_cloud_cover_pct", "units": "%",         "transform": None},
    "HCDC_high":       {"pat": r":HCDC:high cloud layer:",       "variable": "high_cloud_cover_pct","units": "%",         "transform": None},
    "DSWRF_surface":   {"pat": r":DSWRF:surface:",               "variable": "downward_shortwave_w_m2", "units": "W/m^2", "transform": None},
    "HPBL_surface":    {"pat": r":HPBL:surface:",                "variable": "pbl_height_m",        "units": "m",         "transform": None},
    "DPT_2m":          {"pat": r":DPT:2 m above ground:",        "variable": "dewpoint_f",          "units": "F",         "transform": "k_to_f"},
    "DPT_925mb":       {"pat": r":DPT:925 mb:",                  "variable": "dewpoint_925mb_f",    "units": "F",         "transform": "k_to_f"},
    "TMP_925mb":       {"pat": r":TMP:925 mb:",                  "variable": "temp_925mb_f",        "units": "F",         "transform": "k_to_f"},
    "UGRD_925mb":      {"pat": r":UGRD:925 mb:",                 "variable": "wind_u_925mb",        "units": "m/s",       "transform": None},
    "VGRD_925mb":      {"pat": r":VGRD:925 mb:",                 "variable": "wind_v_925mb",        "units": "m/s",       "transform": None},
}

# Compile patterns
for k, v in FIELD_SPECS.items():
    v["regex"] = re.compile(v["pat"], re.IGNORECASE)


def k_to_f(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


TRANSFORMS = {"k_to_f": k_to_f}

_write_lock = threading.Lock()
_session_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_session_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        s.mount("https://", adapter)
        _session_local.session = s
    return _session_local.session


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_urls(run_iso: str, forecast_hour: int) -> tuple[str, str]:
    import pandas as pd
    run = pd.Timestamp(run_iso)
    date_part = run.strftime("%Y%m%d")
    cycle_part = run.strftime("%H")
    base = f"{HRRR_AWS_BASE}/hrrr.{date_part}/conus/hrrr.t{cycle_part}z.{HRRR_PRODUCT}{forecast_hour:02d}.grib2"
    return base, f"{base}.idx"


def fetch_idx(idx_url: str, timeout: float = 20.0, retries: int = 3) -> list[dict[str, Any]] | None:
    """Fetch IDX file. Returns list of {message, byte_offset, byte_end, line}."""
    for attempt in range(retries):
        try:
            r = get_session().get(idx_url, timeout=timeout)
            r.raise_for_status()
            break
        except Exception:
            if attempt + 1 == retries:
                return None
            time.sleep(1.0 * (attempt + 1))
    text = r.text.strip()
    if not text:
        return None
    lines = text.split("\n")
    rows: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            msg = int(parts[0])
            off = int(parts[1])
        except ValueError:
            continue
        rows.append({"message": msg, "byte_offset": off, "byte_end": -1, "line": line})
    for i in range(len(rows) - 1):
        rows[i]["byte_end"] = rows[i + 1]["byte_offset"] - 1
    return rows


def fetch_grib_range(grib_url: str, byte_start: int, byte_end: int,
                     timeout: float = 60.0, retries: int = 3) -> bytes | None:
    rng = f"bytes={byte_start}-{byte_end}" if byte_end >= 0 else f"bytes={byte_start}-"
    for attempt in range(retries):
        try:
            r = get_session().get(grib_url, headers={"Range": rng}, timeout=timeout)
            if r.status_code in (200, 206):
                return r.content
            if 400 <= r.status_code < 500:
                return None
        except Exception:
            pass
        time.sleep(1.0 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# In-memory GRIB decode
# ---------------------------------------------------------------------------

def decode_chicago_point(grib_bytes: bytes) -> dict[str, Any] | None:
    """Decode raw GRIB bytes and extract value at Chicago."""
    import xarray as xr  # lazy import

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        tmp_path = f.name
    try:
        ds = xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        try:
            data_vars = list(ds.data_vars)
            if not data_vars:
                return None
            da = ds[data_vars[0]].squeeze()
            lat_grid = np.asarray(ds["latitude"].values, dtype=float)
            lon_grid = np.asarray(ds["longitude"].values, dtype=float)
            values   = np.squeeze(np.asarray(da.values, dtype=float))
            raw_units = str(da.attrs.get("GRIB_units") or da.attrs.get("units") or "")
            data_var_name = str(da.name)

            lon_t = CHI_LON if CHI_LON >= 0 else CHI_LON + 360.0
            dlat = lat_grid - CHI_LAT
            dlon = lon_grid - lon_t
            idx = np.unravel_index(int(np.argmin(dlat * dlat + dlon * dlon)), lat_grid.shape)
            y, x = int(idx[0]), int(idx[1])
            raw = float(values[y, x])

            y_lo = max(0, y - BOX_RADIUS); y_hi = min(values.shape[0], y + BOX_RADIUS + 1)
            x_lo = max(0, x - BOX_RADIUS); x_hi = min(values.shape[1], x + BOX_RADIUS + 1)
            box  = values[y_lo:y_hi, x_lo:x_hi].ravel()
            box  = box[~np.isnan(box)]
            if box.size == 0:
                return None

            return {
                "raw_value":     raw,
                "raw_units":     raw_units,
                "data_var":      data_var_name,
                "grid_lat":      float(lat_grid[y, x]),
                "grid_lon":      float(lon_grid[y, x]),
                "grid_y":        y,
                "grid_x":        x,
                "box_radius_grid_cells": BOX_RADIUS,
                "box_grid_count": int(box.size),
                "box_raw_mean":  float(box.mean()),
                "box_raw_min":   float(box.min()),
                "box_raw_max":   float(box.max()),
            }
        finally:
            ds.close()
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run(run_iso: str, forecast_hour: int, seen: set,
                out_file) -> dict[str, int]:
    """Download + decode + write all 17 fields for one (run, fh)."""
    import pandas as pd
    grib_url, idx_url = s3_urls(run_iso, forecast_hour)
    stats = {"fields_done": 0, "fields_skipped": 0, "fields_err": 0}

    # Skip if all 17 fields already done for this (run, fh)
    pending_fields = [f for f in FIELD_SPECS if (run_iso, forecast_hour, f) not in seen]
    if not pending_fields:
        stats["fields_skipped"] = len(FIELD_SPECS)
        return stats

    idx_rows = fetch_idx(idx_url)
    if idx_rows is None:
        stats["fields_err"] = len(pending_fields)
        return stats

    # Find byte ranges for each pending field
    field_to_byterange: dict[str, tuple[int, int]] = {}
    for field in pending_fields:
        spec = FIELD_SPECS[field]
        for row in idx_rows:
            if spec["regex"].search(row["line"]):
                field_to_byterange[field] = (row["byte_offset"], row["byte_end"])
                break

    # Fetch each byte range, decode, write
    valid_iso = (pd.Timestamp(run_iso) + pd.Timedelta(hours=forecast_hour)).isoformat()
    for field, (b_start, b_end) in field_to_byterange.items():
        grib_bytes = fetch_grib_range(grib_url, b_start, b_end)
        if grib_bytes is None:
            stats["fields_err"] += 1
            continue
        point = decode_chicago_point(grib_bytes)
        if point is None:
            stats["fields_err"] += 1
            continue

        spec = FIELD_SPECS[field]
        transform = TRANSFORMS.get(spec["transform"]) if spec["transform"] else None
        value      = transform(point["raw_value"])    if transform else point["raw_value"]
        box_mean   = transform(point["box_raw_mean"]) if transform else point["box_raw_mean"]
        box_min    = transform(point["box_raw_min"])  if transform else point["box_raw_min"]
        box_max    = transform(point["box_raw_max"])  if transform else point["box_raw_max"]

        row = {
            "value":          value,
            "units":          spec["units"],
            "variable":       spec["variable"],
            "raw_value":      point["raw_value"],
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
            "box_raw_mean":   point["box_raw_mean"],
            "box_raw_min":    point["box_raw_min"],
            "box_raw_max":    point["box_raw_max"],
            "data_var":       point["data_var"],
            "status":         "ok",
            "city":           "chicago",
            "city_alias":     "chicago",
            "lat":            CHI_LAT,
            "lon":            CHI_LON,
            "run_time":       run_iso,
            "forecast_hour":  forecast_hour,
            "valid_time":     valid_iso,
            "field_name":     field,
        }
        with _write_lock:
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()
        seen.add((run_iso, forecast_hour, field))
        stats["fields_done"] += 1

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-runs",    type=int, default=None,
                        help="Cap to first N (run, fh) pairs.")
    parser.add_argument("--slice-start", type=int, default=0,
                        help="Start index into target list (for parallel workers).")
    parser.add_argument("--slice-end",   type=int, default=None,
                        help="Exclusive end index into target list.")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Internal worker threads (keep at 1 — eccodes is not thread-safe).")
    parser.add_argument("--output",      type=str, default=str(EXTRAS_PATH))
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing extras to find what's already done
    seen: set = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        seen.add((r["run_time"], int(r["forecast_hour"]), r["field_name"]))
                except Exception:
                    continue
    print(f"  Resume: {len(seen)} (run, fh, field) tuples already extracted")

    # Build target list from chicago_hrrr_points.jsonl
    target_pairs: set = set()
    with ORIG_POINTS.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok":
                target_pairs.add((r["run_time"], int(r["forecast_hour"])))
    target_list = sorted(target_pairs)
    if args.max_runs:
        target_list = target_list[: args.max_runs]
    # Slice (for parallel subprocesses)
    end = args.slice_end if args.slice_end is not None else len(target_list)
    target_list = target_list[args.slice_start : end]
    print(f"  Target: {len(target_list)} (run, fh) pairs  "
          f"(slice [{args.slice_start}:{end}])")
    print(f"  Fields: {len(FIELD_SPECS)} per pair")
    print(f"  Workers: {args.workers}")
    print()

    n_done_fields = sum(1 for _ in seen)
    expected_fields = len(target_list) * len(FIELD_SPECS)
    print(f"  Already done: {n_done_fields:,} fields ({n_done_fields/expected_fields*100:.1f}%)")
    print()

    t0 = time.time()
    completed_pairs = 0
    err_pairs = 0
    total_fields_done = 0

    with out_path.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(process_run, run_iso, fh, seen, out): (run_iso, fh)
                for run_iso, fh in target_list
            }
            for fut in as_completed(futures):
                run_iso, fh = futures[fut]
                try:
                    stats = fut.result()
                    total_fields_done += stats["fields_done"]
                    if stats["fields_err"] > 0:
                        err_pairs += 1
                except Exception as exc:
                    err_pairs += 1
                    print(f"    EXC {run_iso} f{fh:02d}: {exc}", flush=True)

                completed_pairs += 1
                if completed_pairs % 50 == 0:
                    elapsed = time.time() - t0
                    rate = completed_pairs / elapsed
                    eta = (len(target_list) - completed_pairs) / max(rate, 0.001)
                    print(f"  [{completed_pairs:>5}/{len(target_list)}]  "
                          f"fields={total_fields_done:>6}  errs={err_pairs:>3}  "
                          f"elapsed={elapsed:>5.0f}s  rate={rate:>4.1f}/s  eta={eta:>5.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print()
    print(f"Sync complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Pairs processed : {completed_pairs}")
    print(f"  Pairs with errors: {err_pairs}")
    print(f"  New fields written: {total_fields_done}")
    print(f"  Output : {out_path}")


if __name__ == "__main__":
    main()
