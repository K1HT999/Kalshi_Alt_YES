"""
Spawn N parallel sync_extra_hrrr.py processes (each gets its own eccodes
context, so no thread-safety issues), monitor their progress, and merge their
output JSONLs into chicago_hrrr_points_extras.jsonl on completion.

Usage
-----
    py research/launch_parallel_hrrr_sync.py --workers 4
    py research/launch_parallel_hrrr_sync.py --workers 8 --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LS   = _HERE.parent
ORIG_POINTS = _LS / "data" / "backfill_chicago" / "chicago_hrrr_points.jsonl"
EXTRAS_PATH = _LS / "data" / "backfill_chicago" / "chicago_hrrr_points_extras.jsonl"
SLICE_DIR   = _LS / "data" / "backfill_chicago" / "_sync_slices"


def count_pairs(path: Path) -> int:
    pairs = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    pairs.add((r["run_time"], int(r["forecast_hour"])))
            except Exception:
                pass
    return len(pairs)


def merge_slices_into_main(slice_dir: Path, main_path: Path) -> int:
    """Append all slice JSONLs into the main extras file, dedup-on-write."""
    seen = set()
    if main_path.exists():
        with main_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add((r["run_time"], int(r["forecast_hour"]), r["field_name"]))
                except Exception:
                    continue
    added = 0
    with main_path.open("a", encoding="utf-8") as out:
        for slice_file in sorted(slice_dir.glob("slice_*.jsonl")):
            with slice_file.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        key = (r["run_time"], int(r["forecast_hour"]), r["field_name"])
                        if key in seen:
                            continue
                        out.write(line if line.endswith("\n") else line + "\n")
                        seen.add(key)
                        added += 1
                    except Exception:
                        continue
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel sync processes (each is single-threaded internally).")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Cap total target pairs (for testing).")
    parser.add_argument("--poll", type=float, default=60.0,
                        help="Seconds between progress checks.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    SLICE_DIR.mkdir(parents=True, exist_ok=True)

    n_pairs = count_pairs(ORIG_POINTS)
    if args.max_runs:
        n_pairs = min(n_pairs, args.max_runs)
    chunk = (n_pairs + args.workers - 1) // args.workers

    print(f"Total target pairs: {n_pairs}")
    print(f"Workers: {args.workers}")
    print(f"Per-worker slice size: {chunk}")
    print()

    if args.dry_run:
        for i in range(args.workers):
            start = i * chunk
            end = min(start + chunk, n_pairs)
            print(f"  worker {i}: slice [{start}:{end}] ({end - start} pairs) → {SLICE_DIR / f'slice_{i:02d}.jsonl'}")
        return

    # Launch workers
    procs = []
    for i in range(args.workers):
        start = i * chunk
        end = min(start + chunk, n_pairs)
        if start >= end:
            continue
        out_path = SLICE_DIR / f"slice_{i:02d}.jsonl"
        log_path = SLICE_DIR / f"slice_{i:02d}.log"
        cmd = [
            "py", "-3", "-u", str(_HERE / "sync_extra_hrrr.py"),
            "--slice-start", str(start),
            "--slice-end", str(end),
            "--output", str(out_path),
        ]
        if args.max_runs:
            cmd += ["--max-runs", str(args.max_runs)]
        log_file = log_path.open("w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(_LS))
        procs.append({"i": i, "start": start, "end": end, "proc": p, "log_path": log_path, "log_file": log_file, "out_path": out_path})
        print(f"  Launched worker {i}: PID={p.pid}  slice [{start}:{end}]  → {out_path.name}")

    print()
    print(f"Monitoring {len(procs)} workers (polling every {args.poll:.0f}s)…")
    t0 = time.time()
    while True:
        time.sleep(args.poll)
        alive = []
        total_rows = 0
        for w in procs:
            still_running = w["proc"].poll() is None
            try:
                rows = sum(1 for _ in w["out_path"].open()) if w["out_path"].exists() else 0
            except Exception:
                rows = 0
            total_rows += rows
            if still_running:
                alive.append(w["i"])
        elapsed = time.time() - t0
        print(f"  [t={elapsed:>5.0f}s] alive={len(alive)}/{len(procs)} workers={alive}  "
              f"total rows so far: {total_rows:,}", flush=True)
        if not alive:
            break

    print()
    print(f"All workers complete in {(time.time() - t0):.0f}s")
    for w in procs:
        w["log_file"].close()
        rc = w["proc"].returncode
        if rc != 0:
            print(f"  ⚠ worker {w['i']} exited with code {rc}; check {w['log_path'].name}")

    # Merge slices into main
    print()
    print(f"Merging slice files into {EXTRAS_PATH.name}…")
    added = merge_slices_into_main(SLICE_DIR, EXTRAS_PATH)
    print(f"  Added {added:,} new rows. Total file:")
    print(f"    Size: {EXTRAS_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"    Rows: {sum(1 for _ in EXTRAS_PATH.open()):,}")


if __name__ == "__main__":
    main()
