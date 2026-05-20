"""
Permutation test for v2 variants.

Reuses the v2 variant walk-forward functions; on each of B iterations,
shuffles `expected_high_f` across dates within the universe and re-runs
the same variant. Reports p-values for actual PnL and ROI vs the null.

Usage
-----
    py research/phase2_permutation.py --variant ml --snapshot-h 12 --permutations 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import kalshi_yes_profitmaxx as pm
import kalshi_yes_profitmaxx_v2 as pmv2


VARIANT_FNS = {
    "baseline":   lambda df, sz: pm.walkforward(df, sz)[1],
    "ml":         pmv2.ml_walkforward,
    "sigma_mod":  pmv2.sigma_mod_walkforward,
    "stake_cap":  pmv2.stake_cap_walkforward,
}


def run_variant(df: pd.DataFrame, variant: str, sizing: str) -> pd.DataFrame:
    fn = VARIANT_FNS[variant]
    return fn(df, sizing)


def shuffle_predictions(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    uniq_dates = df["date"].unique()
    perm = rng.permutation(uniq_dates)
    date_map = dict(zip(uniq_dates, perm))
    pred_by_dt = df.drop_duplicates("date").set_index("date")["expected_high_f"]
    df["expected_high_f"] = df["date"].map(date_map).map(pred_by_dt).to_numpy()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANT_FNS.keys()), required=True)
    parser.add_argument("--snapshot-h", type=int, required=True)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--sizing", default="flat")
    args = parser.parse_args()

    print(f"Loading v2 universe at h={args.snapshot_h}…", flush=True)
    df = pmv2.build_universe_v2(args.snapshot_h)
    print(f"  {len(df)} contracts", flush=True)

    # Actual
    actual = run_variant(df, args.variant, args.sizing)
    n_a = len(actual)
    pnl_a = float(actual["pnl_cents"].sum()) if n_a else 0
    cost_a = float(actual["cost_cents"].sum()) if n_a else 0
    roi_a = pnl_a / cost_a * 100 if cost_a > 0 else 0
    print(f"Actual: n={n_a}  PnL={pnl_a:+.1f}c  ROI={roi_a:+.1f}%", flush=True)

    print(f"\nRunning permutation test: B={args.permutations}…", flush=True)
    rng = np.random.default_rng(42)
    t0 = time.time()
    null_pnls, null_rois = [], []
    for b in range(args.permutations):
        d_shuf = shuffle_predictions(df, rng)
        t = run_variant(d_shuf, args.variant, args.sizing)
        if t.empty:
            null_pnls.append(0); null_rois.append(0)
        else:
            pnl = float(t["pnl_cents"].sum())
            cost = float(t["cost_cents"].sum())
            null_pnls.append(pnl)
            null_rois.append(pnl / cost * 100 if cost > 0 else 0)
        if (b + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (b + 1) * (args.permutations - b - 1)
            print(f"  [{b+1:>3}/{args.permutations}] elapsed={elapsed:>4.0f}s eta={eta:>4.0f}s", flush=True)

    null_pnls = np.array(null_pnls)
    null_rois = np.array(null_rois)
    p_pnl = float((null_pnls >= pnl_a).mean())
    p_roi = float((null_rois >= roi_a).mean())

    print()
    print(f"Actual: n={n_a}  PnL={pnl_a:+.1f}c  ROI={roi_a:+.1f}%")
    print(f"Null  : PnL mean={null_pnls.mean():+.1f}c  median={np.median(null_pnls):+.1f}c  q95={np.percentile(null_pnls,95):+.1f}c")
    print(f"        ROI mean={null_rois.mean():+.1f}%  median={np.median(null_rois):+.1f}%  q95={np.percentile(null_rois,95):+.1f}%")
    print(f"p-value (PnL): {p_pnl:.4f}")
    print(f"p-value (ROI): {p_roi:.4f}")


if __name__ == "__main__":
    main()
