"""Permutation test on the COMBINED (12h+14h dedup union) book for a variant.

For each iteration:
  1. Shuffle expected_high_f at both snapshots independently
  2. Run the variant walk-forward at each snapshot
  3. Combine via 12h-first dedup
  4. Record the combined PnL and ROI
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import kalshi_yes_profitmaxx as pm
import kalshi_yes_profitmaxx_v2 as pmv2
from phase2_permutation import VARIANT_FNS, run_variant, shuffle_predictions


def combine(t12: pd.DataFrame, t14: pd.DataFrame) -> pd.DataFrame:
    if t12.empty: return t14
    if t14.empty: return t12
    keys12 = set(zip(t12["date"].astype(str), t12["ticker"]))
    mask14 = ~t14.apply(lambda r: (str(r["date"]), r["ticker"]) in keys12, axis=1)
    return pd.concat([t12, t14[mask14]], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANT_FNS.keys()), required=True)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--sizing", default="flat")
    args = p.parse_args()

    print(f"Loading v2 universe at 12h and 14h …")
    df12 = pmv2.build_universe_v2(12)
    df14 = pmv2.build_universe_v2(14)
    print(f"  12h: {len(df12)} contracts | 14h: {len(df14)} contracts", flush=True)

    # Actual combined book
    t12 = run_variant(df12, args.variant, args.sizing)
    t14 = run_variant(df14, args.variant, args.sizing)
    actual = combine(t12, t14)
    n_a = len(actual)
    pnl_a = float(actual["pnl_cents"].sum()) if n_a else 0
    cost_a = float(actual["cost_cents"].sum()) if n_a else 0
    roi_a = pnl_a / cost_a * 100 if cost_a > 0 else 0
    print(f"Actual COMBINED ({args.variant}): n={n_a}  PnL={pnl_a:+.1f}c  ROI={roi_a:+.1f}%", flush=True)

    print(f"\nRunning B={args.permutations} combined permutations …", flush=True)
    rng = np.random.default_rng(42)
    t0 = time.time()
    null_pnls, null_rois = [], []
    for b in range(args.permutations):
        d12_shuf = shuffle_predictions(df12, rng)
        d14_shuf = shuffle_predictions(df14, rng)
        p12 = run_variant(d12_shuf, args.variant, args.sizing)
        p14 = run_variant(d14_shuf, args.variant, args.sizing)
        cb = combine(p12, p14)
        if cb.empty:
            null_pnls.append(0); null_rois.append(0)
        else:
            p_ = float(cb["pnl_cents"].sum())
            c_ = float(cb["cost_cents"].sum())
            null_pnls.append(p_)
            null_rois.append(p_ / c_ * 100 if c_ > 0 else 0)
        if (b + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (b + 1) * (args.permutations - b - 1)
            print(f"  [{b+1:>3}/{args.permutations}]  elapsed={elapsed:>4.0f}s  eta={eta:>4.0f}s", flush=True)

    null_pnls = np.array(null_pnls)
    null_rois = np.array(null_rois)
    p_pnl = float((null_pnls >= pnl_a).mean())
    p_roi = float((null_rois >= roi_a).mean())

    print()
    print(f"=== COMBINED-BOOK PERMUTATION RESULT for variant={args.variant} ===")
    print(f"Actual:  n={n_a}  PnL={pnl_a:+.1f}c  ROI={roi_a:+.1f}%")
    print(f"Null PnL: mean={null_pnls.mean():+.1f}c  median={np.median(null_pnls):+.1f}c  "
          f"q05={np.percentile(null_pnls,5):+.1f}c  q95={np.percentile(null_pnls,95):+.1f}c")
    print(f"Null ROI: mean={null_rois.mean():+.1f}%  median={np.median(null_rois):+.1f}%  "
          f"q05={np.percentile(null_rois,5):+.1f}%  q95={np.percentile(null_rois,95):+.1f}%")
    print(f"p-value (PnL): {p_pnl:.4f}")
    print(f"p-value (ROI): {p_roi:.4f}")

    # Save null distribution
    out = _HERE.parent / "appendix" / f"permutation_null_v2_{args.variant}_combined.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"null_pnl_cents": null_pnls, "null_roi_pct": null_rois}).to_csv(out, index=False)
    print(f"Null distribution saved → {out}")


if __name__ == "__main__":
    main()
