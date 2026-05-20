"""
Statistical tests for the calibrated ASOS+HRRR Kalshi YES strategy.

Reports
-------
1.  Universe stats (base rate, market-implied probability)
2.  Binomial test of realized WR vs unconditional base rate
3.  Binomial test of realized WR vs market-implied rate (avg yes_ask paid)
4.  Bootstrap CIs on PnL and ROI (trade-level resample, B = 5000)
5.  Block bootstrap CIs on monthly PnL (1-month blocks, B = 5000)
6.  Monthly t-test:  H0: mean monthly PnL = 0
7.  Wilcoxon signed-rank on monthly PnL (non-parametric robustness)
8.  Permutation test: shuffle the model predictions across dates and re-run
    the entire walk-forward N times.  Tests whether selection is driven by
    predictive power or by selection of cheap contracts alone.
9.  Direction split (greater vs less)
10. Calibration goodness-of-fit (predicted vs realised, by probability bin)

Outputs are written to appendix/ as CSVs and a single summary JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_LS   = _HERE.parent
sys.path.insert(0, str(_HERE))

import kalshi_yes_profitmaxx as pm
import kalshi_yes_combined  as kc

APPENDIX = _LS / "appendix"
APPENDIX.mkdir(exist_ok=True)

DEFAULT_HOURS = [11, 12, 13, 14, 15]
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pct(x: float, dp: int = 2) -> str:
    return f"{x*100:.{dp}f}%"


def trade_pnl_roi(trades: pd.DataFrame) -> tuple[float, float, float]:
    if trades.empty:
        return 0.0, 0.0, 0.0
    pnl  = float(trades["pnl_cents"].sum())
    cost = float(trades["cost_cents"].sum())
    roi  = pnl / cost if cost > 0 else 0.0
    return pnl, cost, roi


def monthly_pnl(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    df = trades.copy()
    df["_month"] = df["date"].str[:7]
    return df.groupby("_month")["pnl_cents"].sum()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def universe_stats(universes: dict[int, pd.DataFrame]) -> dict:
    out = {}
    months_all = set()
    for h, df in universes.items():
        out[f"n_contracts_{h:02d}h"] = len(df)
        out[f"base_rate_{h:02d}h"]   = float(df["yes_won"].mean())
        out[f"avg_ask_{h:02d}h"]     = float(df["yes_ask"].mean())
        out[f"n_dates_{h:02d}h"]     = int(df["date"].nunique())
        months_all |= set(df["_month"].unique())
    out["n_months"] = len(months_all)
    return out


def binomial_tests(trades: pd.DataFrame, base_rate: float) -> dict:
    n     = len(trades)
    wins  = int(trades["yes_won"].sum())
    if n == 0:
        return {"n": 0}

    # vs unconditional base rate
    b1 = stats.binomtest(wins, n, p=base_rate, alternative="greater")
    # vs market-implied probability (avg yes_ask)
    market_p = float(trades["yes_ask"].mean())
    b2 = stats.binomtest(wins, n, p=market_p, alternative="greater")

    return {
        "n":                 n,
        "wins":              wins,
        "realised_wr":       wins / n,
        "vs_base_rate":      {"p_null": base_rate, "p_value": b1.pvalue},
        "vs_market_implied": {"p_null": market_p,  "p_value": b2.pvalue},
    }


def bootstrap_pnl_roi(trades: pd.DataFrame, B: int = 5000) -> dict:
    if trades.empty:
        return {}
    pnls = trades["pnl_cents"].to_numpy()
    cost = trades["cost_cents"].to_numpy()
    n    = len(pnls)
    pnl_samples, roi_samples = [], []
    for _ in range(B):
        idx = RNG.integers(0, n, size=n)
        p   = pnls[idx].sum()
        c   = cost[idx].sum()
        pnl_samples.append(p)
        roi_samples.append(p / c if c > 0 else 0)
    return {
        "B":           B,
        "pnl_mean":    float(np.mean(pnl_samples)),
        "pnl_lo95":    float(np.percentile(pnl_samples, 2.5)),
        "pnl_hi95":    float(np.percentile(pnl_samples, 97.5)),
        "roi_mean":    float(np.mean(roi_samples)),
        "roi_lo95":    float(np.percentile(roi_samples, 2.5)),
        "roi_hi95":    float(np.percentile(roi_samples, 97.5)),
        "p_pnl_pos":   float(np.mean(np.array(pnl_samples) > 0)),
    }


def block_bootstrap_monthly(monthly: pd.Series, B: int = 5000) -> dict:
    if len(monthly) == 0:
        return {}
    vals = monthly.to_numpy()
    n    = len(vals)
    means    = []
    sharpes  = []
    totals   = []
    for _ in range(B):
        idx = RNG.integers(0, n, size=n)
        sample = vals[idx]
        means.append(sample.mean())
        sd = sample.std(ddof=1)
        sharpes.append(sample.mean() / sd if sd > 0 else 0)
        totals.append(sample.sum())
    return {
        "B":               B,
        "monthly_mean":    float(np.mean(means)),
        "monthly_mean_lo": float(np.percentile(means, 2.5)),
        "monthly_mean_hi": float(np.percentile(means, 97.5)),
        "monthly_sharpe_mean": float(np.mean(sharpes)),
        "sharpe_lo95":     float(np.percentile(sharpes, 2.5)),
        "sharpe_hi95":     float(np.percentile(sharpes, 97.5)),
        "total_lo95":      float(np.percentile(totals, 2.5)),
        "total_hi95":      float(np.percentile(totals, 97.5)),
        "p_mean_pos":      float(np.mean(np.array(means) > 0)),
        "p_sharpe_pos":    float(np.mean(np.array(sharpes) > 0)),
    }


def monthly_t_test(monthly: pd.Series) -> dict:
    if len(monthly) < 4:
        return {"n_months": len(monthly)}
    t_stat, p_two = stats.ttest_1samp(monthly, 0.0)
    # one-sided: PnL > 0
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    w_stat, w_p_two = stats.wilcoxon(monthly, alternative="two-sided")
    w_p_one = w_p_two / 2 if monthly.median() > 0 else 1 - w_p_two / 2
    return {
        "n_months":             len(monthly),
        "mean_monthly_pnl":     float(monthly.mean()),
        "std_monthly_pnl":      float(monthly.std()),
        "t_stat":               float(t_stat),
        "t_p_one_sided":        float(p_one),
        "wilcoxon_stat":        float(w_stat),
        "wilcoxon_p_one_sided": float(w_p_one),
    }


def permutation_test(sizing: str, hours: list[int], B: int = 200) -> dict:
    """Shuffle expected_high_f across dates within each snapshot universe,
       re-run the entire walk-forward, collect null OOS PnL/ROI.

       If the model is non-informative, shuffling predictions should NOT
       degrade results.  If the model is informative, shuffled runs should
       cluster around zero (or negative) ROI.
    """
    print(f"  Running permutation test: B = {B}  (this is the slow one)", flush=True)
    raw = {h: pm.build_universe(h) for h in hours}

    # Actual run
    def make_combined(universes: dict[int, pd.DataFrame]) -> pd.DataFrame:
        books = {}
        for h, d in universes.items():
            _, t, _ = pm.walkforward(d, sizing)
            if not t.empty:
                t = t.assign(snapshot_h=h)
            books[h] = t
        return kc.combine_books(books)

    actual = make_combined(raw)
    actual_pnl, _, actual_roi = trade_pnl_roi(actual)

    null_pnls, null_rois = [], []
    t0 = time.time()
    for b in range(B):
        shuf = {}
        for h, d in raw.items():
            d = d.copy()
            uniq_dates = d["date"].unique()
            perm_dates = RNG.permutation(uniq_dates)
            date_map   = dict(zip(uniq_dates, perm_dates))
            pred_by_dt = d.drop_duplicates("date").set_index("date")["expected_high_f"]
            d["expected_high_f"] = d["date"].map(date_map).map(pred_by_dt).to_numpy()
            shuf[h] = d

        comb = make_combined(shuf)
        pnl, _, roi = trade_pnl_roi(comb)
        null_pnls.append(pnl)
        null_rois.append(roi)
        if (b + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (b + 1) * (B - b - 1)
            print(f"    [{b+1:>4}/{B}]  elapsed {elapsed:>5.0f}s  eta {eta:>5.0f}s",
                  flush=True)

    null_pnls = np.array(null_pnls)
    null_rois = np.array(null_rois)
    p_pnl = float(np.mean(null_pnls >= actual_pnl))
    p_roi = float(np.mean(null_rois >= actual_roi))
    return {
        "B":             B,
        "actual_pnl":    float(actual_pnl),
        "actual_roi":    float(actual_roi),
        "null_pnl_mean": float(null_pnls.mean()),
        "null_pnl_std":  float(null_pnls.std()),
        "null_pnl_q05":  float(np.percentile(null_pnls, 5)),
        "null_pnl_q50":  float(np.percentile(null_pnls, 50)),
        "null_pnl_q95":  float(np.percentile(null_pnls, 95)),
        "null_roi_mean": float(null_rois.mean()),
        "null_roi_std":  float(null_rois.std()),
        "null_roi_q05":  float(np.percentile(null_rois, 5)),
        "null_roi_q50":  float(np.percentile(null_rois, 50)),
        "null_roi_q95":  float(np.percentile(null_rois, 95)),
        "p_value_pnl":   p_pnl,
        "p_value_roi":   p_roi,
        "null_pnls":     null_pnls.tolist(),
        "null_rois":     null_rois.tolist(),
    }


def direction_split(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for d, g in trades.groupby("threshold_direction"):
        n  = len(g)
        w  = int(g["yes_won"].sum())
        pnl = float(g["pnl_cents"].sum())
        cost = float(g["cost_cents"].sum())
        rows.append({
            "direction":  d,
            "n":          n,
            "wins":       w,
            "wr":         w / n,
            "pnl_cents":  pnl,
            "cost_cents": cost,
            "roi":        pnl / cost if cost > 0 else 0,
            "avg_ask":    float(g["yes_ask"].mean()),
        })
    return pd.DataFrame(rows)


def calibration_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    bins   = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01]
    trades = trades.copy()
    trades["p_bin"] = pd.cut(trades["p_model"], bins=bins, include_lowest=True)
    out = trades.groupby("p_bin", observed=True).agg(
        n=("yes_won", "size"),
        mean_p=("p_model", "mean"),
        realised=("yes_won", "mean"),
        avg_ask=("yes_ask", "mean"),
        pnl_cents=("pnl_cents", "sum"),
    ).reset_index()
    out["calibration_gap"] = out["realised"] - out["mean_p"]
    out["p_bin"] = out["p_bin"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizing", choices=["flat", "kelly_full", "kelly_half", "kelly_quarter"],
                        default="flat")
    parser.add_argument("--permutations", type=int, default=200,
                        help="Number of permutation iterations (default 200).")
    parser.add_argument("--hours", type=str, default="",
                        help=f"Comma-separated snapshot hours (default {DEFAULT_HOURS})")
    args = parser.parse_args()

    hours = (sorted(int(x.strip()) for x in args.hours.split(",") if x.strip())
             if args.hours else DEFAULT_HOURS)

    print("=" * 80)
    print(f"  STATISTICAL TESTS — sizing = {args.sizing}, hours = {hours}, "
          f"B_perm = {args.permutations}")
    print("=" * 80)

    # Build universes and per-snapshot books
    universes = {h: pm.build_universe(h) for h in hours}
    books = {}
    for h, df in universes.items():
        _, t, _ = pm.walkforward(df, args.sizing)
        if not t.empty:
            t = t.assign(snapshot_h=h)
        books[h] = t
    combined = kc.combine_books(books)

    summary: dict = {
        "sizing":             args.sizing,
        "hours":              hours,
        "permutation_iters":  args.permutations,
    }

    # ── 1: Universe stats ─────────────────────────────────────────────────
    summary["universe"] = universe_stats(universes)
    print("\n1.  UNIVERSE")
    for h in hours:
        print(f"    {h:>2}h  n = {summary['universe'][f'n_contracts_{h:02d}h']:>4}  "
              f"base-rate YES = {pct(summary['universe'][f'base_rate_{h:02d}h'])}  "
              f"avg ask = ${summary['universe'][f'avg_ask_{h:02d}h']:.3f}")
    print(f"    Months covered: {summary['universe']['n_months']}")

    # ── 2: Headline book stats ────────────────────────────────────────────
    pnl, cost, roi = trade_pnl_roi(combined)
    n_wins = int(combined["yes_won"].sum())
    print(f"\n2.  COMBINED BOOK")
    print(f"    n = {len(combined)}   {n_wins}W / {len(combined)-n_wins}L  "
          f"WR = {pct(n_wins/len(combined))}")
    print(f"    PnL = {pnl:+.1f}¢   Cost = {cost:.1f}¢   ROI = {pct(roi)}")
    summary["combined_book"] = {
        "n": len(combined), "wins": n_wins,
        "losses": len(combined) - n_wins,
        "wr": n_wins / len(combined),
        "pnl_cents": pnl, "cost_cents": cost, "roi": roi,
    }

    # ── 3: Binomial tests ────────────────────────────────────────────────
    print(f"\n3.  BINOMIAL TESTS (one-sided, H0: WR ≤ p_null)")
    # Universe-level base rate: average YES win rate across all eligible
    # contracts in every snapshot universe used.
    total_wins = sum(int(u["yes_won"].sum()) for u in universes.values())
    total_n    = sum(len(u) for u in universes.values())
    base_rate  = total_wins / total_n if total_n > 0 else 0
    bt2 = binomial_tests(combined, base_rate=base_rate)
    print(f"    vs base rate ({pct(base_rate)}):       "
          f"p = {bt2['vs_base_rate']['p_value']:.4f}")
    print(f"    vs market-implied "
          f"({pct(bt2['vs_market_implied']['p_null'])}): "
          f"p = {bt2['vs_market_implied']['p_value']:.4f}")
    summary["binomial"] = bt2

    # ── 4: Trade-level bootstrap ─────────────────────────────────────────
    print(f"\n4.  TRADE-LEVEL BOOTSTRAP (B = 5000)")
    boot = bootstrap_pnl_roi(combined, B=5000)
    print(f"    PnL:  mean {boot['pnl_mean']:+.1f}¢   "
          f"95% CI [{boot['pnl_lo95']:+.1f}, {boot['pnl_hi95']:+.1f}]")
    print(f"    ROI:  mean {pct(boot['roi_mean'])}    "
          f"95% CI [{pct(boot['roi_lo95'])}, {pct(boot['roi_hi95'])}]")
    print(f"    P(PnL > 0) = {pct(boot['p_pnl_pos'])}")
    summary["trade_bootstrap"] = boot

    # ── 5: Monthly block bootstrap ───────────────────────────────────────
    monthly = monthly_pnl(combined)
    print(f"\n5.  MONTHLY BLOCK BOOTSTRAP (B = 5000)")
    mboot = block_bootstrap_monthly(monthly, B=5000)
    print(f"    Mean monthly PnL: {mboot['monthly_mean']:+.1f}¢   "
          f"95% CI [{mboot['monthly_mean_lo']:+.1f}, {mboot['monthly_mean_hi']:+.1f}]")
    print(f"    Monthly Sharpe:    {mboot['monthly_sharpe_mean']:.2f}     "
          f"95% CI [{mboot['sharpe_lo95']:.2f}, {mboot['sharpe_hi95']:.2f}]")
    print(f"    P(mean monthly PnL > 0) = {pct(mboot['p_mean_pos'])}")
    summary["monthly_bootstrap"] = mboot

    # ── 6 & 7: parametric and non-parametric monthly tests ───────────────
    print(f"\n6.  PARAMETRIC + NON-PARAMETRIC MONTHLY TESTS")
    mt = monthly_t_test(monthly)
    print(f"    t-test  : t = {mt.get('t_stat', float('nan')):+.2f}   "
          f"p (one-sided, mean > 0) = {mt.get('t_p_one_sided', float('nan')):.4f}")
    print(f"    Wilcoxon: W = {mt.get('wilcoxon_stat', float('nan')):.1f}   "
          f"p (one-sided, median > 0) = {mt.get('wilcoxon_p_one_sided', float('nan')):.4f}")
    summary["monthly_tests"] = mt

    # ── 8: Permutation test ──────────────────────────────────────────────
    print(f"\n7.  PERMUTATION TEST (shuffle predictions across dates)")
    perm = permutation_test(args.sizing, hours, B=args.permutations)
    print(f"    Actual OOS PnL : {perm['actual_pnl']:+.1f}¢   "
          f"actual ROI: {pct(perm['actual_roi'])}")
    print(f"    Null PnL:  mean = {perm['null_pnl_mean']:+.1f}¢   "
          f"median = {perm['null_pnl_q50']:+.1f}¢   "
          f"95th pctile = {perm['null_pnl_q95']:+.1f}¢")
    print(f"    Null ROI:  mean = {pct(perm['null_roi_mean'])}   "
          f"median = {pct(perm['null_roi_q50'])}   "
          f"95th pctile = {pct(perm['null_roi_q95'])}")
    print(f"    p-value (PnL): {perm['p_value_pnl']:.4f}")
    print(f"    p-value (ROI): {perm['p_value_roi']:.4f}")
    summary["permutation"] = {k: v for k, v in perm.items()
                              if k not in ("null_pnls", "null_rois")}

    # Save the null distribution
    pd.DataFrame({"null_pnl_cents": perm["null_pnls"],
                  "null_roi":       perm["null_rois"]}
                 ).to_csv(APPENDIX / "permutation_null_distribution.csv", index=False)

    # ── 9: Direction split ───────────────────────────────────────────────
    print(f"\n8.  DIRECTION SPLIT")
    ds = direction_split(combined)
    for _, r in ds.iterrows():
        print(f"    {r['direction']:>8}: n = {int(r['n']):>3}  "
              f"WR = {pct(r['wr'])}  ROI = {pct(r['roi'])}  "
              f"avg ask = ${r['avg_ask']:.3f}")
    ds.to_csv(APPENDIX / "direction_split.csv", index=False)
    summary["direction_split"] = ds.to_dict("records")

    # ── 10: Calibration ──────────────────────────────────────────────────
    print(f"\n9.  CALIBRATION (model probability vs realised win rate)")
    cal = calibration_table(combined)
    print(f"    {'p̂ bin':>16}  {'N':>4}  {'mean p̂':>8}  {'realised':>9}  {'gap':>6}")
    for _, r in cal.iterrows():
        print(f"    {r['p_bin']:>16}  {int(r['n']):>4}  "
              f"{r['mean_p']:>8.3f}  {r['realised']:>9.3f}  "
              f"{r['calibration_gap']:>+6.3f}")
    cal.to_csv(APPENDIX / "calibration_table.csv", index=False)
    summary["calibration"] = cal.to_dict("records")

    # ── Save artifacts ───────────────────────────────────────────────────
    combined.to_csv(APPENDIX / "combined_trades.csv", index=False)
    monthly.to_csv(APPENDIX / "monthly_pnl.csv", header=["pnl_cents"])

    with open(APPENDIX / "statistical_tests_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n  Appendix artifacts written to: {APPENDIX}")
    for p in sorted(APPENDIX.glob("*")):
        print(f"    {p.name}")


if __name__ == "__main__":
    main()
