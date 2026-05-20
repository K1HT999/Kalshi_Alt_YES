"""
Baseline Comparison for Kalshi YES Temperature Strategy — Chicago (KXHIGHCHI)
==============================================================================
Establishes a hierarchy of baselines against the 12 PM ASOS+HRRR blend model
to isolate the marginal contribution of each data source.  All strategies use
the same walk-forward methodology (threshold trained on prior months, tested
on the held-out month) so comparisons are apples-to-apples.

Same-day contracts only
-----------------------
  The 60-minute snapshot file contains both same-day and next-day contracts.
  We retain only same-day contracts by deriving:
      settlement_date = (close_time_utc − 8 h).tz_convert("America/Chicago").date
  and filtering to rows where settlement_date == snapshot_date.  Without this
  filter, live intraday observations (ASOS high-so-far) would be compared
  against a *different* day's strike, creating spurious "already-hit" signals.

Strategies evaluated
--------------------
  B0  Random                — buy YES on a random sample (no signal)
  B1  Market price cheap    — buy YES when yes_ask ≤ tuned ceiling, no model
  B2  Already-hit (same-day)— buy YES only when today's intraday high has
                              already cleared the strike (rare: ~3 events)
  B3  Persistence           — yesterday's observed high as today's prediction
  B4  ASOS only             — intraday high-so-far as the forecast
  B5  HRRR only             — NWP afternoon-peak forecast, no live obs
  B6  Blend (ASOS + HRRR)  — max(high_so_far, hrrr_peak): the full strategy

Metrics reported
----------------
  Win rate, Total PnL, ROI on capital deployed, monthly Sharpe ratio,
  % profitable months, pairwise t-test vs blend (monthly ROI series)

Usage
-----
    python research/kalshi_yes_baselines.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_LS   = _HERE.parent

CONTRACTS_JSONL = _LS / "data" / "backfill_chicago" / "chicago_60m.jsonl"
PREDS_CSV       = _HERE / "daily_high_preds.csv"
FLAT_CSV        = _LS / "reports" / "backfill_chicago" / "discovery" / "temperature_path_flat_features.csv"

SNAPSHOT_H  = 12
# Filter, do NOT clip, yes_ask below this floor.  Contracts at yes_ask = 0.00–0.04
# are typically token offers with no real liquidity; treating them as fillable
# creates a phantom favorite-longshot edge of ~+6¢/trade in the (0, 0.02] bucket
# alone (31% of the raw universe) because the assumed 1¢ cost is unrealistic.
MIN_YES_ASK = 0.05
EDGE_SCAN   = np.arange(0.0, 9.5, 0.5)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_universe() -> pd.DataFrame:
    rows = []
    with open(CONTRACTS_JSONL, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["as_of_dt"]      = pd.to_datetime(df["as_of"], utc=True)
    df["local_hour"]    = df["as_of_dt"].dt.tz_convert("America/Chicago").dt.hour
    df["snapshot_date"] = df["as_of_dt"].dt.tz_convert("America/Chicago").dt.date.astype(str)
    df["yes_ask"]       = pd.to_numeric(df["yes_ask"],         errors="coerce")
    df["threshold_value"] = pd.to_numeric(df["threshold_value"], errors="coerce")
    df["volume"]        = pd.to_numeric(df["volume"],          errors="coerce").fillna(0)

    # Derive settlement date from close_time.
    # Contracts close near 06:00 UTC (midnight Chicago).  Subtracting 8 h
    # shifts the result into the prior local calendar day reliably across
    # both CST (UTC-6) and CDT (UTC-5).
    df["close_time_dt"]   = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df["settlement_date"] = (
        df["close_time_dt"] - pd.Timedelta(hours=8)
    ).dt.tz_convert("America/Chicago").dt.date.astype(str)

    noon = df[
        (df["local_hour"] == SNAPSHOT_H) &
        (df["status"] == "finalized") &
        df["threshold_direction"].isin(["greater", "less"]) &
        (df["snapshot_date"] == df["settlement_date"])    # same-day only
    ].copy()
    noon["date"] = noon["snapshot_date"]

    # Merge 12pm predictions
    preds = pd.read_csv(PREDS_CSV)
    p12   = preds[(preds["snapshot_hour"] == SNAPSHOT_H) & (preds["model"] == "ridge")][
        ["date", "expected_high_f", "high_so_far_f", "hrrr_afternoon_peak_f"]
    ]
    noon  = noon.merge(p12, on="date", how="inner")

    # Observed daily high (ground truth)
    flat = pd.read_csv(FLAT_CSV)
    flat["date"] = pd.to_datetime(flat["observed_weather_valid_date"]).dt.date.astype(str)
    obs_map = flat.set_index("date")["observed_max_temp_f"].to_dict()
    noon["observed_high"] = noon["date"].map(obs_map)

    # Yesterday's high for persistence baseline
    obs_series = pd.Series(obs_map).sort_index()
    noon["prev_high"] = noon["date"].map(obs_series.shift(1).to_dict())

    noon = noon.dropna(subset=["observed_high", "yes_ask", "threshold_value", "expected_high_f"])
    # FILTER (not clip) below the floor.  Anything priced below MIN_YES_ASK is
    # treated as not realistically tradeable.
    noon = noon[noon["yes_ask"] >= MIN_YES_ASK].copy()

    # YES outcome (direction-aware)
    noon["yes_won"] = np.where(
        noon["threshold_direction"] == "greater",
        noon["observed_high"] >= noon["threshold_value"],
        noon["observed_high"] <  noon["threshold_value"],
    )

    # Model edges: positive → model predicts YES will win
    def make_edge(df, pred_col):
        return np.where(
            df["threshold_direction"] == "greater",
            df[pred_col] - df["threshold_value"],
            df["threshold_value"] - df[pred_col],
        )

    noon["edge_blend"]   = make_edge(noon, "expected_high_f")
    noon["edge_hrrr"]    = make_edge(noon, "hrrr_afternoon_peak_f")
    noon["edge_asos"]    = make_edge(noon, "high_so_far_f")
    noon["edge_persist"] = make_edge(noon, "prev_high")

    # Flag contracts where YES is already locked in at noon
    noon["already_hit"] = (
        (noon["threshold_direction"] == "greater") &
        (noon["high_so_far_f"] >= noon["threshold_value"])
    )

    noon["_month"] = noon["date"].str[:7]
    return noon.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core trade metrics
# ---------------------------------------------------------------------------

def trade_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0, wins=0, losses=0, win_rate=0.0,
                    total_pnl=0.0, avg_pnl=0.0, total_cost=0.0, roi=None)
    wins   = int(df["yes_won"].sum())
    losses = len(df) - wins
    pnl    = float(((1 - df["yes_ask"]) * df["yes_won"] - df["yes_ask"] * (~df["yes_won"])) * 100).sum() \
             if False else \
             float((np.where(df["yes_won"], (1 - df["yes_ask"]) * 100, -df["yes_ask"] * 100)).sum())
    cost   = float((df["yes_ask"] * 100).sum())
    return dict(
        n=len(df), wins=wins, losses=losses,
        win_rate=round(wins / len(df), 4),
        total_pnl=round(pnl, 1),
        avg_pnl=round(pnl / len(df), 2),
        total_cost=round(cost, 1),
        roi=round(pnl / cost, 4) if cost > 0 else None,
    )


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walkforward_edge(df: pd.DataFrame, edge_col: str,
                     label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward using a model-edge threshold (scanned on training data)."""
    months    = sorted(df["_month"].unique())
    fold_rows = []
    all_sel   = []

    for month in months:
        train = df[df["_month"] < month].dropna(subset=[edge_col])
        test  = df[df["_month"] == month].dropna(subset=[edge_col])
        if len(train) < 30 or test.empty:
            continue

        best: dict | None = None
        for thr in EDGE_SCAN:
            sel = train[train[edge_col] >= thr]
            if len(sel) < 5:
                break
            m     = trade_metrics(sel)
            score = -(m["losses"]) * 1000 + (m["roi"] or 0)
            if best is None or score > best["score"]:
                best = {"thr": thr, "score": score}

        if best is None:
            continue

        sel_test = test[test[edge_col] >= best["thr"]].copy()
        sel_test["wf_threshold"] = best["thr"]
        all_sel.append(sel_test)

        m = trade_metrics(sel_test)
        fold_rows.append(dict(
            strategy=label, month=month,
            threshold=best["thr"], available=len(test),
            **{k: m[k] for k in ["n", "wins", "losses", "win_rate",
                                   "total_pnl", "total_cost", "roi"]},
        ))

    folds  = pd.DataFrame(fold_rows)
    trades = pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame()
    return folds, trades


def walkforward_price(df: pd.DataFrame, max_ask: float,
                      label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward that selects by yes_ask <= threshold (no model edge)."""
    months    = sorted(df["_month"].unique())
    fold_rows = []
    all_sel   = []

    for month in months:
        train = df[df["_month"] < month]
        test  = df[df["_month"] == month]
        if len(train) < 30 or test.empty:
            continue

        # Find best price ceiling on training data
        best: dict | None = None
        for ceiling in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            sel = train[train["yes_ask"] <= ceiling]
            if len(sel) < 5:
                continue
            m     = trade_metrics(sel)
            score = -(m["losses"]) * 1000 + (m["roi"] or 0)
            if best is None or score > best["score"]:
                best = {"ceiling": ceiling, "score": score}

        ceiling = best["ceiling"] if best else max_ask
        sel_test = test[test["yes_ask"] <= ceiling].copy()
        sel_test["wf_threshold"] = ceiling
        all_sel.append(sel_test)

        m = trade_metrics(sel_test)
        fold_rows.append(dict(
            strategy=label, month=month,
            threshold=ceiling, available=len(test),
            **{k: m[k] for k in ["n", "wins", "losses", "win_rate",
                                   "total_pnl", "total_cost", "roi"]},
        ))

    folds  = pd.DataFrame(fold_rows)
    trades = pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame()
    return folds, trades


def walkforward_arb(df: pd.DataFrame,
                    label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Already-hit arbitrage: buy YES when high_so_far already cleared strike."""
    months    = sorted(df["_month"].unique())
    fold_rows = []
    all_sel   = []

    for month in months:
        test = df[df["_month"] == month]
        if test.empty:
            continue
        sel_test = test[test["already_hit"]].copy()
        if sel_test.empty:
            continue
        all_sel.append(sel_test)
        m = trade_metrics(sel_test)
        fold_rows.append(dict(
            strategy=label, month=month,
            threshold=0.0, available=len(test),
            **{k: m[k] for k in ["n", "wins", "losses", "win_rate",
                                   "total_pnl", "total_cost", "roi"]},
        ))

    folds  = pd.DataFrame(fold_rows)
    trades = pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame()
    return folds, trades


def walkforward_random(df: pd.DataFrame, target_n_pct: float,
                       label: str, n_sim: int = 1000) -> pd.DataFrame:
    """Simulate buying YES on a random fraction of contracts."""
    months    = sorted(df["_month"].unique())
    fold_rows = []

    for month in months:
        test = df[df["_month"] == month]
        if test.empty:
            continue
        n_select = max(1, int(len(test) * target_n_pct))
        sim_rois  = []
        sim_pnls  = []
        for _ in range(n_sim):
            idx = np.random.choice(len(test), size=min(n_select, len(test)), replace=False)
            sel = test.iloc[idx]
            m   = trade_metrics(sel)
            sim_rois.append(m["roi"] or 0)
            sim_pnls.append(m["total_pnl"])
        fold_rows.append(dict(
            strategy=label, month=month,
            threshold=np.nan, available=len(test),
            n=n_select, wins=np.nan, losses=np.nan,
            win_rate=np.nan,
            total_pnl=np.mean(sim_pnls),
            total_cost=np.nan,
            roi=np.mean(sim_rois),
        ))

    return pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# Summary stats from folds
# ---------------------------------------------------------------------------

def summary(folds: pd.DataFrame) -> dict:
    total_n    = folds["n"].sum()
    total_pnl  = folds["total_pnl"].sum()
    total_cost = folds["total_cost"].sum() if "total_cost" in folds else np.nan
    total_roi  = total_pnl / total_cost if (pd.notna(total_cost) and total_cost > 0) else np.nan
    wins       = folds["wins"].sum() if folds["wins"].notna().all() else np.nan
    losses     = folds["losses"].sum() if folds["losses"].notna().all() else np.nan
    wr         = wins / total_n if (pd.notna(wins) and total_n > 0) else np.nan
    monthly    = folds["total_pnl"].dropna()
    sharpe     = monthly.mean() / monthly.std() if monthly.std() > 0 else np.nan
    prof_months = (monthly > 0).sum()
    return dict(
        n=int(total_n), wins=wins, losses=losses,
        win_rate=wr, total_pnl=total_pnl,
        total_cost=total_cost, roi=total_roi,
        monthly_sharpe=sharpe, prof_months=int(prof_months),
        n_months=len(folds),
        monthly_pnl=monthly,
    )


def wilson_ci(wins, n, z=1.96):
    if pd.isna(wins) or n == 0:
        return np.nan, np.nan
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(max(0, center - half) * 100, 1), round(min(1, center + half) * 100, 1)


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building universe …")
    df = build_universe()
    print(f"  {len(df):,} contracts  |  {df['date'].nunique()} dates  |  "
          f"{df['_month'].nunique()} months  ({df['_month'].min()}–{df['_month'].max()})")
    print(f"  Directions: {df['threshold_direction'].value_counts().to_dict()}")
    print(f"  Already-hit at noon: {df['already_hit'].sum()} contracts ({df['already_hit'].mean()*100:.1f}%)")
    print(f"  Unconditional YES win rate: {df['yes_won'].mean()*100:.1f}%")

    # Run all strategies
    print("\nRunning walk-forward strategies …")
    strategies: dict[str, tuple[pd.DataFrame, pd.DataFrame | None]] = {}

    f_rand = walkforward_random(df, target_n_pct=0.30, label="B0  Random (30% sample)")
    strategies["B0  Random (30% sample)"]     = (f_rand, None)

    f_price, t_price = walkforward_price(df, max_ask=0.20, label="B1  Market price (cheap YES)")
    strategies["B1  Market price (cheap YES)"] = (f_price, t_price)

    f_arb, t_arb = walkforward_arb(df, label="B2  Already-hit arbitrage")
    strategies["B2  Already-hit arbitrage"]    = (f_arb, t_arb)

    f_pers, t_pers = walkforward_edge(df, "edge_persist", "B3  Persistence (prev day high)")
    strategies["B3  Persistence (prev day high)"] = (f_pers, t_pers)

    f_asos, t_asos = walkforward_edge(df, "edge_asos", "B4  ASOS only (high-so-far)")
    strategies["B4  ASOS only (high-so-far)"]  = (f_asos, t_asos)

    f_hrrr, t_hrrr = walkforward_edge(df, "edge_hrrr", "B5  HRRR only (NWP forecast)")
    strategies["B5  HRRR only (NWP forecast)"] = (f_hrrr, t_hrrr)

    f_blend, t_blend = walkforward_edge(df, "edge_blend", "B6  Blend: ASOS + HRRR")
    strategies["B6  Blend: ASOS + HRRR"]       = (f_blend, t_blend)

    # ── Main comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  STRATEGY COMPARISON — CHICAGO KXHIGHCHI  |  Walk-Forward OOS")
    print("=" * 100)
    print(f"  {'Strategy':36s}  {'Trades':>7}  {'W/L':>9}  {'Win%':>8}  "
          f"{'95% CI':>16}  {'PnL¢':>8}  {'ROI':>8}  {'Sharpe':>7}  {'Prof Mo':>8}")
    print(f"  {'─'*36}  {'─'*7}  {'─'*9}  {'─'*8}  {'─'*16}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*8}")

    summaries: dict[str, dict] = {}
    for name, (folds, _) in strategies.items():
        s = summary(folds)
        summaries[name] = s
        lo, hi = wilson_ci(s["wins"], s["n"])
        ci_str = f"[{lo},{hi}]" if pd.notna(lo) else "   —   "
        wr_str = f"{s['win_rate']*100:.1f}%" if pd.notna(s["win_rate"]) else "  —  "
        roi_str = f"{s['roi']*100:+.1f}%" if pd.notna(s["roi"]) else "  —"
        wl_str  = (f"{int(s['wins'])}W/{int(s['losses'])}L"
                   if pd.notna(s["wins"]) and pd.notna(s["losses"]) else "  —")
        sh_str  = f"{s['monthly_sharpe']:.2f}" if pd.notna(s["monthly_sharpe"]) else "  —"
        pm_str  = f"{s['prof_months']}/{s['n_months']}"
        print(f"  {name:36s}  {s['n']:>7}  {wl_str:>9}  {wr_str:>8}  "
              f"{ci_str:>16}  {s['total_pnl']:>8.0f}  {roi_str:>8}  {sh_str:>7}  {pm_str:>8}")

    # ── Statistical significance vs Blend ─────────────────────────────────
    print("\n" + "=" * 100)
    print("  PAIRWISE t-TEST: monthly ROI vs B6 Blend (two-sided, H0: means equal)")
    print("=" * 100)
    blend_monthly = summaries["B6  Blend: ASOS + HRRR"]["monthly_pnl"]
    print(f"\n  B6 Blend monthly PnL: mean={blend_monthly.mean():.1f}¢  "
          f"std={blend_monthly.std():.1f}¢  n={len(blend_monthly)}")
    print(f"\n  {'Strategy':36s}  {'mean PnL¢':>10}  {'std':>8}  {'t-stat':>8}  {'p-value':>10}  {'sig':>5}")
    print(f"  {'─'*36}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*5}")
    for name, s in summaries.items():
        if name == "B6  Blend: ASOS + HRRR":
            continue
        other = s["monthly_pnl"].dropna()
        if len(other) < 4:
            continue
        # Align on common months
        common = blend_monthly.index.intersection(other.index)
        if len(common) < 4:
            continue
        b_aligned = blend_monthly.loc[common]
        o_aligned = other.loc[common]
        t_stat, p_val = stats.ttest_rel(b_aligned, o_aligned)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else "ns"
        print(f"  {name:36s}  {o_aligned.mean():>10.1f}  {o_aligned.std():>8.1f}  "
              f"{t_stat:>8.2f}  {p_val:>10.4f}  {sig:>5}")

    # ── Data source decomposition ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  DATA SOURCE CONTRIBUTION  (marginal ROI lift)")
    print("=" * 100)
    s_hrrr  = summaries["B5  HRRR only (NWP forecast)"]
    s_asos  = summaries["B4  ASOS only (high-so-far)"]
    s_blend = summaries["B6  Blend: ASOS + HRRR"]
    s_price = summaries["B1  Market price (cheap YES)"]

    hrrr_roi  = (s_hrrr["roi"]  or 0) * 100
    asos_roi  = (s_asos["roi"]  or 0) * 100
    blend_roi = (s_blend["roi"] or 0) * 100
    price_roi = (s_price["roi"] or 0) * 100

    print(f"\n  No data   (market price only):          ROI = {price_roi:+.1f}%")
    print(f"  + ASOS only (high-so-far):              ROI = {asos_roi:+.1f}%   Δ = {asos_roi-price_roi:+.1f}pp vs no-data")
    print(f"  + HRRR only (NWP forecast):             ROI = {hrrr_roi:+.1f}%   Δ = {hrrr_roi-price_roi:+.1f}pp vs no-data")
    print(f"  + Blend (ASOS + HRRR):                  ROI = {blend_roi:+.1f}%   Δ = {blend_roi-hrrr_roi:+.1f}pp vs HRRR alone")
    print(f"\n  ASOS marginal lift over HRRR alone:     {blend_roi - hrrr_roi:+.1f}pp")
    print(f"  HRRR marginal lift over no data:        {hrrr_roi - price_roi:+.1f}pp")
    print(f"  Total model lift over no data:          {blend_roi - price_roi:+.1f}pp")

    print()


if __name__ == "__main__":
    main()
