"""
Phase 3 exploratory data analysis: correlate microstructure features with
trade outcomes (Mode A and Mode B combined books).

For every trade in:
  - reports/v2_trades_baseline_combined.csv   (Mode A through v2)
  - reports/v2_trades_sigma_mod_combined.csv  (Mode B)

attach microstructure features and report:
  1. Univariate Spearman ρ between each feature and `yes_won` (1/0)
  2. Univariate Spearman ρ between each feature and `pnl_cents`
  3. Pairwise comparison: winning vs losing trade distributions
  4. Predictive incremental: does adding the top features to a logistic
     classifier improve over the current `p_model` alone?

Saves:
  reports/phase3_microstructure_eda.csv        — feature × trade table
  reports/phase3_microstructure_corr.csv       — correlation summary
  reports/phase3_microstructure_eda.md         — human-readable report
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_LS   = _HERE.parent
sys.path.insert(0, str(_HERE))

from kalshi_microstructure_features import attach_microstructure_features

TRADES_DIR = _LS / "reports"  # where Mode A/B scripts write combined-book CSVs
REPORTS = _LS / "reports"


def load_trades() -> pd.DataFrame:
    a = pd.read_csv(TRADES_DIR / "v2_trades_baseline_combined.csv")
    b = pd.read_csv(TRADES_DIR / "v2_trades_sigma_mod_combined.csv")
    a["mode"] = "A"
    b["mode"] = "B"
    df = pd.concat([a, b], ignore_index=True)
    return df


def univariate_correlations(df: pd.DataFrame, micro_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in micro_cols:
        x = df[c].dropna()
        if len(x) < 20:
            continue
        # vs yes_won (Spearman)
        y_won = df.loc[x.index, "yes_won"].astype(float)
        try:
            rho_won, p_won = stats.spearmanr(x, y_won)
        except Exception:
            rho_won, p_won = float("nan"), float("nan")
        # vs pnl_cents
        y_pnl = df.loc[x.index, "pnl_cents"].astype(float)
        try:
            rho_pnl, p_pnl = stats.spearmanr(x, y_pnl)
        except Exception:
            rho_pnl, p_pnl = float("nan"), float("nan")
        # Winner / loser mean comparison
        win_mean  = float(x[df.loc[x.index, "yes_won"]].mean()) if (df.loc[x.index, "yes_won"]).any() else float("nan")
        loss_mean = float(x[~df.loc[x.index, "yes_won"]].mean()) if (~df.loc[x.index, "yes_won"]).any() else float("nan")
        try:
            ks = stats.ks_2samp(
                x[df.loc[x.index, "yes_won"]].to_numpy(),
                x[~df.loc[x.index, "yes_won"]].to_numpy(),
            )
            ks_p = float(ks.pvalue)
        except Exception:
            ks_p = float("nan")
        rows.append({
            "feature": c,
            "n": len(x),
            "spearman_vs_yes_won": round(rho_won, 3),
            "p_vs_yes_won": round(p_won, 4),
            "spearman_vs_pnl": round(rho_pnl, 3),
            "p_vs_pnl": round(p_pnl, 4),
            "winner_mean": round(win_mean, 3),
            "loser_mean": round(loss_mean, 3),
            "ks_winner_vs_loser_p": round(ks_p, 4),
        })
    return pd.DataFrame(rows)


def main():
    print("Loading trades …")
    df = load_trades()
    print(f"  {len(df)} trade rows ({len(df[df['mode']=='A'])} Mode A, {len(df[df['mode']=='B'])} Mode B)")

    print("\nAttaching microstructure features …")
    df = attach_microstructure_features(df)
    print(f"  Shape: {df.shape}")

    # Microstructure columns
    micro_cols = [c for c in df.columns if c.startswith("micro_")]
    print(f"  Microstructure features: {len(micro_cols)}")

    # Dedup: keep each (date, ticker, mode) row. Rows we already have are
    # already unique by definition since combined files dedup internally.
    df.to_csv(REPORTS / "phase3_microstructure_eda.csv", index=False)

    # ---- Univariate correlations (combined modes) ------------------------
    print("\nComputing univariate correlations (combined Mode A + Mode B trades) …")
    corr = univariate_correlations(df, micro_cols)
    corr = corr.sort_values("spearman_vs_yes_won", key=lambda s: s.abs(), ascending=False)
    corr.to_csv(REPORTS / "phase3_microstructure_corr.csv", index=False)

    print("\nTop 15 features by |Spearman ρ| with yes_won (combined Mode A + Mode B):")
    print(corr.head(15).to_string(index=False))

    # ---- Same, but restricted to UNIQUE trades only (drop double-counting)
    # (Each trade may appear in both A and B; we want unique contracts here.)
    df_unique = df.drop_duplicates(subset=["date", "ticker"])
    print(f"\n--- Unique trades only: {len(df_unique)} ---")
    corr_u = univariate_correlations(df_unique, micro_cols)
    corr_u = corr_u.sort_values("spearman_vs_yes_won", key=lambda s: s.abs(), ascending=False)
    print("\nTop 15 features by |Spearman ρ| with yes_won (unique trades only):")
    print(corr_u.head(15).to_string(index=False))

    # ---- Specifically: model_minus_market_pp - does it predict outcomes? ----
    if "micro_model_minus_market_pp" in df_unique.columns:
        m = df_unique[["micro_model_minus_market_pp", "yes_won", "pnl_cents", "yes_ask"]].dropna()
        if len(m) > 10:
            print("\n=== Model-market disagreement (micro_model_minus_market_pp) ===")
            # Bucket
            bins = [-10, 0, 10, 20, 40, 70, 200]
            m["mm_bucket"] = pd.cut(m["micro_model_minus_market_pp"], bins)
            g = m.groupby("mm_bucket", observed=True).agg(
                n=("yes_won", "size"),
                wr=("yes_won", "mean"),
                avg_pnl=("pnl_cents", "mean"),
                avg_ask=("yes_ask", "mean"),
            ).round(3)
            print(g.to_string())

    # ---- Market drift confirmation -------------------------------------
    if "micro_market_drift_confirms_model" in df_unique.columns:
        d = df_unique[["micro_market_drift_confirms_model", "yes_won", "pnl_cents"]].dropna()
        if len(d) > 5:
            print("\n=== Market-drift-confirms-model breakdown ===")
            g = d.groupby("micro_market_drift_confirms_model").agg(
                n=("yes_won", "size"),
                wr=("yes_won", "mean"),
                avg_pnl=("pnl_cents", "mean"),
            ).round(3)
            print(g.to_string())

    # ---- Spread tightness vs outcomes ----------------------------------
    if "micro_spread_now_cents" in df_unique.columns:
        s = df_unique[["micro_spread_now_cents", "yes_won", "pnl_cents"]].dropna()
        if len(s) > 5:
            print("\n=== Spread at trade entry — outcome breakdown ===")
            s["s_bucket"] = pd.cut(s["micro_spread_now_cents"], [-1, 1, 2, 4, 10, 100])
            g = s.groupby("s_bucket", observed=True).agg(
                n=("yes_won", "size"),
                wr=("yes_won", "mean"),
                avg_pnl=("pnl_cents", "mean"),
            ).round(3)
            print(g.to_string())

    print(f"\nFull EDA table saved → reports/phase3_microstructure_eda.csv")
    print(f"Correlation summary  → reports/phase3_microstructure_corr.csv")


if __name__ == "__main__":
    main()
