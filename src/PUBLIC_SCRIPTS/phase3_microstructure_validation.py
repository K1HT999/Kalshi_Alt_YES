"""
Phase 3 microstructure validation — two falsification tests.

Test 1: Logistic regression with controls
-----------------------------------------
Question: do microstructure features add predictive power for `yes_won`
OVER the model probability `p_model` alone?

Fit three models on the combined (Mode A + Mode B) trade list:
  M0:  yes_won ~ 1                        (baseline rate)
  M1:  yes_won ~ p_model                  (model probability only)
  M2:  yes_won ~ p_model + microstructure (model + microstructure)

Compare with:
  - In-sample log-loss
  - 5-fold CV log-loss
  - McFadden's pseudo-R²
  - Coefficient significance (statsmodels)

If M2 ≪ M1 on CV log-loss → microstructure adds real predictive power.
If M2 ≈ M1               → microstructure is redundant with p_model.
If M2 > M1               → microstructure is overfitting in-sample.


Test 2: Time-split validation
-----------------------------
Question: do the microstructure filter thresholds (model_minus_market ≥ 10pp,
spread ≤ 2¢, volume_6h ≥ some_floor) generalize out-of-sample?

Split trades by date into:
  Period 1 (first half): in-sample for threshold fitting
  Period 2 (second half): held out for validation

For each candidate filter, report:
  - Period 1 metrics: WR, ROI, n
  - Period 2 metrics: WR, ROI, n (out-of-sample)

If the filter's effect generalizes, Period 2 numbers should resemble
Period 1.  If reversion to baseline, the in-sample correlations were noise.

Saves:
  reports/phase3_validation.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_LS   = _HERE.parent
sys.path.insert(0, str(_HERE))

from kalshi_microstructure_features import attach_microstructure_features

REPORTS  = _LS / "reports"
TRADES_DIR = _LS / "reports"
EDA_CSV  = REPORTS / "phase3_microstructure_eda.csv"


def load_eda() -> pd.DataFrame:
    """Reuse the trades+microstructure CSV produced by phase3_microstructure_eda.py.
       If absent, regenerate on the fly."""
    if EDA_CSV.exists():
        return pd.read_csv(EDA_CSV)
    a = pd.read_csv(TRADES_DIR / "v2_trades_baseline_combined.csv"); a["mode"] = "A"
    b = pd.read_csv(TRADES_DIR / "v2_trades_sigma_mod_combined.csv"); b["mode"] = "B"
    df = pd.concat([a, b], ignore_index=True)
    return attach_microstructure_features(df)


# ---------------------------------------------------------------------------
# Test 1 — logistic regression with controls
# ---------------------------------------------------------------------------

def test1_logistic(df: pd.DataFrame) -> dict:
    """Compare baseline / p_model only / p_model + microstructure."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import log_loss

    # Use unique trades (drop double-counting between Mode A and Mode B)
    df = df.drop_duplicates(subset=["date", "ticker"]).copy()
    y = df["yes_won"].astype(int).to_numpy()

    # Microstructure features to include (drop the obviously-redundant or noisy ones)
    micro_cols = [c for c in df.columns if c.startswith("micro_") and c not in (
        "micro_yes_bid_now", "micro_yes_ask_now", "micro_mid_now",  # trivially correlated with p_model edge
        "micro_market_drift_confirms_model",  # null result from EDA
    )]
    # Drop columns that are mostly NaN or constant
    keep_cols = []
    for c in micro_cols:
        nn = df[c].notna().sum()
        if nn < len(df) * 0.5:
            continue
        if df[c].nunique() < 2:
            continue
        keep_cols.append(c)
    micro_cols = keep_cols

    print(f"  N trades: {len(df)}  ({y.sum()} won, {len(y)-y.sum()} lost)")
    print(f"  Microstructure features after filtering: {len(micro_cols)}")
    print()

    # Build feature matrices
    p_model = df["p_model"].astype(float).to_numpy().reshape(-1, 1)
    X_micro = df[micro_cols].to_numpy()

    imp = SimpleImputer(strategy="median")
    X_micro = imp.fit_transform(X_micro)

    # Standardize microstructure cols (p_model already in [0,1])
    scaler = StandardScaler()
    X_micro_s = scaler.fit_transform(X_micro)
    X_p      = p_model
    X_p_plus = np.hstack([X_p, X_micro_s])

    # M0 — baseline (intercept only)
    p_const = y.mean()
    ll_m0   = log_loss(y, np.full(len(y), p_const))

    # M1 — p_model only
    m1 = LogisticRegression(C=1e6, max_iter=1000)
    m1.fit(X_p, y)
    ll_m1 = log_loss(y, m1.predict_proba(X_p))

    # M2 — p_model + microstructure
    m2 = LogisticRegression(C=1.0, max_iter=2000)  # mild regularization to avoid overfit on n=52
    m2.fit(X_p_plus, y)
    ll_m2 = log_loss(y, m2.predict_proba(X_p_plus))

    # 5-fold CV log-loss
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_m1, cv_m2 = [], []
    for tr, te in kf.split(X_p):
        m1_cv = LogisticRegression(C=1e6, max_iter=1000).fit(X_p[tr], y[tr])
        m2_cv = LogisticRegression(C=1.0, max_iter=2000).fit(X_p_plus[tr], y[tr])
        try:
            cv_m1.append(log_loss(y[te], m1_cv.predict_proba(X_p[te]), labels=[0, 1]))
            cv_m2.append(log_loss(y[te], m2_cv.predict_proba(X_p_plus[te]), labels=[0, 1]))
        except Exception as e:
            print(f"  CV fold skipped: {e}")
    cv_m1_mean = float(np.mean(cv_m1)) if cv_m1 else float("nan")
    cv_m2_mean = float(np.mean(cv_m2)) if cv_m2 else float("nan")

    # Compute McFadden's pseudo-R²
    pseudo_r2_m1 = 1 - ll_m1 / ll_m0 if ll_m0 > 0 else float("nan")
    pseudo_r2_m2 = 1 - ll_m2 / ll_m0 if ll_m0 > 0 else float("nan")

    # Top microstructure coefficients (standardized)
    coef_pairs = sorted(zip(micro_cols, m2.coef_[0][1:]), key=lambda x: abs(x[1]), reverse=True)[:10]

    print(f"  In-sample log-loss:")
    print(f"    M0 (baseline): {ll_m0:.4f}")
    print(f"    M1 (p_model):  {ll_m1:.4f}   pseudo-R² = {pseudo_r2_m1:.3f}")
    print(f"    M2 (+micro):   {ll_m2:.4f}   pseudo-R² = {pseudo_r2_m2:.3f}")
    print()
    print(f"  5-fold CV log-loss:")
    print(f"    M1 (p_model):  {cv_m1_mean:.4f}")
    print(f"    M2 (+micro):   {cv_m2_mean:.4f}")
    print(f"    Δ:             {cv_m2_mean - cv_m1_mean:+.4f}   "
          f"({'M2 better' if cv_m2_mean < cv_m1_mean else 'M1 better or equal'})")
    print()
    print(f"  Top microstructure coefficients in M2 (standardized features):")
    for col, c in coef_pairs:
        print(f"    {col:>40}: {c:+.3f}")

    return {
        "n": len(df),
        "ll_m0": ll_m0, "ll_m1": ll_m1, "ll_m2": ll_m2,
        "pseudo_r2_m1": pseudo_r2_m1, "pseudo_r2_m2": pseudo_r2_m2,
        "cv_m1": cv_m1_mean, "cv_m2": cv_m2_mean,
        "cv_delta": cv_m2_mean - cv_m1_mean,
        "top_coefs": [(c, float(v)) for c, v in coef_pairs],
        "micro_cols_used": micro_cols,
    }


# ---------------------------------------------------------------------------
# Test 2 — time-split validation of microstructure filters
# ---------------------------------------------------------------------------

def book_metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "n": 0, "wins": 0, "wr_pct": 0.0,
                "pnl": 0.0, "cost": 0.0, "roi_pct": 0.0}
    n = len(df)
    wins = int(df["yes_won"].sum())
    pnl = float(df["pnl_cents"].sum())
    cost = float(df["cost_cents"].sum())
    return {
        "label": label, "n": n, "wins": wins, "losses": n - wins,
        "wr_pct": round(wins / n * 100, 2),
        "pnl_cents": round(pnl, 1),
        "cost_cents": round(cost, 1),
        "roi_pct": round(pnl / cost * 100, 1) if cost > 0 else 0,
    }


def test2_time_split(df: pd.DataFrame) -> dict:
    """Split by date into two halves. Apply candidate filters; compare
    in-sample vs out-of-sample metrics for each filter."""
    df = df.drop_duplicates(subset=["date", "ticker"]).copy()
    df["date_ts"] = pd.to_datetime(df["date"])
    median_date = df["date_ts"].quantile(0.5)
    p1 = df[df["date_ts"] <= median_date].copy()
    p2 = df[df["date_ts"] >  median_date].copy()
    print(f"  Total unique trades: {len(df)}")
    print(f"  Split at: {median_date.date()}")
    print(f"  Period 1 (early):  {len(p1)} trades  ({p1['date'].min()} → {p1['date'].max()})")
    print(f"  Period 2 (late):   {len(p2)} trades  ({p2['date'].min()} → {p2['date'].max()})")
    print()

    # Baseline metrics
    base_p1 = book_metrics(p1, "P1 baseline (no filter)")
    base_p2 = book_metrics(p2, "P2 baseline (no filter)")

    filters = [
        ("model_minus_market_pp ≥ 5",
         lambda d: d[d["micro_model_minus_market_pp"] >= 5]),
        ("model_minus_market_pp ≥ 10",
         lambda d: d[d["micro_model_minus_market_pp"] >= 10]),
        ("model_minus_market_pp ≥ 20",
         lambda d: d[d["micro_model_minus_market_pp"] >= 20]),
        ("spread_now ≤ 1¢",
         lambda d: d[d["micro_spread_now_cents"] <= 1]),
        ("spread_now ≤ 2¢",
         lambda d: d[d["micro_spread_now_cents"] <= 2]),
        ("volume_6h ≥ 5000",
         lambda d: d[d["micro_volume_6h_total"] >= 5000]),
        ("volume_6h ≥ 10000",
         lambda d: d[d["micro_volume_6h_total"] >= 10000]),
        ("spread ≤ 2¢ AND model−market ≥ 10",
         lambda d: d[(d["micro_spread_now_cents"] <= 2) & (d["micro_model_minus_market_pp"] >= 10)]),
        ("spread ≤ 1¢ AND model−market ≥ 5",
         lambda d: d[(d["micro_spread_now_cents"] <= 1) & (d["micro_model_minus_market_pp"] >= 5)]),
    ]

    rows = []
    rows.append({"filter": "(baseline P1)", **{k: v for k, v in base_p1.items() if k != "label"}, "period": "P1 (in-sample)"})
    rows.append({"filter": "(baseline P2)", **{k: v for k, v in base_p2.items() if k != "label"}, "period": "P2 (OOS)"})

    for name, fn in filters:
        p1_f = fn(p1)
        p2_f = fn(p2)
        m1 = book_metrics(p1_f, name + " (P1)")
        m2 = book_metrics(p2_f, name + " (P2)")
        rows.append({"filter": name, **{k: v for k, v in m1.items() if k != "label"}, "period": "P1 (in-sample)"})
        rows.append({"filter": name, **{k: v for k, v in m2.items() if k != "label"}, "period": "P2 (OOS)"})

    rows_df = pd.DataFrame(rows)
    print(f"  {'filter':>40}  {'period':>13}  {'N':>4}  {'WR%':>5}  {'PnL¢':>7}  {'ROI%':>7}")
    print("  " + "─" * 90)
    for _, r in rows_df.iterrows():
        wr = r.get("wr_pct", 0)
        pnl = r.get("pnl_cents", 0)
        roi = r.get("roi_pct", 0)
        print(f"  {r['filter']:>40}  {r['period']:>13}  {int(r.get('n',0)):>4}  "
              f"{wr:>4.1f}%  {pnl:>+6.0f}  {roi:>+6.1f}%")

    rows_df.to_csv(REPORTS / "phase3_time_split.csv", index=False)
    return {"split_date": str(median_date.date()),
            "p1_baseline": base_p1, "p2_baseline": base_p2,
            "rows": rows_df}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_eda()
    print(f"Loaded {len(df)} trades with {sum(1 for c in df.columns if c.startswith('micro_'))} microstructure features.")
    print()
    print("=" * 70)
    print("  TEST 1: logistic regression with controls")
    print("=" * 70)
    t1 = test1_logistic(df)
    print()
    print("=" * 70)
    print("  TEST 2: time-split validation")
    print("=" * 70)
    t2 = test2_time_split(df)
    print()

    # Write summary report
    md_path = REPORTS / "phase3_validation.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Phase 3 microstructure — validation tests\n\n")
        f.write("Generated 2026-05-18. Two falsification tests run against the\n")
        f.write("microstructure correlations reported in `phase3_microstructure_eda.md`.\n\n")
        f.write("## Test 1 — Logistic regression with controls\n\n")
        f.write(f"**N trades**: {t1['n']} (unique contracts, dedup across Mode A and Mode B).\n\n")
        f.write("Three nested models compared:\n\n")
        f.write("| Model | Features | In-sample log-loss | Pseudo-R² | 5-fold CV log-loss |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| M0 | intercept only | {t1['ll_m0']:.4f} | 0.000 | — |\n")
        f.write(f"| M1 | `p_model` only | {t1['ll_m1']:.4f} | {t1['pseudo_r2_m1']:.3f} | {t1['cv_m1']:.4f} |\n")
        f.write(f"| M2 | `p_model` + microstructure | {t1['ll_m2']:.4f} | {t1['pseudo_r2_m2']:.3f} | {t1['cv_m2']:.4f} |\n\n")
        delta_cv = t1["cv_delta"]
        if delta_cv < -0.01:
            f.write(f"**ΔCV = {delta_cv:+.4f}: microstructure ADDS predictive power over `p_model` alone (out-of-fold log-loss drops by {-delta_cv:.4f}).**\n\n")
        elif delta_cv > 0.01:
            f.write(f"**ΔCV = {delta_cv:+.4f}: microstructure HURTS out-of-fold log-loss — features are overfitting in-sample.**\n\n")
        else:
            f.write(f"**ΔCV = {delta_cv:+.4f}: roughly neutral; microstructure is largely redundant with `p_model`.**\n\n")
        f.write("Top 10 standardized microstructure coefficients in M2 (positive = predicts win):\n\n")
        f.write("| Feature | Coefficient |\n|---|---|\n")
        for col, c in t1["top_coefs"]:
            f.write(f"| `{col}` | {c:+.3f} |\n")

        f.write("\n## Test 2 — Time-split validation\n\n")
        f.write(f"Trades split at median date {t2['split_date']}.\n")
        f.write(f"Period 1 (in-sample): {t2['p1_baseline']['n']} trades, "
                f"WR {t2['p1_baseline']['wr_pct']}%, ROI {t2['p1_baseline']['roi_pct']}%\n")
        f.write(f"Period 2 (out-of-sample): {t2['p2_baseline']['n']} trades, "
                f"WR {t2['p2_baseline']['wr_pct']}%, ROI {t2['p2_baseline']['roi_pct']}%\n\n")
        f.write("Filter results (apply same filter to both periods; compare):\n\n")
        f.write("| Filter | Period | N | WR% | PnL¢ | ROI% |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, r in t2["rows"].iterrows():
            f.write(f"| {r['filter']} | {r['period']} | {int(r.get('n', 0))} | "
                    f"{r.get('wr_pct', 0):.1f}% | {r.get('pnl_cents', 0):+.0f} | "
                    f"{r.get('roi_pct', 0):+.1f}% |\n")
        f.write("\n")
        f.write("**Interpretation rule**: if a filter's Period 2 (OOS) metrics resemble its Period 1 (in-sample) metrics, the filter generalises. ")
        f.write("If Period 2 reverts to the baseline (no-filter) numbers, the filter was a sample artifact.\n")
    print(f"  Markdown summary → {md_path}")


if __name__ == "__main__":
    main()
