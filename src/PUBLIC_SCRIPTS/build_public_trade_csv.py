"""Build the public trade list CSV from the private microstructure-joined book.

Produces: appendix/oos_trade_list.csv

Schema: one row per trade-entry (per (mode, date, ticker)). Columns:
  - as_of              UTC ISO-8601 timestamp the row was generated
  - snapshot_date      local Chicago calendar date
  - snapshot_h         local Chicago snapshot hour (12 or 14)
  - ticker             Kalshi contract identifier
  - direction          greater | less
  - strike_f           contract strike (degrees Fahrenheit)
  - yes_ask            entry price (USD per share)
  - p_model            post-Platt YES probability
  - ev_cents           model expected value (cents per share)
  - observed_high_f    settlement-day high temperature (degrees Fahrenheit)
  - yes_won            resolution: True if YES settled
  - pnl_cents          per-trade PnL under flat sizing
  - cost_cents         per-trade cost under flat sizing
  - micro_volume_6h    6-hour traded volume prior to snapshot (Mode C input)
  - in_mode_a          1 if trade is in Mode A capped book
  - in_mode_b          1 if trade is in Mode B sigma_mod book
  - in_mode_a_c        1 if trade survives Mode C [vol_6h >= 1000] applied to Mode A
  - in_mode_b_c        1 if trade survives Mode C [vol_6h >= 1000] applied to Mode B (PRODUCTION)

Source:
  KThompsonKalshiAltYes/PRIVATE_chi/appendix/phase3_microstructure_eda.csv

Strips internal columns (wf_mu, wf_sd, wf_thr, p_raw, regime, sd_eff, full HRRR
extras, family-graph metadata, etc.) — keeping only what the public docs
already disclose.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
SOURCE    = ROOT / "PRIVATE_chi" / "appendix" / "phase3_microstructure_eda.csv"
OUT       = ROOT / "appendix" / "oos_trade_list.csv"

VOL_MIN   = 1000


def main():
    df = pd.read_csv(SOURCE)
    print(f"Loaded {len(df)} rows from {SOURCE.name}")
    print(f"  mode breakdown: {df['mode'].value_counts().to_dict()}")

    # Mode C inclusion: needs both data-clean (micro_mid_now exists) AND vol_6h >= 1000
    passes_mode_c = df["micro_mid_now"].notna() & (df["micro_volume_6h_total"].fillna(0) >= VOL_MIN)
    df["_passes_c"] = passes_mode_c

    # Book membership flags
    df["in_mode_a"]   = (df["mode"] == "A").astype(int)
    df["in_mode_b"]   = (df["mode"] == "B").astype(int)
    df["in_mode_a_c"] = ((df["mode"] == "A") & passes_mode_c).astype(int)
    df["in_mode_b_c"] = ((df["mode"] == "B") & passes_mode_c).astype(int)

    # Public-safe column subset.  Withheld: p_raw, wf_mu/sd/thr, regime, sd_eff,
    # all forward-window HRRR extras (the calibrated features that drive Mode B),
    # all Kalshi family-graph fields (event_ticker, threshold_family_key, etc.),
    # minutes_since_open / time_to_close (entry-time fingerprints).
    public_cols = [
        "as_of",
        "snapshot_date",
        "local_hour",
        "ticker",
        "threshold_direction",
        "threshold_value",
        "yes_ask",
        "p_model",
        "ev_cents",
        "observed_high",
        "yes_won",
        "pnl_cents",
        "cost_cents",
        "micro_volume_6h_total",
        "in_mode_a", "in_mode_b", "in_mode_a_c", "in_mode_b_c",
    ]
    out = df[public_cols].copy()
    out = out.rename(columns={
        "local_hour":           "snapshot_h",
        "threshold_direction":  "direction",
        "threshold_value":      "strike_f",
        "observed_high":        "observed_high_f",
        "micro_volume_6h_total":"micro_volume_6h",
    })

    # Sort by entry time
    out["as_of"] = pd.to_datetime(out["as_of"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out.sort_values(["as_of", "ticker"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {len(out)} rows to {OUT}")

    # Sanity check: per-book counts
    print()
    print("Public-CSV book counts (each should match the documented headline):")
    for col, label, expected in [
        ("in_mode_a",   "Mode A raw",                    48),
        ("in_mode_b",   "Mode B σ-mod raw",              46),
        ("in_mode_a_c", "Mode A + Mode C [vol≥1000]",    38),
        ("in_mode_b_c", "Mode B + Mode C [vol≥1000]",    36),
    ]:
        n = int(out[col].sum())
        match = "✓" if n == expected else "✗"
        print(f"  {label:<35s}  CSV={n:>3}  expected={expected:>3}  {match}")

    # Per-book PnL/ROI sanity
    print()
    print("Per-book PnL on the public CSV:")
    print(f"  {'book':<35s}  {'n':>3}  {'pnl_¢':>7}  {'cost_¢':>7}  {'ROI':>8}  {'WR':>6}")
    for col, label in [
        ("in_mode_a",   "Mode A raw"),
        ("in_mode_b",   "Mode B σ-mod raw"),
        ("in_mode_a_c", "Mode A + Mode C [vol≥1000]"),
        ("in_mode_b_c", "Mode B + Mode C [vol≥1000]  ← PRODUCTION"),
    ]:
        sub = out[out[col] == 1]
        n = len(sub)
        pnl = sub["pnl_cents"].sum()
        cost = sub["cost_cents"].sum()
        roi = pnl / cost if cost > 0 else 0
        wr = sub["yes_won"].mean() if n > 0 else 0
        print(f"  {label:<35s}  {n:>3}  {pnl:>+7.0f}  {cost:>7.0f}  {roi*100:>+7.1f}%  {wr*100:>5.1f}%")


if __name__ == "__main__":
    main()
