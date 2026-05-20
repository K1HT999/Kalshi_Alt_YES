# Weather Data + Microstructure → Kalshi Temperature Markets

**A reproducible study of two backtesting artifacts that look like alpha, and three orthogonal signal layers that survive them.**

Author: K. Thompson
Sample: June 2024 – April 2026 (22 months)
Markets: Kalshi KXHIGHCHI- (Chicago daily-high YES/NO contracts)
This project pairs three public data sources against the **KXHIGHCHI** Kalshi market (daily high temperature, Chicago):

1. **NOAA HRRR** — NWP forecast of the day's high temperature
2. **Iowa Mesonet ASOS** — surface observations through the trading snapshot
3. **Kalshi candlestick history** — 60-minute price / volume / spread / open interest evolution per contract

Three independent enhancement layers compose to a single strategy. Each adds ROI over the layer below it; each survives the same permutation test (p < 0.005 on the base layer); the combination at full stack delivers **+208% OOS ROI and Sharpe 0.46** with 36 trades over 22 months.

> *Permutation test (B = 200): all base modes pass at p < 0.005. The headline lift from each successive layer is verified across a P1 / P2 time-split.*

The full research paper is in [`KALTYES_Public.md`](KALTYES_Public.md).

---

## The three signal layers

| Layer | Source | What it does | Marginal lift |
|---|---|---|---|
| **Mode A** | ASOS + HRRR `max_temp_f` | Calibrated Gaussian-shell EV walk-forward on the temperature blend; capped `yes_ask ≤ $0.40` | foundation, **+122% ROI** |
| **Mode B** | + HRRR wind, cloud, pressure, precip | Row-wise σ inflation on volatile days using a regime score | **+15 pp** vs A → +137% |
| **Mode C** | + Kalshi candlestick data | Post-filter: drop NaN-data contracts; require `volume_6h ≥ 1000` | **+46 pp** vs B → +208% |

The layers are **orthogonal**: Mode B uses weather features, Mode C uses market-microstructure features. They can be applied separately or stacked.

---

## The four configurations worth shipping

| Config | Data needed | Trades | WR | ROI | Sharpe |
|---|---|---|---|---|---|
| **Mode A raw** | Just temperature blend | 48 | 18.8% | +122% | 0.34 |
| **Mode B raw** | Mode A + HRRR extras (17 vars) | 46 | 19.6% | +137% | 0.37 |
| **Mode A + Mode C** [vol≥1000] | Mode A + Kalshi candles | **38** | **23.7%** | **+183%** | **0.43** |
| **Mode B + Mode C** [vol≥1000] | Mode A + HRRR extras + Kalshi candles | **36** | **25.0%** | **+208%** | **0.46** |

For minimal operational complexity, **Mode A + Mode C** is the recommended deployment (only Kalshi candles needed beyond the basic Mode A inputs). For maximum performance, **Mode B + Mode C** stacks both enhancements.

The aggressive tight-filter version (Mode B + Mode C with `spread ≤ 4¢` added) reaches +267% ROI / Sharpe 0.49 on 28 trades — highest in-sample but smallest sample.

---

## Headline numbers — Mode B + Mode C [vol≥1000]

| Metric | Value |
|---|---|
| Sample | 22 months, Jun 2024 – Apr 2026, Chicago |
| Snapshots | 12pm + 2pm America/Chicago (unioned, 12pm-first dedup) |
| Universe filter | same-day contracts; `yes_ask ∈ [$0.05, $0.40]`; `volume_6h_total ≥ 1000` |
| OOS trades | **36** |
| Win rate | **25.0%** (vs 11.1% universe base rate) |
| OOS PnL | **+608¢** per $1 unit |
| OOS ROI | **+208%** on 292¢ deployed |
| Monthly Sharpe | **0.46** |
| Profitable months | 6 / 15 |

Time-split (P1 / P2) — both halves above their respective baselines:
- P1: 18 trades, **+92.3% ROI** (Mode B raw P1 was +42.2%)
- P2: 18 trades, **+341.2% ROI** (Mode B raw P2 was +257.1%)

The contract-by-contract trade list — every entry timestamp, entry price, and resolution — is in [`appendix/oos_trade_list.csv`](appendix/oos_trade_list.csv).

---

## The story in six acts

**Act 1 — A backtest that looked too good.** Initial walk-forward results showed 96% win rates and +348% ROI. They were artifacts.

**Act 2 — Two methodology bugs.**
1. **Next-day contract contamination.** Each Kalshi snapshot captures both today's *and* tomorrow's contracts. A naive date-keyed join compares today's intraday observations against tomorrow's strike, manufacturing fake "already-hit" arbitrage.
2. **Phantom-fillable cheap contracts.** Floors that *clip* `yes_ask` (e.g. 0.00 → 0.01) treat unfillable token offers as tradeable. ~31% of the post-fix universe sits in this bucket.

**Act 3 — What survives.** Same-day filter + `yes_ask ≥ $0.05` floor + the Gaussian-shell calibrated EV walk-forward. 57-trade book at +73.7% ROI, permutation p = 0.005.

**Act 4 — Mode A (capped book).** Calibration diagnostic shows the model is overconfident above `p̂ = 0.40`. Capping `yes_ask ≤ $0.40` excludes the unreliable regime. **48 trades, +122% ROI, Sharpe 0.34.**

**Act 5 — Mode B (regime-modulated σ).** A second HRRR-data layer (wind, cloud, pressure, precip) feeds a regime score that inflates σ on volatile days. **46 trades, +137% ROI, Sharpe 0.37.** Closes the calibration gap in the (0.20, 0.40] bin from −0.15 to −0.01.

**Act 6 — Mode C (microstructure filter).** Kalshi candlestick data exposes a third layer: contracts with cumulative 6-hour volume below ~1,000 are systematic underperformers regardless of model strength. Filtering them out lifts the Mode A book by +61 pp ROI and the Mode B book by +71 pp. **36-trade Mode B + Mode C book delivers +208% ROI at Sharpe 0.46.**

Full narrative + statistical detail: [`KALTYES_Public.md`](KALTYES_Public.md).

---

## Honest accounting of Mode C

Mode C is *two* effects bundled together. Be precise about each:

**Effect 1 — Data-availability cleanup.** 5 of 48 Mode A trades (and 5 of 46 Mode B trades) have `NaN` microstructure data. These are contracts the 60-minute candlestick feed doesn't have an entry for at the snapshot time (mostly an old ticker-format-change issue from Sept-Oct 2024). All 5 happen to be losers. Production naturally skips trades where you can't read the order book, so this isn't a "signal" — it's a production constraint. The 43-trade / 41-trade "data-clean" baselines reflect what production actually trades.

**Effect 2 — Liquidity filter.** On the data-clean baseline, requiring `volume_6h_total ≥ 1000` drops an additional 5 trades and lifts ROI another ~31 pp (Mode A: 151% → 183%; Mode B: 171% → 208%). This is a real microstructure signal — it engages with variance in the data and survives the time split.

The breakdown for Mode A:

| Step | Trades | ROI | Cumulative lift vs raw Mode A |
|---|---|---|---|
| Mode A raw | 48 | +122% | — |
| + Mode C data-clean | 43 | +151% | +29 pp (data artifact) |
| + Mode C `vol ≥ 1000` | 38 | +183% | +61 pp |
| + Mode C tight `vol≥1000 ∧ spr≤4` | 31 | +215% | +93 pp |

Same pattern for Mode B. The +29 pp from "data cleanup" is real for production (you can't trade contracts you can't see) but isn't a tradeable signal. The remaining +32 pp from the liquidity filter is the actual microstructure contribution.

---

## What this repo contains (and doesn't)

**Public:**
- This README and the full paper ([`KALTYES_Public.md`](KALTYES_Public.md))
- A data-dependency README explaining ASOS / HRRR / Kalshi acquisition ([`data/README.md`](data/README.md))
- The contract-by-contract OOS trade list — every timestamp, entry, and resolution
  ([`appendix/oos_trade_list.csv`](appendix/oos_trade_list.csv))
- Aggregated statistical artifacts for every shipping configuration (calibration
  tables, monthly PnL series, permutation null distributions, Mode C threshold
  sweep, microstructure feature correlations, machine-readable summary)
- A subset of the research code: the no-skill baselines, the HRRR-extras
  byte-range downloader, the microstructure feature extractor, the
  statistical-test machinery, the OOS-trade-list builder
  ([`research/PUBLIC_SCRIPTS/`](research/PUBLIC_SCRIPTS/))

**Not public** (withheld for the author and private collaborators):
- The walk-forward strategy code: point forecast, Gaussian shell, Platt
  recalibration, EV-cents threshold scan, sizing rule. Lives in
  `research/PRIVATE_SCRIPTS/`, not distributed.
- The σ-modulator implementation and the exact regime-score coefficient values
- The Mode C filter sweep (which thresholds were tried, which won)
- The exact α value for σ-modulation (public default 0.5 is defensible; production differs)
- Exact volume / spread thresholds tuned for production (public default 1000 / 4 is defensible)
- Exact calibration parameters and per-fold EV thresholds
- Direction-specific results and high-`p̂` × high-`yes_ask` stake-cap rule
- The full research paper with operational rules and per-fold coefficients

The published OOS trade list is the **audit trail** — a reviewer can verify
internal consistency of every headline number (PnL, ROI, win rate, Sharpe,
calibration gaps) directly from the CSV without needing the strategy code.

---

## Repository layout

```
.
├── README.md                                    this file
├── KALTYES_Public.md                            the full public research paper
├── appendix/                                    statistical artifacts + the OOS trade CSV
│   ├── oos_trade_list.csv                       every trade entered, with entry + resolution
│   ├── calibration_table*.csv                   post-Platt calibration per variant
│   ├── monthly_pnl*.csv                         per-month PnL series
│   ├── permutation_null_distribution*.csv       null distributions (B = 200)
│   ├── permutation_null_v2_sigma_mod_combined.csv  Mode B perm null
│   ├── phase2_alpha_sweep.csv                   Mode B α-robustness sweep
│   ├── phase2_monthly_comparison.csv            Mode A vs Mode B monthly
│   ├── phase2_sigma_mod_calibration.csv         post-σ-mod calibration table
│   ├── phase3_final_summary.csv                 A/B × C combination grid
│   ├── phase3_filter_sweep_combos_full.csv      Mode C threshold sweep
│   ├── phase3_microstructure_corr.csv           Spearman correlations
│   └── statistical_tests_summary*.json          machine-readable test outputs
├── data/
│   └── README.md                                data acquisition: ASOS / HRRR / Kalshi
└── research/
    ├── PUBLIC_SCRIPTS/                          published code (boring utilities)
    │   ├── README.md                            what's here, what's withheld
    │   ├── kalshi_yes_baselines.py              the seven no-skill baselines
    │   ├── kalshi_microstructure_features.py    candlestick feature extractor
    │   ├── sync_extra_hrrr.py                   HRRR byte-range downloader
    │   ├── launch_parallel_hrrr_sync.py         12-worker sync orchestrator
    │   ├── extract_extra_hrrr.py                GRIB → JSONL helper
    │   ├── statistical_tests.py                 permutation / bootstrap / monthly tests
    │   ├── phase2_permutation.py                per-snapshot permutation runner
    │   ├── phase2_permutation_combined.py       combined-book permutation runner
    │   ├── phase3_microstructure_eda.py         Spearman correlation EDA
    │   ├── phase3_microstructure_validation.py  ML-overfit demonstration
    │   └── build_public_trade_csv.py            built oos_trade_list.csv
    └── PRIVATE_SCRIPTS/                         NOT included in this repo — see Public/Private split above
```

---

## Reproducing the published numbers

The published artifacts (trade list, calibration tables, null distributions,
summary JSON) reproduce all headline numbers as long as your replication uses
the same data inputs (ASOS KORD, HRRR, Kalshi KXHIGHCHI). Three things you can
do with what's published:

**1. Verify internal consistency.**

```bash
# Verify each book's PnL/ROI/WR from the trade list alone
py research/PUBLIC_SCRIPTS/build_public_trade_csv.py
# (regenerates appendix/oos_trade_list.csv from source; verify it matches)
```

**2. Re-run the no-skill baselines on your own data.**

```bash
py research/PUBLIC_SCRIPTS/kalshi_yes_baselines.py
# Random, market-price, already-hit, persistence, ASOS-only, HRRR-only, blend.
# All should be flat or negative; positive results suggest universe-filter bugs.
```

**3. Re-run the microstructure EDA / validation** on your own trade book to
   confirm the +0.286 CV-log-loss overfit (the result that motivates a
   hand-picked filter rather than an ML approach):

```bash
py research/PUBLIC_SCRIPTS/phase3_microstructure_eda.py
py research/PUBLIC_SCRIPTS/phase3_microstructure_validation.py
```

The walk-forward strategy itself is in `research/PRIVATE_SCRIPTS/` and is not
distributed. The statistical-test code (`statistical_tests.py`,
`phase2_permutation*.py`) is included as published reference material but
will not run standalone because it imports the strategy code — see
[`research/PUBLIC_SCRIPTS/README.md`](research/PUBLIC_SCRIPTS/README.md).

---

## Honest caveats

- **n is small.** 36-trade headline (Mode B + Mode C). Confidence intervals are wide.
- **Single city, single market series, 22 months.** Cross-city / longer-horizon untested.
- **Fill quality unverified.** Backtest assumes execution at displayed `yes_ask`. Most edge concentrates at `yes_ask ≤ $0.10`.
- **Three universe-tuning decisions are stacked** in the headline (`yes_ask ≤ $0.40` cap; σ-mod α value; `volume_6h ≥ 1000` floor). Each is independently justified; together they should be treated cautiously.
- **Mode C data-clean is partly a sample artifact.** The 5 NaN contracts in our sample all happen to be losers. If a future replication has different NaN contracts, the data-cleanup lift could be different.
- **Sharpe ceiling.** All configurations max out around Sharpe 0.49 (half-Kelly). The right-skewed monthly PnL distribution caps Sharpe regardless of mode.

If you can falsify the permutation result on more data, please do.
