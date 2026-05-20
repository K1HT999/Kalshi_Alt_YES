# PUBLIC_SCRIPTS — what's here and why

This folder holds the subset of the research code that is published with the
public package. The full pipeline (~22 scripts) is held in a sibling
`PRIVATE_SCRIPTS/` folder that is **not** distributed.

The split follows one rule: a script is **public** iff its source code does
not reveal the production strategy — its features, its calibrated parameters,
its EV grid, its sizing rule, its σ-modulator coefficients, its
microstructure filter thresholds.

Concretely, that means:

- The walk-forward strategy code, the σ-modulator implementation, the
  Mode C filter sweep, and the point-forecast pipeline live in
  `PRIVATE_SCRIPTS/` only.
- The data-acquisition utilities, no-skill baselines, microstructure
  feature extractor, and the statistical-test machinery live here.

## Files in this folder

### Standalone — runs without any PRIVATE_SCRIPTS dependency

| Script | What it does |
|---|---|
| `kalshi_yes_baselines.py` | Computes the seven no-skill baselines (random, market-price, already-hit arbitrage, persistence, ASOS-only, HRRR-only, blend) under the same universe filter used by the strategy. **The credibility play: these baselines must be flat or negative; if any shows positive ROI under your replication, the artifact removal hasn't been done correctly.** See [`docs/03-artifacts-found.md`] in the private companion for details. |
| `kalshi_microstructure_features.py` | The `MicrostructureExtractor` class — loads a 60-minute Kalshi candlestick JSONL, builds 33 features per (ticker, snapshot_time) lookup. Pure feature engineering; no filter / no model. |
| `sync_extra_hrrr.py` | Downloader for the 17 HRRR variables (wind, cloud, pressure, precipitation, plus 925-mb fields) via NOAA AWS byte-range. Idempotent + resumable. |
| `launch_parallel_hrrr_sync.py` | 12-worker orchestrator for the HRRR sync. Wraps `sync_extra_hrrr.py` in subprocess pool to dodge eccodes thread-unsafety. |
| `extract_extra_hrrr.py` | Helper that decodes already-cached GRIB byte buffers into JSONL output. |
| `phase3_microstructure_eda.py` | Univariate Spearman correlations of microstructure features against `yes_won`. Output is in `appendix/phase3_microstructure_corr.csv`. |
| `phase3_microstructure_validation.py` | Demonstrates that a 33-feature logistic regression on `p_model + all micro features` overfits catastrophically (5-fold CV log-loss worsens by +0.286 vs `p_model` alone). The reason Mode C must be a hand-picked filter, not a learned model. |
| `build_public_trade_csv.py` | The script that built `appendix/oos_trade_list.csv` from the private microstructure-joined trade book. Pure data-shaping; no alpha. |

### Documentation-only — depends on PRIVATE_SCRIPTS to run

These three scripts implement the statistical-test methodology that produced
the public results. They cannot be executed standalone because they import
from the strategy code in `PRIVATE_SCRIPTS/`. They are included as
**reference material** so readers can audit the test methodology.

| Script | What it implements | Why it's published |
|---|---|---|
| `statistical_tests.py` | Permutation test, trade-level bootstrap, monthly block bootstrap, t-test / Wilcoxon, calibration diagnostic, direction split | The test methodology is the credibility layer of the result; readers should be able to inspect exactly how the p-values, CIs, and calibration gaps were computed. |
| `phase2_permutation.py` | Per-snapshot permutation runner (Mode B variants) | Same as above. |
| `phase2_permutation_combined.py` | Combined-book permutation for the 12+14 union | Same as above. |

Each begins with `import kalshi_yes_profitmaxx as pm` (and v2). Without the
strategy module, the imports fail. **They are illustrative, not runnable.**

## What's withheld

`PRIVATE_SCRIPTS/` contains the alpha-bearing code:

| Withheld script | Why |
|---|---|
| `kalshi_yes_profitmaxx.py` | The Mode A walk-forward: residual fit, Gaussian shell, Platt recalibration, EV-cents threshold scan, sizing. The exact EV grid (`np.arange(0.0, 30.0, 2.0)`), loss-aversion coefficient (0.5), and ask cap ($0.40) are in this file. |
| `kalshi_yes_profitmaxx_v2.py` | Mode B σ-modulator with exact regime-score coefficient values and the σ-mod multiplier α. The `ml`, `stake_cap`, and `sigma_mod_walkforward` variants. |
| `hrrr_extra_features.py` | Forward-window feature builders (wind, precip, baro, bonus) — exposes which HRRR aggregates the regime score consumes and how. |
| `daily_high_prediction.py` / `_v2.py` | Point-forecast pipeline + the full feature vector (dewpoint depression, warmup-so-far, overnight low, multi-window deltas). |
| `kalshi_yes_combined.py` | N-snapshot dedup logic. |
| `phase3_filter_sweep.py` / `_broad.py` / `_final_summary.py` | The Mode C threshold sweep — exposes exactly which volume / spread thresholds were tested. |
| `phase2_diagnostics.py` | α-sweep + bootstrap on the σ-modulator. |
| `phase2_combine_variants.py` | Variant aggregator. |
| `kalshi_yes_strategy.py` | Earlier-iteration strategy script (predecessor to profitmaxx). |

## Audit trail

The aggregated outputs of the withheld scripts are still public, in
`appendix/`:

- `oos_trade_list.csv` — every OOS trade (timestamp, entry, resolution) under
  all four shipping configurations
- `calibration_table*.csv` — post-Platt calibration tables under each variant
- `monthly_pnl*.csv` — per-month PnL series
- `permutation_null_distribution*.csv` — full null distributions from the
  permutation tests (B = 200)
- `phase3_final_summary.csv` — the A/B/C combination grid
- `phase3_microstructure_corr.csv` — Spearman correlations for Mode C features
- `phase3_filter_sweep_combos_full.csv` — Mode C threshold-sweep results
- `statistical_tests_summary.json` — machine-readable summary

A reviewer can verify internal consistency of every headline number from
these files alone, even without the strategy code.
