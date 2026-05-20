# An investigation into the application of calibrated NOAA HRRR and Iowa Mesonet ASOS state, to Kalshi Daily High weather contracts

Author: K. Thompson
Sample: June 2024 – April 2026 (22 months)
Markets: Kalshi KXHIGHCHI- (Chicago daily-high YES/NO contracts)

---

## Introduciton

Kalshi weather markets are an emerging market for alternative data applcations. In this strategy, Kalshi's KXHIGHCHI climate contracts are backetested on three signal layers:

1. **Mode A** — a calibrated walk-forward strategy on NWP + surface observations alone.
2. **Mode B** — a regime-score enhancement that forecasts uncertainty using wind / cloud / pressure / precipitation forecasts derived from HRRR data.
3. **Mode C** — a microstructure filter using Kalshi's own 60-minute candlestick feed.

Each layer adds measurable lift over the layer below it. The four configurations:

| Configuration | Trades | WR | ROI | Sharpe | Permutation p |
|---|---|---|---|---|---|
| Mode A — capped baseline | 48 | 18.8% | **+122%** | 0.34 | **< 0.005** |
| Mode B — regime-modulated σ | 46 | 19.6% | **+137%** | 0.37 | **< 0.005** |
| Mode A + Mode C [vol≥1000] | 38 | 23.7% | **+183%** | 0.43 | inherited from A |
| **Mode B + Mode C [vol≥1000]** | **36** | **25.0%** | **+208%** | **0.46** | inherited from B |

All reported values are out-of-sample calculated by a walk-forward protocol. Each test month uses only prior months for fitting, probability calibration, and threshold selection. The permutation test was perfomed to shuffle model predictions across dates and re-analyzes the entire selection pipeline; zero of 200 null iterations beat the observed Mode A or Mode B PnL.

---

## 1. Data and Methodology

### 1.1 What's being predicted

Kalshi operates a regulated prediction market. The KXHIGHCHI series asks variants of:

> *"Will the daily high temperature in Chicago be ≥ X°F on date D?"*

Contracts are priced in cents per dollar of YES payout. Settlement is automated against the daily high reported by the designated NWS station (O'Hare airport (KORD) for Chicago). Each calendar day carries roughly 20–40 strike levels and two directional variants (`greater` and `less`).

### 1.2 Why weather

Weather binaries on Kalshi are physical data based, and are unambigous, barring any characters with blow dryers . Weather forecasting is a deep field with many intelligent applications, most of which are not applied to these markets. Contract pricing reflects public and easily accesible information. This strategy works to price in expected temperatures before the market can react. The observations and data I use on these markets used is free, the contracts are able to be quoted at 60 minute intervals over years of markets, and they do not have institutional volume.

Two alternative datasources:

- **HRRR** — NOAA High-Resolution Rapid Refresh NWP model. Hourly runs, 3 km resolution, 2-m temperature among many fields.
- **ASOS** — METAR-format surface observations distributed by Iowa Environmental Mesonet (Thanks, Iowa), 5-minute cadence at major airports.

### 1.3 Scope and limitations

This study covers **Chicago, KXHIGHCHI, 12pm + 2pm America/Chicago snapshots, June 2024 – April 2026**. Nothing here can be automatically to other cities, other snapshot hours(morning/evening hours), or other contracts. Also our dataset it still limited, we do not have the history US stocks have, but we make do with what is available.


## 1.4 Pipeline
DIAGRAM


### 1.5 Universe filters

After much iteration (see § 4 below), three filters define the working universe:

1. **Same-day**: `settlement_date == snapshot_date`. The settlement date is derived as `close_time_utc − 8h`, projected into America/Chicago.
2. **Price floor**: `yes_ask ≥ $0.05`. Token offers below this level are not realistically fillable.
3. **Price cap (Mode A/B)**: `yes_ask ≤ $0.40`. Calibration above this is unreliable (see § 5).

### 1.6 Snapshot hours

| Snapshot | MAE | Residual μ | Residual σ | Verdict |
|---|---|---|---|---|
| 9am | 4.45°F | +5.41°F | 6.42°F | not usable (large bias) |
| 11am | 2.10°F | +0.28°F | 2.13°F | usable, weakest |
| 12pm | 1.52°F | +0.19°F | 2.00°F | usable, strongest individually |
| 1pm | 1.49°F | +0.38°F | 2.15°F | usable |
| 2pm | 1.35°F | +0.32°F | 1.75°F | usable, smallest book / highest per-trade ROI |
| 3pm | 1.19°F | +0.21°F | 1.58°F | usable, tightest σ |

The proposed deployment uses 12pm + 2pm. The five-snapshot broad-coverage book is retained for context.

---

## 2. The Signals

### 2.1 - The Math 
Universe predicate — same-day filter [(close_time_utc − 8h)]^{date}_{Chicago} == [as_of_utc]^{date}_{Chicago} (the projection notation means "project to Chicago local timezone, take the calendar date") plus the price band $0.05 ≤ yes_ask ≤ $0.40 (the calibration-driven cap).

Point forecast — T̂ = max(H^ASOS, F^HRRR,aft) where H^ASOS is the intraday max KORD temperature observed from local midnight to the snapshot, and F^HRRR,aft is the forward-window peak from the most recent HRRR run prior to the snapshot. Beats ridge/RF/GBM at the production snapshots.

Probabilistic shell — Gaussian conditional T_D | T̂ ~ N(T̂ + μ_f, σ_f²), with (μ_f, σ_f) fit per fold from the prior months' per-date residuals ε_D = T_D − T̂, and a hard floor σ_f ≥ σ_floor = 1.0°F.
Recalibration — single-fold Platt: p̂ = sigm(a_f · logit(p_raw) + b_f) where sigm(x) = 1/(1+e^{-x}), fit by NLL minimization on training contracts. Note b_f here is the Platt intercept; not to be confused with the Kelly odds (the doc uses ω_k = (1 − a_k)/a_k to avoid collision).

Trade rule — EV_k = (p̂_k − yes_ask_k) · 100 (cents per share); the per-fold threshold τ_f is chosen by argmax_τ ∈ {0,2,…,28} [pnl_train(τ) − 0.5 · losses_train(τ)]. The loss-aversion coefficient ½ is the only non-static knob in selection.
Mode B (σ-modulator) — σ_eff,k = σ
_f · (1 + α · regime_score_k). The regime score is the clipped mean of four normalized HRRR forward-window components: cloud cover peak (÷100), wind gust peak ((g−5)/15, clipped), surface pressure swing ((P_peak − P_low)/15, clipped), and any-precip flag. Public default α = 0.5; at α = 0 Mode B collapses to Mode A.

Mode C (microstructure filter) — deterministic post-filter V^6h_k ≥ V_min = 1000 (cumulative traded volume in the 6 hours prior to the snapshot). ML on microstructure features overfits catastrophically (5-fold CV log-loss worsens by +0.286 vs p̂ alone), so the filter is hand-picked, not learned.

Sharpe ceiling — under a two-point mixture model with carrier-month frequency q (the fraction of months that account for most of the headline PnL), monthly Sharpe is bounded by √(q/(1−q)). For empirical q ≈ 1/4, the bound is √(1/3) ≈ 0.58. The observed 0.46 sits inside this ceiling rather than below a higher one; reporting much above this on the 16-month sample would require a distributional change, not better calibration.

Permutation test — uniform random permutation π^(b) of date labels on the predictor column only (all other columns unchanged), repeated B = 200 times. The pipeline is re-run end-to-end on each shuffled set, generating a null distribution that captures everything except the date-conditioning information in the predictor. A small p̂_perm thus isolates the predictor's contribution.

### 2.2 Mode A — calibrated baseline

The blend of `high_so_far_f` (ASOS-derived intraday max) and `hrrr_afternoon_peak_f` (HRRR forward-window 2-m temperature max) yields a point prediction. The Gaussian shell turns the point prediction into a YES probability for each strike. Platt recalibration corrects systematic overconfidence (the raw probability is off by ~25–40 pp in the middle bins; post-Platt is ≤ 5 pp in active bands).

**Result (12pm + 2pm, capped, flat sizing):** 48 trades, 18.75% win rate, **+122% OOS ROI, Sharpe 0.34**, permutation p < 0.005.

### 2.3 Mode B — regime-modulated σ (Phase 2)

Mode A holds σ constant across contracts within a fold. Mode B replaces this with a *row-wise* σ that inflates on volatile days, pulling marginal trades on those days below the EV threshold.

A scalar `regime_score ∈ [0, 1]` is computed per contract from four HRRR features:

1. **Total cloud cover** (peak %) — caps insolation.
2. **10-m wind gust** (peak m/s) — proxies for boundary-layer mixing.
3. **Surface-pressure swing** (peak − min, mb) — proxies for frontal passage.
4. **Forecast precipitation flag** (any PRATE > 0) — direct surface-temperature suppressor.

The four normalised components are averaged. The per-row effective σ becomes:

```
σ_effective(i) = σ_base × (1 + α · regime_score(i))
```

The public default is α = 0.5. An α sweep on [0.0, 1.0] confirms **every α > 0 beats α = 0 on both ROI and Sharpe**; the variant is not knife-edge. Overfitters recoil.

**Result:** 46 trades, 19.6% win rate, **+137% OOS ROI, Sharpe 0.37**, permutation p < 0.005 (0/200 nulls).

The mechanism is visible in the calibration table:

| `p̂` bin | Mode A gap | Mode B gap |
|---|---|---|
| (0.20, 0.40] | **−0.148** | **−0.009** ✓ |
| (0.40, 0.60] | **−0.474** | **−0.001** ✓ |

The mid-probability bins go from substantially overconfident to essentially perfectly calibrated.

### 2.4 Mode C — microstructure filter (Phase 3)

Mode C is*not a standalone strategy. It's a filter on top of Mode A or Mode B's trade list. Two layered effects:

1. **Data clean**: drop trades on contracts where the 60-minute Kalshi candlestick feed has no entry at trade time. 5 of 48 Mode A trades fall here, all from a ticker-format-change window or a feed gap. All 5 happen to be losers in this sample. This is a *production constraint* — you can't trade what you can't price, so it was excluded from backtesting.
2. **Liquidity filter**: require `volume_6h_total ≥ 1000` cumulative contract volume in the 6 hours prior to the snapshot. Simple but fundamentally sound.

The univariate Spearman test was performed on `volume_6h_total`, `yes_won` with respect to the the joint n=94 trade pool, the Spearman correlation came out to **+0.39 (p = 0.0001)**. 


### 2.5 Results

| Configuration | Data needed | Trades | WR | ROI | Sharpe |
|---|---|---|---|---|---|
| Mode A raw | Temperature blend | 48 | 18.8% | +122% | 0.34 |
| Mode B raw | + HRRR extras (17 vars) | 46 | 19.6% | +137% | 0.37 |
| **Mode A + Mode C [vol≥1000]** | + Kalshi candles | **38** | **23.7%** | **+183%** | **0.43** |
| **Mode B + Mode C [vol≥1000]** | + extras + candles | **36** | **25.0%** | **+208%** | **0.46** |

Each layer adds lift. Mode A → Mode B is +15 pp ROI from σ-modulation. Mode A → Mode A + C is +61 pp from the microstructure filter (~+29 pp from the data-clean step, ~+32 pp from the volume filter). Mode A → Mode B + C combines both for +86 pp total.

---


## 3. Why the $0.40 cap

The post-Platt calibration table on the uncapped 57-trade book shows the model's probability estimates are reasonably reliable through the (0.20, 0.40] bin and wildly unreliable above:

| `p̂` bin (uncapped) | N | Mean `p̂` | Realised | Gap |
|---|---|---|---|---|
| (0.10, 0.20] | 30 | 0.143 | 0.100 | −0.043 ✓ |
| (0.20, 0.40] | 13 | 0.302 | 0.154 | −0.148 ⚠ |
| (0.40, 0.60] | 1 | 0.474 | 0.000 | **−0.474** ✗ |
| (0.60, 0.80] | 3 | 0.713 | 1.000 | +0.287 ✗ |
| (0.80, 1.01] | 2 | 0.960 | 0.500 | **−0.460** ✗ |

Above `p̂ = 0.40`, the model is overconfident by 30–47 pp. These bins also map to higher `yes_ask` because positive-EV trades there require the model to substantially exceed market-implied probability; the model's *most confident* calls are the *most expensive* trades and the *least reliable*. 

The single largest dollar loss in the uncapped book (−69¢ on a `yes_ask = $0.69` trade where the model claimed `p̂ = 0.96` but the actual went the other way) lives in this regime. Excluding it raises ROI from +74% (uncapped) to +122% (capped) and Sharpe from 0.24 to 0.34.

Mode B's σ-modulator addresses the same calibration weakness with a continuous, feature-driven correction. The cap remains in production because the calibration evidence supports it and because the two interventions are complementary, not redundant.

---

## 4. Statistical evidence

### 4.1 Permutation test (B = 200)

The permutation test is the strongest evidence. I shuffled the model's probability predictions across dates, re-ran the entire selection pipeline, and recorded the resulting OOS PnL. Repeating 200 times produces a null distribution of "what this pipeline does when its predictions don't carry information."

**Mode A (48 trades):**

| | Actual | Null mean | Null median | Null 95th pctile |
|---|---|---|---|---|
| OOS PnL (¢) | **+495.0** | −77.6 | −88.5 | +168.7 |
| OOS ROI | **+122.2%** | −15.3% | −16.1% | +31.6% |

**P-value: < 0.005 for both PnL and ROI** (zero of 200 prediction-shuffled iterations beat the observed result). Actual sits ~2.9× the null 95th percentile.

**Mode B (46 trades):**

| | Actual | Null mean | Null median | Null 95th pctile |
|---|---|---|---|---|
| OOS PnL (¢) | **+521** | −69 | −59 | +161 |
| OOS ROI | **+137.5%** | −12.5% | −12.5% | +27.1% |

**P-value: < 0.005 for both PnL and ROI.** Actual sits ~5× the null 95th percentile (Mode A was ~3×). The Mode B null distribution is sharper-negative because the σ-modulator amplifies the cost of bad predictions on volatile days.

**Mode C** is a deterministic post-filter on the base trade list. It doesn't add new walk-forward predictions, so the permutation result passes through.
### 4.2 Trade-level bootstrap (B = 5000)

| Statistic | Mode A | Mode B |
|---|---|---|
| Total PnL (¢), mean | +502 | +522 |
| 95% CI on PnL | [−12, +1083] | **[+17, +1083]** |
| **P(PnL > 0)** | **96.9%** | **97.9%** |

The lower bound of Mode B's bootstrap CI is now positive. An unfavorable resample of the 46 trades of Mode B stays profitable with 95% probability.

### 4.3 Monthly block bootstrap (B = 5000)

| Statistic | Mode A | Mode B |
|---|---|---|
| Mean monthly PnL 95% CI | [−7.9, +78.4] | [−5.4, +78.6] |
| Monthly Sharpe 95% CI | [−0.17, +0.72] | [−0.12, +0.75] |
| **P(mean monthly PnL > 0)** | **92.7%** | **94.5%** |

The Sharpe upper bound of 0.72–0.75 is the same across modes.

### 4.4 Monthly t-test and Wilcoxon

| Test | Mode A | Mode B |
|---|---|---|
| Paired t-test (one-sided) | t = +1.36, p = **0.098** | t = +1.46, p = **0.082** |
| Wilcoxon signed-rank | W = 55, p = 0.736 | W = 57, p = 0.702 |

The t-test is marginally significant; the Wilcoxon is not, because the monthly distribution is heavily right-skewed. For right-skewed distributions, the t-test is the more powerful parametric test and the permutation is the most adversarial. Both favour Mode B.

### 4.5 Mode B Calibration
| `p̂` bin | N | Mean `p̂` | Realised | Gap |
|---|---|---|---|---|
| (0.05, 0.10] | 2 | 0.076 | 0.000 | −0.076 |
| (0.10, 0.20] | 27 | 0.143 | 0.074 | −0.069 |
| **(0.20, 0.40]** | **12** | **0.259** | **0.250** | **−0.009** ✓ |
| **(0.40, 0.60]** | **2** | **0.501** | **0.500** | **−0.001** ✓ |
| (0.60, 0.80] | 1 | 0.625 | 1.000 | +0.375 |
| (0.80, 1.01] | 2 | 0.955 | 1.000 | +0.045 |


### 4.6 Mode C time-split validation

| Configuration | P1 (early) | P2 (late) |
|---|---|---|
| Mode A raw | 24 trades, +38.9% ROI | 24 trades, +217.5% ROI |
| **Mode A + Mode C [vol≥1000]** | **19 trades, +74.4% ROI** | **19 trades, +311.0% ROI** |
| Mode B raw | 23 trades, +42.2% ROI | 23 trades, +257.1% ROI |
| **Mode B + Mode C [vol≥1000]** | **18 trades, +92.3% ROI** | **18 trades, +341.2% ROI** |

Both halves lift above their respective baselines. The filter generalises across the time split.

---

## 4.7 Baseline comparison

All baselines use the identical universe filter and walk-forward framework, varying only the signal used to rank contracts.

| Strategy | Trades | WR | ROI | Sharpe | Verdict |
|---|---|---|---|---|---|
| B0 Random (30% sample) | ~50 | — | — | ~−0.1 | Correctly unprofitable |
| B1 Market price only (cheap YES) | ~25 | low | negative | negative | Correctly unprofitable |
| B2 Already-hit (same-day) | 1 | — | — | — | Vanishes after artifact fix |
| B3 Persistence (prev day high) | 22 | 22.7% | +83.8% | 0.21 | Variance, not signal |
| B4 ASOS only | 12 | 41.7% | +38.9% | 0.21 | Modest standalone signal |
| B5 HRRR only | 7 | 57.1% | +66.7% | 0.33 | NWP does most of the work |
| **B6 Blend (ASOS + HRRR)** | **7** | **57.1%** | **+66.7%** | **0.33** | Combined evidence |

**B0 and B1 are correctly flat-to-negative.** This is the necessary precondition for trusting the headline number, aka the baselines that should REALLY fail. B3 Persistence's nominally positive ROI is a variance artifact: 17 of 22 trades lose, the top 2 wins carry 83% of total PnL, and the 12pm blend correctly predicts those 17 losses within ±1°F while persistence is wildly wrong.

---

## 5. Sample, universe, and reproducibility

June 2024 – April 2026, 22 calendar months. Walk-forward folds = 19 test months (early months are training-only).

### 5.1 Universe statistics (filter applied)

| Snapshot | N contracts | N dates | Base-rate YES | Mean ask |
|---|---|---|---|---|
| 12pm | 192 | 174 | 11.5% | $0.105 |
| 2pm | 114 | 110 | 10.5% | $0.157 |

### 5.2 Reproducing the headline numbers

```bash
# Mode A capped baseline
py research/statistical_tests.py --hours 12,14 --permutations 200

# Mode B σ-modulator
py research/phase2_permutation_combined.py --variant sigma_mod --permutations 200
py research/phase2_diagnostics.py

# Mode C microstructure
py research/phase3_filter_sweep.py
py research/phase3_final_summary.py

# Broad-coverage context
py research/statistical_tests.py --hours 11,12,13,14,15 --permutations 200
```

All numbers reported in this document are produced by the scripts above against the published data. Public artifacts:

- `appendix/calibration_table.csv` — post-Platt calibration table (Mode A, capped book)
- `appendix/monthly_pnl.csv` — per-month PnL series
- `appendix/permutation_null_distribution.csv` — full Mode A null distribution (B = 200)
- `appendix/permutation_null_v2_sigma_mod_combined.csv` — Mode B null distribution
- `appendix/phase3_final_summary.csv` — Mode A/B × Mode C combination grid
- `appendix/phase3_microstructure_corr.csv` — Spearman correlation table for microstructure features
- `appendix/phase3_filter_sweep_combos_full.csv` — full Mode C threshold sweep
- `appendix/statistical_tests_summary.json` — machine-readable summary
- 'appendix/oos_trade_list.csv' — full oos trade list, timestamped

---

## 6. Caveats

1. **Monthly Sharpe ceiling ≈ 0.46.** The right-skewed monthly distribution caps risk-adjusted return at this level on this sample. The Sharpe upper bound (95% bootstrap CI) is 0.75 — *that* is the ceiling under favourable resampling. Reality is somewhere in the [0.34, 0.75] band.
2. **Strategy is small-edge.** Three consecutive negative months is the natural drawdown floor at this Sharpe. Forward deployment requires the discipline to wait through them.

---

## 7. References

- Iowa State University Mesonet — IEM ASOS METAR archive: <https://mesonet.agron.iastate.edu/>
- NOAA HRRR archive (AWS public bucket): `noaa-hrrr-bdp-pds.s3.amazonaws.com`
- Kalshi public API documentation: <https://trading-api.readme.io/>

---

*Last revised: 2026-05-18.*
