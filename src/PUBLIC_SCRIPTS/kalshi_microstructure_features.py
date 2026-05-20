"""
Kalshi candlestick microstructure feature extraction.

For each (ticker, snapshot_time) entry in the 60-minute candlestick file, the
contract has a full lifetime of prior snapshots. This module pre-loads the
per-ticker timeline once, then exposes a builder that produces a row of
microstructure features at any requested snapshot time.

Feature groups
--------------
* Price momentum: mid_yes drift over 1h / 3h / 6h prior to snapshot
* Price volatility: std + drawdown over 6h
* Volume: recent activity, burst-vs-lifetime z-score
* Spread / liquidity: current and 6h-avg spread, tight-flag, bid-zero-flag
* Open interest: growth over 3h, OI-to-volume ratio
* In-spread skew: where the mid sits between bid and ask
* (Optional) Model-market disagreement: requires `p_model` from the strategy

Public API
----------
    micro = MicrostructureExtractor(jsonl_path)
    feats = micro.features_at("KXHIGHCHI-26APR13-T76", as_of_iso)

Each call returns a flat dict of feature → float / int.

Usage from a strategy script (sketch)
-------------------------------------
    micro = MicrostructureExtractor(CONTRACTS_JSONL)
    df["spread_now_cents"] = df.apply(
        lambda r: micro.features_at(r["ticker"], r["as_of"]).get("spread_now_cents", np.nan),
        axis=1,
    )

Cost
----
First load of the JSONL is ~5 seconds and ~150 MB RAM. After that all
feature lookups are O(log n) per ticker via sorted timestamps.
"""
from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_LS   = _HERE.parent
DEFAULT_JSONL = _LS / "data" / "backfill_chicago" / "chicago_60m.jsonl"


class MicrostructureExtractor:
    """Loads the 60-minute candlestick JSONL once and provides fast feature
    lookups per (ticker, snapshot_time)."""

    def __init__(self, jsonl_path: Path | str = DEFAULT_JSONL):
        self.path = Path(jsonl_path)
        # per-ticker sorted lists of (as_of_ts_seconds, row_dict)
        self._timelines: dict[str, list[tuple[float, dict]]] = {}
        self._timestamps: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        per_ticker: dict[str, list[tuple[float, dict]]] = defaultdict(list)
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                t = r.get("ticker")
                a = r.get("as_of")
                if t is None or a is None:
                    continue
                # parse timestamp to seconds (avoid full pandas datetime cost in loop)
                try:
                    ts = pd.Timestamp(a).timestamp()
                except Exception:
                    continue
                per_ticker[t].append((ts, r))
        # sort each timeline, build sidecar timestamp index
        for t, rows in per_ticker.items():
            rows.sort(key=lambda x: x[0])
            self._timelines[t] = rows
            self._timestamps[t] = [r[0] for r in rows]

    # --- accessor helpers -------------------------------------------------

    def _slice_back(self, ticker: str, as_of_ts: float, hours: float
                    ) -> list[dict]:
        """Return all rows for `ticker` with timestamp in [as_of - hours, as_of]."""
        ts_list = self._timestamps.get(ticker, [])
        if not ts_list:
            return []
        lo_ts = as_of_ts - hours * 3600.0
        lo = bisect.bisect_left(ts_list, lo_ts)
        hi = bisect.bisect_right(ts_list, as_of_ts)
        return [self._timelines[ticker][i][1] for i in range(lo, hi)]

    def _row_at_or_before(self, ticker: str, as_of_ts: float) -> dict | None:
        ts_list = self._timestamps.get(ticker, [])
        if not ts_list:
            return None
        i = bisect.bisect_right(ts_list, as_of_ts) - 1
        if i < 0:
            return None
        return self._timelines[ticker][i][1]

    def lifetime_hourly_volume_avg(self, ticker: str) -> float:
        rows = self._timelines.get(ticker, [])
        if not rows:
            return float("nan")
        vols = [r[1].get("volume") for r in rows]
        vols = [v for v in vols if v is not None and not math.isnan(float(v) if isinstance(v, (int, float)) else float("nan"))]
        if not vols:
            return float("nan")
        return float(np.mean(vols))

    # --- main feature builder ---------------------------------------------

    def features_at(self, ticker: str, as_of: str | pd.Timestamp,
                    p_model: float | None = None,
                    snapshot_h: int | None = None) -> dict[str, Any]:
        """
        Return a flat dict of microstructure features for `ticker` at snapshot
        time `as_of`. Missing inputs => NaN/None for affected outputs.

        If `p_model` is provided, also compute model-market disagreement
        signals.  `snapshot_h` is unused but accepted so callers can pass it
        without breaking.
        """
        if isinstance(as_of, str):
            as_of_ts = pd.Timestamp(as_of).timestamp()
        else:
            as_of_ts = as_of.timestamp()

        out: dict[str, Any] = {}

        # The "now" row — the row at the snapshot itself
        now = self._row_at_or_before(ticker, as_of_ts)
        if now is None:
            return out  # contract not in data / no prior history

        # ---- Current state ------------------------------------------------
        out["spread_now_cents"]      = _safe_float(now.get("spread_cents"))
        out["mid_now"]               = _safe_float(now.get("mid_yes"))
        out["yes_bid_now"]           = _safe_float(now.get("yes_bid"))
        out["yes_ask_now"]           = _safe_float(now.get("yes_ask"))
        out["volume_now"]            = _safe_float(now.get("volume"))
        out["open_interest_now"]     = _safe_float(now.get("open_interest"))
        out["minutes_since_open"]    = _safe_float(now.get("minutes_since_open"))
        out["time_to_close_minutes"] = _safe_float(now.get("time_to_close_minutes"))

        # Where the mid sits inside the bid/ask range
        bid = out["yes_bid_now"]
        ask = out["yes_ask_now"]
        mid = out["mid_now"]
        if bid is not None and ask is not None and mid is not None and ask > bid:
            out["mid_skew_in_spread"] = (mid - bid) / (ask - bid)
        else:
            out["mid_skew_in_spread"] = float("nan")

        # Flags
        out["bid_zero_flag"] = 1 if (bid is not None and bid <= 0.0) else 0
        out["spread_tight_flag"] = 1 if (out["spread_now_cents"] is not None and out["spread_now_cents"] <= 1.0) else 0

        # ---- Look-back windows --------------------------------------------
        for hours in (1, 3, 6):
            window = self._slice_back(ticker, as_of_ts, hours)
            # Strip rows with missing mid
            mids = [_safe_float(r.get("mid_yes")) for r in window]
            mids = [m for m in mids if m is not None]
            vols = [_safe_float(r.get("volume")) for r in window]
            vols = [v for v in vols if v is not None]
            spreads = [_safe_float(r.get("spread_cents")) for r in window]
            spreads = [s for s in spreads if s is not None]
            ois = [_safe_float(r.get("open_interest")) for r in window]
            ois = [o for o in ois if o is not None]

            # Price drift: mid_now − mid_at_window_start
            if mids and len(mids) >= 2:
                drift_pp = (mids[-1] - mids[0]) * 100  # in cents
                out[f"mid_change_{hours}h_pp"] = drift_pp
            else:
                out[f"mid_change_{hours}h_pp"] = float("nan")

            # Volume sum
            if vols:
                out[f"volume_{hours}h_total"] = float(sum(vols))
            else:
                out[f"volume_{hours}h_total"] = 0.0

            # Spread average + max
            if spreads:
                out[f"spread_{hours}h_avg"] = float(np.mean(spreads))
                out[f"spread_{hours}h_max"] = float(max(spreads))
            else:
                out[f"spread_{hours}h_avg"] = float("nan")
                out[f"spread_{hours}h_max"] = float("nan")

            # OI growth over window
            if len(ois) >= 2:
                out[f"oi_growth_{hours}h"] = ois[-1] - ois[0]
                if ois[0] > 0:
                    out[f"oi_growth_{hours}h_pct"] = (ois[-1] - ois[0]) / ois[0] * 100
                else:
                    out[f"oi_growth_{hours}h_pct"] = float("nan")
            else:
                out[f"oi_growth_{hours}h"] = float("nan")
                out[f"oi_growth_{hours}h_pct"] = float("nan")

        # ---- 6h volatility & drawdown ------------------------------------
        window6 = self._slice_back(ticker, as_of_ts, 6)
        mids6 = [_safe_float(r.get("mid_yes")) for r in window6]
        mids6 = [m for m in mids6 if m is not None]
        if len(mids6) >= 3:
            out["mid_std_6h"] = float(np.std(mids6, ddof=1)) * 100  # in cents
            peak = max(mids6)
            out["mid_drawdown_from_peak_6h_pp"] = (mids6[-1] - peak) * 100
            out["mid_max_6h"] = peak * 100
            out["mid_min_6h"] = min(mids6) * 100
        else:
            out["mid_std_6h"] = float("nan")
            out["mid_drawdown_from_peak_6h_pp"] = float("nan")
            out["mid_max_6h"] = float("nan")
            out["mid_min_6h"] = float("nan")

        # ---- Volume burst vs lifetime ------------------------------------
        avg = self.lifetime_hourly_volume_avg(ticker)
        cur_vol = out["volume_now"]
        if avg and avg > 0 and cur_vol is not None:
            out["volume_burst_ratio"] = float(cur_vol) / avg
        else:
            out["volume_burst_ratio"] = float("nan")

        # OI-to-volume ratio
        if out["volume_now"] and out["volume_now"] > 0 and out["open_interest_now"] is not None:
            out["oi_to_volume_ratio"] = out["open_interest_now"] / out["volume_now"]
        else:
            out["oi_to_volume_ratio"] = float("nan")

        # ---- Model-market disagreement (if p_model supplied) -------------
        if p_model is not None and out["mid_now"] is not None:
            edge_pp = (p_model - out["mid_now"]) * 100  # cents of mispricing per side
            out["model_minus_market_pp"] = edge_pp
            # Has the market been MOVING toward or away from the model edge?
            drift_3h = out.get("mid_change_3h_pp", float("nan"))
            if not math.isnan(drift_3h):
                # If model_p > market mid, edge is positive; market moving UP
                # toward model is "confirming" the edge
                out["market_drift_confirms_model"] = (
                    1 if (edge_pp > 0 and drift_3h > 0) or (edge_pp < 0 and drift_3h < 0) else 0
                )
            else:
                out["market_drift_confirms_model"] = None
        else:
            out["model_minus_market_pp"] = None
            out["market_drift_confirms_model"] = None

        return out


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


# ---------------------------------------------------------------------------
# Convenience: build features for an entire trade list
# ---------------------------------------------------------------------------

def attach_microstructure_features(
    trade_df: pd.DataFrame,
    jsonl_path: Path | str = DEFAULT_JSONL,
    ticker_col: str = "ticker",
    as_of_col: str = "as_of",
    p_model_col: str | None = "p_model",
) -> pd.DataFrame:
    """Vectorised helper: returns the input DataFrame with microstructure
    columns added. Single pass through the 60m JSONL."""
    print(f"Loading {jsonl_path} …", flush=True)
    micro = MicrostructureExtractor(jsonl_path)
    print(f"  Loaded {len(micro._timelines):,} tickers", flush=True)

    out = trade_df.copy()
    feature_rows = []
    for _, r in out.iterrows():
        p = float(r[p_model_col]) if (p_model_col and p_model_col in r and pd.notna(r[p_model_col])) else None
        feats = micro.features_at(r[ticker_col], r[as_of_col], p_model=p)
        feature_rows.append(feats)
    feat_df = pd.DataFrame(feature_rows).add_prefix("micro_")
    return pd.concat([out.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    # CLI smoke test: load and dump features for a known ticker
    import sys
    micro = MicrostructureExtractor()
    print(f"Loaded {len(micro._timelines):,} tickers from {DEFAULT_JSONL.name}")
    if len(sys.argv) >= 3:
        ticker, as_of = sys.argv[1], sys.argv[2]
        feats = micro.features_at(ticker, as_of)
        print(f"\nFeatures for {ticker} at {as_of}:")
        for k, v in feats.items():
            print(f"  {k:>32}: {v}")
    else:
        # Default smoke test on a known winning trade
        ticker = "KXHIGHCHI-26APR13-T76"   # the 2026-04-13 +95¢ winner
        as_of  = "2026-04-13T19:00:00+00:00"  # 14h Chicago = 19 UTC
        feats = micro.features_at(ticker, as_of, p_model=0.78)
        print(f"\nFeatures for {ticker} at {as_of} (with p_model=0.78):")
        for k, v in feats.items():
            print(f"  {k:>32}: {v}")
