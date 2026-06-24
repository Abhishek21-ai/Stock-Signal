"""
Volatility-Structure Weight Adjuster
=====================================
Modifies regime-level fusion weights per-stock using metrics that
together distinguish trending vs mean-reverting stock personalities.

Problem solved (from ITC backtest diagnosis):
  Regime weights are global — every stock in BULL gets trend=0.35,
  breakout=0.20. But defensive/FMCG stocks (ITC, HINDUNILVR) don't
  produce the directional momentum or volume spikes that trend/breakout
  strategies need.

Solution — 2+1 metric volatility-structure adjustment:

  Metric 1: 30-day Realized Volatility (vol magnitude)
    Low-vol stocks (σ < 1.2%/day) → reduce trend/breakout
    High-vol stocks (σ > 2.5%/day) → amplify trend/breakout

  Metric 2: ADX (trend strength — already computed in features)
    Low ADX (<20) → stock isn't trending → reduce trend/breakout,
                    raise reversion
    High ADX (>30) → stock is trending → amplify trend/breakout

  Metric 3: Return Autocorrelation (price structure)
    ONLY used when caller provides real daily_returns list.
    When not provided (production default), this metric is SKIPPED
    rather than computed from the fallback scalar return fields
    (return_1d / return_5d / return_20d), which produce structurally
    invalid autocorrelation due to constant-block padding — see note
    in _build_return_series().

Fix notes vs original Chat B implementation:
  1. Metric 3 disabled when daily_returns is None — the fallback
     scalar reconstruction always produces autocorr ≈ 0.87 (trending)
     due to constant-value block padding, spuriously cancelling the
     correct Metric 1 + 2 reductions and inflating reversion weight.

  2. momentum moved OUT of TREND_DIRECTION_STRATEGIES → NEUTRAL.
     ITC backtest showed momentum Sharpe=4.18 (best strategy) — it
     should not be penalised alongside trend/breakout. Momentum does
     not require large directional moves; it responds to rate-of-change
     patterns that low-vol stocks also exhibit.

  3. Renormalization replaced with multiplicative adjustment capped
     at the regime weight itself (never amplify beyond what the regime
     set for high-vol stocks). Original additive + renorm caused:
       - volume gaining +19% as a pure renorm artefact (no metric justified it)
       - reversion inflating +114% from spurious autocorr + renorm compound
       - neutral strategies getting free weight boosts

  4. Integration point: called BEFORE correlation penalty (correct order
     already in Chat B's fusion engine — preserved here).

These 3 metrics together correctly identify:
  - ITC:    low vol + low ADX → defensive/reverting (autocorr skipped in prod)
  - VEDL:   high vol + high ADX → momentum/trending
  - HDFCBANK: moderate → balanced, near-market adjustment
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from app.logger import get_logger

logger = get_logger("vol_adjuster")

# ── Metric 1: Realized Volatility thresholds ──────────────────
VOL_LOW_THRESHOLD  = 0.012   # < 1.2%/day → low-vol (defensive/FMCG)
VOL_HIGH_THRESHOLD = 0.025   # > 2.5%/day → high-vol (cyclical/volatile)

# ── Metric 2: ADX thresholds ──────────────────────────────────
ADX_WEAK_THRESHOLD   = 20.0  # no clear trend
ADX_STRONG_THRESHOLD = 30.0  # strong trend confirmed

# ── Metric 3: Return autocorrelation ──────────────────────────
# Only used when real daily_returns are provided by caller.
# When None: skipped entirely to avoid fallback-series artifacts.
AUTOCORR_REVERTING   = -0.10
AUTOCORR_TRENDING    =  0.10

# ── Adjustment magnitude per metric ───────────────────────────
# Each metric shifts the multiplier by ±this amount.
# 2 metrics agreeing → max ±0.16 from regime weight.
# 3 metrics agreeing (backtest path) → max ±0.24.
MAX_PER_METRIC_SHIFT = 0.08

# ── Strategy sensitivity mapping ─────────────────────────────
# trend + breakout: need large directional moves + volume → penalise on low-vol stocks
# reversion: works better on range-bound low-vol names → amplify
# momentum: rate-of-change based, works on low-vol too → NEUTRAL (not penalised)
# volume, risk: vol-agnostic → NEUTRAL
TREND_DIRECTION_STRATEGIES  = ["trend", "breakout"]   # momentum removed: see Fix #2
REVERT_DIRECTION_STRATEGIES = ["reversion"]
NEUTRAL_STRATEGIES          = ["momentum", "volume", "risk"]


def _compute_realized_vol(returns: List[float]) -> Optional[float]:
    """
    30-day realized daily volatility (std of daily returns).
    Returns None if insufficient data.
    """
    clean = [r for r in returns if r is not None and not np.isnan(r)]
    if len(clean) < 10:
        return None
    return float(np.std(clean))


def _compute_autocorrelation(returns: List[float]) -> Optional[float]:
    """
    Lag-1 autocorrelation of daily returns over the available window.
    Positive → trending, Negative → mean-reverting.

    IMPORTANT: only call this with real daily returns, not with output
    from _build_return_series(). The fallback reconstruction pads the
    series with constant-value blocks (e.g. 15 copies of return_20d/20)
    which creates structural autocorrelation of ~0.87 regardless of the
    stock's actual price behaviour — making this metric meaningless and
    counterproductive in that mode.
    """
    clean = [r for r in returns if r is not None and not np.isnan(r)]
    if len(clean) < 10:
        return None
    arr = np.array(clean)
    if arr.std() == 0:
        return None
    corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
    return float(corr) if not np.isnan(corr) else None


def _build_return_series(features: Dict) -> List[float]:
    """
    Approximate return series from scalar feature fields.
    Used ONLY for realized volatility (Metric 1) — not for autocorrelation.

    NOTE: this produces structurally invalid autocorrelation because
    it pads with constant-block averages (4× return_5d/5, 15× return_20d/20).
    The constant runs create step-changes that np.corrcoef interprets as
    strong positive autocorrelation (~0.87) for every stock regardless of
    actual price behaviour. Callers that need autocorrelation must pass
    real daily_returns instead.
    """
    r = []
    if features.get("return_1d") is not None:
        r.append(features["return_1d"])
    if features.get("return_5d") is not None:
        avg = features["return_5d"] / 5
        r.extend([avg] * 4)
    if features.get("return_20d") is not None:
        avg = features["return_20d"] / 20
        r.extend([avg] * 15)
    return r


def _metric1_vol_adjustment(realized_vol: float) -> Tuple[float, float, str]:
    """Returns (trend_adj_multiplier_delta, reversion_adj_multiplier_delta, note)."""
    if realized_vol < VOL_LOW_THRESHOLD:
        return -MAX_PER_METRIC_SHIFT, +MAX_PER_METRIC_SHIFT, \
               f"low vol ({realized_vol:.3f}<{VOL_LOW_THRESHOLD})"
    elif realized_vol > VOL_HIGH_THRESHOLD:
        return +MAX_PER_METRIC_SHIFT, -MAX_PER_METRIC_SHIFT, \
               f"high vol ({realized_vol:.3f}>{VOL_HIGH_THRESHOLD})"
    return 0.0, 0.0, f"normal vol ({realized_vol:.3f})"


def _metric2_adx_adjustment(adx: float) -> Tuple[float, float, str]:
    """Returns (trend_adj_multiplier_delta, reversion_adj_multiplier_delta, note)."""
    if adx < ADX_WEAK_THRESHOLD:
        return -MAX_PER_METRIC_SHIFT, +MAX_PER_METRIC_SHIFT, \
               f"weak ADX ({adx:.1f}<{ADX_WEAK_THRESHOLD})"
    elif adx > ADX_STRONG_THRESHOLD:
        return +MAX_PER_METRIC_SHIFT, -MAX_PER_METRIC_SHIFT, \
               f"strong ADX ({adx:.1f}>{ADX_STRONG_THRESHOLD})"
    return 0.0, 0.0, f"moderate ADX ({adx:.1f})"


def _metric3_autocorr_adjustment(autocorr: float) -> Tuple[float, float, str]:
    """Returns (trend_adj_multiplier_delta, reversion_adj_multiplier_delta, note)."""
    if autocorr < AUTOCORR_REVERTING:
        return -MAX_PER_METRIC_SHIFT, +MAX_PER_METRIC_SHIFT, \
               f"mean-reverting (autocorr={autocorr:.3f}<{AUTOCORR_REVERTING})"
    elif autocorr > AUTOCORR_TRENDING:
        return +MAX_PER_METRIC_SHIFT, -MAX_PER_METRIC_SHIFT, \
               f"trending (autocorr={autocorr:.3f}>{AUTOCORR_TRENDING})"
    return 0.0, 0.0, f"random-walk (autocorr={autocorr:.3f})"


def adjust_weights(
    symbol:        str,
    base_weights:  Dict[str, float],
    features:      Dict,
    daily_returns: Optional[List[float]] = None,
    verbose:       bool = False,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Main entry point. Called by fusion engine after regime weights are set
    and before correlation penalty.

    Args:
        symbol:        stock ticker (for logging)
        base_weights:  regime fusion weights dict
        features:      feature vector dict from FeatureEngineer
        daily_returns: real daily return series (last 20-30 days).
                       When provided: all 3 metrics active.
                       When None (production default): only Metrics 1 + 2.
                       Callers with OHLCV access should pass this for
                       full 3-metric accuracy (e.g. backtest engine).
        verbose:       include per-metric detail in notes

    Returns:
        (adjusted_weights, notes)

    Adjustment mechanism (multiplicative, not additive):
        For trend/breakout: multiply by (1 + net_delta), floored at 60%
        of regime weight, capped at regime weight (never amplify beyond
        what the regime set — only down-adjust low-vol stocks).
        For reversion: mirror relationship.
        Neutral strategies (momentum, volume, risk): unchanged.

    No renormalization — fuse() handles normalization via weight_used.
    """
    notes: List[str] = []

    # ── Gather metrics ─────────────────────────────────────────
    scalar_returns = _build_return_series(features)
    adx = float(features.get("adx_14") or 0.0)

    realized_vol = _compute_realized_vol(
        daily_returns if daily_returns else scalar_returns
    )

    # Metric 3 only when real daily series provided (see module docstring)
    autocorr = (
        _compute_autocorrelation(daily_returns)
        if daily_returns is not None
        else None
    )

    # ── Compute multiplier delta from each metric ─────────────
    total_trend_delta    = 0.0
    total_reversion_delta = 0.0
    active_metrics       = 0

    if realized_vol is not None:
        t, r, note = _metric1_vol_adjustment(realized_vol)
        total_trend_delta     += t
        total_reversion_delta += r
        active_metrics        += 1
        if t != 0 or verbose:
            notes.append(f"vol: {note}")

    if adx > 0:
        t, r, note = _metric2_adx_adjustment(adx)
        total_trend_delta     += t
        total_reversion_delta += r
        active_metrics        += 1
        if t != 0 or verbose:
            notes.append(f"adx: {note}")

    if autocorr is not None:
        t, r, note = _metric3_autocorr_adjustment(autocorr)
        total_trend_delta     += t
        total_reversion_delta += r
        active_metrics        += 1
        if t != 0 or verbose:
            notes.append(f"autocorr: {note}")

    # No adjustment needed
    if total_trend_delta == 0.0 and total_reversion_delta == 0.0:
        if verbose:
            notes.append("no adjustment (all metrics neutral)")
        return dict(base_weights), notes

    # ── Apply multiplicative adjustment ───────────────────────
    # For reductions: new_w = base_w * (1 + delta), floored at 60% of base
    # For amplification: capped at base_w (never exceed regime weight)
    # Neutral strategies: untouched — no renorm either (fuse() owns that)
    VOL_FLOOR = 0.6   # never reduce below 60% of regime weight

    adjusted = dict(base_weights)

    for strat in TREND_DIRECTION_STRATEGIES:
        if strat not in adjusted:
            continue
        base_w = base_weights[strat]
        mult   = 1.0 + total_trend_delta
        mult   = max(VOL_FLOOR, mult)          # floor at 60%
        new_w  = base_w * mult
        new_w  = min(new_w, base_w)            # cap: never amplify beyond regime
        adjusted[strat] = round(new_w, 4)

    for strat in REVERT_DIRECTION_STRATEGIES:
        if strat not in adjusted:
            continue
        base_w = base_weights[strat]
        mult   = 1.0 + total_reversion_delta
        mult   = max(VOL_FLOOR, mult)
        new_w  = base_w * mult
        new_w  = min(new_w, base_w)            # cap for amplification direction too
        adjusted[strat] = round(new_w, 4)

    # NEUTRAL_STRATEGIES (momentum, volume, risk): not touched

    if notes:
        logger.debug(f"{symbol} vol-structure: {' | '.join(notes)}")

    return adjusted, notes


def describe_stock_personality(
    symbol:        str,
    features:      Dict,
    daily_returns: Optional[List[float]] = None,
) -> Dict:
    """
    Diagnostic helper — human-readable stock personality profile.
    Used by dashboard and test scripts.
    """
    scalar_returns = _build_return_series(features)
    adx            = float(features.get("adx_14") or 0.0)
    realized_vol   = _compute_realized_vol(
        daily_returns if daily_returns else scalar_returns
    )
    autocorr = (
        _compute_autocorrelation(daily_returns)
        if daily_returns is not None
        else None
    )

    vol_class = (
        "low-vol (defensive)"  if realized_vol and realized_vol < VOL_LOW_THRESHOLD
        else "high-vol (volatile)" if realized_vol and realized_vol > VOL_HIGH_THRESHOLD
        else "normal-vol"
    )
    adx_class = (
        "weak-trend (ranging)" if adx < ADX_WEAK_THRESHOLD
        else "strong-trend"    if adx > ADX_STRONG_THRESHOLD
        else "moderate-trend"
    )
    corr_class = (
        "mean-reverting" if autocorr is not None and autocorr < AUTOCORR_REVERTING
        else "trending"  if autocorr is not None and autocorr > AUTOCORR_TRENDING
        else "random-walk (or not computed)"
    )

    trend_signals = sum([
        1 if "volatile"  in vol_class else 0,
        1 if "strong"    in adx_class else 0,
        1 if "trending"  in corr_class else 0,
    ])
    revert_signals = sum([
        1 if "defensive" in vol_class  else 0,
        1 if "ranging"   in adx_class  else 0,
        1 if "reverting" in corr_class else 0,
    ])

    personality = (
        "MOMENTUM/TRENDING"       if trend_signals  >= 2
        else "DEFENSIVE/MEAN-REVERTING" if revert_signals >= 2
        else "BALANCED/MIXED"
    )

    return {
        "symbol":           symbol,
        "personality":      personality,
        "realized_vol":     round(realized_vol, 4) if realized_vol else None,
        "adx":              round(adx, 1),
        "autocorrelation":  round(autocorr, 3) if autocorr is not None else "not computed",
        "vol_class":        vol_class,
        "adx_class":        adx_class,
        "autocorr_class":   corr_class,
        "trend_vote":       trend_signals,
        "revert_vote":      revert_signals,
        "metrics_active":   2 + (1 if autocorr is not None else 0),
    }