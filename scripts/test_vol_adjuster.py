"""
Volatility-Structure Weight Adjuster Test

Usage:
    python scripts/test_vol_adjuster.py

Tests:
  1. Individual metrics (vol, ADX, autocorr)
  2. ITC defensive profile with real daily_returns (all 3 metrics)
  3. VEDL momentum profile with real daily_returns
  4. Neutral stock — no adjustment
  5. Production path (no daily_returns) — Metric 3 disabled, no spurious autocorr
  6. momentum untouched — ITC momentum was best strategy (Sharpe 4.18)
  7. No renorm — neutral strategies (volume, risk) unchanged
  8. ITC vs VEDL divergence (same BULL regime, different weights)
  9. describe_stock_personality()
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app.regime.vol_adjuster import (
    adjust_weights, describe_stock_personality,
    _compute_realized_vol, _compute_autocorrelation, _build_return_series,
    VOL_LOW_THRESHOLD, VOL_HIGH_THRESHOLD,
    ADX_WEAK_THRESHOLD, ADX_STRONG_THRESHOLD,
    AUTOCORR_REVERTING, AUTOCORR_TRENDING,
    NEUTRAL_STRATEGIES, TREND_DIRECTION_STRATEGIES,
)

BULL_WEIGHTS = {
    "trend": 0.35, "momentum": 0.25, "reversion": 0.10,
    "breakout": 0.20, "volume": 0.10,
}


def make_returns(n=30, mean=0.0005, std=0.008, autocorr=0.0):
    np.random.seed(42)
    noise = np.random.normal(mean, std, n)
    if autocorr == 0:
        return list(noise)
    series = [noise[0]]
    for i in range(1, n):
        series.append(autocorr * series[-1] + (1 - abs(autocorr)) * noise[i])
    return series


def itc_features():
    returns = make_returns(n=30, std=0.007, autocorr=-0.2)
    return {"adx_14": 15.0, "atr_pct": 1.1,
            "return_1d": returns[-1], "return_5d": sum(returns[-5:]),
            "return_20d": sum(returns[-20:])}, returns


def vedl_features():
    returns = make_returns(n=30, std=0.030, autocorr=+0.2)
    return {"adx_14": 35.0, "atr_pct": 3.5,
            "return_1d": returns[-1], "return_5d": sum(returns[-5:]),
            "return_20d": sum(returns[-20:])}, returns


def neutral_features():
    returns = make_returns(n=30, std=0.015, autocorr=0.0)
    return {"adx_14": 22.0, "atr_pct": 2.0,
            "return_1d": returns[-1], "return_5d": sum(returns[-5:]),
            "return_20d": sum(returns[-20:])}, returns


def test_metrics():
    print("\n── Test 1: Individual metrics ────────────────────────")
    low_vol  = _compute_realized_vol(make_returns(std=0.007))
    high_vol = _compute_realized_vol(make_returns(std=0.030))
    assert low_vol  < VOL_LOW_THRESHOLD,  f"low vol {low_vol:.4f} not < {VOL_LOW_THRESHOLD}"
    assert high_vol > VOL_HIGH_THRESHOLD, f"high vol {high_vol:.4f} not > {VOL_HIGH_THRESHOLD}"
    print(f"  ✅ Realized vol: low={low_vol:.4f} high={high_vol:.4f}")

    ac_rev = _compute_autocorrelation(make_returns(autocorr=-0.3))
    ac_trn = _compute_autocorrelation(make_returns(autocorr=+0.3))
    assert ac_rev < AUTOCORR_REVERTING, f"reverting autocorr {ac_rev:.3f} not < {AUTOCORR_REVERTING}"
    assert ac_trn > AUTOCORR_TRENDING,  f"trending autocorr {ac_trn:.3f} not > {AUTOCORR_TRENDING}"
    print(f"  ✅ Autocorrelation: reverting={ac_rev:.3f} trending={ac_trn:.3f}")


def test_itc_with_real_returns():
    print("\n── Test 2: ITC defensive profile (real daily_returns) ─")
    feat, returns = itc_features()
    adj, notes = adjust_weights("ITC", BULL_WEIGHTS, feat, daily_returns=returns, verbose=True)

    print(f"  trend:     {BULL_WEIGHTS['trend']:.3f} → {adj['trend']:.3f}")
    print(f"  breakout:  {BULL_WEIGHTS['breakout']:.3f} → {adj['breakout']:.3f}")
    print(f"  reversion: {BULL_WEIGHTS['reversion']:.3f} → {adj['reversion']:.3f}")
    print(f"  momentum:  {BULL_WEIGHTS['momentum']:.3f} → {adj['momentum']:.3f}")
    for n in notes: print(f"  → {n}")

    assert adj["trend"]    < BULL_WEIGHTS["trend"],    "trend should decrease"
    assert adj["breakout"] < BULL_WEIGHTS["breakout"], "breakout should decrease"
    # reversion may or may not increase depending on autocorr — both valid
    print("  ✅ ITC: trend↓ breakout↓ with real returns")


def test_vedl_with_real_returns():
    print("\n── Test 3: VEDL momentum profile (real daily_returns) ─")
    feat, returns = vedl_features()
    adj, notes = adjust_weights("VEDL", BULL_WEIGHTS, feat, daily_returns=returns, verbose=True)

    for n in notes: print(f"  → {n}")
    # High vol + strong ADX → no reduction applied (capped at base)
    # trend/breakout stay at regime weight
    assert adj["trend"]    == BULL_WEIGHTS["trend"],    "VEDL trend should stay at regime weight (cap)"
    assert adj["breakout"] == BULL_WEIGHTS["breakout"], "VEDL breakout should stay at regime weight (cap)"
    print("  ✅ VEDL: trend/breakout unchanged (high-vol stock, cap prevents amplification)")


def test_neutral_no_adjustment():
    print("\n── Test 4: Neutral stock — no adjustment ─────────────")
    feat, returns = neutral_features()
    adj, notes = adjust_weights("NEUTRAL", BULL_WEIGHTS, feat, daily_returns=returns, verbose=True)
    print(f"  Notes: {notes}")
    for strat, base_w in BULL_WEIGHTS.items():
        diff = abs(adj[strat] - base_w)
        assert diff < 0.02, f"{strat}: expected ~{base_w:.3f} got {adj[strat]:.3f}"
    print("  ✅ Neutral stock: weights unchanged (all metrics mid-range)")


def test_production_path_no_daily_returns():
    print("\n── Test 5: Production path — no daily_returns ────────")
    # This is what fuse() calls — no daily_returns kwarg
    feat, _ = itc_features()
    adj, notes = adjust_weights("ITC_PROD", BULL_WEIGHTS, feat)  # no daily_returns

    # Verify fallback autocorr bug does NOT fire
    fallback_series = _build_return_series(feat)
    fallback_ac = _compute_autocorrelation(fallback_series)
    print(f"  Fallback autocorr would be: {fallback_ac:.4f} (always ~0.87 — structurally invalid)")

    # In production: only vol + ADX fire for ITC (both correctly reduce trend/breakout)
    # Autocorr metric is SKIPPED → no spurious +0.08 cancellation
    assert adj["trend"]    < BULL_WEIGHTS["trend"],    "trend should decrease (vol+ADX fire)"
    assert adj["breakout"] < BULL_WEIGHTS["breakout"], "breakout should decrease"

    has_autocorr_note = any("autocorr" in n for n in notes)
    assert not has_autocorr_note, f"autocorr metric should not fire in production path, got: {notes}"

    print(f"  trend:    {BULL_WEIGHTS['trend']:.3f} → {adj['trend']:.3f}")
    print(f"  breakout: {BULL_WEIGHTS['breakout']:.3f} → {adj['breakout']:.3f}")
    print(f"  Notes: {notes}")
    print("  ✅ Production path: Metric 3 disabled, only valid vol+ADX adjustments apply")


def test_momentum_untouched():
    print("\n── Test 6: momentum is neutral (not penalised) ───────")
    assert "momentum" not in TREND_DIRECTION_STRATEGIES, \
        "momentum should NOT be in TREND_DIRECTION_STRATEGIES"
    assert "momentum" in NEUTRAL_STRATEGIES, \
        "momentum should be in NEUTRAL_STRATEGIES"

    feat, returns = itc_features()
    adj, _ = adjust_weights("ITC", BULL_WEIGHTS, feat, daily_returns=returns)

    assert adj["momentum"] == BULL_WEIGHTS["momentum"], (
        f"momentum changed {BULL_WEIGHTS['momentum']} → {adj['momentum']} "
        f"— it should be untouched (ITC momentum Sharpe=4.18 in backtest)"
    )
    print(f"  momentum: {BULL_WEIGHTS['momentum']:.3f} → {adj['momentum']:.3f} (unchanged)")
    print("  ✅ momentum correctly left in neutral group")


def test_no_renorm_neutral_unchanged():
    print("\n── Test 7: No renorm — neutral strategies unchanged ──")
    feat, returns = itc_features()
    adj, _ = adjust_weights("ITC", BULL_WEIGHTS, feat, daily_returns=returns)

    for strat in NEUTRAL_STRATEGIES:
        if strat in BULL_WEIGHTS:
            assert adj[strat] == BULL_WEIGHTS[strat], (
                f"{strat} changed {BULL_WEIGHTS[strat]} → {adj[strat]} "
                f"— neutral strategies should be untouched, no renorm artefacts"
            )
            print(f"  {strat}: {BULL_WEIGHTS[strat]:.4f} → {adj[strat]:.4f} ✅ unchanged")
    print("  ✅ No renorm bleed — volume/risk/momentum exactly at regime weight")


def test_itc_vs_vedl_divergence():
    print("\n── Test 8: ITC vs VEDL — same BULL regime ────────────")
    itc_feat,  itc_ret   = itc_features()
    vedl_feat, vedl_ret  = vedl_features()

    itc_adj,  _ = adjust_weights("ITC",  BULL_WEIGHTS, itc_feat,  daily_returns=itc_ret)
    vedl_adj, _ = adjust_weights("VEDL", BULL_WEIGHTS, vedl_feat, daily_returns=vedl_ret)

    print(f"  {'strategy':<12} {'BULL base':>10} {'ITC':>10} {'VEDL':>10}")
    print(f"  {'-'*44}")
    for s in BULL_WEIGHTS:
        print(f"  {s:<12} {BULL_WEIGHTS[s]:>10.3f} {itc_adj[s]:>10.3f} {vedl_adj[s]:>10.3f}")

    assert itc_adj["trend"]    < vedl_adj["trend"],    "ITC trend should be lower than VEDL"
    assert itc_adj["breakout"] < vedl_adj["breakout"], "ITC breakout should be lower than VEDL"
    assert itc_adj["momentum"] == vedl_adj["momentum"] == BULL_WEIGHTS["momentum"], \
        "momentum should be equal for both (neutral)"
    print("  ✅ Same BULL regime → correct divergence: ITC trend/breakout lower than VEDL")


def test_personality_profile():
    print("\n── Test 9: describe_stock_personality() ──────────────")
    itc_feat,  itc_ret  = itc_features()
    vedl_feat, vedl_ret = vedl_features()

    itc_p  = describe_stock_personality("ITC",  itc_feat,  daily_returns=itc_ret)
    vedl_p = describe_stock_personality("VEDL", vedl_feat, daily_returns=vedl_ret)

    print(f"  ITC  personality: {itc_p['personality']}  (metrics_active={itc_p['metrics_active']})")
    print(f"  VEDL personality: {vedl_p['personality']} (metrics_active={vedl_p['metrics_active']})")

    assert itc_p["personality"]  == "DEFENSIVE/MEAN-REVERTING"
    assert vedl_p["personality"] == "MOMENTUM/TRENDING"

    # Production profile (no daily_returns) — autocorr shows as 'not computed'
    itc_prod = describe_stock_personality("ITC_PROD", itc_feat)
    assert itc_prod["autocorrelation"] == "not computed"
    assert itc_prod["metrics_active"] == 2
    print(f"  ITC (production): autocorr='not computed', metrics_active=2")
    print("  ✅ Personality profiles correct; production profile shows 2 active metrics")


def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — Vol-Structure Adjuster Test")
    print(f"{'='*60}")
    print(f"  Vol:     low<{VOL_LOW_THRESHOLD} | high>{VOL_HIGH_THRESHOLD}")
    print(f"  ADX:     weak<{ADX_WEAK_THRESHOLD} | strong>{ADX_STRONG_THRESHOLD}")
    print(f"  Autocorr: reverting<{AUTOCORR_REVERTING} | trending>{AUTOCORR_TRENDING}")
    print(f"  Metric 3 only active when daily_returns provided")

    test_metrics()
    test_itc_with_real_returns()
    test_vedl_with_real_returns()
    test_neutral_no_adjustment()
    test_production_path_no_daily_returns()
    test_momentum_untouched()
    test_no_renorm_neutral_unchanged()
    test_itc_vs_vedl_divergence()
    test_personality_profile()

    print(f"\n{'='*60}")
    print("  ✅ Volatility-Structure Weight Adjuster verified")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()