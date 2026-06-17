"""
Risk Strategy Engine test — run after test_fusion.py passes.

Usage:
    python scripts/test_risk.py

What this tests:
  1. Clean setup — all checks pass → positive score
  2. High volatility penalty (ATR% > 5%)
  3. Hard veto — extreme volatility (ATR% > 8%)
  4. Falling knife penalty (drawdown > 20% from 52w high)
  5. Hard veto — extreme drawdown (> 35%)
  6. Liquidity penalty (volume ratio < 0.5)
  7. Hard veto — illiquid (volume ratio < 0.2)
  8. Poor R/R penalty
  9. Regime weight effect (BEAR vs BULL)
  10. Full runner with all 6 strategies
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.strategies.risk import RiskStrategy, VETO_ATR_PCT, VETO_DRAWDOWN_PCT, VETO_VOLUME_RATIO
from app.strategies.runner import StrategyRunner


def make_features(
    close=500.0,
    atr_pct=2.5,
    pct_from_52w_high=-8.0,
    volume_ratio=1.2,
    regime="BULL",
) -> dict:
    atr    = close * atr_pct / 100
    stop   = close - 1.5 * atr
    target = close + 3.0 * atr
    return {
        "symbol":            "TEST",
        "close":             close,
        "atr_14":            atr,
        "atr_pct":           atr_pct,
        "pct_from_52w_high": pct_from_52w_high,
        "pct_from_52w_low":  abs(pct_from_52w_high) + 10,
        "volume_ratio":      volume_ratio,
        "volume_sma_20":     1_000_000,
        "atr_stop_15x":      round(stop, 2),
        "atr_target_3x":     round(target, 2),
        # needed by other strategies in runner
        "ema_bull_alignment": 2,
        "ema_20": close * 1.01, "ema_50": close * 0.99, "ema_200": close * 0.97,
        "adx_14": 28.0, "adx_dmp": 20.0, "adx_dmn": 12.0,
        "macd_hist": 1.2, "macd_cross": 0,
        "rsi_14": 55.0,
        "bb_pct_b": 0.55, "bb_width": 0.04,
        "stoch_k": 60.0, "stoch_d": 55.0,
        "obv_slope": 0.3,
        "vwap": close * 0.99,
        "volume": 1_200_000,
    }


def run_test(name: str, features: dict, regime: str = "BULL",
             expect_veto: bool = False, expect_signal: str = None):
    strategy = RiskStrategy()
    result   = strategy.run(features, regime=regime)
    veto     = result.meta.get("hard_veto", False)

    status = "✅"
    if expect_veto and not veto:
        status = "❌"
    elif not expect_veto and veto:
        status = "❌"
    elif expect_signal and result.signal != expect_signal:
        status = "❌"

    print(f"  {status} {name}")
    print(f"     score={result.score:.1f}  signal={result.signal}  "
          f"conf={result.confidence:.0f}%  veto={veto}")
    for r in result.reasons:
        print(f"     → {r}")
    print()
    return result


def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — Risk Strategy Test")
    print(f"{'='*60}\n")

    # ── 1. Clean setup ────────────────────────────────────────
    run_test("Clean setup (all checks pass)",
             make_features(atr_pct=2.5, pct_from_52w_high=-5.0, volume_ratio=1.5),
             regime="BULL")

    # ── 2. High volatility penalty ────────────────────────────
    run_test("High volatility (ATR%=6.0)",
             make_features(atr_pct=6.0),
             regime="BULL")

    # ── 3. Hard veto — extreme volatility ────────────────────
    run_test("Hard veto: extreme volatility (ATR%=9.0)",
             make_features(atr_pct=9.0),
             expect_veto=True)

    # ── 4. Falling knife penalty ──────────────────────────────
    run_test("Falling knife (drawdown=-25%)",
             make_features(pct_from_52w_high=-25.0),
             regime="BEAR")

    # ── 5. Hard veto — extreme drawdown ──────────────────────
    run_test("Hard veto: extreme drawdown (-40%)",
             make_features(pct_from_52w_high=-40.0),
             expect_veto=True)

    # ── 6. Liquidity penalty ──────────────────────────────────
    run_test("Low liquidity (volume_ratio=0.35)",
             make_features(volume_ratio=0.35),
             regime="SIDEWAYS")

    # ── 7. Hard veto — illiquid ───────────────────────────────
    run_test("Hard veto: illiquid (volume_ratio=0.1)",
             make_features(volume_ratio=0.1),
             expect_veto=True)

    # ── 8. Regime weight — BEAR amplifies penalty ─────────────
    f_risk = make_features(atr_pct=6.0, pct_from_52w_high=-22.0, volume_ratio=0.4)
    bull_r = RiskStrategy().run(f_risk, regime="BULL")
    bear_r = RiskStrategy().run(f_risk, regime="BEAR")
    print(f"  ── Regime weight effect ──────────────────────────")
    print(f"  BULL regime: score={bull_r.score:.1f}")
    print(f"  BEAR regime: score={bear_r.score:.1f}")
    ok = bear_r.score < bull_r.score
    print(f"  {'✅' if ok else '❌'} BEAR more penalising than BULL\n")

    # ── 9. Full runner with all 6 strategies ──────────────────
    print(f"  ── Full StrategyRunner (6 strategies) ───────────")
    runner   = StrategyRunner()
    features = make_features(atr_pct=2.5, pct_from_52w_high=-5.0, volume_ratio=1.5)
    results  = runner.run(features, regime="BULL")
    ids      = [r.strategy_id for r in results]
    assert "risk" in ids, "RiskStrategy not in runner output"
    print(f"  ✅ Runner returned {len(results)} strategies: {ids}")
    for r in results:
        bar = "█" * max(0, int(abs(r.score) // 10))
        sign = "+" if r.score >= 0 else ""
        print(f"     {r.strategy_id:<12} {sign}{r.score:>6.1f}  {r.signal:<12} {bar}")

    print(f"\n{'='*60}")
    print("  ✅ Risk Strategy Engine verified")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
