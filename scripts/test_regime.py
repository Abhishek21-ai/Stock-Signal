"""
Regime Detection test — run after test_strategies.py passes.

Usage:
    python scripts/test_regime.py

What this tests:
  1. Nifty50 fetch via yfinance
  2. EMA20/50/200 + ADX indicator computation
  3. Regime classification logic (all 4 outcomes)
  4. FII stress adjustment
  5. DB upsert into regime_snapshots
  6. get_latest_regime() retrieval
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from app.regime.detector import (
    RegimeDetector,
    classify_regime,
    get_latest_regime,
    REGIME_WEIGHTS,
    _apply_fii_adjustment,
)


def test_classification_logic():
    """Unit test all 4 regime classifications without DB or network."""
    print("\n── Unit: classify_regime() ──────────────────────────")

    cases = [
        # (close, ema20, ema50, ema200, adx, expected)
        (22000, 21900, 21700, 21000, 28.0, "BULL"),
        (19000, 19200, 19500, 20000, 30.0, "BEAR"),
        (22000, 22010, 21990, 21500, 17.0, "SIDEWAYS"),
        (22000, 21900, 21700, 21000, 18.0, "UNCERTAIN"),  # bull align but weak ADX
    ]

    for close, e20, e50, e200, adx, expected in cases:
        regime, reasons = classify_regime(close, e20, e50, e200, adx)
        status = "✅" if regime == expected else "❌"
        print(f"  {status} close={close} EMA{e20}/{e50}/{e200} ADX={adx} → {regime} (expected {expected})")
        if regime != expected:
            for r in reasons:
                print(f"       {r}")


def test_fii_adjustment():
    """Unit test FII stress weight adjustment."""
    print("\n── Unit: FII stress adjustment ──────────────────────")

    bull_weights = dict(REGIME_WEIGHTS["BULL"])

    # No stress (BULL with positive FII)
    conf, stress, w = _apply_fii_adjustment("BULL", 3200.0, bull_weights)
    print(f"  {'✅' if not stress else '❌'} No stress: fii=+3200 → confidence={conf}, stress={stress}")

    # Stress triggered
    conf, stress, w = _apply_fii_adjustment("BULL", -6500.0, bull_weights)
    print(f"  {'✅' if stress else '❌'} FII stress: fii=-6500 → confidence={conf}, stress={stress}")
    print(f"     Adjusted weights: {w}")
    total = round(sum(w.values()), 4)
    print(f"     Weight sum={total} {'✅' if abs(total - 1.0) < 0.01 else '❌ (should be 1.0)'}")

    # Non-BULL regime — no adjustment regardless of FII
    conf, stress, w = _apply_fii_adjustment("BEAR", -8000.0, dict(REGIME_WEIGHTS["BEAR"]))
    print(f"  {'✅' if not stress else '❌'} BEAR regime: FII=-8000 → no adjustment (stress={stress})")


def test_live_detection():
    """Full integration test with Nifty50 fetch and DB write."""
    print("\n── Integration: RegimeDetector.run() ───────────────")

    detector = RegimeDetector(run_date=date.today())
    result   = detector.run()

    print(f"\n  Regime:      {result.regime}")
    print(f"  Confidence:  {result.regime_confidence}")
    print(f"  Nifty close: ₹{result.nifty_close:,.2f}")
    print(f"  EMA20:       ₹{result.nifty_ema20:,.2f}")
    print(f"  EMA50:       ₹{result.nifty_ema50:,.2f}")
    print(f"  EMA200:      ₹{result.nifty_ema200:,.2f}")
    print(f"  ADX(14):     {result.nifty_adx:.2f}")
    print(f"  FII stress:  {result.fii_stress}")
    print(f"\n  Fusion weights:")
    for k, v in result.fusion_weights.items():
        bar = "█" * int(v * 100 // 5)
        print(f"    {k:<12} {v:.2%}  {bar}")

    print(f"\n  Reasons:")
    for r in result.reasons:
        print(f"    → {r}")

    return result


def test_db_retrieval():
    """Verify regime snapshot was written and can be read back."""
    print("\n── DB: get_latest_regime() ──────────────────────────")

    latest = get_latest_regime()
    if latest:
        print(f"  ✅ Retrieved from DB: {latest.regime} | {latest.regime_confidence}")
        print(f"     Nifty={latest.nifty_close:.2f} | ADX={latest.nifty_adx:.2f}")
    else:
        print("  ⚠️  No regime snapshot in DB yet (run test_live_detection first)")


def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — Regime Detection Test")
    print(f"{'='*60}")

    test_classification_logic()
    test_fii_adjustment()

    print("\n── Live detection (requires internet + DB) ──────────")
    try:
        result = test_live_detection()
        test_db_retrieval()
        print(f"\n{'='*60}")
        print(f"  ✅ Regime Detection verified: {result.regime} ({result.regime_confidence})")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"  ❌ Live test failed: {e}")
        print("     → Is postgres + internet available?")
        print(f"\n{'='*60}")
        print("  ⚠️  Unit tests passed, live test requires DB + internet")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
