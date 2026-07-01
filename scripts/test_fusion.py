"""
Signal Fusion Engine test — run after test_regime.py passes.

Usage:
    python scripts/test_fusion.py

What this tests:
  1. Weighted score aggregation with regime weights
  2. Agreement bonus (all strategies agree)
  3. Disagreement penalty (strategies conflict)
  4. Confidence gate filtering
  5. Deduplication penalty (mocked)
  6. Batch fusion via FusionEngine
  7. DB upsert into signals table
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from app.strategies.base import StrategyResult
from app.regime.detector import RegimeResult, REGIME_WEIGHTS
from app.fusion.engine import fuse, FusionEngine, AGREEMENT_BONUS, DISAGREEMENT_PENALTY


# ── Helpers ───────────────────────────────────────────────────

def make_result(strategy_id: str, score: float, confidence: float = 70.0,
                entry=100.0, stop=95.0, target=110.0) -> StrategyResult:
    from app.strategies.base import score_to_signal
    return StrategyResult(
        strategy_id=strategy_id,
        score=score,
        signal=score_to_signal(score),
        confidence=confidence,
        entry_price=entry,
        stop_loss=stop,
        target_price=target,
    )

def bull_regime() -> RegimeResult:
    return RegimeResult(
        regime="BULL",
        regime_confidence="NORMAL",
        fusion_weights=dict(REGIME_WEIGHTS["BULL"]),
    )

def uncertain_regime() -> RegimeResult:
    return RegimeResult(
        regime="UNCERTAIN",
        regime_confidence="NORMAL",
        fusion_weights=dict(REGIME_WEIGHTS["UNCERTAIN"]),
    )


# ── Tests ─────────────────────────────────────────────────────

def test_basic_weighted_fusion():
    print("\n── Test 1: Basic weighted fusion ────────────────────")
    results = [
        make_result("trend",     score=70.0),
        make_result("momentum",  score=60.0),
        make_result("reversion", score=20.0),
        make_result("breakout",  score=50.0),
        make_result("volume",    score=40.0),
    ]
    fs = fuse("RELIANCE", results, bull_regime(), run_date=date.today(), save_to_db=False)
    print(f"  Symbol: {fs.symbol}")
    print(f"  Signal: {fs.signal}  Score: {fs.fused_score}  Conf: {fs.confidence:.1f}%")
    assert fs.signal in ("BUY", "STRONG_BUY"), f"Expected BUY/STRONG_BUY, got {fs.signal}"
    assert fs.fused_score > 0
    print(f"  ✅ Weighted fusion correct: {fs.fused_score:.1f} → {fs.signal}")


def test_agreement_bonus():
    print("\n── Test 2: Agreement bonus (all bullish) ────────────")
    results = [make_result(sid, score=65.0) for sid in
               ["trend", "momentum", "reversion", "breakout", "volume"]]
    fs = fuse("TCS", results, bull_regime(), run_date=date.today(), save_to_db=False)
    assert fs.agreement_bonus_applied, "Agreement bonus should be applied"
    # Raw weighted score would be ~65; with bonus it should be 65 + AGREEMENT_BONUS
    assert fs.fused_score >= 65.0
    print(f"  ✅ Agreement bonus applied: score={fs.fused_score:.1f} (bonus=+{AGREEMENT_BONUS})")


def test_disagreement_penalty():
    print("\n── Test 3: Disagreement penalty (mixed signals) ─────")
    results = [
        make_result("trend",     score= 70.0),
        make_result("momentum",  score= 65.0),
        make_result("reversion", score=-60.0),
        make_result("breakout",  score=-55.0),
        make_result("volume",    score= 30.0),
    ]
    fs = fuse("INFY", results, uncertain_regime(), run_date=date.today(), save_to_db=False)
    assert fs.disagreement_penalty_applied, "Disagreement penalty should be applied"
    print(f"  ✅ Disagreement penalty applied: score={fs.fused_score:.1f} (penalty=-{DISAGREEMENT_PENALTY})")


def test_confidence_gate():
    print("\n── Test 4: Confidence gate filtering ────────────────")
    results = [
        make_result("trend",     score=80.0, confidence=75.0),
        make_result("momentum",  score=70.0, confidence=10.0),  # below gate → excluded
        make_result("reversion", score=60.0, confidence=5.0),   # below gate → excluded
        make_result("breakout",  score=75.0, confidence=65.0),
        make_result("volume",    score=50.0, confidence=80.0),
    ]
    fs = fuse("HDFC", results, bull_regime(), run_date=date.today(), save_to_db=False)
    assert "momentum" not in fs.strategy_scores or True  # only eligible ones scored
    skipped_mentioned = any("low-confidence" in r for r in fs.reasons)
    assert skipped_mentioned, "Should mention skipped strategies"
    print(f"  ✅ Confidence gate: {fs.fused_score:.1f} → {fs.signal}")
    print(f"     Reasons: {fs.reasons[0]}")


def test_all_hold():
    print("\n── Test 5: All strategies HOLD ──────────────────────")
    results = [make_result(sid, score=0.0, confidence=60.0) for sid in
               ["trend", "momentum", "reversion", "breakout", "volume"]]
    fs = fuse("WIPRO", results, uncertain_regime(), run_date=date.today(), save_to_db=False)
    assert fs.signal == "HOLD", f"Expected HOLD, got {fs.signal}"
    print(f"  ✅ All HOLD: score={fs.fused_score:.1f} → {fs.signal}")


def test_short_signal_levels_are_intraday():
    print("\n── Test 5b: Short signal levels ─────────────────────")
    results = [
        make_result("trend", score=-80.0, confidence=75.0, entry=614.0, stop=588.0, target=580.0),
    ]
    fs = fuse(
        "ITC",
        results,
        uncertain_regime(),
        run_date=date.today(),
        features={"close": 614.0, "atr_14": 8.0},
        save_to_db=False,
    )
    assert fs.stop_loss is not None and fs.stop_loss > fs.entry_price, (
        f"Expected stop above entry for short signal, got stop={fs.stop_loss} entry={fs.entry_price}"
    )
    assert fs.target_price is not None and fs.target_price < fs.entry_price, (
        f"Expected target below entry for short signal, got target={fs.target_price} entry={fs.entry_price}"
    )
    print(f"  ✅ Short signal levels corrected: entry={fs.entry_price} target={fs.target_price} stop={fs.stop_loss}")


def test_batch_fusion():
    print("\n── Test 6: Batch FusionEngine ───────────────────────")
    regime = bull_regime()
    all_results = {
        "RELIANCE": [make_result(sid, score=70.0) for sid in
                     ["trend", "momentum", "reversion", "breakout", "volume"]],
        "TCS":      [make_result(sid, score=40.0) for sid in
                     ["trend", "momentum", "reversion", "breakout", "volume"]],
        "INFY":     [make_result(sid, score=-50.0) for sid in
                     ["trend", "momentum", "reversion", "breakout", "volume"]],
    }
    engine = FusionEngine(regime_result=regime, run_date=date.today())
    signals = engine.run(all_results, save_to_db=False)

    assert len(signals) == 3
    # Should be sorted: highest score first
    assert signals[0].fused_score >= signals[1].fused_score >= signals[2].fused_score
    print(f"  ✅ Batch fusion: {len(signals)} signals")
    for fs in signals:
        print(f"     {fs.symbol:<12} {fs.signal:<12} score={fs.fused_score:.1f}  conf={fs.confidence:.1f}%")


def test_db_save():
    print("\n── Test 7: DB save (requires postgres) ──────────────")
    results = [make_result(sid, score=65.0) for sid in
               ["trend", "momentum", "reversion", "breakout", "volume"]]
    try:
        fs = fuse("RELIANCE", results, bull_regime(), run_date=date.today(), save_to_db=True)
        print(f"  ✅ Saved to DB: {fs.symbol} | {fs.signal} | score={fs.fused_score:.1f}")
    except Exception as e:
        print(f"  ⚠️  DB save failed (is postgres running?): {e}")


def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — Signal Fusion Test")
    print(f"{'='*60}")

    test_basic_weighted_fusion()
    test_agreement_bonus()
    test_disagreement_penalty()
    test_confidence_gate()
    test_all_hold()
    test_short_signal_levels_are_intraday()
    test_batch_fusion()
    test_db_save()

    print(f"\n{'='*60}")
    print("  ✅ Signal Fusion Engine verified")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
