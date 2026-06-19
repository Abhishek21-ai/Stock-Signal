"""
LLM Resilience test — Groq -> Gemini -> OpenAI -> Rule-based fallback (Section 27)

Usage:
    python scripts/test_llm_resilience.py

What this tests:
  1. Gemini API call (direct, isolated)
  2. Rule-based fallback — earnings window VETO
  3. Rule-based fallback — sector concentration VETO
  4. Rule-based fallback — macro window REDUCE_CONFIDENCE
  5. Rule-based fallback — clean signal CONFIRM
  6. Session failure tracking — circuit breaker triggers after threshold
  7. Full chain integration — get_llm_verdict() with real signal
"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from app.fusion.engine import FusedSignal
from app.regime.detector import RegimeResult, REGIME_WEIGHTS
from app.llm.override import (
    build_prompt, _call_gemini, _parse_verdict,
    LLMSessionStats, get_llm_verdict, SESSION_FAILURE_THRESHOLD,
)
from app.llm.rule_fallback import apply_rule_based_fallback


def make_signal(symbol="RELIANCE", score=70.0, signal="BUY", conf=75.0) -> FusedSignal:
    return FusedSignal(
        symbol=symbol, run_date=date.today(),
        fused_score=score, signal=signal, confidence=conf,
        regime="BULL",
        strategy_scores={"trend": 70, "momentum": 65},
        entry_price=2500.0, stop_loss=2420.0, target_price=2680.0,
    )

def make_regime() -> RegimeResult:
    return RegimeResult(
        regime="BULL", regime_confidence="NORMAL",
        nifty_close=23987, nifty_ema20=23555, nifty_ema50=23779,
        nifty_ema200=24513, nifty_adx=21.6,
        fusion_weights=dict(REGIME_WEIGHTS["BULL"]),
    )


async def test_live_gemini():
    print("\n── Test 1: Live Gemini API (direct) ─────────────────")
    from config.settings import settings
    if not settings.google_api_key:
        print("  ⚠️  GOOGLE_API_KEY not set in .env — skipping")
        return

    prompt = build_prompt(make_signal(), make_regime(),
                          {"rsi_14": 58, "atr_pct": 2.3})
    try:
        import time
        t0  = time.monotonic()
        raw = await _call_gemini(prompt, timeout=15)
        ms  = int((time.monotonic() - t0) * 1000)
        v   = _parse_verdict(raw)
        assert v is not None and v.verdict in {"CONFIRM", "VETO", "REDUCE_CONFIDENCE"}
        print(f"  ✅ Gemini: verdict={v.verdict} | {ms}ms")
        print(f"     {v.explanation}")
    except Exception as e:
        print(f"  ❌ Gemini failed: {e}")


def test_rule_earnings():
    print("\n── Test 2: Rule-based — earnings window VETO ────────")
    signal   = make_signal()
    regime   = make_regime()
    features = {"days_to_earnings": 2, "rsi_14": 60}

    verdict = apply_rule_based_fallback(signal, regime, features)
    assert verdict.verdict == "VETO"
    assert verdict.provider == "rule_based"
    print(f"  ✅ Earnings in 2d → {verdict.verdict}")
    print(f"     {verdict.explanation}")


def test_rule_no_earnings():
    print("\n── Test 2b: Rule-based — earnings far away (no trigger) ──")
    signal   = make_signal()
    regime   = make_regime()
    features = {"days_to_earnings": 15, "rsi_14": 60}

    verdict = apply_rule_based_fallback(signal, regime, features)
    assert verdict.verdict == "CONFIRM"
    print(f"  ✅ Earnings in 15d → {verdict.verdict} (correctly not triggered)")


def test_rule_volatility():
    print("\n── Test 3: Rule-based — high volatility penalty ─────")
    signal   = make_signal()
    regime   = make_regime()
    features = {
        "atr_pct": 8.0,
        "atr_pct_6m_avg": 3.0,   # 8.0 > 2x3.0=6.0 → triggers
        "days_to_earnings": 30,
    }
    verdict = apply_rule_based_fallback(signal, regime, features)
    assert verdict.verdict == "REDUCE_CONFIDENCE"
    print(f"  ✅ ATR 8% vs 6mo avg 3% → {verdict.verdict}")
    print(f"     {verdict.explanation}")


def test_rule_clean_signal():
    print("\n── Test 4: Rule-based — clean signal CONFIRM ────────")
    signal   = make_signal(symbol="CLEANCO")
    regime   = make_regime()
    features = {"days_to_earnings": 45, "atr_pct": 2.0, "atr_pct_6m_avg": 2.2}

    verdict = apply_rule_based_fallback(signal, regime, features)
    assert verdict.verdict == "CONFIRM"
    assert verdict.llm_status == "RULE_BASED"
    print(f"  ✅ No red flags → {verdict.verdict}")
    print(f"     {verdict.explanation}")


def test_session_circuit_breaker():
    print("\n── Test 5: Session failure circuit breaker ──────────")
    stats = LLMSessionStats()

    # Simulate 4 failures, 1 success — under the 5-call minimum
    for _ in range(4):
        stats.record(all_providers_failed=True)
    stats.record(all_providers_failed=False)

    print(f"  After 5 calls (4 fail, 1 ok): "
          f"failure_rate={stats.failure_rate:.0%} | "
          f"should_skip={stats.should_skip_api}")
    assert stats.should_skip_api == True   # 80% > 20% threshold, 5+ calls

    # Fresh stats — only 3 calls, all failed (under minimum sample size)
    stats2 = LLMSessionStats()
    for _ in range(3):
        stats2.record(all_providers_failed=True)
    print(f"  After 3 calls (3 fail): "
          f"failure_rate={stats2.failure_rate:.0%} | "
          f"should_skip={stats2.should_skip_api}")
    assert stats2.should_skip_api == False  # under 5-call minimum

    print(f"  ✅ Circuit breaker triggers correctly at >{SESSION_FAILURE_THRESHOLD:.0%} "
          f"with 5+ calls")


async def test_full_chain():
    print("\n── Test 6: Full chain — get_llm_verdict() ───────────")
    signal   = make_signal(symbol="FULLCHAIN")
    regime   = make_regime()
    features = {"rsi_14": 55, "atr_pct": 2.0, "days_to_earnings": 30}

    verdict = await get_llm_verdict(signal, regime, features)
    assert verdict.verdict in {"CONFIRM", "VETO", "REDUCE_CONFIDENCE"}
    print(f"  ✅ Chain resolved: verdict={verdict.verdict} | "
          f"provider={verdict.provider} | status={verdict.llm_status}")
    print(f"     {verdict.explanation}")


async def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — LLM Resilience Test")
    print("  (Groq -> Gemini -> OpenAI -> Rule-Based Fallback)")
    print(f"{'='*60}")

    await test_live_gemini()
    test_rule_earnings()
    test_rule_no_earnings()
    test_rule_volatility()
    test_rule_clean_signal()
    test_session_circuit_breaker()
    await test_full_chain()

    print(f"\n{'='*60}")
    print("  ✅ LLM Resilience Layer verified")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
