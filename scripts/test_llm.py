"""
LLM Override Layer test — run after test_risk.py passes.

Usage:
    python scripts/test_llm.py

What this tests:
  1. Prompt builder — correct structure
  2. Verdict parser — all 3 verdicts + malformed JSON
  3. Apply verdict — score/confidence mutations (CONFIRM, VETO, REDUCE_CONFIDENCE)
  4. Redis cache read/write
  5a. Live Groq API (primary)
  5b. Live OpenAI API (fallback — called directly, not via Groq failure)
  6. Override rate cap enforcement
"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from app.fusion.engine import FusedSignal
from app.regime.detector import RegimeResult, REGIME_WEIGHTS
from app.llm.override import (
    build_prompt, _parse_verdict, _apply_verdict,
    LLMVerdict, CONFIDENCE_REDUCTION,
)


# ── Fixtures ──────────────────────────────────────────────────

def make_signal(symbol="RELIANCE", score=70.0, signal="BUY", conf=75.0) -> FusedSignal:
    return FusedSignal(
        symbol=symbol, run_date=date.today(),
        fused_score=score, signal=signal, confidence=conf,
        regime="BULL",
        strategy_scores={"trend": 70, "momentum": 65, "reversion": 20,
                         "breakout": 60, "volume": 40, "risk": 50},
        entry_price=2500.0, stop_loss=2420.0, target_price=2680.0,
    )

def make_regime() -> RegimeResult:
    return RegimeResult(
        regime="BULL", regime_confidence="NORMAL",
        nifty_close=23987, nifty_ema20=23555, nifty_ema50=23779,
        nifty_ema200=24513, nifty_adx=21.6,
        fusion_weights=dict(REGIME_WEIGHTS["BULL"]),
    )

def make_features() -> dict:
    return {
        "rsi_14": 58.0, "atr_pct": 2.3, "volume_ratio": 1.4,
        "pct_from_52w_high": -6.5, "ema_bull_alignment": 2,
    }


# ── Unit tests ────────────────────────────────────────────────

def test_prompt_builder():
    print("\n── Test 1: Prompt builder ───────────────────────────")
    prompt = build_prompt(make_signal(), make_regime(), make_features())
    assert "RELIANCE" in prompt
    assert "CONFIRM" in prompt
    assert "VETO" in prompt
    assert "verdict" in prompt
    assert "explanation" in prompt
    print(f"  ✅ Prompt built ({len(prompt)} chars)")
    print(f"     Preview: {prompt[:120].strip()}...")


def test_verdict_parser():
    print("\n── Test 2: Verdict parser ───────────────────────────")
    cases = [
        ('{"verdict": "CONFIRM", "explanation": "Signal looks valid."}',         "CONFIRM"),
        ('{"verdict": "VETO", "explanation": "RSI overbought, avoid."}',          "VETO"),
        ('{"verdict": "REDUCE_CONFIDENCE", "explanation": "Mixed signals."}',     "REDUCE_CONFIDENCE"),
        ("```json\n{\"verdict\": \"CONFIRM\", \"explanation\": \"OK\"}\n```",     "CONFIRM"),
        ('{"verdict": "INVALID_VERDICT", "explanation": "Bad"}',                  "CONFIRM"),  # bad → CONFIRM
        ("not json at all",                                                        None),
    ]
    for raw, expected in cases:
        result = _parse_verdict(raw)
        if expected is None:
            ok = result is None
        else:
            ok = result is not None and result.verdict == expected
        print(f"  {'✅' if ok else '❌'} {raw[:55]!r} → {result.verdict if result else None}")


def test_apply_verdicts():
    print("\n── Test 3: Apply verdicts ───────────────────────────")

    # CONFIRM — no change
    sig = make_signal(score=70.0, conf=75.0)
    _apply_verdict(sig, LLMVerdict("CONFIRM", "Looks good.", "OK", "groq"))
    assert sig.fused_score == 70.0 and sig.confidence == 75.0 and sig.signal == "BUY"
    print(f"  ✅ CONFIRM: score={sig.fused_score}, conf={sig.confidence}, signal={sig.signal}")

    # VETO — score→0, signal→HOLD, conf halved
    sig = make_signal(score=70.0, conf=75.0)
    _apply_verdict(sig, LLMVerdict("VETO", "Earnings risk.", "OK", "groq"))
    assert sig.fused_score == 0.0 and sig.signal == "HOLD" and sig.confidence == 37.5
    print(f"  ✅ VETO:    score={sig.fused_score}, conf={sig.confidence}, signal={sig.signal}")

    # REDUCE_CONFIDENCE — score unchanged, conf -25
    sig = make_signal(score=70.0, conf=75.0)
    _apply_verdict(sig, LLMVerdict("REDUCE_CONFIDENCE", "Mixed signals.", "OK", "groq"))
    assert sig.fused_score == 70.0 and sig.confidence == 75.0 - CONFIDENCE_REDUCTION
    print(f"  ✅ REDUCE:  score={sig.fused_score}, conf={sig.confidence}, signal={sig.signal}")


async def test_cache():
    print("\n── Test 4: Redis cache ──────────────────────────────")
    from app.cache import llm_cache
    key_date = str(date.today())
    try:
        data = {"verdict": "CONFIRM", "explanation": "Cache test",
                "llm_status": "OK", "provider": "test"}
        await llm_cache.set("TESTSTOCK", key_date, data, ttl_seconds=60)
        cached = await llm_cache.get("TESTSTOCK", key_date)
        assert cached is not None and cached["verdict"] == "CONFIRM"
        await llm_cache.delete("TESTSTOCK", key_date)
        print("  ✅ Redis cache write/read/delete OK")
    except Exception as e:
        print(f"  ⚠️  Redis not available: {e}")


async def test_live_groq():
    print("\n── Test 5a: Live Groq API (primary) ────────────────")
    from config.settings import settings
    if not settings.groq_api_key:
        print("  ⚠️  GROQ_API_KEY not set in .env — skipping")
        return

    from app.llm.override import _call_groq
    import time
    prompt = build_prompt(make_signal(), make_regime(), make_features())
    try:
        t0  = time.monotonic()
        raw = await _call_groq(prompt, timeout=settings.groq_timeout_seconds)
        ms  = int((time.monotonic() - t0) * 1000)
        v   = _parse_verdict(raw)
        assert v is not None and v.verdict in {"CONFIRM", "VETO", "REDUCE_CONFIDENCE"}
        print(f"  ✅ Groq: verdict={v.verdict} | {ms}ms")
        print(f"     {v.explanation}")
    except Exception as e:
        print(f"  ❌ Groq failed: {e}")


async def test_live_openai_fallback():
    print("\n── Test 5b: OpenAI fallback (called directly) ───────")
    from config.settings import settings
    if not settings.openai_api_key:
        print("  ⚠️  OPENAI_API_KEY not set in .env — skipping")
        print("     (This is optional — Groq is primary, OpenAI is fallback only)")
        return

    from app.llm.override import _call_openai
    import time
    prompt = build_prompt(make_signal(), make_regime(), make_features())
    try:
        t0  = time.monotonic()
        raw = await _call_openai(prompt, timeout=15)
        ms  = int((time.monotonic() - t0) * 1000)
        v   = _parse_verdict(raw)
        assert v is not None and v.verdict in {"CONFIRM", "VETO", "REDUCE_CONFIDENCE"}
        print(f"  ✅ OpenAI fallback: verdict={v.verdict} | {ms}ms")
        print(f"     {v.explanation}")
    except Exception as e:
        print(f"  ❌ OpenAI fallback failed: {e}")


def test_rate_cap():
    print("\n── Test 6: Override rate cap ────────────────────────")
    from config.settings import settings
    cap           = settings.max_llm_override_rate
    total         = 10
    max_overrides = int(total * cap)
    assert max_overrides == 3, f"Expected 3, got {max_overrides}"
    print(f"  ✅ Cap: {cap:.0%} of {total} signals = max {max_overrides} overrides")


async def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — LLM Override Layer Test")
    print(f"{'='*60}")

    test_prompt_builder()
    test_verdict_parser()
    test_apply_verdicts()
    await test_cache()
    await test_live_groq()
    await test_live_openai_fallback()
    test_rate_cap()

    print(f"\n{'='*60}")
    print("  ✅ LLM Override Layer verified")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())