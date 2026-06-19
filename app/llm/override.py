"""
LLM Override Layer — Section 10 & 27

Flow per stock:
  1. Check Redis cache (key: llm_override:{symbol}:{date}, TTL 24h) — skip API if hit
  2. Build structured prompt from FusedSignal + RegimeResult + features
  3. Call Groq (llama-3.3-70b-versatile) with 10s timeout       [primary]
  4. Fallback to Gemini (gemini-1.5-flash) if Groq fails         [secondary, free tier]
  5. Fallback to OpenAI gpt-4o-mini if Gemini also fails         [tertiary]
  6. If ALL three fail → rule-based fallback (Section 27.1)      [deterministic, no API]
  7. Apply verdict to FusedSignal score/confidence
  8. Cache result in Redis (24h TTL)
  9. Update daily_signals row in DB (llm_override, llm_status, llm_explanation)

Override rate cap (Section 10.3):
  Max 30% of signals may be overridden per run (settings.max_llm_override_rate).

Verdict effects:
  CONFIRM          → no score change, llm_override=NONE
  VETO             → score set to 0, signal → HOLD, confidence halved
  REDUCE_CONFIDENCE→ confidence reduced by 25%

Session-level failure tracking (Section 24.1):
  If >20% of LLM calls in a session time out, all subsequent calls in that
  session skip straight to rule-based fallback (no more API attempts wasted).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import httpx

from app.cache import llm_cache
from app.db import get_sync_db
from app.fusion.engine import FusedSignal
from app.llm.rule_fallback import apply_rule_based_fallback
from app.logger import get_logger
from app.regime.detector import RegimeResult
from config.settings import settings

logger = get_logger("llm")

# ── Constants ─────────────────────────────────────────────────
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
OPENAI_URL  = "https://api.openai.com/v1/chat/completions"
GEMINI_URL  = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-lite:generateContent"
)
LLM_CACHE_TTL = 86_400          # 24 hours
CONFIDENCE_REDUCTION = 25.0     # pct points removed on REDUCE_CONFIDENCE

VALID_VERDICTS = {"CONFIRM", "VETO", "REDUCE_CONFIDENCE"}

# Session-level failure tracking (Section 24.1: >20% triggers fallback mode)
SESSION_FAILURE_THRESHOLD = 0.20


# ── Data classes ──────────────────────────────────────────────

@dataclass
class LLMVerdict:
    verdict: str                 # CONFIRM | VETO | REDUCE_CONFIDENCE
    explanation: str
    llm_status: str              # OK | TIMEOUT | FALLBACK | ERROR | CACHED | RULE_BASED
    provider: str                # groq | gemini | openai | cache | rule_based | none
    latency_ms: int = 0

    def to_dict(self) -> Dict:
        return {
            "verdict":     self.verdict,
            "explanation": self.explanation,
            "llm_status":  self.llm_status,
            "provider":    self.provider,
            "latency_ms":  self.latency_ms,
        }


# ── Prompt builder ────────────────────────────────────────────

def build_prompt(
    signal: FusedSignal,
    regime: RegimeResult,
    features: Dict,
) -> str:
    """
    Builds a structured prompt that gives the LLM just enough context
    to confirm or override the quant signal. Kept concise to minimise
    latency and cost.
    """
    rr = "N/A"
    if signal.entry_price and signal.stop_loss and signal.target_price:
        risk   = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.target_price - signal.entry_price)
        rr     = f"{reward/risk:.2f}" if risk > 0 else "N/A"

    return f"""You are a senior Indian equity analyst reviewing an algorithmic trading signal.
Respond ONLY in valid JSON. No markdown, no explanation outside JSON.

## Signal Summary
Symbol:         {signal.symbol}
Signal:         {signal.signal}
Quant Score:    {signal.fused_score:.1f} / 100
Confidence:     {signal.confidence:.1f}%
Market Regime:  {regime.regime} (Nifty ADX={regime.nifty_adx:.1f})

## Price Levels
Entry:          ₹{signal.entry_price or 'N/A'}
Stop Loss:      ₹{signal.stop_loss or 'N/A'}
Target:         ₹{signal.target_price or 'N/A'}
Risk/Reward:    {rr}

## Key Indicators
RSI(14):        {features.get('rsi_14', 'N/A')}
ATR%:           {features.get('atr_pct', 'N/A')}
Volume Ratio:   {features.get('volume_ratio', 'N/A')}
% from 52w High:{features.get('pct_from_52w_high', 'N/A')}
EMA Alignment:  {features.get('ema_bull_alignment', 'N/A')}/3

## Strategy Breakdown
{json.dumps(signal.strategy_scores, indent=2)}

## Your Task
Review this signal for the NSE Indian market context.
Consider: earnings season, sector rotation, FII flows (regime={regime.regime}), 
liquidity, and whether the risk/reward is justified.

Respond with EXACTLY this JSON structure:
{{
  "verdict": "CONFIRM" | "VETO" | "REDUCE_CONFIDENCE",
  "explanation": "One concise sentence (max 25 words) explaining your decision"
}}

CONFIRM          = signal looks valid, proceed as-is
VETO             = serious red flag, do not trade this signal
REDUCE_CONFIDENCE= signal has merit but uncertainty warrants lower confidence
"""


# ── API callers ───────────────────────────────────────────────

async def _call_groq(prompt: str, timeout: int) -> str:
    """Call Groq API. Returns raw response text."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       settings.groq_model,
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  120,
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def _call_gemini(prompt: str, timeout: int = 15) -> str:
    """
    Call Google Gemini API (free tier via AI Studio).
    Returns raw response text.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{GEMINI_URL}?key={settings.google_api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature":     0.1,
                    "maxOutputTokens": 120,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def _call_openai(prompt: str, timeout: int) -> str:
    """Call OpenAI fallback. Returns raw response text."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            OPENAI_URL,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       "gpt-4o-mini",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  120,
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_verdict(raw: str) -> Optional[LLMVerdict]:
    """
    Parse LLM JSON response into LLMVerdict.
    Returns None if parsing fails.
    """
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.strip("`").lstrip("json").strip()
        data    = json.loads(clean)
        verdict = data.get("verdict", "").strip().upper()
        if verdict not in VALID_VERDICTS:
            logger.warning(f"Unexpected verdict: {verdict!r} — defaulting to CONFIRM")
            verdict = "CONFIRM"
        return LLMVerdict(
            verdict=verdict,
            explanation=str(data.get("explanation", ""))[:300],
            llm_status="OK",
            provider="",
        )
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e} | raw={raw!r}")
        return None


# ── Session-level failure tracking (Section 24.1) ──────────────

class LLMSessionStats:
    """
    Tracks success/failure across a single pipeline run.
    If failure rate exceeds 20%, subsequent calls skip API attempts
    and go straight to rule-based fallback to save time.
    """
    def __init__(self):
        self.total_calls   = 0
        self.failed_calls  = 0   # all providers failed (not counting cache hits)

    def record(self, all_providers_failed: bool) -> None:
        self.total_calls += 1
        if all_providers_failed:
            self.failed_calls += 1

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    @property
    def should_skip_api(self) -> bool:
        """After at least 5 calls, if failure rate exceeds threshold, skip API."""
        return self.total_calls >= 5 and self.failure_rate > SESSION_FAILURE_THRESHOLD


# ── Main per-symbol override ──────────────────────────────────

async def get_llm_verdict(
    signal: FusedSignal,
    regime: RegimeResult,
    features: Dict,
    session_stats: Optional[LLMSessionStats] = None,
) -> LLMVerdict:
    """
    Get LLM verdict for a single signal.
    Chain: cache -> Groq -> Gemini -> OpenAI -> rule-based fallback.
    """
    run_date = str(signal.run_date)

    # ── 1. Cache check ────────────────────────────────────────
    cached = await llm_cache.get(signal.symbol, run_date)
    if cached:
        logger.debug(f"{signal.symbol}: LLM cache hit")
        return LLMVerdict(
            verdict=cached["verdict"],
            explanation=cached["explanation"],
            llm_status="CACHED",
            provider="cache",
        )

    # ── Session circuit breaker — skip straight to rule-based ──
    if session_stats and session_stats.should_skip_api:
        logger.warning(
            f"{signal.symbol}: session failure rate "
            f"{session_stats.failure_rate:.0%} > {SESSION_FAILURE_THRESHOLD:.0%} "
            f"— skipping API, using rule-based fallback"
        )
        verdict = apply_rule_based_fallback(signal, regime, features)
        if session_stats:
            session_stats.record(all_providers_failed=False)  # not an API failure
        return verdict

    prompt = build_prompt(signal, regime, features)
    t0     = time.monotonic()
    verdict = None

    # ── 2. Groq (primary) ─────────────────────────────────────
    if settings.groq_api_key:
        try:
            raw     = await _call_groq(prompt, timeout=settings.groq_timeout_seconds)
            verdict = _parse_verdict(raw)
            if verdict:
                verdict.provider   = "groq"
                verdict.latency_ms = int((time.monotonic() - t0) * 1000)
                logger.info(
                    f"{signal.symbol}: Groq verdict={verdict.verdict} "
                    f"({verdict.latency_ms}ms)"
                )
        except Exception as e:
            logger.warning(f"{signal.symbol}: Groq failed ({e}) — trying Gemini")

    # ── 3. Gemini (secondary, free tier) ──────────────────────
    if verdict is None and settings.google_api_key:
        try:
            raw     = await _call_gemini(prompt, timeout=15)
            verdict = _parse_verdict(raw)
            if verdict:
                verdict.provider   = "gemini"
                verdict.llm_status = "OK"
                verdict.latency_ms = int((time.monotonic() - t0) * 1000)
                logger.info(
                    f"{signal.symbol}: Gemini verdict={verdict.verdict} "
                    f"({verdict.latency_ms}ms)"
                )
        except Exception as e:
            logger.warning(f"{signal.symbol}: Gemini failed ({e}) — trying OpenAI")

    # ── 4. OpenAI (tertiary) ──────────────────────────────────
    if verdict is None and settings.openai_api_key:
        try:
            raw     = await _call_openai(prompt, timeout=15)
            verdict = _parse_verdict(raw)
            if verdict:
                verdict.provider   = "openai"
                verdict.llm_status = "FALLBACK"
                verdict.latency_ms = int((time.monotonic() - t0) * 1000)
                logger.info(
                    f"{signal.symbol}: OpenAI fallback verdict={verdict.verdict} "
                    f"({verdict.latency_ms}ms)"
                )
        except Exception as e:
            logger.warning(f"{signal.symbol}: OpenAI also failed: {e}")

    # ── 5. All 3 providers failed → rule-based fallback (§27.1) ─
    all_failed = verdict is None
    if all_failed:
        logger.warning(
            f"{signal.symbol}: Groq + Gemini + OpenAI all failed — "
            f"using rule-based fallback (Section 27.1)"
        )
        verdict = apply_rule_based_fallback(signal, regime, features)
        verdict.latency_ms = int((time.monotonic() - t0) * 1000)

    if session_stats:
        session_stats.record(all_providers_failed=all_failed)

    # ── 6. Cache result (skip caching rule-based / errors) ─────
    if verdict.llm_status in ("OK", "FALLBACK"):
        await llm_cache.set(
            signal.symbol, run_date, verdict.to_dict(),
            ttl_seconds=LLM_CACHE_TTL,
        )

    return verdict


def _apply_verdict(signal: FusedSignal, verdict: LLMVerdict) -> FusedSignal:
    """
    Mutate signal in-place based on LLM verdict.
    Returns the same signal object for chaining.
    """
    if verdict.verdict == "VETO":
        logger.info(f"{signal.symbol}: VETO ({verdict.provider}) — {verdict.explanation}")
        signal.fused_score = 0.0
        signal.signal      = "HOLD"
        signal.confidence  = max(0.0, signal.confidence / 2)
        signal.reasons.append(f"LLM VETO [{verdict.provider}]: {verdict.explanation}")

    elif verdict.verdict == "REDUCE_CONFIDENCE":
        logger.info(
            f"{signal.symbol}: REDUCE_CONFIDENCE ({verdict.provider}) — {verdict.explanation}"
        )
        signal.confidence = max(0.0, signal.confidence - CONFIDENCE_REDUCTION)
        signal.reasons.append(
            f"LLM reduced confidence [{verdict.provider}]: {verdict.explanation}"
        )

    else:  # CONFIRM
        signal.reasons.append(f"LLM confirmed [{verdict.provider}]: {verdict.explanation}")

    return signal


def _update_db(
    symbol: str,
    run_date: date,
    verdict: LLMVerdict,
) -> None:
    """Update llm_override fields in daily_signals table."""
    override_map = {
        "VETO":              "VETO",
        "REDUCE_CONFIDENCE": "REDUCE_CONFIDENCE",
        "CONFIRM":           "NONE",
    }
    db_override = override_map.get(verdict.verdict, "NONE")

    if verdict.llm_status == "TIMEOUT":
        db_override = "TIMEOUT"
    elif verdict.llm_status in ("ERROR", "RULE_BASED"):
        db_override = "FALLBACK"

    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE daily_signals
                SET llm_override    = %s::llm_override_type,
                    llm_status      = %s,
                    llm_explanation = %s
                WHERE stock = %s AND date = %s
                """,
                (
                    db_override,
                    f"{verdict.llm_status}:{verdict.provider}",
                    verdict.explanation,
                    symbol,
                    run_date,
                ),
            )
    except Exception as e:
        logger.error(f"Failed to update LLM fields for {symbol}: {e}")


# ── Batch override engine ─────────────────────────────────────

class LLMOverrideEngine:
    """
    Called by pipeline.py Stage 6.
    Processes all fused signals through LLM override with rate cap.
    """

    def __init__(
        self,
        regime: RegimeResult,
        features_map: Dict[str, Dict],
        run_date: Optional[date] = None,
    ):
        self.regime       = regime
        self.features_map = features_map
        self.run_date     = run_date or date.today()
        self.session_stats = LLMSessionStats()

    async def run(self, signals: List[FusedSignal]) -> List[FusedSignal]:
        """
        Process all signals. Only BUY/STRONG_BUY signals go to LLM review.
        SELL/HOLD signals are passed through unchanged.
        Override rate cap: max 30% of total signals.
        """
        total          = len(signals)
        max_overrides  = int(total * settings.max_llm_override_rate)
        override_count = 0
        provider_counts: Dict[str, int] = {}

        buy_signals  = [s for s in signals if "BUY" in s.signal]
        pass_signals = [s for s in signals if "BUY" not in s.signal]

        logger.info(
            f"LLM override: {len(buy_signals)} BUY signals to review | "
            f"max_overrides={max_overrides}"
        )

        for signal in buy_signals:
            features = self.features_map.get(signal.symbol, {})

            verdict = await get_llm_verdict(
                signal, self.regime, features, session_stats=self.session_stats
            )
            _apply_verdict(signal, verdict)
            _update_db(signal.symbol, self.run_date, verdict)

            provider_counts[verdict.provider] = provider_counts.get(verdict.provider, 0) + 1

            if verdict.verdict in ("VETO", "REDUCE_CONFIDENCE"):
                override_count += 1

            if override_count >= max_overrides and max_overrides > 0:
                logger.warning(
                    f"LLM override rate cap hit ({override_count}/{max_overrides}) "
                    f"— remaining signals passed through"
                )
                break

        all_signals = buy_signals + pass_signals
        priority = {"STRONG_BUY": 0, "BUY": 1, "HOLD": 2, "SELL": 3, "STRONG_SELL": 4}
        all_signals.sort(key=lambda s: (priority.get(s.signal, 5), -s.fused_score))

        logger.info(
            f"LLM override complete: {override_count} overrides | "
            f"BUY={sum(1 for s in all_signals if 'BUY' in s.signal)} | "
            f"providers={provider_counts} | "
            f"session_failure_rate={self.session_stats.failure_rate:.0%}"
        )
        return all_signals