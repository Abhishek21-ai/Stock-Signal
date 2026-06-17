"""
LLM Override Layer — Section 10 & 27

Flow per stock:
  1. Check Redis cache (key: llm_override:{symbol}:{date}, TTL 24h) — skip API if hit
  2. Build structured prompt from FusedSignal + RegimeResult + features
  3. Call Groq (llama-3.3-70b-versatile) with 10s timeout
  4. Fallback to OpenAI gpt-4o-mini if Groq fails/times out
  5. Parse LLM response → LLMVerdict (CONFIRM | VETO | REDUCE_CONFIDENCE)
  6. Apply verdict to FusedSignal score/confidence
  7. Cache result in Redis (24h TTL)
  8. Update daily_signals row in DB (llm_override, llm_status, llm_explanation)

Override rate cap (Section 10.3):
  Max 30% of signals may be overridden per run (settings.max_llm_override_rate).
  If cap is hit, remaining signals get llm_override=NONE with status=FALLBACK.

Verdict effects:
  CONFIRM          → no score change, llm_override=NONE
  VETO             → score set to 0, signal → HOLD, confidence halved
  REDUCE_CONFIDENCE→ confidence reduced by 25%
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from app.cache import llm_cache
from app.db import get_sync_db
from app.fusion.engine import FusedSignal
from app.logger import get_logger
from app.regime.detector import RegimeResult
from app.strategies.base import score_to_signal
from config.settings import settings

logger = get_logger("llm")

# ── Constants ─────────────────────────────────────────────────
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
LLM_CACHE_TTL = 86_400          # 24 hours
CONFIDENCE_REDUCTION = 25.0     # pct points removed on REDUCE_CONFIDENCE

VALID_VERDICTS = {"CONFIRM", "VETO", "REDUCE_CONFIDENCE"}


# ── Data classes ──────────────────────────────────────────────

@dataclass
class LLMVerdict:
    verdict: str                 # CONFIRM | VETO | REDUCE_CONFIDENCE
    explanation: str
    llm_status: str              # OK | TIMEOUT | FALLBACK | ERROR | CACHED
    provider: str                # groq | openai | cache | none
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


def _parse_verdict(raw: str) -> LLMVerdict | None:
    """
    Parse LLM JSON response into LLMVerdict.
    Returns None if parsing fails.
    """
    try:
        # Strip markdown fences if present
        clean = raw.strip().strip("```json").strip("```").strip()
        data  = json.loads(clean)
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


# ── Main per-symbol override ──────────────────────────────────

async def get_llm_verdict(
    signal: FusedSignal,
    regime: RegimeResult,
    features: Dict,
) -> LLMVerdict:
    """
    Get LLM verdict for a single signal.
    Checks cache first, then Groq, then OpenAI fallback.
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

    prompt = build_prompt(signal, regime, features)
    t0     = time.monotonic()

    # ── 2. Groq (primary) ─────────────────────────────────────
    verdict = None
    provider = "groq"
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
        logger.warning(f"{signal.symbol}: Groq failed ({e}) — trying OpenAI fallback")

    # ── 3. OpenAI fallback ────────────────────────────────────
    if verdict is None and settings.openai_api_key:
        provider = "openai"
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
            logger.warning(f"{signal.symbol}: OpenAI fallback also failed: {e}")

    # ── 4. Final fallback — CONFIRM with error status ─────────
    if verdict is None:
        verdict = LLMVerdict(
            verdict="CONFIRM",
            explanation="LLM unavailable — quant signal used as-is",
            llm_status="ERROR",
            provider="none",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        logger.warning(f"{signal.symbol}: Both LLM providers failed — defaulting CONFIRM")

    # ── 5. Cache result ───────────────────────────────────────
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
        logger.info(f"{signal.symbol}: LLM VETO — {verdict.explanation}")
        signal.fused_score = 0.0
        signal.signal      = "HOLD"
        signal.confidence  = max(0.0, signal.confidence / 2)
        signal.reasons.append(f"LLM VETO: {verdict.explanation}")

    elif verdict.verdict == "REDUCE_CONFIDENCE":
        logger.info(f"{signal.symbol}: LLM REDUCE_CONFIDENCE — {verdict.explanation}")
        signal.confidence = max(0.0, signal.confidence - CONFIDENCE_REDUCTION)
        signal.reasons.append(f"LLM reduced confidence: {verdict.explanation}")

    else:  # CONFIRM
        signal.reasons.append(f"LLM confirmed: {verdict.explanation}")

    return signal


def _update_db(
    symbol: str,
    run_date: date,
    verdict: LLMVerdict,
) -> None:
    """Update llm_override fields in daily_signals table."""
    # Map verdict → DB enum
    override_map = {
        "VETO":              "VETO",
        "REDUCE_CONFIDENCE": "REDUCE_CONFIDENCE",
        "CONFIRM":           "NONE",
    }
    db_override = override_map.get(verdict.verdict, "NONE")

    # Map status → if timeout/error use TIMEOUT/FALLBACK enums
    if verdict.llm_status in ("TIMEOUT",):
        db_override = "TIMEOUT"
    elif verdict.llm_status in ("ERROR",):
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
                    verdict.llm_status,
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

    async def run(self, signals: List[FusedSignal]) -> List[FusedSignal]:
        """
        Process all signals. Only BUY/STRONG_BUY signals go to LLM.
        SELL/HOLD signals are passed through unchanged.
        Override rate cap: max 30% of total signals.
        """
        total          = len(signals)
        max_overrides  = int(total * settings.max_llm_override_rate)
        override_count = 0

        # Only send BUY-side signals to LLM (cost + latency control)
        buy_signals  = [s for s in signals if "BUY" in s.signal]
        pass_signals = [s for s in signals if "BUY" not in s.signal]

        logger.info(
            f"LLM override: {len(buy_signals)} BUY signals to review | "
            f"max_overrides={max_overrides}"
        )

        for signal in buy_signals:
            features = self.features_map.get(signal.symbol, {})

            verdict = await get_llm_verdict(signal, self.regime, features)
            _apply_verdict(signal, verdict)
            _update_db(signal.symbol, self.run_date, verdict)

            if verdict.verdict in ("VETO", "REDUCE_CONFIDENCE"):
                override_count += 1

            # Rate cap
            if override_count >= max_overrides:
                logger.warning(
                    f"LLM override rate cap hit ({override_count}/{max_overrides}) "
                    f"— remaining signals passed through"
                )
                break

        all_signals = buy_signals + pass_signals
        # Re-sort by score
        priority = {"STRONG_BUY": 0, "BUY": 1, "HOLD": 2, "SELL": 3, "STRONG_SELL": 4}
        all_signals.sort(key=lambda s: (priority.get(s.signal, 5), -s.fused_score))

        logger.info(
            f"LLM override complete: {override_count} overrides | "
            f"BUY={sum(1 for s in all_signals if 'BUY' in s.signal)}"
        )
        return all_signals
