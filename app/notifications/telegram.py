"""
Telegram Notification Dispatcher — Section 14.2 (paper trading interface)

This is the primary way you'll see signals day to day during paper
trading — the dashboard is a nice-to-have, this is the thing that
actually pings you when a signal fires.

Public API (called by app/pipeline.py Stage 10):
    await send_signal_digest(signals, regime, run_date)

Also exposed for manual/ad-hoc use:
    await send_text(message)                  — raw text, chunked if long
    await send_health_alert(alert)             — used by app/monitoring/health.py
    await send_trade_lifecycle_event(...)      — used by app/trades/lifecycle.py

Design notes:
  - Telegram hard-caps messages at 4096 chars. Long digests are split
    into multiple messages rather than truncated, so nothing is silently
    dropped.
  - All network calls go through httpx with a short timeout + one retry,
    since a notification failure must never take down the pipeline
    (Stage 10 already wraps this in try/except, but we don't rely on that
    alone — failures here are logged and swallowed).
  - position_size_shares / position_value_inr / risk_amount_inr are
    attached dynamically onto FusedSignal by the Portfolio Manager and
    are NOT part of the dataclass definition, so every access here uses
    getattr() with a safe default — a signal that was HOLD'd or rejected
    before reaching the portfolio stage won't have them set.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import List, Optional, TYPE_CHECKING

import httpx

from app.logger import get_logger
from config.settings import settings

if TYPE_CHECKING:
    from app.fusion.engine import FusedSignal
    from app.regime.detector import RegimeResult

logger = get_logger("notifications.telegram")

TELEGRAM_API_BASE   = "https://api.telegram.org"
TELEGRAM_MAX_CHARS  = 4096
REQUEST_TIMEOUT_SEC = 10.0
MAX_RETRIES         = 1

SIGNAL_EMOJI = {
    "STRONG_BUY":  "🟢🟢",
    "BUY":         "🟢",
    "HOLD":        "⚪",
    "SELL":        "🔴",
    "STRONG_SELL": "🔴🔴",
}

REGIME_EMOJI = {
    "BULL":      "📈",
    "BEAR":      "📉",
    "SIDEWAYS":  "↔️",
    "UNCERTAIN": "❓",
}


# ── Low-level send primitives ──────────────────────────────────

def _chunk_message(text: str, limit: int = TELEGRAM_MAX_CHARS) -> List[str]:
    """
    Split text into Telegram-safe chunks, breaking on line boundaries
    so a single stock's block is never cut in half mid-line.
    """
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    current = ""
    for line in text.split("\n"):
        # +1 for the newline that will be re-added
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


async def send_text(message: str, parse_mode: str = "Markdown") -> bool:
    """
    Send a raw text message to the configured Telegram chat.
    Returns True if all chunks sent successfully, False otherwise.
    Never raises — failures are logged and swallowed, since a broken
    notification must never break the pipeline that triggered it.
    """
    if not settings.telegram_enabled:
        logger.debug("Telegram not configured — skipping send")
        return False

    url = f"{TELEGRAM_API_BASE}/bot{settings.telegram_bot_token}/sendMessage"
    chunks = _chunk_message(message)
    all_ok = True

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SEC) as client:
        for i, chunk in enumerate(chunks):
            ok = await _send_one_chunk(client, url, chunk, parse_mode)
            all_ok = all_ok and ok
            # Telegram rate limit: ~1 msg/sec per chat is safe for multi-part digests
            if i < len(chunks) - 1:
                await asyncio.sleep(1.0)

    return all_ok


async def _send_one_chunk(
    client: httpx.AsyncClient,
    url: str,
    text: str,
    parse_mode: str,
) -> bool:
    payload = {
        "chat_id":    settings.telegram_chat_id,
        "text":       text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return True

            # Markdown parse errors are common with special chars in
            # LLM-generated reasons text — retry once as plain text
            # rather than dropping the whole message.
            if resp.status_code == 400 and parse_mode != "":
                logger.warning(
                    f"Telegram rejected Markdown (400) — retrying as plain text"
                )
                payload["parse_mode"] = ""
                continue

            logger.warning(
                f"Telegram send failed: {resp.status_code} {resp.text[:200]}"
            )
        except httpx.TimeoutException:
            logger.warning(f"Telegram send timed out (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"Telegram send error: {e}")

    return False


# ── Message formatting ─────────────────────────────────────────

def _fmt_price(value: Optional[float]) -> str:
    return f"₹{value:,.2f}" if value is not None else "—"


def _fmt_signal_block(signal: "FusedSignal") -> str:
    emoji  = SIGNAL_EMOJI.get(signal.signal, "⚪")
    shares = getattr(signal, "position_size_shares", None)
    value  = getattr(signal, "position_value_inr", None)
    risk   = getattr(signal, "risk_amount_inr", None)

    lines = [
        f"{emoji} *{signal.symbol}* — {signal.signal}",
        f"   Score: {signal.fused_score:+.1f}  Confidence: {signal.confidence:.0f}%",
        f"   Entry: {_fmt_price(signal.entry_price)}  "
        f"Stop: {_fmt_price(signal.stop_loss)}  "
        f"Target: {_fmt_price(signal.target_price)}",
    ]

    if shares:
        lines.append(
            f"   Size: {shares} shares ({_fmt_price(value)}) "
            f"| Risk: {_fmt_price(risk)}"
        )

    if getattr(signal, "correlation_adjusted", False):
        lines.append("   ⚠️ correlation-adjusted")

    # Show only the single most informative reason inline — full list
    # is in the DB / dashboard, the Telegram digest stays scannable.
    if signal.reasons:
        top_reason = signal.reasons[0]
        if len(top_reason) > 100:
            top_reason = top_reason[:97] + "..."
        lines.append(f"   ↳ {top_reason}")

    return "\n".join(lines)


def _build_digest(
    signals: List["FusedSignal"],
    regime: Optional["RegimeResult"],
    run_date: date,
) -> str:
    regime_name = regime.regime if regime else "UNKNOWN"
    regime_icon = REGIME_EMOJI.get(regime_name, "❓")

    buys  = [s for s in signals if "BUY"  in s.signal]
    sells = [s for s in signals if "SELL" in s.signal]

    header = [
        f"📊 *Stock Signal Platform — {run_date.strftime('%d %b %Y')}*",
        f"{regime_icon} Regime: *{regime_name}*"
        + (f" ({regime.regime_confidence})" if regime else ""),
        f"Signals: {len(buys)} BUY, {len(sells)} SELL",
        "",
    ]

    body: List[str] = []
    if buys:
        body.append("*── BUY SIGNALS ──*")
        for s in sorted(buys, key=lambda x: -x.fused_score):
            body.append(_fmt_signal_block(s))
            body.append("")

    if sells:
        body.append("*── SELL SIGNALS ──*")
        for s in sorted(sells, key=lambda x: x.fused_score):
            body.append(_fmt_signal_block(s))
            body.append("")

    return "\n".join(header + body).rstrip()


# ── Public API: called by app/pipeline.py Stage 10 ─────────────

async def send_signal_digest(
    signals: List["FusedSignal"],
    regime: Optional["RegimeResult"],
    run_date: date,
) -> bool:
    """
    Format and send the daily signal digest.
    `signals` should already be filtered to actionable (non-HOLD) signals
    by the caller — see app/pipeline.py `_run_notifications`.
    """
    if not signals:
        logger.info("No actionable signals — skipping digest send")
        return True

    message = _build_digest(signals, regime, run_date)
    ok = await send_text(message)

    if ok:
        logger.info(f"Telegram digest sent — {len(signals)} signals")
    else:
        logger.warning("Telegram digest failed to send")

    return ok


# ── Public API: called by app/monitoring/health.py ──────────────

async def send_health_alert(
    severity: str,
    metric: str,
    message: str,
) -> bool:
    """
    Send a single health/monitoring alert.
    Kept deliberately terse — these fire mid-pipeline and shouldn't
    flood the chat with the same formatting weight as a signal digest.
    """
    icon = "🚨" if severity == "CRITICAL" else "⚠️"
    text = f"{icon} *{severity}* — `{metric}`\n{message}"
    return await send_text(text)


# ── Public API: called by app/trades/lifecycle.py ───────────────

async def send_trade_lifecycle_event(
    symbol: str,
    event: str,        # ENTERED | TARGET_HIT | STOP_HIT | EXPIRED
    price: float,
    pnl_pct: Optional[float] = None,
) -> bool:
    """Notify on individual trade state transitions (paper trading log)."""
    icons = {
        "ENTERED":    "▶️",
        "TARGET_HIT": "🎯",
        "STOP_HIT":   "🛑",
        "EXPIRED":    "⏱️",
    }
    icon = icons.get(event, "ℹ️")
    text = f"{icon} *{symbol}* — {event} @ {_fmt_price(price)}"
    if pnl_pct is not None:
        sign = "+" if pnl_pct >= 0 else ""
        text += f"  ({sign}{pnl_pct:.2f}%)"
    return await send_text(text)


# ── Connectivity check ───────────────────────────────────────────

async def test_connection() -> bool:
    """
    Verify the bot token + chat ID actually work, via Telegram's
    getMe endpoint plus a real test send. Used by scripts/test_telegram.py.
    """
    if not settings.telegram_enabled:
        logger.warning("Telegram not configured (missing token or chat_id)")
        return False

    url = f"{TELEGRAM_API_BASE}/bot{settings.telegram_bot_token}/getMe"
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SEC) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.error(f"Telegram getMe failed: {resp.status_code} {resp.text}")
                return False
            bot_info = resp.json().get("result", {})
            logger.info(f"Telegram bot connected: @{bot_info.get('username', '?')}")
    except Exception as e:
        logger.error(f"Telegram connectivity check failed: {e}")
        return False

    return await send_text("✅ Stock Signal Platform — Telegram connection test successful")
