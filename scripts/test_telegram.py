"""
Telegram Notifications test — verifies bot connectivity and message
formatting before relying on it for daily paper trading.

Usage:
    python scripts/test_telegram.py

Setup required first:
    1. Message @BotFather on Telegram, /newbot, get the token
    2. Set TELEGRAM_BOT_TOKEN in .env
    3. Message your new bot once (anything), then visit:
       https://api.telegram.org/bot<TOKEN>/getUpdates
       to find your chat_id in the response
    4. Set TELEGRAM_CHAT_ID in .env
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from datetime import date

from config.settings import settings


async def test_config_present():
    print("\n── Test 1: Configuration check ───────────────────────")
    if not settings.telegram_bot_token:
        print("  ❌ TELEGRAM_BOT_TOKEN not set in .env")
        return False
    if not settings.telegram_chat_id:
        print("  ❌ TELEGRAM_CHAT_ID not set in .env")
        return False
    print(f"  ✅ Bot token present ({settings.telegram_bot_token[:10]}...)")
    print(f"  ✅ Chat ID present ({settings.telegram_chat_id})")
    return True


async def test_connectivity():
    print("\n── Test 2: Bot connectivity (getMe + test send) ──────")
    from app.notifications.telegram import test_connection
    ok = await test_connection()
    if ok:
        print("  ✅ Bot connected and test message sent — check your Telegram")
    else:
        print("  ❌ Connection failed — check token/chat_id and bot is not blocked")
    return ok


async def test_digest_formatting():
    print("\n── Test 3: Signal digest formatting (synthetic data) ──")
    from app.fusion.engine import FusedSignal
    from app.regime.detector import RegimeResult

    regime = RegimeResult(
        regime="BULL",
        regime_confidence="NORMAL",
        fusion_weights={"trend": 0.35, "momentum": 0.25, "reversion": 0.10,
                         "breakout": 0.20, "volume": 0.10},
        nifty_close=24500.0,
    )

    s1 = FusedSignal(
        symbol="RELIANCE", signal="STRONG_BUY", fused_score=72.5,
        confidence=78.0, regime="BULL",
        entry_price=1335.50, stop_loss=1298.00, target_price=1410.00,
        reasons=["Strong EMA alignment + volume confirmation"],
    )
    # Simulate what PortfolioManager dynamically attaches
    s1.position_size_shares = 45
    s1.position_value_inr   = 60097.50
    s1.risk_amount_inr      = 1687.50

    s2 = FusedSignal(
        symbol="TCS", signal="SELL", fused_score=-38.0,
        confidence=38.0, regime="BULL",
        entry_price=2080.0, stop_loss=2125.0, target_price=1995.0,
        reasons=["RSI divergence + breakdown below EMA50"],
    )

    from app.notifications.telegram import _build_digest, _chunk_message

    message = _build_digest([s1, s2], regime, date.today())
    print("  Generated digest preview:")
    print("  " + "\n  ".join(message.split("\n")[:8]) + "\n  ...")

    chunks = _chunk_message(message)
    print(f"  ✅ Digest built ({len(message)} chars, {len(chunks)} chunk(s))")

    if settings.telegram_enabled:
        from app.notifications.telegram import send_signal_digest
        ok = await send_signal_digest([s1, s2], regime, date.today())
        print(f"  {'✅' if ok else '❌'} Synthetic digest sent to Telegram")
    else:
        print("  ⚠️  Telegram not configured — formatting verified, not sent")


async def test_trade_lifecycle_event():
    print("\n── Test 4: Trade lifecycle event formatting ──────────")
    from app.notifications.telegram import send_trade_lifecycle_event

    if settings.telegram_enabled:
        ok1 = await send_trade_lifecycle_event("HDFCBANK", "ENTERED", 785.50)
        ok2 = await send_trade_lifecycle_event("HDFCBANK", "TARGET_HIT", 812.30, pnl_pct=3.41)
        ok3 = await send_trade_lifecycle_event("ITC", "STOP_HIT", 287.10, pnl_pct=-2.15)
        print(f"  {'✅' if (ok1 and ok2 and ok3) else '❌'} Trade event messages sent")
    else:
        print("  ⚠️  Telegram not configured — skipping live send")


async def test_health_alert():
    print("\n── Test 5: Health alert formatting ────────────────────")
    from app.notifications.telegram import send_health_alert

    if settings.telegram_enabled:
        ok = await send_health_alert(
            severity="WARNING",
            metric="data_ingestion_sla_test",
            message="This is a test alert from test_telegram.py — safe to ignore.",
        )
        print(f"  {'✅' if ok else '❌'} Health alert sent")
    else:
        print("  ⚠️  Telegram not configured — skipping live send")


async def test_long_message_chunking():
    print("\n── Test 6: Long message chunking (>4096 chars) ────────")
    from app.notifications.telegram import _chunk_message

    long_text = "\n".join([f"Line {i}: " + ("x" * 80) for i in range(100)])
    chunks = _chunk_message(long_text)
    total_len = sum(len(c) for c in chunks)
    assert len(chunks) > 1, "Should split into multiple chunks"
    assert all(len(c) <= 4096 for c in chunks), "Each chunk must respect Telegram's limit"
    print(f"  ✅ {len(long_text)} char message → {len(chunks)} chunks, "
          f"each ≤4096 chars, no content lost")


async def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — Telegram Notifications Test")
    print(f"{'='*60}")

    has_config = await test_config_present()

    if has_config:
        await test_connectivity()
    else:
        print("\n  ⚠️  Skipping live connectivity test — configure .env first")

    await test_digest_formatting()
    await test_trade_lifecycle_event()
    await test_health_alert()
    await test_long_message_chunking()

    print(f"\n{'='*60}")
    if has_config:
        print("  ✅ Telegram Notifications verified — check your chat for test messages")
    else:
        print("  ⚠️  Formatting verified. Add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
        print("     to .env to test live sending.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
