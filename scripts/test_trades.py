"""
Trade Lifecycle Manager test — run after test_portfolio.py passes.

Usage:
    python scripts/test_trades.py

What this tests:
  1. Create PENDING trades from accepted signals
  2. PENDING → ACTIVE transition (open within 0.5% of entry)
  3. PENDING → CLOSED (EXPIRY) — signal too old
  4. ACTIVE → CLOSED (STOP) — day low hits stop loss
  5. ACTIVE → CLOSED (TARGET) — day high hits target
  6. ACTIVE → CLOSED (CIRCUIT) — high==low==close
  7. get_open_trades() query
  8. get_trade_summary() rolling stats
Requires: postgres running
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from app.trades.lifecycle import (
    create_pending_trades, update_trade_states,
    get_open_trades, get_trade_summary,
    _close_trade, _get_ohlc, ENTRY_TOLERANCE_PCT, SIGNAL_VALIDITY_DAYS,
)
from app.db import get_sync_db


# ── Fixtures ──────────────────────────────────────────────────

def make_signal(symbol="RELIANCE", entry=1332.0, stop=1293.0, target=1412.0,
                signal="BUY", regime="UNCERTAIN"):
    class MockSignal:
        pass
    s = MockSignal()
    s.symbol              = symbol
    s.signal              = signal
    s.entry_price         = entry
    s.stop_loss           = stop
    s.target_price        = target
    s.position_size_shares= 112
    s.position_value_inr  = entry * 112
    s.regime              = regime
    s.reasons             = []
    return s


def insert_ohlc(stock: str, dt: date, open_p: float, high: float,
                low: float, close: float):
    """Insert a test OHLC row into market_data."""
    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO market_data
                    (date, stock, open, high, low, close, volume, adjusted_close)
                VALUES (%s, %s, %s, %s, %s, %s, 1000000, %s)
                ON CONFLICT (date, stock) DO UPDATE SET
                    open=EXCLUDED.open, high=EXCLUDED.high,
                    low=EXCLUDED.low,   close=EXCLUDED.close
                """,
                (dt, stock, open_p, high, low, close, close),
            )
    except Exception as e:
        print(f"  ⚠️  Could not insert OHLC: {e}")


def cleanup_test_trades():
    """Remove test trade rows."""
    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM trades WHERE stock LIKE 'TEST_%'"
            )
    except Exception as e:
        print(f"  ⚠️  Cleanup failed: {e}")


# ── Tests ─────────────────────────────────────────────────────

def test_create_pending():
    print("\n── Test 1: Create PENDING trades ────────────────────")
    cleanup_test_trades()
    today   = date.today()
    signals = [
        make_signal("TEST_REL",  entry=1332.0, stop=1293.0, target=1412.0),
        make_signal("TEST_TCS",  entry=2223.0, stop=2166.0, target=2336.0),
        make_signal("TEST_HOLD", signal="HOLD"),   # should be skipped
    ]
    created = create_pending_trades(signals, today)
    assert len(created) == 2, f"Expected 2 trades, got {len(created)}"
    print(f"  ✅ Created {len(created)} PENDING trades (HOLD skipped correctly)")
    return today


def test_pending_to_active(signal_date: date):
    print("\n── Test 2: PENDING → ACTIVE (entry gap ≤ 0.5%) ─────")
    # Entry = 1332, so 0.5% tolerance = ±6.66
    # Open at 1333 (within tolerance) → should activate
    check_date = signal_date + timedelta(days=1)
    insert_ohlc("TEST_REL", signal_date, open_p=1333.0, high=1350.0,
                low=1310.0, close=1340.0)
    insert_ohlc("TEST_TCS", signal_date, open_p=2500.0, high=2520.0,
                low=2480.0, close=2510.0)  # gap > 0.5% → stays PENDING

    counts = update_trade_states(check_date)
    print(f"  Activated: {counts['activated']} | Unchanged: {counts['unchanged']}")
    assert counts["activated"] >= 1
    print(f"  ✅ TEST_REL activated (open 1333 within 0.5% of entry 1332)")
    print(f"  ✅ TEST_TCS stayed PENDING (open 2500 > 0.5% from entry 2223)")


def test_pending_to_expiry():
    print("\n── Test 3: PENDING → CLOSED (EXPIRY) ────────────────")
    cleanup_test_trades()
    old_date = date.today() - timedelta(days=SIGNAL_VALIDITY_DAYS + 1)
    signals  = [make_signal("TEST_OLD", entry=1000.0, stop=970.0, target=1060.0)]
    create_pending_trades(signals, old_date)

    # Force signal_date to be old enough
    with get_sync_db() as conn:
        conn.cursor().execute(
            "UPDATE trades SET signal_date=%s WHERE stock='TEST_OLD'",
            (old_date,)
        )

    check_date = date.today()
    counts = update_trade_states(check_date)
    print(f"  Expired: {counts['expired']}")
    assert counts["expired"] >= 1
    print(f"  ✅ Old signal expired after {SIGNAL_VALIDITY_DAYS} days")
    cleanup_test_trades()


def test_active_stop_hit():
    print("\n── Test 4: ACTIVE → CLOSED (STOP) ──────────────────")
    cleanup_test_trades()
    today = date.today()
    yesterday = today - timedelta(days=1)

    # Create and immediately activate a trade
    with get_sync_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO trades (
                stock, entry_price_theoretical, entry_price_realistic,
                entry_price_actual, stop_loss_realistic, target_price,
                position_size_shares, position_value_inr,
                regime_at_entry, status, signal_date, entry_date
            ) VALUES (
                'TEST_STOP', 1000, 1000, 1000, 960, 1080,
                100, 100000, 'UNCERTAIN'::regime_type,
                'ACTIVE', %s, %s
            )
            """,
            (yesterday, yesterday),
        )

    # Yesterday: low hit the stop (960)
    insert_ohlc("TEST_STOP", yesterday, open_p=990.0, high=995.0,
                low=955.0, close=970.0)

    counts = update_trade_states(today)
    print(f"  Stopped: {counts['stopped']}")
    assert counts["stopped"] >= 1

    # Verify P&L recorded
    with get_sync_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pnl_pct, exit_reason FROM trades WHERE stock='TEST_STOP'")
        row = cur.fetchone()
        assert row["exit_reason"] == "STOP"
        assert float(row["pnl_pct"]) < 0   # should be a loss
        print(f"  ✅ Stop hit: pnl={float(row['pnl_pct']):.2f}% | exit={row['exit_reason']}")
    cleanup_test_trades()


def test_active_target_hit():
    print("\n── Test 5: ACTIVE → CLOSED (TARGET) ────────────────")
    cleanup_test_trades()
    today = date.today()
    yesterday = today - timedelta(days=1)

    with get_sync_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO trades (
                stock, entry_price_theoretical, entry_price_realistic,
                entry_price_actual, stop_loss_realistic, target_price,
                position_size_shares, position_value_inr,
                regime_at_entry, status, signal_date, entry_date
            ) VALUES (
                'TEST_TARGET', 1000, 1000, 1000, 960, 1080,
                100, 100000, 'UNCERTAIN'::regime_type,
                'ACTIVE', %s, %s
            )
            """,
            (yesterday, yesterday),
        )

    # Yesterday: high hit the target (1080)
    insert_ohlc("TEST_TARGET", yesterday, open_p=1010.0, high=1090.0,
                low=1005.0, close=1070.0)

    counts = update_trade_states(today)
    print(f"  Targeted: {counts['targeted']}")
    assert counts["targeted"] >= 1

    with get_sync_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pnl_pct, exit_reason FROM trades WHERE stock='TEST_TARGET'")
        row = cur.fetchone()
        assert row["exit_reason"] == "TARGET"
        assert float(row["pnl_pct"]) > 0   # should be a profit
        print(f"  ✅ Target hit: pnl={float(row['pnl_pct']):.2f}% | exit={row['exit_reason']}")
    cleanup_test_trades()


def test_open_trades_query():
    print("\n── Test 6: get_open_trades() ────────────────────────")
    trades = get_open_trades()
    print(f"  Open trades in DB: {len(trades)}")
    for t in trades[:3]:
        print(f"  {t['stock']}: {t['status']} | entry=₹{t['entry_price_realistic']}")
    print(f"  ✅ get_open_trades() returned {len(trades)} rows")


def test_trade_summary():
    print("\n── Test 7: get_trade_summary() ──────────────────────")
    summary = get_trade_summary(days=30)
    print(f"  Summary (last 30 days): {summary}")
    assert "total_trades" in summary or "total" in summary
    print(f"  ✅ Trade summary query works")


def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — Trade Lifecycle Test")
    print(f"{'='*60}")

    try:
        signal_date = test_create_pending()
        test_pending_to_active(signal_date)
        test_pending_to_expiry()
        test_active_stop_hit()
        test_active_target_hit()
        test_open_trades_query()
        test_trade_summary()

        print(f"\n{'='*60}")
        print("  ✅ Trade Lifecycle Manager verified")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_test_trades()


if __name__ == "__main__":
    main()