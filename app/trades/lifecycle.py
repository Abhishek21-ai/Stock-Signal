"""
Trade Lifecycle Manager — Section 22
Implements the full SIGNAL → PENDING → ACTIVE → CLOSED state machine.

State transitions (Section 22.2):
  SIGNAL  → PENDING  : Signal passes portfolio constraints → trade row created
  PENDING → ACTIVE   : Next session open within 0.5% of realistic entry
  PENDING → CLOSED   : valid_until passes without entry (exit_reason=EXPIRY)
  ACTIVE  → CLOSED   : stop loss hit (STOP), target hit (TARGET),
                       circuit breaker (CIRCUIT), or manual close (MANUAL)

Two daily jobs:
  1. create_pending_trades()   — called by pipeline after portfolio stage
  2. update_trade_states()     — called by scheduler at 08:30 IST (pre-market)
                                 checks yesterday's prices against open trades
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

from app.db import get_sync_db
from app.logger import get_logger
from config.settings import settings

logger = get_logger("trades")

# ── Constants ─────────────────────────────────────────────────
ENTRY_TOLERANCE_PCT  = 0.005   # 0.5% — PENDING→ACTIVE trigger
SIGNAL_VALIDITY_DAYS = 3       # PENDING expires after N days without entry


@dataclass
class TradeRow:
    """Lightweight representation of a trades table row."""
    id:                       int
    trade_uuid:               str
    stock:                    str
    status:                   str        # PENDING | ACTIVE | CLOSED
    entry_price_theoretical:  float
    entry_price_realistic:    float
    entry_price_actual:       Optional[float]
    stop_loss_realistic:      float
    target_price:             float
    position_size_shares:     int
    position_value_inr:       float
    regime_at_entry:          str
    signal_date:              date
    entry_date:               Optional[date]
    exit_date:                Optional[date]
    exit_reason:              Optional[str]
    pnl_pct:                  Optional[float]
    pnl_inr:                  Optional[float]


# ── Stage 1: Create PENDING trades after pipeline ────────────

def create_pending_trades(
    accepted_signals: list,
    run_date: date,
) -> List[str]:
    """
    Called by pipeline.py after portfolio stage.
    Creates a PENDING trade row for each accepted BUY signal.
    Returns list of created trade UUIDs.
    """
    created = []
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            for signal in accepted_signals:
                if "BUY" not in signal.signal:
                    continue

                # Get signal_id from daily_signals
                cursor.execute(
                    "SELECT id FROM daily_signals WHERE stock=%s AND date=%s",
                    (signal.symbol, run_date),
                )
                row = cursor.fetchone()
                signal_id = row["id"] if row else None

                valid_until = run_date + timedelta(days=SIGNAL_VALIDITY_DAYS)

                # Upsert — don't duplicate if pipeline reruns same day
                cursor.execute(
                    """
                    INSERT INTO trades (
                        signal_id, stock,
                        entry_price_theoretical, entry_price_realistic,
                        stop_loss_theoretical,   stop_loss_realistic,
                        target_price,
                        position_size_shares,    position_value_inr,
                        regime_at_entry,
                        status, signal_date
                    ) VALUES (
                        %s, %s,
                        %s, %s,
                        %s, %s,
                        %s,
                        %s, %s,
                        %s::regime_type,
                        'PENDING', %s
                    )
                    ON CONFLICT DO NOTHING
                    RETURNING trade_uuid::text
                    """,
                    (
                        signal_id,
                        signal.symbol,
                        signal.entry_price,
                        signal.entry_price,       # realistic = theoretical until filled
                        signal.stop_loss,
                        signal.stop_loss,
                        signal.target_price,
                        getattr(signal, "position_size_shares", 0),
                        getattr(signal, "position_value_inr", 0),
                        signal.regime,
                        run_date,
                    ),
                )
                r = cursor.fetchone()
                if r:
                    created.append(r["trade_uuid"])
                    logger.info(
                        f"Trade created: {signal.symbol} | PENDING | "
                        f"entry=₹{signal.entry_price} | "
                        f"stop=₹{signal.stop_loss} | "
                        f"target=₹{signal.target_price} | "
                        f"shares={getattr(signal, 'position_size_shares', 0)}"
                    )
    except Exception as e:
        logger.error(f"Failed to create pending trades: {e}")

    logger.info(f"Trade creation: {len(created)} new PENDING trades")
    return created


# ── Stage 2: Daily state update (pre-market) ─────────────────

def update_trade_states(check_date: date) -> Dict[str, int]:
    """
    Called daily by scheduler at 08:30 IST.
    Processes yesterday's OHLC against all open trades.

    Returns counts: {activated, expired, stopped, targeted, unchanged}
    """
    counts = {"activated": 0, "expired": 0, "stopped": 0,
              "targeted": 0, "unchanged": 0}
    price_date = check_date - timedelta(days=1)   # yesterday's data

    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()

            # ── 1. PENDING → ACTIVE or EXPIRY ────────────────
            cursor.execute(
                """
                SELECT t.id, t.stock, t.entry_price_realistic,
                       t.stop_loss_realistic, t.target_price,
                       t.position_size_shares, t.signal_date
                FROM trades t
                WHERE t.status = 'PENDING'
                """
            )
            pending = cursor.fetchall()

            for trade in pending:
                stock      = trade["stock"]
                entry      = float(trade["entry_price_realistic"])
                signal_dt  = trade["signal_date"]

                # Check expiry first
                expiry_date = signal_dt + timedelta(days=SIGNAL_VALIDITY_DAYS)
                if check_date > expiry_date:
                    _close_trade(cursor, trade["id"], check_date,
                                 exit_price=entry, exit_reason="EXPIRY",
                                 pnl_pct=0.0, pnl_inr=0.0)
                    counts["expired"] += 1
                    logger.info(f"{stock}: PENDING→CLOSED (EXPIRY after {SIGNAL_VALIDITY_DAYS}d)")
                    continue

                # Check if yesterday's open was within entry tolerance
                ohlc = _get_ohlc(cursor, stock, price_date)
                if not ohlc:
                    counts["unchanged"] += 1
                    continue

                open_price = ohlc["open"]
                gap_pct    = abs(open_price - entry) / entry

                if gap_pct <= ENTRY_TOLERANCE_PCT:
                    # Activate — use yesterday's open as actual entry
                    cursor.execute(
                        """
                        UPDATE trades SET
                            status             = 'ACTIVE',
                            entry_price_actual = %s,
                            entry_date         = %s,
                            updated_at         = NOW()
                        WHERE id = %s
                        """,
                        (open_price, price_date, trade["id"]),
                    )
                    counts["activated"] += 1
                    logger.info(
                        f"{stock}: PENDING→ACTIVE | "
                        f"entry=₹{open_price:.2f} (gap={gap_pct:.3%})"
                    )
                else:
                    counts["unchanged"] += 1

            # ── 2. ACTIVE → CLOSED (stop/target/circuit) ─────
            cursor.execute(
                """
                SELECT t.id, t.stock,
                       t.entry_price_actual, t.entry_price_realistic,
                       t.stop_loss_realistic, t.target_price,
                       t.position_size_shares, t.position_value_inr,
                       t.entry_date, t.regime_at_entry
                FROM trades t
                WHERE t.status = 'ACTIVE'
                """
            )
            active = cursor.fetchall()

            for trade in active:
                stock      = trade["stock"]
                entry      = float(trade["entry_price_actual"] or
                                   trade["entry_price_realistic"])
                stop       = float(trade["stop_loss_realistic"])
                target     = float(trade["target_price"])
                shares     = int(trade["position_size_shares"] or 0)

                ohlc = _get_ohlc(cursor, stock, price_date)
                if not ohlc:
                    counts["unchanged"] += 1
                    continue

                day_high  = ohlc["high"]
                day_low   = ohlc["low"]
                day_close = ohlc["close"]

                exit_price  = None
                exit_reason = None

                # Check circuit breaker (high==low==close = no trade)
                if day_high == day_low and day_high == day_close:
                    exit_price  = day_close
                    exit_reason = "CIRCUIT"

                # Stop loss hit (day low touched stop)
                elif day_low <= stop:
                    exit_price  = stop
                    exit_reason = "STOP"

                # Target hit (day high touched target)
                elif day_high >= target:
                    exit_price  = target
                    exit_reason = "TARGET"

                if exit_price and exit_reason:
                    pnl_pct = (exit_price - entry) / entry * 100 if entry else 0
                    pnl_inr = (exit_price - entry) * shares if shares else 0
                    entry_dt = trade["entry_date"]
                    duration = (price_date - entry_dt).days if entry_dt else 0

                    _close_trade(cursor, trade["id"], price_date,
                                 exit_price=exit_price,
                                 exit_reason=exit_reason,
                                 pnl_pct=round(pnl_pct, 4),
                                 pnl_inr=round(pnl_inr, 2),
                                 duration_days=duration)

                    if exit_reason == "STOP":
                        counts["stopped"] += 1
                    elif exit_reason == "TARGET":
                        counts["targeted"] += 1
                    else:
                        counts["expired"] += 1

                    logger.info(
                        f"{stock}: ACTIVE→CLOSED ({exit_reason}) | "
                        f"exit=₹{exit_price:.2f} | "
                        f"pnl={pnl_pct:+.2f}% (₹{pnl_inr:+,.0f})"
                    )
                else:
                    counts["unchanged"] += 1

    except Exception as e:
        logger.error(f"Trade state update failed: {e}")

    logger.info(
        f"Trade update complete | activated={counts['activated']} | "
        f"expired={counts['expired']} | stopped={counts['stopped']} | "
        f"targeted={counts['targeted']} | unchanged={counts['unchanged']}"
    )
    return counts


# ── Helpers ───────────────────────────────────────────────────

def _get_ohlc(cursor, stock: str, price_date: date) -> Optional[Dict]:
    """Fetch OHLC row for a stock on a given date."""
    cursor.execute(
        """
        SELECT open, high, low, close
        FROM market_data
        WHERE stock = %s AND date = %s
        """,
        (stock, price_date),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "open":  float(row["open"]),
        "high":  float(row["high"]),
        "low":   float(row["low"]),
        "close": float(row["close"]),
    }


def _close_trade(
    cursor,
    trade_id: int,
    exit_date: date,
    exit_price: float,
    exit_reason: str,
    pnl_pct: float,
    pnl_inr: float,
    duration_days: int = 0,
) -> None:
    """Update trade row to CLOSED status."""
    cursor.execute(
        """
        UPDATE trades SET
            status        = 'CLOSED',
            exit_date     = %s,
            exit_price    = %s,
            exit_reason   = %s::exit_reason,
            pnl_pct       = %s,
            pnl_inr       = %s,
            duration_days = %s,
            updated_at    = NOW()
        WHERE id = %s
        """,
        (exit_date, exit_price, exit_reason,
         pnl_pct, pnl_inr, duration_days, trade_id),
    )


# ── Query helpers for portfolio manager ──────────────────────

def get_open_trades(status: Optional[str] = None) -> List[Dict]:
    """
    Returns all PENDING + ACTIVE trades.
    Used by PortfolioManager.get_active_positions().
    """
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            query = """
                SELECT stock, status, entry_price_actual, entry_price_realistic,
                       stop_loss_realistic, target_price,
                       position_size_shares, position_value_inr,
                       regime_at_entry, signal_date, entry_date
                FROM trades
                WHERE status IN ('PENDING', 'ACTIVE')
            """
            if status:
                query += f" AND status = '{status}'"
            cursor.execute(query)
            return [dict(r) for r in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch open trades: {e}")
        return []


def get_trade_summary(days: int = 30) -> Dict:
    """
    Returns rolling performance summary for the last N days.
    Used by monitoring layer and dashboard.
    """
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*)                                      AS total,
                    COUNT(*) FILTER (WHERE pnl_pct > 0)           AS winners,
                    COUNT(*) FILTER (WHERE pnl_pct <= 0)          AS losers,
                    ROUND(AVG(pnl_pct)::numeric, 4)               AS avg_pnl_pct,
                    ROUND(SUM(pnl_inr)::numeric, 2)               AS total_pnl_inr,
                    ROUND(MAX(pnl_pct)::numeric, 4)               AS best_trade_pct,
                    ROUND(MIN(pnl_pct)::numeric, 4)               AS worst_trade_pct,
                    COUNT(*) FILTER (WHERE exit_reason = 'TARGET') AS targets_hit,
                    COUNT(*) FILTER (WHERE exit_reason = 'STOP')   AS stops_hit
                FROM trades
                WHERE status = 'CLOSED'
                  AND exit_date >= CURRENT_DATE - INTERVAL '%s days'
                """,
                (days,),
            )
            row = cursor.fetchone()
            if not row or not row["total"]:
                return {"total": 0, "message": f"No closed trades in last {days} days"}

            total   = int(row["total"])
            winners = int(row["winners"] or 0)
            return {
                "period_days":    days,
                "total_trades":   total,
                "winners":        winners,
                "losers":         int(row["losers"] or 0),
                "win_rate_pct":   round(winners / total * 100, 1) if total else 0,
                "avg_pnl_pct":    float(row["avg_pnl_pct"] or 0),
                "total_pnl_inr":  float(row["total_pnl_inr"] or 0),
                "best_trade_pct": float(row["best_trade_pct"] or 0),
                "worst_trade_pct":float(row["worst_trade_pct"] or 0),
                "targets_hit":    int(row["targets_hit"] or 0),
                "stops_hit":      int(row["stops_hit"] or 0),
            }
    except Exception as e:
        logger.error(f"Trade summary failed: {e}")
        return {"error": str(e)}
