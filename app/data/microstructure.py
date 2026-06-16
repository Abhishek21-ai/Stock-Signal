"""
Market Microstructure Filters — Section 19
Applied after signal fusion, before notifications.

Checks:
  19.1 Circuit breaker detection
  19.2 Gap open classification (handled by gap_detector.py pre-market)
  19.5 Promoter pledge risk
  19.6 Macro event window (confidence reduction)
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from app.db import get_sync_db
from app.logger import get_logger

logger = get_logger("microstructure")


def check_circuit_breaker(symbol: str, run_date: date) -> bool:
    """
    Returns True if stock hit upper/lower circuit on run_date.
    Circuit = High == Low (frozen price) or price at ±20% from prev close.
    Section 19.1: circuit-hit stocks → signal = CIRCUIT_HIT, skip all strategies.
    """
    with get_sync_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT t.high, t.low, t.close, y.close as prev_close
            FROM market_data t
            LEFT JOIN market_data y
                ON y.stock = t.stock
                AND y.date = (
                    SELECT MAX(date) FROM market_data
                    WHERE stock = %s AND date < %s
                )
            WHERE t.stock = %s AND t.date = %s
            """,
            (symbol, run_date, symbol, run_date),
        )
        row = cursor.fetchone()

    if not row or not row["prev_close"]:
        return False

    # Frozen price = circuit
    if row["high"] == row["low"]:
        logger.warning(f"{symbol}: Circuit breaker detected (High==Low on {run_date})")
        return True

    # ±20% move from prev close
    pct_change = abs(row["close"] - row["prev_close"]) / row["prev_close"]
    if pct_change >= 0.195:   # 19.5% threshold (just under 20% circuit)
        logger.warning(f"{symbol}: Near-circuit move detected ({pct_change:.1%} on {run_date})")
        return True

    return False


def classify_gap_open(symbol: str, run_date: date) -> Optional[str]:
    """
    Classifies overnight gap (Section 19.2).
    Returns: None | 'CHASE_RISK' | 'STOP_INVALIDATED'
    """
    with get_sync_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT t.open, t.close as today_open,
                   y.close as prev_close,
                   y.high as prev_high, y.low as prev_low
            FROM market_data t
            LEFT JOIN market_data y ON y.stock = t.stock
                AND y.date = (
                    SELECT MAX(date) FROM market_data
                    WHERE stock = %s AND date < %s
                )
            WHERE t.stock = %s AND t.date = %s
            """,
            (symbol, run_date, symbol, run_date),
        )
        row = cursor.fetchone()

    if not row or not row["prev_close"]:
        return None

    gap_pct = (row["today_open"] - row["prev_close"]) / row["prev_close"]

    if gap_pct > 0.015:     # +1.5% gap up
        return "CHASE_RISK"
    if gap_pct < -0.015:    # -1.5% gap down
        return "STOP_INVALIDATED"
    return None


def get_promoter_pledge_risk(symbol: str) -> tuple[Optional[float], str]:
    """
    Returns (pledge_pct, risk_level) from promoter_data table.
    Section 19.5: HIGH > 50%, CRITICAL > 75% → reduce confidence or veto.
    """
    with get_sync_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT pledged_pct FROM promoter_data
            WHERE stock = %s
            ORDER BY quarter DESC
            LIMIT 1
            """,
            (symbol,),
        )
        row = cursor.fetchone()

    if not row or row["pledged_pct"] is None:
        return None, "UNKNOWN"

    pct = float(row["pledged_pct"])
    if pct >= 75:
        return pct, "CRITICAL"
    if pct >= 50:
        return pct, "HIGH"
    return pct, "LOW"


def is_macro_window_active(run_date: date, window_days: int = 2) -> tuple[bool, Optional[str]]:
    """
    Returns (is_active, event_type) if a HIGH-impact macro event is within window.
    Section 19.6: reduce all confidence by 20% during macro windows.
    """
    with get_sync_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT event_type, event_name FROM macro_events
            WHERE event_date BETWEEN %s AND %s
              AND expected_impact = 'HIGH'
            ORDER BY event_date
            LIMIT 1
            """,
            (run_date, run_date + timedelta(days=window_days)),
        )
        row = cursor.fetchone()

    if row:
        return True, row["event_type"]
    return False, None


def apply_microstructure_filters(
    symbol: str,
    run_date: date,
    confidence_pct: float,
) -> dict:
    """
    Runs all microstructure checks and returns adjusted signal metadata.
    Called by signal fusion layer for each stock.
    """
    result = {
        "circuit_hit": False,
        "gap_open_type": None,
        "macro_window_active": False,
        "macro_event_type": None,
        "promoter_pledge_pct": None,
        "pledge_risk_flag": "UNKNOWN",
        "liquidity_warning": False,
        "adjusted_confidence": confidence_pct,
        "suppress_signal": False,
    }

    # 1. Circuit breaker
    if check_circuit_breaker(symbol, run_date):
        result["circuit_hit"] = True
        result["suppress_signal"] = True
        return result

    # 2. Gap open
    gap = classify_gap_open(symbol, run_date)
    result["gap_open_type"] = gap
    if gap == "STOP_INVALIDATED":
        result["suppress_signal"] = True
        return result
    if gap == "CHASE_RISK":
        result["adjusted_confidence"] *= 0.75   # -25% confidence

    # 3. Promoter pledge
    pledge_pct, pledge_risk = get_promoter_pledge_risk(symbol)
    result["promoter_pledge_pct"] = pledge_pct
    result["pledge_risk_flag"] = pledge_risk
    if pledge_risk == "CRITICAL":
        result["suppress_signal"] = True
        return result
    if pledge_risk == "HIGH":
        result["adjusted_confidence"] *= 0.80   # -20% confidence

    # 4. Macro window
    macro_active, macro_type = is_macro_window_active(run_date)
    result["macro_window_active"] = macro_active
    result["macro_event_type"] = macro_type
    if macro_active:
        result["adjusted_confidence"] *= 0.80   # -20% confidence

    return result
