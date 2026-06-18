"""
NSE holiday and market schedule guard.
Section 19.3: pipeline must not run on NSE trading holidays.

Uses sync DB (psycopg2) to avoid SQLAlchemy text() complexity in the holiday check.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

from app.db import get_sync_db
from app.logger import get_logger

logger = get_logger("market_calendar")


async def is_nse_holiday(check_date: date) -> bool:
    """Returns True if check_date is an NSE holiday or weekend."""
    # Weekend check first (no DB needed)
    if check_date.weekday() >= 5:
        logger.info(f"{check_date} is a weekend — market closed")
        return True

    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM nse_holiday_calendar WHERE holiday_date = %s",
                (check_date,),
            )
            if cur.fetchone():
                logger.info(f"{check_date} is an NSE holiday")
                return True
    except Exception as e:
        # If DB unavailable, don't block the pipeline
        logger.warning(f"Holiday check failed ({e}) — assuming market open")

    return False


async def is_fno_expiry(check_date: date) -> bool:
    """Returns True if check_date is an F&O expiry day."""
    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM macro_events WHERE event_date = %s AND event_type = 'FNO_EXPIRY'",
                (check_date,),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


async def get_active_macro_events(check_date: date, window_days: int = 2) -> List[dict]:
    """Returns HIGH-impact macro events within window_days of check_date."""
    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_date, event_type, event_name, expected_impact
                FROM macro_events
                WHERE event_date BETWEEN %s AND %s
                  AND expected_impact = 'HIGH'
                ORDER BY event_date
                """,
                (check_date, check_date + timedelta(days=window_days)),
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []