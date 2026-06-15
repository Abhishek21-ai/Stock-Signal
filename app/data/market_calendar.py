"""
NSE holiday and market schedule guard.
Section 19.3: pipeline must not run on NSE trading holidays.
"""
from __future__ import annotations

from datetime import date

from app.db import get_async_db
from app.logger import get_logger

logger = get_logger("market_calendar")


async def is_nse_holiday(check_date: date) -> bool:
    """Returns True if check_date is an NSE holiday."""
    async with get_async_db() as db:
        result = await db.execute(
            "SELECT 1 FROM nse_holiday_calendar WHERE holiday_date = :d",
            {"d": check_date},
        )
        row = result.fetchone()
        if row:
            logger.info(f"{check_date} is an NSE holiday")
            return True
    return False


async def is_fno_expiry(check_date: date) -> bool:
    """Returns True if check_date is an F&O expiry day (Section 19.6)."""
    async with get_async_db() as db:
        result = await db.execute(
            """
            SELECT 1 FROM macro_events
            WHERE event_date = :d AND event_type = 'FNO_EXPIRY'
            """,
            {"d": check_date},
        )
        return result.fetchone() is not None


async def get_active_macro_events(check_date: date, window_days: int = 2) -> list:
    """
    Returns HIGH-impact macro events within window_days of check_date.
    Section 19.6: reduce confidence by 20% within 2 days.
    """
    async with get_async_db() as db:
        result = await db.execute(
            """
            SELECT event_date, event_type, event_name, expected_impact
            FROM macro_events
            WHERE event_date BETWEEN :d AND :d + INTERVAL ':w days'
              AND expected_impact = 'HIGH'
            ORDER BY event_date
            """,
            {"d": check_date, "w": window_days},
        )
        return [dict(r) for r in result.fetchall()]
