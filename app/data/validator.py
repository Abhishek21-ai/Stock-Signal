"""
Post-ingestion data quality checks.
Section 5.3: runs after upsert to catch DB-level anomalies.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

from app.db import get_sync_db
from app.logger import get_logger

logger = get_logger("validator")


def check_missing_dates(symbol: str, run_date: date, lookback_days: int = 10) -> List[date]:
    """
    Detects unexpected gaps in market_data for a symbol.
    Skips known NSE holidays.
    Returns list of missing dates.
    """
    with get_sync_db() as conn:
        cursor = conn.cursor()

        # Get all dates we have for this stock in lookback window
        cursor.execute(
            """
            SELECT date FROM market_data
            WHERE stock = %s AND date >= %s
            ORDER BY date
            """,
            (symbol, run_date - timedelta(days=lookback_days)),
        )
        have_dates = {row["date"] for row in cursor.fetchall()}

        # Get NSE holidays in same window
        cursor.execute(
            """
            SELECT holiday_date FROM nse_holiday_calendar
            WHERE holiday_date >= %s AND holiday_date <= %s
            """,
            (run_date - timedelta(days=lookback_days), run_date),
        )
        holidays = {row["holiday_date"] for row in cursor.fetchall()}

    # Build expected trading days (Mon–Fri, excluding holidays)
    missing = []
    check = run_date - timedelta(days=lookback_days)
    while check <= run_date:
        if check.weekday() < 5 and check not in holidays and check not in have_dates:
            missing.append(check)
        check += timedelta(days=1)

    return missing


def get_data_coverage_report(run_date: date) -> Dict[str, dict]:
    """
    Returns per-stock coverage stats for dashboard Section 24.2.
    """
    with get_sync_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                stock,
                COUNT(*) as total_rows,
                MIN(date) as first_date,
                MAX(date) as last_date,
                MAX(date) = %s as is_current
            FROM market_data
            GROUP BY stock
            ORDER BY stock
            """,
            (run_date,),
        )
        rows = cursor.fetchall()

    return {
        row["stock"]: {
            "total_rows": row["total_rows"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "is_current": row["is_current"],
        }
        for row in rows
    }


def get_latest_ohlcv(symbol: str, n: int = 252) -> List[dict]:
    """
    Fetch latest n rows from market_data for a symbol.
    Used by feature engineering layer.
    """
    with get_sync_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT date, open, high, low, close, volume, adjusted_close
            FROM market_data
            WHERE stock = %s
            ORDER BY date DESC
            LIMIT %s
            """,
            (symbol, n),
        )
        rows = cursor.fetchall()

    # Return in ascending order for TA calculations
    return list(reversed([dict(r) for r in rows]))
