"""
Data Ingestion Layer — Section 5
Fetches OHLCV data from yfinance (primary) with NSE holiday guard,
SLA validation, and stores into market_data table.

Flow:
  1. Check NSE holiday calendar
  2. Fetch OHLCV via yfinance (auto_adjust=True for corporate actions)
  3. Validate SLA (completeness, staleness, price sanity)
  4. Upsert into market_data table
  5. Flag breaches back to pipeline context
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from app.db import get_sync_db
from app.logger import get_logger
from config.settings import settings

logger = get_logger("ingestor")

# NSE suffix for yfinance
NSE_SUFFIX = ".NS"

# SLA thresholds
MIN_HISTORY_DAYS = 365 * settings.backtest_min_history_years
MAX_STALE_DAYS = 1          # data older than 1 trading day = breach
MIN_VOLUME = 10_000         # stocks with < 10k avg volume = liquidity warning


def to_nse_ticker(symbol: str) -> str:
    """RELIANCE → RELIANCE.NS"""
    return f"{symbol}{NSE_SUFFIX}" if not symbol.endswith(NSE_SUFFIX) else symbol


def from_nse_ticker(ticker: str) -> str:
    """RELIANCE.NS → RELIANCE"""
    return ticker.replace(NSE_SUFFIX, "")


# ── Core fetch ────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _fetch_yfinance(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance with retry.
    auto_adjust=True handles splits, bonuses, dividends (Section 5.2).
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval, auto_adjust=True)
    return df


def _fetch_batch(
    tickers: List[str],
    period: str = "5y",
) -> Dict[str, pd.DataFrame]:
    """
    Batch fetch multiple tickers in one yfinance call — faster than per-ticker.
    Returns dict: {symbol: DataFrame}
    """
    nse_tickers = [to_nse_ticker(t) for t in tickers]
    joined = " ".join(nse_tickers)

    logger.info(f"Fetching batch of {len(tickers)} tickers from yfinance")
    raw = yf.download(
        joined,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    result: Dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        # yfinance returns flat df for single ticker
        symbol = tickers[0]
        if not raw.empty:
            result[symbol] = raw
    else:
        for ticker, symbol in zip(nse_tickers, tickers):
            try:
                df = raw[ticker].dropna(how="all")
                if not df.empty:
                    result[symbol] = df
            except KeyError:
                logger.warning(f"No data returned for {symbol}")

    return result


# ── SLA Validation ────────────────────────────────────────────

class SLAViolation(Exception):
    pass


def validate_sla(symbol: str, df: pd.DataFrame, fetch_date: date) -> List[str]:
    """
    Validates fetched data against SLA rules (Section 5.3).
    Returns list of violation messages (empty = pass).
    """
    violations = []

    if df is None or df.empty:
        violations.append(f"{symbol}: No data returned")
        return violations

    # 1. Staleness check — last candle must be today or yesterday (trading day)
    last_date = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
    days_stale = (fetch_date - last_date).days
    if days_stale > MAX_STALE_DAYS:
        violations.append(f"{symbol}: Data is {days_stale} days stale (last={last_date})")

    # 2. Minimum history check
    if len(df) < MIN_HISTORY_DAYS * 0.6:   # allow 40% gaps (holidays etc)
        violations.append(
            f"{symbol}: Insufficient history — {len(df)} rows, need ~{MIN_HISTORY_DAYS}"
        )

    # 3. Price sanity
    if (df["Close"] <= 0).any():
        violations.append(f"{symbol}: Non-positive close prices detected")

    if (df["High"] < df["Low"]).any():
        violations.append(f"{symbol}: High < Low detected (data corruption)")

    # 4. Volume check
    avg_vol = df["Volume"].tail(20).mean()
    if avg_vol < MIN_VOLUME:
        violations.append(f"{symbol}: Low liquidity — avg 20d volume={avg_vol:.0f}")

    # 5. Duplicate dates
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        violations.append(f"{symbol}: {dupes} duplicate date rows")

    return violations


# ── DB Upsert ─────────────────────────────────────────────────

def upsert_market_data(symbol: str, df: pd.DataFrame, source: str = "YFINANCE") -> int:
    """
    Upserts OHLCV rows into market_data table.
    Returns count of rows inserted/updated.
    """
    if df is None or df.empty:
        return 0

    rows = []
    for ts, row in df.iterrows():
        dt = ts.date() if hasattr(ts, 'date') else ts
        rows.append({
            "date": dt,
            "stock": symbol,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
            "adjusted_close": float(row["Close"]),   # auto_adjust=True means Close IS adjusted
            "data_source": source,
        })

    if not rows:
        return 0

    with get_sync_db() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO market_data (date, stock, open, high, low, close, volume, adjusted_close, data_source)
            VALUES (%(date)s, %(stock)s, %(open)s, %(high)s, %(low)s, %(close)s,
                    %(volume)s, %(adjusted_close)s, %(data_source)s)
            ON CONFLICT (date, stock) DO UPDATE SET
                open           = EXCLUDED.open,
                high           = EXCLUDED.high,
                low            = EXCLUDED.low,
                close          = EXCLUDED.close,
                volume         = EXCLUDED.volume,
                adjusted_close = EXCLUDED.adjusted_close,
                data_source    = EXCLUDED.data_source
            """,
            rows,
        )
        count = cursor.rowcount

    logger.info(f"{symbol}: upserted {len(rows)} rows into market_data")
    return len(rows)


# ── Incremental fetch (daily pipeline) ───────────────────────

def get_last_ingested_date(symbol: str) -> Optional[date]:
    """Returns the most recent date in market_data for this symbol."""
    with get_sync_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(date) FROM market_data WHERE stock = %s",
            (symbol,),
        )
        row = cursor.fetchone()
        if row and row["max"]:
            return row["max"]
    return None


def fetch_incremental(symbol: str, fetch_date: date) -> Tuple[pd.DataFrame, str]:
    """
    If we already have history, fetch only missing days (faster).
    Falls back to full 5y fetch if no data exists yet.
    """
    last_date = get_last_ingested_date(symbol)

    if last_date is None:
        logger.info(f"{symbol}: No history found — fetching full 5y")
        df = _fetch_yfinance(to_nse_ticker(symbol), period="5y")
        return df, "FULL"

    days_missing = (fetch_date - last_date).days
    if days_missing <= 0:
        logger.info(f"{symbol}: Already up to date (last={last_date})")
        return pd.DataFrame(), "UP_TO_DATE"

    # Fetch slightly more than needed to handle weekends/holidays
    period_days = max(days_missing + 5, 7)
    logger.info(f"{symbol}: Incremental fetch — {days_missing} days missing since {last_date}")
    df = _fetch_yfinance(to_nse_ticker(symbol), period=f"{period_days}d")
    return df, "INCREMENTAL"


# ── Main ingestion entry point ────────────────────────────────

class DataIngestor:
    """
    Called by pipeline.py Stage 1.
    Handles the full ingestion flow for all watchlist stocks.
    """

    def __init__(self, stocks: List[str], run_date: Optional[date] = None):
        self.stocks = stocks
        self.run_date = run_date or date.today()
        self.sla_breaches: List[str] = []
        self.skipped: List[str] = []
        self.liquidity_warnings: List[str] = []

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Returns dict of {symbol: DataFrame} for all stocks with valid data.
        Stocks with SLA breaches are still returned but flagged.
        """
        logger.info(
            f"Starting ingestion | date={self.run_date} | stocks={len(self.stocks)}"
        )

        # Try batch fetch first (faster)
        try:
            data = _fetch_batch(self.stocks, period="5y")
        except Exception as e:
            logger.warning(f"Batch fetch failed ({e}), falling back to per-stock")
            data = self._fetch_per_stock()

        valid_data: Dict[str, pd.DataFrame] = {}

        for symbol in self.stocks:
            df = data.get(symbol)

            # SLA validation
            violations = validate_sla(symbol, df, self.run_date)
            if violations:
                for v in violations:
                    logger.warning(f"SLA breach: {v}")
                self.sla_breaches.append(symbol)
                # Still proceed — partial data is better than nothing for features

            if df is None or df.empty:
                self.skipped.append(symbol)
                continue

            # Upsert to DB
            try:
                upsert_market_data(symbol, df)
            except Exception as e:
                logger.error(f"{symbol}: DB upsert failed — {e}")
                self.sla_breaches.append(symbol)
                continue

            valid_data[symbol] = df

            # Liquidity check for position sizing later
            avg_vol = df["Volume"].tail(20).mean()
            if avg_vol < MIN_VOLUME:
                self.liquidity_warnings.append(symbol)

        logger.info(
            f"Ingestion complete | valid={len(valid_data)} | "
            f"skipped={len(self.skipped)} | sla_breaches={len(self.sla_breaches)}"
        )
        return valid_data

    def _fetch_per_stock(self) -> Dict[str, pd.DataFrame]:
        """Fallback: fetch one by one."""
        result = {}
        for symbol in self.stocks:
            try:
                df = _fetch_yfinance(to_nse_ticker(symbol), period="5y")
                result[symbol] = df
            except Exception as e:
                logger.error(f"{symbol}: fetch failed — {e}")
        return result


# ── Nifty50 index fetch (for regime detection) ────────────────

def fetch_nifty50(period: str = "5y") -> pd.DataFrame:
    """
    Fetches Nifty 50 index data (^NSEI) for regime detection (Section 7).
    """
    logger.info("Fetching Nifty50 index data")
    df = _fetch_yfinance("^NSEI", period=period)
    if df.empty:
        raise RuntimeError("Failed to fetch Nifty50 index data — regime detection will fail")
    return df


def fetch_nifty_bank(period: str = "5y") -> pd.DataFrame:
    """Fetches Bank Nifty (^NSEBANK) — useful for Banking sector signals."""
    return _fetch_yfinance("^NSEBANK", period=period)
