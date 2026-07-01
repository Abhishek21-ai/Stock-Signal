"""
Feature Engineering Layer — Section 6
Computes all 13 technical indicators from OHLCV data using pandas_ta.

Indicators:
  Trend:     EMA(20), EMA(50), EMA(200)
  Momentum:  RSI(14), MACD(12,26,9), Stochastic(14,3)
  Volatility: ATR(14), Bollinger Bands(20,2)
  Trend Str: ADX(14)
  Volume:    OBV, Volume SMA(20), VWAP

All features stored in Redis cache (TTL 1h) keyed by symbol+date.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from app.data.validator import get_latest_ohlcv
from app.logger import get_logger

logger = get_logger("features")

# Minimum rows needed for all indicators to be valid
MIN_ROWS_REQUIRED = 210   # EMA200 needs 200+ rows


def build_dataframe(rows: List[dict]) -> pd.DataFrame:
    """Convert DB rows → DataFrame ready for pandas_ta."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all 13 indicators on the DataFrame.
    Returns the same df with indicator columns appended.
    """
    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Insufficient data: {len(df)} rows, need {MIN_ROWS_REQUIRED}"
        )

    # ── Trend: EMAs ──────────────────────────────────────────
    df["ema_20"]  = ta.ema(df["Close"], length=20)
    df["ema_50"]  = ta.ema(df["Close"], length=50)
    df["ema_200"] = ta.ema(df["Close"], length=200)

    # EMA alignment score: +1 per bullish alignment layer
    # close > ema20 > ema50 > ema200 = fully aligned bull
    df["ema_bull_alignment"] = (
        (df["Close"]   > df["ema_20"]).astype(int) +
        (df["ema_20"]  > df["ema_50"]).astype(int) +
        (df["ema_50"]  > df["ema_200"]).astype(int)
    )

    # ── Momentum: RSI ────────────────────────────────────────
    df["rsi_14"] = ta.rsi(df["Close"], length=14)

    # RSI zone classification
    df["rsi_zone"] = pd.cut(
        df["rsi_14"],
        bins=[0, 30, 45, 55, 70, 100],
        labels=["OVERSOLD", "WEAK", "NEUTRAL", "STRONG", "OVERBOUGHT"],
    )

    # ── Momentum: MACD ───────────────────────────────────────
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd"]        = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"]   = macd["MACDh_12_26_9"]

    # MACD crossover signal: +1 bullish cross, -1 bearish cross
    df["macd_cross"] = 0
    df.loc[(df["macd"] > df["macd_signal"]) &
           (df["macd"].shift(1) <= df["macd_signal"].shift(1)), "macd_cross"] = 1
    df.loc[(df["macd"] < df["macd_signal"]) &
           (df["macd"].shift(1) >= df["macd_signal"].shift(1)), "macd_cross"] = -1

    # ── Momentum: Stochastic ─────────────────────────────────
    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]

    # ── Volatility: ATR ──────────────────────────────────────
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # ATR % of close (normalised volatility)
    df["atr_pct"] = df["atr_14"] / df["Close"] * 100

    # ── Volatility: Bollinger Bands ──────────────────────────
    # Column names vary by pandas_ta version — detect dynamically
    bb = ta.bbands(df["Close"], length=20, std=2)
    bb_upper_col = [c for c in bb.columns if c.startswith("BBU")][0]
    bb_mid_col   = [c for c in bb.columns if c.startswith("BBM")][0]
    bb_lower_col = [c for c in bb.columns if c.startswith("BBL")][0]
    df["bb_upper"] = bb[bb_upper_col]
    df["bb_mid"]   = bb[bb_mid_col]
    df["bb_lower"] = bb[bb_lower_col]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # %B: where price sits within the band (0=lower, 1=upper)
    df["bb_pct_b"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── Trend Strength: ADX ──────────────────────────────────
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["adx_14"]  = adx["ADX_14"]
    df["adx_dmp"] = adx["DMP_14"]   # +DI
    df["adx_dmn"] = adx["DMN_14"]   # -DI

    # ADX trend classification
    df["adx_trend"] = pd.cut(
        df["adx_14"],
        bins=[0, 20, 25, 40, 100],
        labels=["WEAK", "DEVELOPING", "STRONG", "VERY_STRONG"],
    )

    # ── Volume: OBV ──────────────────────────────────────────
    df["obv"] = ta.obv(df["Close"], df["Volume"])

    # OBV trend (rising = accumulation)
    df["obv_ema"] = ta.ema(df["obv"], length=20)
    df["obv_trend"] = (df["obv"] > df["obv_ema"]).astype(int)

    # ── Volume: Volume SMA ───────────────────────────────────
    df["volume_sma_20"] = ta.sma(df["Volume"], length=20)
    df["volume_ratio"]  = df["Volume"] / df["volume_sma_20"]   # >1.5 = high volume

    # ── Volume: VWAP ─────────────────────────────────────────
    # Daily VWAP — meaningful for intraday; we use rolling 20d VWAP for swing
    df["vwap_20"] = (
        (df["Close"] * df["Volume"]).rolling(20).sum() /
        df["Volume"].rolling(20).sum()
    )
    df["price_vs_vwap"] = df["Close"] / df["vwap_20"] - 1   # % above/below VWAP

    # ── Composite: Price momentum ─────────────────────────────
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)
    df["return_60d"] = df["Close"].pct_change(60)

    # 52-week high/low proximity
    df["high_52w"] = df["Close"].rolling(252).max()
    df["low_52w"]  = df["Close"].rolling(252).min()
    df["pct_from_52w_high"] = (df["Close"] - df["high_52w"]) / df["high_52w"] * 100
    df["pct_from_52w_low"]  = (df["Close"] - df["low_52w"])  / df["low_52w"]  * 100

    return df


def extract_latest_features(df: pd.DataFrame, symbol: str) -> Dict:
    """
    Extracts the latest row of computed features as a flat dict.
    This is what gets cached in Redis and consumed by strategy engines.
    """
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else latest

    return {
        "symbol": symbol,
        "date":   str(df.index[-1].date()),
        "close":  float(latest["Close"]),
        "volume": float(latest["Volume"]),

        # ── EMAs ──────────────────────────────────────────────
        "ema_20":            _safe(latest, "ema_20"),
        "ema_50":            _safe(latest, "ema_50"),
        "ema_200":           _safe(latest, "ema_200"),
        "ema_bull_alignment": int(_safe(latest, "ema_bull_alignment", 0)),

        # ── RSI ───────────────────────────────────────────────
        "rsi_14":   _safe(latest, "rsi_14"),
        "rsi_zone": str(latest.get("rsi_zone", "NEUTRAL")),

        # ── MACD ─────────────────────────────────────────────
        "macd":        _safe(latest, "macd"),
        "macd_signal": _safe(latest, "macd_signal"),
        "macd_hist":   _safe(latest, "macd_hist"),
        "macd_cross":  int(_safe(latest, "macd_cross", 0)),

        # ── Stochastic ───────────────────────────────────────
        "stoch_k": _safe(latest, "stoch_k"),
        "stoch_d": _safe(latest, "stoch_d"),

        # ── ATR ──────────────────────────────────────────────
        "atr_14":  _safe(latest, "atr_14"),
        "atr_pct": _safe(latest, "atr_pct"),

        # ── Bollinger Bands ───────────────────────────────────
        "bb_upper":  _safe(latest, "bb_upper"),
        "bb_lower":  _safe(latest, "bb_lower"),
        "bb_width":  _safe(latest, "bb_width"),
        "bb_pct_b":  _safe(latest, "bb_pct_b"),

        # ── ADX ──────────────────────────────────────────────
        "adx_14":   _safe(latest, "adx_14"),
        "adx_dmp":  _safe(latest, "adx_dmp"),
        "adx_dmn":  _safe(latest, "adx_dmn"),
        "adx_trend": str(latest.get("adx_trend", "WEAK")),

        # ── Volume ───────────────────────────────────────────
        "obv":           _safe(latest, "obv"),
        "obv_trend":     int(_safe(latest, "obv_trend", 0)),
        "volume_sma_20": _safe(latest, "volume_sma_20"),
        "volume_ratio":  _safe(latest, "volume_ratio"),
        "vwap_20":       _safe(latest, "vwap_20"),
        "price_vs_vwap": _safe(latest, "price_vs_vwap"),

        # ── Returns ──────────────────────────────────────────
        "return_1d":  _safe(latest, "return_1d"),
        "return_5d":  _safe(latest, "return_5d"),
        "return_20d": _safe(latest, "return_20d"),
        "return_60d": _safe(latest, "return_60d"),

        # ── 52-week ──────────────────────────────────────────
        "pct_from_52w_high": _safe(latest, "pct_from_52w_high"),
        "pct_from_52w_low":  _safe(latest, "pct_from_52w_low"),

        # ── ATR-based price levels (for signal generation) ───
        # BUY direction (long): stop below entry, target above entry
        "atr_stop_1x":  round(float(latest["Close"]) - float(_safe(latest, "atr_14", 0)), 2),
        "atr_stop_15x": round(float(latest["Close"]) - 1.5 * float(_safe(latest, "atr_14", 0)), 2),
        "atr_stop_2x":  round(float(latest["Close"]) - 2.0 * float(_safe(latest, "atr_14", 0)), 2),
        "atr_target_2x": round(float(latest["Close"]) + 2.0 * float(_safe(latest, "atr_14", 0)), 2),
        "atr_target_3x": round(float(latest["Close"]) + 3.0 * float(_safe(latest, "atr_14", 0)), 2),

        # SELL direction (short): stop above entry, target below entry
        "atr_stop_1x_sell":   round(float(latest["Close"]) + float(_safe(latest, "atr_14", 0)), 2),
        "atr_stop_15x_sell":  round(float(latest["Close"]) + 1.5 * float(_safe(latest, "atr_14", 0)), 2),
        "atr_stop_2x_sell":   round(float(latest["Close"]) + 2.0 * float(_safe(latest, "atr_14", 0)), 2),
        "atr_target_2x_sell": round(float(latest["Close"]) - 2.0 * float(_safe(latest, "atr_14", 0)), 2),
        "atr_target_3x_sell": round(float(latest["Close"]) - 3.0 * float(_safe(latest, "atr_14", 0)), 2),
    }


def _safe(row, col: str, default: float = float("nan")) -> float:
    """Safe column getter — returns default on missing/NaN."""
    val = row.get(col, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return default


# ── Main entry point ─────────────────────────────────────────

class FeatureEngineer:
    """
    Called by pipeline.py Stage 2.
    Computes features for all stocks and caches results.
    """

    def __init__(self, lookback_rows: int = 300):
        self.lookback_rows = lookback_rows
        self._cache: Dict[str, Dict] = {}   # in-memory for pipeline session

    def run(self, stocks: List[str], run_date: date) -> Dict[str, Dict]:
        """
        Returns dict of {symbol: feature_dict} for all stocks.
        Skips stocks with insufficient history.
        """
        logger.info(f"Engineering features for {len(stocks)} stocks")
        results = {}

        for symbol in stocks:
            try:
                features = self._compute_for_stock(symbol)
                if features:
                    results[symbol] = features
                    self._cache[symbol] = features
            except Exception as e:
                logger.error(f"{symbol}: feature engineering failed — {e}")

        logger.info(f"Features computed for {len(results)}/{len(stocks)} stocks")
        return results

    def _compute_for_stock(self, symbol: str) -> Optional[Dict]:
        rows = get_latest_ohlcv(symbol, n=self.lookback_rows)
        if len(rows) < MIN_ROWS_REQUIRED:
            logger.warning(
                f"{symbol}: only {len(rows)} rows in DB, need {MIN_ROWS_REQUIRED} — skipping"
            )
            return None

        df = build_dataframe(rows)
        df = compute_features(df)
        features = extract_latest_features(df, symbol)

        logger.info(
            f"{symbol}: features OK | close={features['close']} | "
            f"rsi={features['rsi_14']:.1f} | adx={features['adx_14']:.1f} | "
            f"ema_align={features['ema_bull_alignment']}"
        )
        return features

    def get(self, symbol: str) -> Optional[Dict]:
        return self._cache.get(symbol)