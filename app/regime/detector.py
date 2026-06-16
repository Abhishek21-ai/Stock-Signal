"""
Regime Detection Engine — Section 7
Classifies the market into one of three regimes daily using Nifty 50 as benchmark.

Regime Classification (Section 7.1):
  BULL      — EMA20 > EMA50 > EMA200 AND ADX > 25
  BEAR      — EMA20 < EMA50 < EMA200 AND ADX > 25
  SIDEWAYS  — ADX < 20 AND price within 1.5% of 20-day mean
  UNCERTAIN — everything else (transitional)

FII/DII Integration (Section 19.4):
  If rolling_5d_fii_sum < -5000 crore AND regime is BULL:
  → downgrade regime_confidence to UNCERTAIN and reduce BUY weights by 15%

Dynamic Fusion Weights by Regime (Section 7.2):
  Returned as part of RegimeResult so the fusion engine can consume directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from app.data.ingestor import fetch_nifty50
from app.db import get_sync_db
from app.logger import get_logger

logger = get_logger("regime")


# ── Regime weight profiles (Section 7.2) ─────────────────────
REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "BULL": {
        "trend":     0.30,
        "momentum":  0.25,
        "reversion": 0.10,
        "breakout":  0.20,
        "volume":    0.10,
        "risk":      0.05,
    },
    "BEAR": {
        "trend":     0.10,
        "momentum":  0.15,
        "reversion": 0.20,
        "breakout":  0.10,
        "volume":    0.10,
        "risk":      0.35,
    },
    "SIDEWAYS": {
        "trend":     0.10,
        "momentum":  0.10,
        "reversion": 0.35,
        "breakout":  0.15,
        "volume":    0.10,
        "risk":      0.20,
    },
    "UNCERTAIN": {
        "trend":     0.20,
        "momentum":  0.18,
        "reversion": 0.18,
        "breakout":  0.16,
        "volume":    0.10,
        "risk":      0.18,
    },
}

# FII stress threshold (Section 19.4)
FII_STRESS_THRESHOLD_CRORE = -5000.0
FII_BUY_WEIGHT_REDUCTION   = 0.15    # reduce BUY weights by 15% under FII stress


@dataclass
class RegimeResult:
    regime: str                          # BULL | BEAR | SIDEWAYS | UNCERTAIN
    regime_confidence: str               # NORMAL | UNCERTAIN (FII-adjusted)
    nifty_close:  float = 0.0
    nifty_ema20:  float = 0.0
    nifty_ema50:  float = 0.0
    nifty_ema200: float = 0.0
    nifty_adx:    float = 0.0
    fii_net_crore: Optional[float] = None
    rolling_5d_fii_sum: Optional[float] = None
    fii_stress: bool = False
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "regime":               self.regime,
            "regime_confidence":    self.regime_confidence,
            "nifty_close":          round(self.nifty_close, 2),
            "nifty_ema20":          round(self.nifty_ema20, 2),
            "nifty_ema50":          round(self.nifty_ema50, 2),
            "nifty_ema200":         round(self.nifty_ema200, 2),
            "nifty_adx":            round(self.nifty_adx, 2),
            "fii_net_crore":        self.fii_net_crore,
            "rolling_5d_fii_sum":   self.rolling_5d_fii_sum,
            "fii_stress":           self.fii_stress,
            "fusion_weights":       self.fusion_weights,
            "reasons":              self.reasons,
        }


# ── Core classification logic ─────────────────────────────────

def classify_regime(
    close: float,
    ema20: float,
    ema50: float,
    ema200: float,
    adx: float,
) -> tuple[str, list[str]]:
    """
    Pure function: classifies regime from indicator values.
    Returns (regime, reasons).
    """
    reasons = []

    # ── Bull: full EMA alignment upward + strong trend ────────
    if ema20 > ema50 > ema200 and adx > 25:
        reasons.append(f"EMA20({ema20:.0f}) > EMA50({ema50:.0f}) > EMA200({ema200:.0f}) — full bull alignment")
        reasons.append(f"ADX={adx:.1f} > 25 — trend is strong")
        return "BULL", reasons

    # ── Bear: full EMA alignment downward + strong trend ──────
    if ema20 < ema50 < ema200 and adx > 25:
        reasons.append(f"EMA20({ema20:.0f}) < EMA50({ema50:.0f}) < EMA200({ema200:.0f}) — full bear alignment")
        reasons.append(f"ADX={adx:.1f} > 25 — trend is strong")
        return "BEAR", reasons

    # ── Sideways: weak trend + price near 20-day mean ─────────
    mean_20 = ema20    # EMA20 ≈ 20-day mean for this check
    price_dev = abs(close - mean_20) / mean_20 if mean_20 > 0 else 1.0
    if adx < 20 and price_dev <= 0.015:
        reasons.append(f"ADX={adx:.1f} < 20 — no clear trend")
        reasons.append(f"Price {price_dev:.2%} from EMA20 — ranging market")
        return "SIDEWAYS", reasons

    # ── Uncertain: transitional / partial alignment ────────────
    if ema20 > ema50 > ema200:
        reasons.append(f"Bull EMA alignment but ADX={adx:.1f} < 25 — weak trend")
    elif ema20 < ema50 < ema200:
        reasons.append(f"Bear EMA alignment but ADX={adx:.1f} < 25 — weak trend")
    else:
        reasons.append(f"Mixed EMA signals — transitional market")
    reasons.append("Regime: UNCERTAIN — using balanced fusion weights")

    return "UNCERTAIN", reasons


def _get_fii_data(run_date: date) -> tuple[Optional[float], Optional[float]]:
    """
    Fetch latest FII net flow and 5-day rolling sum from fii_dii_flows table.
    Returns (fii_net_crore, rolling_5d_fii_sum).
    """
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT fii_net_crore, rolling_5d_fii_sum
                FROM fii_dii_flows
                WHERE date <= %s
                ORDER BY date DESC
                LIMIT 1
                """,
                (run_date,),
            )
            row = cursor.fetchone()
            if row:
                return float(row["fii_net_crore"] or 0), float(row["rolling_5d_fii_sum"] or 0)
    except Exception as e:
        logger.warning(f"Could not fetch FII data: {e}")
    return None, None


def _apply_fii_adjustment(
    regime: str,
    rolling_5d: Optional[float],
    weights: Dict[str, float],
) -> tuple[str, bool, Dict[str, float]]:
    """
    Section 19.4: if FII stress detected during BULL regime,
    downgrade confidence and reduce BUY strategy weights.
    Returns (regime_confidence, fii_stress, adjusted_weights).
    """
    if regime != "BULL" or rolling_5d is None:
        return "NORMAL", False, weights

    if rolling_5d < FII_STRESS_THRESHOLD_CRORE:
        logger.warning(
            f"FII stress detected: rolling_5d_fii_sum={rolling_5d:.0f} crore "
            f"(threshold={FII_STRESS_THRESHOLD_CRORE:.0f}) — downgrading regime confidence"
        )
        # Reduce trend + momentum (the BUY-driving strategies) by 15%
        adjusted = dict(weights)
        reduction = FII_BUY_WEIGHT_REDUCTION
        adjusted["trend"]    = max(0, adjusted["trend"]    * (1 - reduction))
        adjusted["momentum"] = max(0, adjusted["momentum"] * (1 - reduction))
        # Redistribute the removed weight to risk penalty
        redistributed = weights["trend"] * reduction + weights["momentum"] * reduction
        adjusted["risk"] = min(1.0, adjusted["risk"] + redistributed)
        # Renormalise to sum to 1.0
        total = sum(adjusted.values())
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}
        return "UNCERTAIN", True, adjusted

    return "NORMAL", False, weights


def _compute_nifty_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Compute EMA20/50/200 and ADX14 on Nifty50 OHLCV dataframe."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    ema20  = ta.ema(close, length=20)
    ema50  = ta.ema(close, length=50)
    ema200 = ta.ema(close, length=200)
    adx_df = ta.adx(high, low, close, length=14)

    adx_col = [c for c in adx_df.columns if c.startswith("ADX")][0]

    def last(series) -> float:
        val = series.iloc[-1]
        return float(val) if not pd.isna(val) else 0.0

    return {
        "close":  float(close.iloc[-1]),
        "ema20":  last(ema20),
        "ema50":  last(ema50),
        "ema200": last(ema200),
        "adx":    last(adx_df[adx_col]),
    }


def _save_regime_snapshot(result: RegimeResult, run_date: date) -> None:
    """Upsert regime snapshot into regime_snapshots table."""
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO regime_snapshots (
                    date, regime, nifty_close,
                    nifty_ema20, nifty_ema50, nifty_ema200, nifty_adx,
                    fii_net_crore, dii_net_crore, rolling_5d_fii_sum,
                    regime_confidence
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (date) DO UPDATE SET
                    regime             = EXCLUDED.regime,
                    nifty_close        = EXCLUDED.nifty_close,
                    nifty_ema20        = EXCLUDED.nifty_ema20,
                    nifty_ema50        = EXCLUDED.nifty_ema50,
                    nifty_ema200       = EXCLUDED.nifty_ema200,
                    nifty_adx          = EXCLUDED.nifty_adx,
                    fii_net_crore      = EXCLUDED.fii_net_crore,
                    rolling_5d_fii_sum = EXCLUDED.rolling_5d_fii_sum,
                    regime_confidence  = EXCLUDED.regime_confidence
                """,
                (
                    run_date,
                    result.regime,
                    result.nifty_close,
                    result.nifty_ema20,
                    result.nifty_ema50,
                    result.nifty_ema200,
                    result.nifty_adx,
                    result.fii_net_crore,
                    None,                      # dii_net_crore — scraped separately
                    result.rolling_5d_fii_sum,
                    result.regime_confidence,
                ),
            )
        logger.info(f"Regime snapshot saved: {result.regime} ({result.regime_confidence})")
    except Exception as e:
        logger.error(f"Failed to save regime snapshot: {e}")


# ── Main entry point ──────────────────────────────────────────

class RegimeDetector:
    """
    Called by pipeline.py Stage 3.
    Fetches Nifty50 data, computes indicators, classifies regime,
    applies FII adjustment, saves to DB, returns RegimeResult.
    """

    def __init__(self, run_date: Optional[date] = None):
        self.run_date = run_date or date.today()

    def run(self) -> RegimeResult:
        logger.info(f"Detecting market regime for {self.run_date}")

        # ── 1. Fetch Nifty50 data ─────────────────────────────
        try:
            nifty_df = fetch_nifty50(period="2y")
        except Exception as e:
            logger.error(f"Failed to fetch Nifty50: {e} — defaulting to UNCERTAIN")
            return self._uncertain_fallback(reason=f"Nifty50 fetch failed: {e}")

        if len(nifty_df) < 210:
            logger.warning(f"Nifty50 only {len(nifty_df)} rows — insufficient for EMA200")
            return self._uncertain_fallback(reason="Insufficient Nifty50 history for EMA200")

        # ── 2. Compute indicators ─────────────────────────────
        try:
            indic = _compute_nifty_indicators(nifty_df)
        except Exception as e:
            logger.error(f"Indicator computation failed: {e}")
            return self._uncertain_fallback(reason=f"Indicator error: {e}")

        # ── 3. Classify regime ────────────────────────────────
        regime, reasons = classify_regime(
            close=indic["close"],
            ema20=indic["ema20"],
            ema50=indic["ema50"],
            ema200=indic["ema200"],
            adx=indic["adx"],
        )

        # ── 4. Get fusion weights for this regime ─────────────
        weights = dict(REGIME_WEIGHTS[regime])

        # ── 5. FII/DII adjustment (Section 19.4) ─────────────
        fii_net, fii_rolling = _get_fii_data(self.run_date)
        regime_confidence, fii_stress, weights = _apply_fii_adjustment(
            regime, fii_rolling, weights
        )

        if fii_stress:
            reasons.append(
                f"FII stress: 5d net flow = ₹{fii_rolling:.0f} Cr — "
                "BUY weights reduced 15%, regime confidence → UNCERTAIN"
            )

        result = RegimeResult(
            regime=regime,
            regime_confidence=regime_confidence,
            nifty_close=indic["close"],
            nifty_ema20=indic["ema20"],
            nifty_ema50=indic["ema50"],
            nifty_ema200=indic["ema200"],
            nifty_adx=indic["adx"],
            fii_net_crore=fii_net,
            rolling_5d_fii_sum=fii_rolling,
            fii_stress=fii_stress,
            fusion_weights=weights,
            reasons=reasons,
        )

        logger.info(
            f"Regime: {regime} | Confidence: {regime_confidence} | "
            f"Nifty={indic['close']:.0f} | EMA20={indic['ema20']:.0f} | "
            f"EMA50={indic['ema50']:.0f} | EMA200={indic['ema200']:.0f} | "
            f"ADX={indic['adx']:.1f} | FII stress={fii_stress}"
        )

        # ── 6. Persist to DB ──────────────────────────────────
        _save_regime_snapshot(result, self.run_date)

        return result

    def _uncertain_fallback(self, reason: str = "") -> RegimeResult:
        """Returns UNCERTAIN regime with balanced weights when detection fails."""
        return RegimeResult(
            regime="UNCERTAIN",
            regime_confidence="UNCERTAIN",
            fusion_weights=dict(REGIME_WEIGHTS["UNCERTAIN"]),
            reasons=[reason] if reason else ["Fallback to UNCERTAIN regime"],
        )


def get_latest_regime(run_date: Optional[date] = None) -> Optional[RegimeResult]:
    """
    Fetch the most recent regime snapshot from DB.
    Used by pipeline stages that need regime without re-running detection.
    """
    check_date = run_date or date.today()
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT regime, regime_confidence,
                       nifty_close, nifty_ema20, nifty_ema50, nifty_ema200, nifty_adx,
                       fii_net_crore, rolling_5d_fii_sum
                FROM regime_snapshots
                WHERE date <= %s
                ORDER BY date DESC
                LIMIT 1
                """,
                (check_date,),
            )
            row = cursor.fetchone()
    except Exception as e:
        logger.error(f"Could not fetch latest regime: {e}")
        return None

    if not row:
        return None

    regime = row["regime"]
    weights = dict(REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["UNCERTAIN"]))

    return RegimeResult(
        regime=regime,
        regime_confidence=row["regime_confidence"] or "NORMAL",
        nifty_close=float(row["nifty_close"] or 0),
        nifty_ema20=float(row["nifty_ema20"] or 0),
        nifty_ema50=float(row["nifty_ema50"] or 0),
        nifty_ema200=float(row["nifty_ema200"] or 0),
        nifty_adx=float(row["nifty_adx"] or 0),
        fii_net_crore=float(row["fii_net_crore"]) if row["fii_net_crore"] else None,
        rolling_5d_fii_sum=float(row["rolling_5d_fii_sum"]) if row["rolling_5d_fii_sum"] else None,
        fusion_weights=weights,
    )
