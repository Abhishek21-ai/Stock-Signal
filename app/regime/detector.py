"""
Market Regime Detector — Section 7

Exports (matched to test_regime.py):
  RegimeDetector        — pipeline class, .run() → RegimeResult
  RegimeResult          — dataclass
  classify_regime()     — pure function (unit-testable)
  get_latest_regime()   — DB read
  REGIME_WEIGHTS        — per-regime fusion weights dict
  _apply_fii_adjustment — FII stress logic (unit-testable)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas_ta as ta

from app.data.ingestor import fetch_nifty50
from app.db import get_sync_db
from app.logger import get_logger

logger = get_logger("regime_detector")

REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "BULL": {
        "trend": 0.25, "momentum": 0.30, "reversion": 0.15,
        "breakout": 0.20, "volume": 0.10,
    },
    "BEAR": {
        "trend": 0.0, "momentum": 0.10, "reversion": 0.70,
        "breakout": 0.0, "volume": 0.20,
    },
    "SIDEWAYS": {
        "trend": 0.05, "momentum": 0.20, "reversion": 0.60,
        "breakout": 0.0, "volume": 0.15,
    },
    "UNCERTAIN": {
        "trend": 0.10, "momentum": 0.20, "reversion": 0.50,
        "breakout": 0.10, "volume": 0.10,
    },
}

FII_STRESS_THRESHOLD = -5000.0


@dataclass
class RegimeResult:
    regime:            str
    regime_confidence: str                   # NORMAL | UNCERTAIN
    fusion_weights:    Dict[str, float]
    nifty_close:       float = 0.0
    nifty_ema20:       float = 0.0
    nifty_ema50:       float = 0.0
    nifty_ema200:      float = 0.0
    nifty_adx:         float = 0.0
    fii_net_crore:     Optional[float] = None
    fii_stress:        bool = False
    reasons:           List[str] = field(default_factory=list)


def classify_regime(
    close:   float,
    ema_20:  float,
    ema_50:  float,
    ema_200: float,
    adx:     float,
) -> Tuple[str, List[str]]:
    """
    Pure classification — no IO. Unit-testable.
    Rules:
      BULL:      EMA20>EMA50>EMA200 AND ADX>=20
      BEAR:      EMA20<EMA50<EMA200 AND ADX>=20
      SIDEWAYS:  ADX<20 (trendless)
      UNCERTAIN: mixed EMA alignment
    """
    reasons = []
    bull_ema     = ema_20 > ema_50 > ema_200
    bear_ema     = ema_20 < ema_50 < ema_200
    strong_trend = adx >= 20

    if not strong_trend:
        # Mixed EMA (no clean stack) + weak ADX = SIDEWAYS (oscillating)
        # Clean EMA alignment but weak ADX = UNCERTAIN (trend forming but unconfirmed)
        if not bull_ema and not bear_ema:
            regime = "SIDEWAYS"
            reasons.append(f"ADX={adx:.1f} < 20 and mixed EMA — sideways market")
        else:
            regime = "UNCERTAIN"
            reasons.append(f"ADX={adx:.1f} < 20 — EMA aligned but trend unconfirmed")
        return regime, reasons

    if bull_ema:
        regime = "BULL"
        reasons.append(f"EMA bullish stack: {ema_20:.0f}>{ema_50:.0f}>{ema_200:.0f}")
        reasons.append(f"ADX={adx:.1f} confirms trend")
    elif bear_ema:
        regime = "BEAR"
        reasons.append(f"EMA bearish stack: {ema_20:.0f}<{ema_50:.0f}<{ema_200:.0f}")
        reasons.append(f"ADX={adx:.1f} confirms trend")
    else:
        regime = "UNCERTAIN"
        reasons.append("Mixed EMA alignment despite strong ADX")

    return regime, reasons


def _apply_fii_adjustment(
    regime:         str,
    fii_rolling_5d: float,
    weights:        Dict[str, float],
) -> Tuple[str, bool, Dict[str, float]]:
    """
    Apply FII stress adjustment to fusion weights.
    Only triggers for BULL regime when FII selling > threshold.
    Returns (confidence, stress_applied, adjusted_weights).
    """
    if regime != "BULL" or fii_rolling_5d >= FII_STRESS_THRESHOLD:
        return "NORMAL", False, weights

    adjusted = dict(weights)
    adjusted["trend"]     = max(0.10, adjusted.get("trend",     0.35) - 0.10)
    adjusted["breakout"]  = max(0.05, adjusted.get("breakout",  0.20) - 0.05)
    adjusted["reversion"] =           adjusted.get("reversion", 0.10) + 0.08
    adjusted["volume"]    =           adjusted.get("volume",    0.10) + 0.07

    # Renormalise
    total    = sum(adjusted.values())
    adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

    return "UNCERTAIN", True, adjusted


def _get_fii_rolling_sum(run_date: date) -> Optional[float]:
    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT rolling_5d_fii_sum FROM fii_dii_flows "
                "WHERE date <= %s ORDER BY date DESC LIMIT 1",
                (run_date,),
            )
            row = cur.fetchone()
            return float(row["rolling_5d_fii_sum"]) if row and row["rolling_5d_fii_sum"] else None
    except Exception:
        return None


def _save_regime_snapshot(run_date: date, result: RegimeResult) -> None:
    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO regime_snapshots (
                    date, regime, nifty_close, nifty_ema20, nifty_ema50, nifty_ema200,
                    nifty_adx, fii_net_crore, regime_confidence
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date) DO UPDATE SET
                    regime=EXCLUDED.regime,
                    nifty_close=EXCLUDED.nifty_close,
                    nifty_adx=EXCLUDED.nifty_adx,
                    regime_confidence=EXCLUDED.regime_confidence
                """,
                (
                    run_date, result.regime,
                    result.nifty_close, result.nifty_ema20,
                    result.nifty_ema50, result.nifty_ema200,
                    result.nifty_adx, result.fii_net_crore,
                    result.regime_confidence,
                ),
            )
    except Exception as e:
        logger.error(f"Failed to save regime snapshot: {e}")


def get_latest_regime() -> Optional[RegimeResult]:
    """Read most recent regime snapshot from DB."""
    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM regime_snapshots ORDER BY date DESC LIMIT 1")
            row = cur.fetchone()
        if not row:
            return None
        regime = row["regime"]
        return RegimeResult(
            regime=regime,
            regime_confidence=row.get("regime_confidence") or "NORMAL",
            fusion_weights=dict(REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["UNCERTAIN"])),
            nifty_close=float(row["nifty_close"] or 0),
            nifty_ema20=float(row["nifty_ema20"] or 0),
            nifty_ema50=float(row["nifty_ema50"] or 0),
            nifty_ema200=float(row["nifty_ema200"] or 0),
            nifty_adx=float(row["nifty_adx"] or 0),
            fii_net_crore=float(row["fii_net_crore"]) if row.get("fii_net_crore") else None,
        )
    except Exception as e:
        logger.error(f"get_latest_regime failed: {e}")
        return None


class RegimeDetector:
    def __init__(self, run_date: Optional[date] = None):
        self.run_date = run_date or date.today()

    def run(self) -> RegimeResult:
        try:
            nifty_df = fetch_nifty50(period="2y")
        except Exception as e:
            logger.error(f"Nifty fetch failed: {e}")
            return RegimeResult(
                regime="UNCERTAIN", regime_confidence="UNCERTAIN",
                fusion_weights=dict(REGIME_WEIGHTS["UNCERTAIN"]),
                reasons=[f"Nifty fetch failed: {e}"],
            )

        close   = nifty_df["Close"]
        high    = nifty_df["High"]
        low     = nifty_df["Low"]

        ema_20  = float(ta.ema(close, length=20).iloc[-1])
        ema_50  = float(ta.ema(close, length=50).iloc[-1])
        ema_200 = float(ta.ema(close, length=200).iloc[-1])
        adx_df  = ta.adx(high, low, close, length=14)
        adx     = float(adx_df["ADX_14"].iloc[-1])
        nifty_close = float(close.iloc[-1])

        regime, reasons = classify_regime(nifty_close, ema_20, ema_50, ema_200, adx)

        fii_sum = _get_fii_rolling_sum(self.run_date)
        base_weights = dict(REGIME_WEIGHTS[regime])
        conf, fii_stress, weights = _apply_fii_adjustment(
            regime, fii_sum or 0.0, base_weights
        )
        if fii_stress:
            reasons.append(f"FII 5d rolling ₹{fii_sum:.0f}Cr → stress mode, weights adjusted")

        result = RegimeResult(
            regime=regime,
            regime_confidence=conf,
            fusion_weights=weights,
            nifty_close=nifty_close,
            nifty_ema20=ema_20,
            nifty_ema50=ema_50,
            nifty_ema200=ema_200,
            nifty_adx=adx,
            fii_net_crore=fii_sum,
            fii_stress=fii_stress,
            reasons=reasons,
        )

        logger.info(f"Regime: {regime} ({conf}) | Nifty={nifty_close:.0f} | ADX={adx:.1f}")
        _save_regime_snapshot(self.run_date, result)
        return result
