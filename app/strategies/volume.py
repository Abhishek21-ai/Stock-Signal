"""
Strategy 5: Volume Profile — Section 8.5
Logic: OBV trend + volume ratio + VWAP position + accumulation/distribution candles

Volume is a leading indicator — smart money leaves footprints.
Works across all regimes.

Regime weights:
  BULL:      weight=0.8
  BEAR:      weight=0.8
  SIDEWAYS:  weight=1.0
  UNCERTAIN: weight=0.8
"""
from __future__ import annotations

from typing import Dict

from app.strategies.base import BaseStrategy, StrategyResult, score_to_signal

REGIME_WEIGHTS = {"BULL": 0.8, "BEAR": 0.8, "SIDEWAYS": 1.0, "UNCERTAIN": 0.8}
REGIME_GATES = {
    "BULL": 50,
    "UNCERTAIN": 40,
    "SIDEWAYS": 0,
    "BEAR": 50,
}


class VolumeProfileStrategy(BaseStrategy):
    strategy_id = "volume"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        score = 0.0
        reasons = []
        close = features.get("close", 0)

        # ── 1. OBV trend ───────────────────────────────────────
        obv_trend = features.get("obv_trend", 0)
        if obv_trend == 1:
            score += 25
            reasons.append("OBV trending up — accumulation phase")
        elif obv_trend == -1:
            score -= 25
            reasons.append("OBV trending down — distribution phase")

        # ── 2. Volume & Price Action (Fix for the falling knife) ──
        vol_ratio = features.get("volume_ratio", 1.0)
        high = features.get("high", close)
        low = features.get("low", close)

        # Calculate candle shape: 1.0 means closed at absolute high, 0.0 means absolute low
        candle_range = high - low
        close_strength = (close - low) / candle_range if candle_range > 0 else 0.5

        if vol_ratio > 1.5:
            if close_strength > 0.7:
                score += 30
                reasons.append(f"High volume ({vol_ratio:.1f}x) and strong close — accumulation")
            elif close_strength < 0.3:
                score -= 40
                reasons.append(f"High volume ({vol_ratio:.1f}x) but weak close — heavy distribution")
            else:
                score -= 10
                reasons.append(f"High volume ({vol_ratio:.1f}x) but indecisive close — warning")
        elif vol_ratio < 0.7 and score > 0:
            score *= 0.5
            reasons.append(f"Low volume ({vol_ratio:.1f}x) — reducing conviction")

        # ── 3. VWAP position ──────────────────────────────────
        price_vs_vwap = features.get("price_vs_vwap", 0) or 0
        if price_vs_vwap > 0.015:
            score += 15
            reasons.append(f"Price comfortably above VWAP (+{price_vs_vwap:.1%})")
        elif price_vs_vwap < -0.015:
            score -= 15
            reasons.append(f"Price rejected below VWAP ({price_vs_vwap:.1%})")

        # ── 4. Overbought Exhaustion Filter ───────────────────
        rsi = features.get("rsi_14", 50)
        if rsi > 70 and score > 0:
            score *= 0.1
            reasons.append(f"Volume spike but RSI is extremely overbought ({rsi:.1f}) — avoiding blow-off top")
        elif rsi < 35 and vol_ratio > 1.5 and close_strength > 0.7:
            score += 30
            reasons.append(f"Strong capitulation/reversal at oversold RSI ({rsi:.1f})")

        # ── 5. Regime weight ──────────────────────────────────
        weight = REGIME_WEIGHTS.get(regime, 1.0)
        score  = max(-100.0, min(100.0, score * weight))

        atr    = features.get("atr_14", close * 0.02)
        entry  = close
        stop   = features.get("atr_stop_1x", close - atr)
        target = features.get("atr_target_2x", close + 2 * atr)

        gate = REGIME_GATES.get(regime, 0)

        if score > 0 and score < gate:
            return StrategyResult(
                strategy_id=self.strategy_id,
                score=0.0,
                signal="HOLD",
                confidence=0.0,
                reasons=reasons + [
                    f"Regime gate: bullish conviction {score:.1f} < {gate}"
                ],
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                target_price=round(target, 2),
            )

        return StrategyResult(
            strategy_id=self.strategy_id,
            score=round(score, 2),
            signal=score_to_signal(score),
            confidence=min(100.0, abs(score)),
            reasons=reasons,
            entry_price=round(entry, 2),
            stop_loss=round(stop, 2),
            target_price=round(target, 2),
        )