"""
Strategy 5: Volume Profile — Section 8.5
Logic: OBV trend + volume ratio + VWAP position + volume-price divergence

Volume is a leading indicator — smart money leaves footprints.
Works across all regimes.

Regime weights:
  BULL:      weight=1.1
  BEAR:      weight=1.1  (volume confirms both directions)
  SIDEWAYS:  weight=1.2
  UNCERTAIN: weight=1.0
"""
from __future__ import annotations

from typing import Dict

from app.strategies.base import BaseStrategy, StrategyResult, score_to_signal

REGIME_WEIGHTS = {"BULL": 1.1, "BEAR": 1.1, "SIDEWAYS": 1.2, "UNCERTAIN": 1.0}


class VolumeProfileStrategy(BaseStrategy):
    strategy_id = "volume"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        score = 0.0
        reasons = []
        close = features.get("close", 0)

        # ── 1. OBV trend (max ±35 pts) ───────────────────────
        obv_trend = features.get("obv_trend", 0)
        ret_20d   = features.get("return_20d", 0) or 0
        ret_60d   = features.get("return_60d", 0) or 0

        if obv_trend == 1:
            score += 25
            reasons.append("OBV above EMA — accumulation phase")
            # Confirm with price momentum
            if ret_20d > 0:
                score += 10
                reasons.append("OBV + price both rising — strong confirmation")
            else:
                reasons.append("OBV rising but price lagging — potential coiling")
        else:
            score -= 20
            reasons.append("OBV below EMA — distribution phase")
            if ret_20d < 0:
                score -= 10
                reasons.append("OBV + price both falling — confirmed distribution")

        # ── 2. Volume ratio (max ±25 pts) ─────────────────────
        vol_ratio = features.get("volume_ratio", 1.0) or 1.0

        if vol_ratio >= 2.5:
            # Very high volume — check direction
            if ret_20d > 0:
                score += 25
                reasons.append(f"Climactic volume buy ({vol_ratio:.1f}x) with rising price")
            else:
                score -= 20
                reasons.append(f"Climactic volume sell ({vol_ratio:.1f}x) with falling price")
        elif vol_ratio >= 1.5:
            if ret_20d > 0:
                score += 15
                reasons.append(f"Above avg volume ({vol_ratio:.1f}x) confirms upward move")
            else:
                score -= 10
                reasons.append(f"Above avg volume ({vol_ratio:.1f}x) confirms downward move")
        elif vol_ratio < 0.5:
            score *= 0.6
            reasons.append(f"Very low volume ({vol_ratio:.1f}x) — move not institutional")

        # ── 3. VWAP position (max ±20 pts) ────────────────────
        price_vs_vwap = features.get("price_vs_vwap", 0) or 0

        if price_vs_vwap > 0.03:
            score += 20
            reasons.append(f"Price {price_vs_vwap:.1%} above VWAP — buyers in control")
        elif price_vs_vwap > 0.01:
            score += 8
            reasons.append(f"Price marginally above VWAP ({price_vs_vwap:.1%})")
        elif price_vs_vwap < -0.03:
            score -= 20
            reasons.append(f"Price {price_vs_vwap:.1%} below VWAP — sellers in control")
        elif price_vs_vwap < -0.01:
            score -= 8
            reasons.append(f"Price marginally below VWAP ({price_vs_vwap:.1%})")

        # ── 4. Volume-price divergence (max ±15 pts) ──────────
        # Rising price + falling volume = weak move (divergence bearish)
        # Falling price + rising volume = capitulation (contrarian bullish)
        if ret_60d and ret_60d > 0.10 and vol_ratio < 0.8:
            score -= 15
            reasons.append("Bullish divergence warning: price up 60d but volume drying")
        elif ret_60d and ret_60d < -0.10 and vol_ratio > 1.5:
            score += 15
            reasons.append("Capitulation signal: price down but volume spike — potential bottom")

        # ── 5. Regime weight ──────────────────────────────────
        weight = REGIME_WEIGHTS.get(regime, 1.0)
        score  = max(-100, min(100, score * weight))

        atr    = features.get("atr_14", close * 0.02)
        entry  = close
        stop   = features.get("atr_stop_1x", close - atr)
        target = features.get("atr_target_2x", close + 2 * atr)

        return StrategyResult(
            strategy_id=self.strategy_id,
            score=score,
            signal=score_to_signal(score),
            confidence=min(100, abs(score)),
            reasons=reasons,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            meta={
                "regime_weight": weight,
                "obv_trend":     obv_trend,
                "vol_ratio":     vol_ratio,
                "price_vs_vwap": price_vs_vwap,
            },
        )
