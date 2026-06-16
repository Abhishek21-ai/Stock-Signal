"""
Strategy 1: Trend Following — Section 8.1
Logic: EMA alignment + ADX trend strength + MACD confirmation

Regime weights (Section 7.3):
  BULL:     weight=1.3  (trend strategies get boosted)
  SIDEWAYS: weight=0.6
  BEAR:     weight=0.5
  UNCERTAIN: weight=0.8
"""
from __future__ import annotations

from typing import Dict

from app.strategies.base import BaseStrategy, StrategyResult, score_to_signal

REGIME_WEIGHTS = {"BULL": 1.3, "BEAR": 0.5, "SIDEWAYS": 0.6, "UNCERTAIN": 0.8}


class TrendFollowingStrategy(BaseStrategy):
    strategy_id = "trend"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        score = 0.0
        reasons = []
        close = features.get("close", 0)

        # ── 1. EMA Alignment (max ±45 pts) ───────────────────
        alignment = features.get("ema_bull_alignment", 0)
        ema_20  = features.get("ema_20",  close)
        ema_50  = features.get("ema_50",  close)
        ema_200 = features.get("ema_200", close)

        if alignment == 3:
            score += 45
            reasons.append("Full bull EMA alignment (close>EMA20>EMA50>EMA200)")
        elif alignment == 2:
            score += 25
            reasons.append("Partial bull EMA alignment (2/3 layers)")
        elif alignment == 1:
            score += 5
            reasons.append("Weak EMA alignment (1/3 layers)")
        else:
            # Check bear alignment
            bear_align = (
                (close   < ema_20)  +
                (ema_20  < ema_50)  +
                (ema_50  < ema_200)
            )
            if bear_align == 3:
                score -= 45
                reasons.append("Full bear EMA alignment")
            elif bear_align >= 2:
                score -= 25
                reasons.append("Partial bear EMA alignment")
            else:
                reasons.append("No clear EMA trend")

        # ── 2. ADX Trend Strength (max ±30 pts) ───────────────
        adx    = features.get("adx_14", 0)
        adx_dp = features.get("adx_dmp", 0)
        adx_dn = features.get("adx_dmn", 0)

        if adx >= 25:
            di_direction = adx_dp - adx_dn
            if di_direction > 5:
                adx_score = min(30, (adx - 25) * 1.5 + 15)
                score += adx_score
                reasons.append(f"Strong bullish trend ADX={adx:.1f} +DI>{adx_dp:.1f}")
            elif di_direction < -5:
                adx_score = min(30, (adx - 25) * 1.5 + 15)
                score -= adx_score
                reasons.append(f"Strong bearish trend ADX={adx:.1f} -DI>{adx_dn:.1f}")
            else:
                reasons.append(f"Strong trend ADX={adx:.1f} but direction unclear")
        elif adx >= 20:
            reasons.append(f"Developing trend ADX={adx:.1f}")
            score += 5 if adx_dp > adx_dn else -5
        else:
            reasons.append(f"Weak/no trend ADX={adx:.1f} — trend strategy less reliable")
            score *= 0.5   # halve score in trendless market

        # ── 3. MACD Confirmation (max ±25 pts) ───────────────
        macd_hist  = features.get("macd_hist",  0)
        macd_cross = features.get("macd_cross", 0)

        if macd_cross == 1:
            score += 20
            reasons.append("MACD bullish crossover")
        elif macd_cross == -1:
            score -= 20
            reasons.append("MACD bearish crossover")
        elif macd_hist > 0:
            score += min(10, macd_hist * 0.5)
            reasons.append(f"MACD histogram positive ({macd_hist:.2f})")
        elif macd_hist < 0:
            score += max(-10, macd_hist * 0.5)
            reasons.append(f"MACD histogram negative ({macd_hist:.2f})")

        # ── 4. Regime weight ──────────────────────────────────
        weight = REGIME_WEIGHTS.get(regime, 0.8)
        score  = max(-100, min(100, score * weight))

        # ── Entry / Stop / Target ─────────────────────────────
        atr = features.get("atr_14", close * 0.02)
        entry  = close
        stop   = features.get("atr_stop_15x", close - 1.5 * atr)
        target = features.get("atr_target_3x", close + 3.0 * atr)

        return StrategyResult(
            strategy_id=self.strategy_id,
            score=score,
            signal=score_to_signal(score),
            confidence=min(100, abs(score)),
            reasons=reasons,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            meta={"regime_weight": weight, "adx": adx, "ema_alignment": alignment},
        )
