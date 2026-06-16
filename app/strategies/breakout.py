"""
Strategy 4: Breakout — Section 8.4
Logic: 52-week high proximity + BB squeeze + volume confirmation

Regime weights:
  BULL:      weight=1.3
  UNCERTAIN: weight=1.0
  SIDEWAYS:  weight=1.1  (breakout from consolidation)
  BEAR:      weight=0.5
"""
from __future__ import annotations

from typing import Dict

from app.strategies.base import BaseStrategy, StrategyResult, score_to_signal

REGIME_WEIGHTS = {"BULL": 1.3, "BEAR": 0.5, "SIDEWAYS": 1.1, "UNCERTAIN": 1.0}


class BreakoutStrategy(BaseStrategy):
    strategy_id = "breakout"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        score = 0.0
        reasons = []
        close = features.get("close", 0)

        # ── 1. 52-week high proximity (max ±40 pts) ───────────
        pct_from_high = features.get("pct_from_52w_high", -50) or -50
        pct_from_low  = features.get("pct_from_52w_low",   50) or 50

        if pct_from_high >= -2:           # within 2% of 52w high
            score += 40
            reasons.append(f"Price at/near 52w high ({pct_from_high:.1f}%) — breakout zone")
        elif pct_from_high >= -5:
            score += 25
            reasons.append(f"Price approaching 52w high ({pct_from_high:.1f}%)")
        elif pct_from_high >= -10:
            score += 10
            reasons.append(f"Price within 10% of 52w high")
        elif pct_from_low <= 5:           # near 52w low = breakdown risk
            score -= 30
            reasons.append(f"Price near 52w low ({pct_from_low:.1f}% above) — breakdown risk")
        else:
            reasons.append(f"Price {pct_from_high:.1f}% from 52w high — no breakout setup")

        # ── 2. Bollinger Band squeeze (max +25 pts) ───────────
        # Squeeze = low BB width = energy building for breakout
        bb_width = features.get("bb_width", 0.1) or 0.1
        bb_pct_b = features.get("bb_pct_b", 0.5) or 0.5

        if bb_width < 0.04:
            score += 25
            reasons.append(f"BB squeeze (width={bb_width:.3f}) — coiling for breakout")
        elif bb_width < 0.07:
            score += 12
            reasons.append(f"Moderate BB squeeze (width={bb_width:.3f})")

        # Breakout direction from band
        if bb_pct_b > 0.90:
            score += 15
            reasons.append("Price breaking above upper BB")
        elif bb_pct_b < 0.10:
            score -= 15
            reasons.append("Price breaking below lower BB — breakdown")

        # ── 3. Volume confirmation (max +20 pts) ──────────────
        vol_ratio = features.get("volume_ratio", 1.0) or 1.0

        if vol_ratio >= 2.0:
            score += 20
            reasons.append(f"High volume breakout confirmation ({vol_ratio:.1f}x avg)")
        elif vol_ratio >= 1.5:
            score += 12
            reasons.append(f"Above-average volume ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.7 and score > 0:
            score *= 0.7
            reasons.append(f"Low volume ({vol_ratio:.1f}x) — breakout unconfirmed, reducing confidence")

        # ── 4. OBV trend (max +15 pts) ────────────────────────
        obv_trend = features.get("obv_trend", 0)
        if obv_trend == 1 and score > 0:
            score += 15
            reasons.append("OBV trending up — institutional accumulation")
        elif obv_trend == 0 and score > 20:
            score -= 10
            reasons.append("OBV not confirming breakout")

        # ── 5. Regime weight ──────────────────────────────────
        weight = REGIME_WEIGHTS.get(regime, 1.0)
        score  = max(-100, min(100, score * weight))

        atr    = features.get("atr_14", close * 0.02)
        entry  = close
        # Breakout stop: just below the breakout level (2x ATR buffer)
        stop   = features.get("atr_stop_2x", close - 2 * atr)
        target = features.get("atr_target_3x", close + 3 * atr)

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
                "regime_weight":   weight,
                "pct_from_52w_high": pct_from_high,
                "bb_width":        bb_width,
                "vol_ratio":       vol_ratio,
            },
        )
