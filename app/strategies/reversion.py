"""
Strategy 3: Mean Reversion — Optimized Edition
"""
from __future__ import annotations

from typing import Dict
from app.strategies.base import BaseStrategy, StrategyResult, score_to_signal

REGIME_WEIGHTS = {"BULL": 0.7, "BEAR": 0.7, "SIDEWAYS": 1.5, "UNCERTAIN": 1.2}


class MeanReversionStrategy(BaseStrategy):
    strategy_id = "reversion"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        score = 0.0
        reasons = []
        close = features.get("close", 0)

        # ── 1. Bollinger Band position (Slightly boosted base scores) ──
        bb_pct_b = features.get("bb_pct_b", 0.5) or 0.5

        if bb_pct_b < 0.05:
            score += 60  # Boosted from 50
            reasons.append(f"Price outside lower Bollinger Band (%B={bb_pct_b:.2f})")
        elif bb_pct_b < 0.20:
            score += 35  # Boosted from 30
            reasons.append(f"Price near lower BB (%B={bb_pct_b:.2f})")
        elif bb_pct_b > 0.95:
            score -= 55  # Boosted from -45
            reasons.append(f"Price outside upper Bollinger Band (%B={bb_pct_b:.2f})")
        elif bb_pct_b > 0.80:
            score -= 30
            reasons.append(f"Price near upper BB (%B={bb_pct_b:.2f})")

        # Band width check
        bb_width = features.get("bb_width", 0.1) or 0.1
        if bb_width < 0.03:
            score *= 0.4  # Slightly more restrictive on tight squeezes
            reasons.append(f"BB squeeze ({bb_width:.3f}) — avoiding breakout")

        # ── 2. Distance from EMA50 ───────────────────────────
        ema_50 = features.get("ema_50", close)
        if ema_50 and ema_50 > 0:
            dev_pct = (close - ema_50) / ema_50
            if dev_pct < -0.04:
                score += 25
                reasons.append(f"Price deep deviation below EMA50 ({dev_pct:.1%})")
            elif dev_pct > 0.04:
                score -= 25

        # ── 3. RSI confirmation ───────────────────────────────
        rsi = features.get("rsi_14", 50)
        if rsi < 35:
            score += 20
        elif rsi > 65:
            score -= 20

        # ── 4. Suppress if strong trend (Loosened penalty slightly) ──
        adx = features.get("adx_14", 0)
        if adx > 32:
            score *= 0.5
            reasons.append(f"ADX extreme trend ({adx:.1f}) — throttling reversion")

        # ── 5. Regime weight and execution boundaries ────────
        weight = REGIME_WEIGHTS.get(regime, 1.2)
        score  = max(-100, min(100, score * weight))

        atr    = features.get("atr_14", close * 0.02)
        entry  = close

        # Use direction-appropriate stop/target
        if score >= 0:                                                                   # BUY signal       
            stop = features.get("atr_stop_15x", close - 1.5 * atr)

            # FIX: Ensure your profit target preserves a positive risk/reward ratio
            min_take_profit = close + (1.5 * atr) # Match risk 1:1 minimum
            if ema_50 and ema_50 > min_take_profit:
                target = float(ema_50)
            else:
                target = min_take_profit

        else:                                                                           # SELL signal
            stop = round(close + max(1.0 * atr, 1.0), 2)

            if ema_50 and float(ema_50) < close:
                target = round(float(ema_50), 2)
            else:
                target = round(close - max(1.5 * atr, 1.0), 2)

        return StrategyResult(
            strategy_id=self.strategy_id,
            score=round(score, 2),
            signal=score_to_signal(score),
            confidence=min(100, abs(score)),
            reasons=reasons,
            entry_price=round(entry, 2),
            stop_loss=round(stop, 2),
            target_price=round(target, 2),
            meta={"regime_weight": weight},
        )