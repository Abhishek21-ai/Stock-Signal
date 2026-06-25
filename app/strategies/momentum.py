"""
Strategy 2: Momentum — Section 8.2
Logic: RSI zones + Stochastic + short-term price momentum

Regime weights:
  BULL:      weight=1.2
  BEAR:      weight=0.7  (momentum in bear = countertrend risk)
  SIDEWAYS:  weight=1.0
  UNCERTAIN: weight=0.9
"""
from __future__ import annotations

from typing import Dict

from app.strategies.base import BaseStrategy, StrategyResult, score_to_signal

REGIME_WEIGHTS = {"BULL": 1.2, "BEAR": 0.7, "SIDEWAYS": 1.0, "UNCERTAIN": 0.9}


class MomentumStrategy(BaseStrategy):
    strategy_id = "momentum"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        score = 0.0
        reasons = []
        close = features.get("close", 0)

        # ── 1. RSI Zone (max ±40 pts) ────────────────────────
        rsi  = features.get("rsi_14", 50)
        zone = features.get("rsi_zone", "NEUTRAL")

        if zone == "OVERSOLD":           # RSI < 30
            score += 35
            reasons.append(f"RSI oversold at {rsi:.1f} — bounce candidate")
        elif zone == "WEAK":             # 30–45
            score += 15
            reasons.append(f"RSI recovering from weakness at {rsi:.1f}")
        elif zone == "STRONG":           # 55–70
            score += 20
            reasons.append(f"RSI in momentum zone at {rsi:.1f}")
        elif zone == "OVERBOUGHT":       # > 70
            score -= 25
            reasons.append(f"RSI overbought at {rsi:.1f} — pullback risk")
        else:
            reasons.append(f"RSI neutral at {rsi:.1f}")

        # RSI divergence proxy: RSI falling while price rising = bearish
        ret_5d = features.get("return_5d", 0)
        if ret_5d and ret_5d > 0.02 and rsi < 45:
            score -= 10
            reasons.append("Possible bearish RSI divergence")
        elif ret_5d and ret_5d < -0.02 and rsi > 55:
            score += 10
            reasons.append("Possible bullish RSI divergence")

        # ── 2. Stochastic (max ±25 pts) ──────────────────────
        stoch_k = features.get("stoch_k", 50)
        stoch_d = features.get("stoch_d", 50)

        if stoch_k < 20 and stoch_d < 20:
            score += 25
            reasons.append(f"Stochastic oversold K={stoch_k:.1f} D={stoch_d:.1f}")
        elif stoch_k > 80 and stoch_d > 80:
            score -= 20
            reasons.append(f"Stochastic overbought K={stoch_k:.1f} D={stoch_d:.1f}")
        elif stoch_k > stoch_d and stoch_k < 80:
            score += 12
            reasons.append(f"Stochastic bullish cross K={stoch_k:.1f}")
        elif stoch_k < stoch_d and stoch_k > 20:
            score -= 12
            reasons.append(f"Stochastic bearish cross K={stoch_k:.1f}")

        # ── 3. Short-term price momentum (max ±20 pts) ────────
        ret_1d  = features.get("return_1d",  0) or 0
        ret_20d = features.get("return_20d", 0) or 0

        # 20d momentum
        if ret_20d > 0.05:
            score += 15
            reasons.append(f"Strong 20d momentum +{ret_20d:.1%}")
        elif ret_20d > 0.02:
            score += 8
            reasons.append(f"Positive 20d momentum +{ret_20d:.1%}")
        elif ret_20d < -0.05:
            score -= 15
            reasons.append(f"Weak 20d momentum {ret_20d:.1%}")
        elif ret_20d < -0.02:
            score -= 8
            reasons.append(f"Negative 20d momentum {ret_20d:.1%}")

        # 1d follow-through
        if ret_1d > 0.02:
            score += 5
            reasons.append(f"Strong 1d follow-through +{ret_1d:.1%}")
        elif ret_1d < -0.02:
            score -= 5
            reasons.append(f"Weak 1d {ret_1d:.1%}")

        # ── 4. VWAP position ─────────────────────────────────
        price_vs_vwap = features.get("price_vs_vwap", 0) or 0
        if price_vs_vwap > 0.02:
            score += 8
            reasons.append(f"Price {price_vs_vwap:.1%} above VWAP")
        elif price_vs_vwap < -0.02:
            score -= 8
            reasons.append(f"Price {price_vs_vwap:.1%} below VWAP")

        # ── 5. ADX confirmation gate (Fix A) ─────────────────
        # Momentum signals require actual directional movement to be valid.
        # RSI/Stochastic reaching extreme levels during a low-ADX drift
        # (common in IT stocks between earnings) produces false signals.
        # Without trend confirmation, these are noise, not momentum.
        adx = features.get("adx_14", 0) or 0
        if adx < 15:
            # Flat market — momentum signals are near-random, halve score
            score *= 0.4
            reasons.append(f"ADX={adx:.1f} < 15 — no trend, momentum signal unreliable")
        elif adx < 20:
            score *= 0.65
            reasons.append(f"ADX={adx:.1f} < 20 — weak trend, reducing momentum confidence")
        elif adx > 28:
            # Strong trend confirms momentum signal is genuine
            score *= 1.15
            score  = max(-100, min(100, score))
            reasons.append(f"ADX={adx:.1f} > 28 — trend confirmed, momentum signal strengthened")

        # ── 6. Regime weight ──────────────────────────────────
        weight = REGIME_WEIGHTS.get(regime, 0.9)
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
            meta={"regime_weight": weight, "rsi": rsi, "stoch_k": stoch_k},
        )