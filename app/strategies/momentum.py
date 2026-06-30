"""
Strategy 2: Momentum — Velocity Expansion Edition
Logic: RSI Trend Invalidation + Stochastic Overbought Continuation + Rate of Change Acceleration
"""
from __future__ import annotations

from typing import Dict
from app.strategies.base import BaseStrategy, StrategyResult, score_to_signal

REGIME_WEIGHTS = {"BULL": 1.4, "BEAR": 0.4, "SIDEWAYS": 0.6, "UNCERTAIN": 1.0}


class MomentumStrategy(BaseStrategy):
    strategy_id = "momentum"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        score = 0.0
        reasons = []
        close = features.get("close", 0)

        # ── 1. Pure Momentum RSI Zones (No bottom fishing!) ──
        rsi = features.get("rsi_14", 50)
        
        if rsi >= 55 and rsi <= 70:
            score += 40
            reasons.append(f"RSI in strong velocity expansion zone ({rsi:.1f})")
        elif rsi > 70:
            score += 25  # High velocity asset continuation
            reasons.append(f"RSI overbought ({rsi:.1f}) — riding strong upward trend")
        elif rsi < 40:
            score -= 35  # Penalize weak momentum severely
            reasons.append(f"RSI weak ({rsi:.1f}) — absolute momentum dead")

        # ── 2. Stochastic Acceleration (Buy the crossover on strength) ──
        stoch_k = features.get("stoch_k", 50)
        stoch_d = features.get("stoch_d", 50)

        if stoch_k > stoch_d and stoch_k > 50:
            score += 20
            reasons.append(f"Stochastic bullish crossover in power zone (K={stoch_k:.1f})")
        elif stoch_k < stoch_d and stoch_k < 50:
            score -= 20

        # ── 3. Multi-Window Rate of Change Acceleration ──
        ret_1d  = features.get("return_1d",  0) or 0
        ret_5d  = features.get("return_5d",  0) or 0
        ret_20d = features.get("return_20d", 0) or 0

        # 20d structural trend alignment
        if ret_20d > 0.04:
            score += 20
            reasons.append(f"Backed by strong structural 20d return (+{ret_20d:.1%})")
        
        # 5d vs 1d velocity matrix
        if ret_1d > 0.015 and ret_5d > 0.03:
            score += 20
            reasons.append("Short-term velocity accelerating upwards")

        # ── 4. VWAP Support Layer ──
        price_vs_vwap = features.get("price_vs_vwap", 0) or 0
        if price_vs_vwap > 0.01:
            score += 15
            reasons.append(f"Price accelerating comfortably above VWAP (+{price_vs_vwap:.1%})")

        # ── 5. Regime Multiplier & Risk Target Mapping ──
        weight = REGIME_WEIGHTS.get(regime, 1.0)
        score  = max(-100, min(100, score * weight))

        atr    = features.get("atr_14", close * 0.02)
        entry  = close
        
        # Tighten stop loss to 1.2x ATR to cut slow moving momentum failures early
        stop   = close - (1.2 * atr)
        target = close + (2.5 * atr) # Extend target to let high-velocity runs capture alpha

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