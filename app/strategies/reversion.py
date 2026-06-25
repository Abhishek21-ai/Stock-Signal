"""
Strategy 3: Mean Reversion — Section 8.3
Logic: Bollinger Band extremes + RSI extremes + distance from EMA

Best in SIDEWAYS/UNCERTAIN regime. Suppressed in strong trends.

Regime weights:
  SIDEWAYS:  weight=1.4  (reversion thrives here)
  UNCERTAIN: weight=1.1
  BULL:      weight=0.6  (don't fade bull trends)
  BEAR:      weight=0.6
"""
from __future__ import annotations

from typing import Dict

from app.strategies.base import BaseStrategy, StrategyResult, score_to_signal

REGIME_WEIGHTS = {"BULL": 0.6, "BEAR": 0.6, "SIDEWAYS": 1.4, "UNCERTAIN": 1.1}


class MeanReversionStrategy(BaseStrategy):
    strategy_id = "reversion"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        score = 0.0
        reasons = []
        close = features.get("close", 0)

        # ── 1. Bollinger Band position (max ±50 pts) ──────────
        bb_pct_b = features.get("bb_pct_b", 0.5) or 0.5
        bb_lower = features.get("bb_lower", close)
        bb_upper = features.get("bb_upper", close)
        bb_mid   = features.get("bb_mid",   close)

        if bb_pct_b < 0.05:          # price at/below lower band
            score += 50
            reasons.append(f"Price at lower Bollinger Band (%B={bb_pct_b:.2f}) — reversion setup")
        elif bb_pct_b < 0.20:
            score += 30
            reasons.append(f"Price near lower BB (%B={bb_pct_b:.2f})")
        elif bb_pct_b > 0.95:        # price at/above upper band
            score -= 45
            reasons.append(f"Price at upper Bollinger Band (%B={bb_pct_b:.2f}) — overbought")
        elif bb_pct_b > 0.80:
            score -= 25
            reasons.append(f"Price near upper BB (%B={bb_pct_b:.2f})")
        else:
            reasons.append(f"Price mid-band (%B={bb_pct_b:.2f}) — no reversion signal")

        # Band width: narrow bands = volatility squeeze = breakout coming, not reversion
        bb_width = features.get("bb_width", 0.1) or 0.1
        if bb_width < 0.03:
            score *= 0.5
            reasons.append(f"BB squeeze detected (width={bb_width:.3f}) — breakout likely, reducing reversion score")

        # ── 2. Distance from EMA50 (max ±25 pts) ─────────────
        ema_50 = features.get("ema_50", close)
        if ema_50 and ema_50 > 0:
            dev_pct = (close - ema_50) / ema_50

            if dev_pct < -0.05:      # >5% below EMA50
                score += 20
                reasons.append(f"Price {dev_pct:.1%} below EMA50 — oversold vs mean")
            elif dev_pct < -0.03:
                score += 10
                reasons.append(f"Price {dev_pct:.1%} below EMA50")
            elif dev_pct > 0.05:     # >5% above EMA50
                score -= 20
                reasons.append(f"Price {dev_pct:.1%} above EMA50 — extended")
            elif dev_pct > 0.03:
                score -= 10
                reasons.append(f"Price {dev_pct:.1%} above EMA50")

        # ── 3. RSI confirmation (max ±20 pts) ─────────────────
        rsi = features.get("rsi_14", 50)
        if rsi < 35:
            score += 20
            reasons.append(f"RSI confirms oversold at {rsi:.1f}")
        elif rsi < 45:
            score += 8
            reasons.append(f"RSI weak at {rsi:.1f} supports reversion buy")
        elif rsi > 65:
            score -= 20
            reasons.append(f"RSI confirms overbought at {rsi:.1f}")
        elif rsi > 55:
            score -= 8
            reasons.append(f"RSI elevated at {rsi:.1f}")

        # ── 4. Suppress if strong trend (ADX check) ──────────
        adx = features.get("adx_14", 0)
        if adx > 30:
            score *= 0.4
            reasons.append(f"Strong trend ADX={adx:.1f} — suppressing reversion signal")
        elif adx > 25:
            score *= 0.7
            reasons.append(f"Moderate trend ADX={adx:.1f} — reducing reversion confidence")

        # ── 5. EMA trend filter — no long reversion in bear structure (Fix B)
        # Mean reversion BUY only makes sense when price structure is neutral
        # or recovering. In a confirmed downtrend (EMA20 < EMA50 < EMA200),
        # oversold stocks often become more oversold (falling knife).
        # Only suppress LONG (positive) reversion scores — short reversion
        # (overbought in uptrend) is still valid.
        ema_20  = features.get("ema_20",  close) or close
        ema_50  = features.get("ema_50",  close) or close
        ema_200 = features.get("ema_200", close) or close

        bear_alignment = ema_20 < ema_50 and ema_50 < ema_200
        bull_alignment = ema_20 > ema_50 and ema_50 > ema_200

        if score > 0 and bear_alignment:
            # Full bear EMA stack — buying reversion into downtrend is dangerous
            score *= 0.25
            reasons.append(
                f"EMA bear stack (EMA20={ema_20:.0f}<EMA50={ema_50:.0f}<EMA200={ema_200:.0f}) "
                f"— suppressing reversion long (catching falling knife risk)"
            )
        elif score > 0 and ema_20 < ema_50:
            # Partial bear alignment — reduce but don't eliminate
            score *= 0.55
            reasons.append(
                f"EMA20 < EMA50 ({ema_20:.0f} < {ema_50:.0f}) "
                f"— reducing reversion long confidence"
            )
        elif score < 0 and bull_alignment:
            # Full bull EMA stack — shorting reversion into uptrend is dangerous
            score *= 0.25
            reasons.append(
                f"EMA bull stack — suppressing reversion short (fading bull trend risk)"
            )

        # ── 6. Regime weight ──────────────────────────────────
        weight = REGIME_WEIGHTS.get(regime, 1.1)
        score  = max(-100, min(100, score * weight))

        atr    = features.get("atr_14", close * 0.02)
        entry  = close
        # Reversion target = EMA50 (mean)
        target = round(float(ema_50), 2) if ema_50 else round(close + 2 * atr, 2)
        stop   = features.get("atr_stop_15x", close - 1.5 * atr)

        return StrategyResult(
            strategy_id=self.strategy_id,
            score=score,
            signal=score_to_signal(score),
            confidence=min(100, abs(score)),
            reasons=reasons,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            meta={"regime_weight": weight, "bb_pct_b": bb_pct_b, "adx": adx},
        )