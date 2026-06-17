"""
Strategy 6: Risk Engine — Section 8.6
Acts as a gatekeeper/penalty layer within the strategy pipeline.

Checks (in order):
  1. Volatility risk   — ATR% too high = position too dangerous
  2. Drawdown risk     — price too far from 52w high = avoid catching falling knife
  3. Liquidity risk    — volume below threshold = can't enter/exit cleanly
  4. Stop validity     — ATR stop would be > MAX_STOP_PCT below entry
  5. Risk/Reward ratio — target must be at least RR_MIN × risk

Scoring:
  Starts at 0. Each passed check adds positive points (up to +40 total).
  Each failed check subtracts penalty points (can go negative → SELL signal = risk veto).
  A hard VETO (-100) is issued if ATR% > VETO_VOLATILITY or volume < VETO_LIQUIDITY.

Regime weights:
  BULL:      0.7  (risk engine less penalising in strong uptrend)
  BEAR:      1.5  (risk engine very penalising in downtrend)
  SIDEWAYS:  1.2
  UNCERTAIN: 1.0
"""
from __future__ import annotations

from typing import Dict

from app.strategies.base import BaseStrategy, StrategyResult, score_to_signal

# ── Thresholds ────────────────────────────────────────────────
MAX_ATR_PCT         = 5.0    # ATR% above this = high volatility penalty
VETO_ATR_PCT        = 8.0    # ATR% above this = hard veto (too dangerous)
MAX_DRAWDOWN_PCT    = -20.0  # % from 52w high; below this = falling knife penalty
VETO_DRAWDOWN_PCT   = -35.0  # below this = hard veto
MIN_VOLUME_RATIO    = 0.5    # volume/avg_volume below this = illiquid penalty
VETO_VOLUME_RATIO   = 0.2    # below this = hard veto (can't trade)
MAX_STOP_PCT        = 7.0    # stop loss > this % below entry = invalid setup
RR_MIN              = 1.5    # minimum risk/reward ratio required

REGIME_WEIGHTS = {
    "BULL":      0.7,
    "BEAR":      1.5,
    "SIDEWAYS":  1.2,
    "UNCERTAIN": 1.0,
}


class RiskStrategy(BaseStrategy):
    strategy_id = "risk"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        score   = 0.0
        reasons = []
        close   = features.get("close", 0.0)
        atr     = features.get("atr_14", close * 0.02)
        atr_pct = features.get("atr_pct", (atr / close * 100) if close else 2.0)

        # ── Check 1: Volatility (ATR%) ────────────────────────
        if atr_pct > VETO_ATR_PCT:
            return self._hard_veto(
                f"Volatility veto: ATR%={atr_pct:.1f}% > {VETO_ATR_PCT}% — too dangerous",
                close, atr, regime,
            )
        elif atr_pct > MAX_ATR_PCT:
            score -= 30
            reasons.append(f"High volatility: ATR%={atr_pct:.1f}% > {MAX_ATR_PCT}% — penalty")
        elif atr_pct < 1.0:
            score -= 10
            reasons.append(f"Very low volatility: ATR%={atr_pct:.1f}% — stock may be illiquid/stuck")
        else:
            score += 15
            reasons.append(f"Volatility OK: ATR%={atr_pct:.1f}%")

        # ── Check 2: Drawdown from 52w high ──────────────────
        dd_pct = features.get("pct_from_52w_high", 0.0)
        if dd_pct < VETO_DRAWDOWN_PCT:
            return self._hard_veto(
                f"Drawdown veto: {dd_pct:.1f}% from 52w high < {VETO_DRAWDOWN_PCT}% — falling knife",
                close, atr, regime,
            )
        elif dd_pct < MAX_DRAWDOWN_PCT:
            score -= 25
            reasons.append(f"Significant drawdown: {dd_pct:.1f}% from 52w high — caution")
        elif dd_pct > -5.0:
            score += 15
            reasons.append(f"Near 52w high: {dd_pct:.1f}% — strong price action")
        else:
            score += 5
            reasons.append(f"Moderate drawdown: {dd_pct:.1f}% from 52w high — acceptable")

        # ── Check 3: Liquidity (volume ratio) ────────────────
        vol_ratio = features.get("volume_ratio", 1.0)
        if vol_ratio < VETO_VOLUME_RATIO:
            return self._hard_veto(
                f"Liquidity veto: volume_ratio={vol_ratio:.2f} < {VETO_VOLUME_RATIO} — untradeable",
                close, atr, regime,
            )
        elif vol_ratio < MIN_VOLUME_RATIO:
            score -= 20
            reasons.append(f"Low liquidity: volume_ratio={vol_ratio:.2f} < {MIN_VOLUME_RATIO}")
        elif vol_ratio > 1.5:
            score += 10
            reasons.append(f"Strong volume: ratio={vol_ratio:.2f} — good liquidity")
        else:
            score += 5
            reasons.append(f"Adequate volume: ratio={vol_ratio:.2f}")

        # ── Check 4: Stop loss validity ───────────────────────
        stop = features.get("atr_stop_15x", close - 1.5 * atr)
        if close > 0:
            stop_pct = abs(close - stop) / close * 100
            if stop_pct > MAX_STOP_PCT:
                score -= 20
                reasons.append(
                    f"Wide stop: {stop_pct:.1f}% below entry > {MAX_STOP_PCT}% — reduces position size"
                )
            else:
                score += 5
                reasons.append(f"Valid stop: {stop_pct:.1f}% below entry")

        # ── Check 5: Risk/Reward ratio ────────────────────────
        target = features.get("atr_target_3x", close + 3.0 * atr)
        risk   = abs(close - stop)   if close > stop  else atr
        reward = abs(target - close) if target > close else atr * 2

        if risk > 0:
            rr = reward / risk
            if rr < RR_MIN:
                score -= 15
                reasons.append(f"Poor R/R: {rr:.2f} < {RR_MIN} minimum")
            elif rr >= 3.0:
                score += 10
                reasons.append(f"Excellent R/R: {rr:.2f}")
            else:
                score += 5
                reasons.append(f"Acceptable R/R: {rr:.2f}")

        # ── Regime weight ─────────────────────────────────────
        weight = REGIME_WEIGHTS.get(regime, 1.0)
        # Risk scores are penalties — amplify in BEAR, dampen in BULL
        # Positive scores get dampened in BEAR (fewer safe setups); capped always
        if score > 0:
            score = score * (2.0 - weight)   # BULL(0.7)→*1.3, BEAR(1.5)→*0.5
        else:
            score = score * weight            # BEAR amplifies penalties

        score = max(-100.0, min(100.0, score))

        return StrategyResult(
            strategy_id=self.strategy_id,
            score=round(score, 2),
            signal=score_to_signal(score),
            confidence=min(100.0, abs(score) + 20),   # risk engine always somewhat confident
            reasons=reasons,
            entry_price=close,
            stop_loss=round(stop, 2),
            target_price=round(target, 2),
            meta={
                "atr_pct":      round(atr_pct, 2),
                "drawdown_pct": round(dd_pct, 2),
                "volume_ratio": round(vol_ratio, 2),
                "regime_weight": weight,
            },
        )

    def _hard_veto(
        self, reason: str, close: float, atr: float, regime: str
    ) -> StrategyResult:
        """Returns a STRONG_SELL with score=-100 to veto the signal."""
        stop   = round(close - 1.5 * atr, 2)
        target = round(close + 3.0 * atr, 2)
        return StrategyResult(
            strategy_id=self.strategy_id,
            score=-100.0,
            signal="STRONG_SELL",
            confidence=100.0,
            reasons=[f"⛔ HARD VETO — {reason}"],
            entry_price=close,
            stop_loss=stop,
            target_price=target,
            meta={"hard_veto": True, "veto_reason": reason},
        )
