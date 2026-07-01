"""
Signal Fusion Engine — Section 9
Now includes Section 23 dynamic correlation penalty, applied to
regime weights before the weighted-sum step.

Exports (matched to test_fusion.py):
  fuse()               — fuse single stock → FusedSignal
  FusionEngine         — batch wrapper
  FusedSignal          — output dataclass
  AGREEMENT_BONUS      — int constant
  DISAGREEMENT_PENALTY — int constant
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from app.regime.detector import RegimeResult, REGIME_WEIGHTS
from app.strategies.base import StrategyResult, score_to_signal
from app.db import get_sync_db
from app.logger import get_logger
from app.correlation.engine import apply_correlation_penalty   # ── NEW (Section 23)
from app.regime.vol_adjuster import adjust_weights as vol_adjust_weights  # ── Vol-structure adjustment

logger = get_logger("fusion")

AGREEMENT_BONUS      = 8
DISAGREEMENT_PENALTY = 12
CONFIDENCE_GATE      = 20.0  # raised from 15 — reversion now fires selectively
SIGNAL_THRESHOLD     = 20.0


@dataclass
class FusedSignal:
    symbol:                       str
    signal:                       str
    fused_score:                  float
    confidence:                   float
    regime:                       str
    strategy_scores:              Dict[str, float] = field(default_factory=dict)
    entry_price:                  Optional[float] = None
    stop_loss:                    Optional[float] = None
    target_price:                 Optional[float] = None
    reasons:                      List[str] = field(default_factory=list)
    agreement_bonus_applied:      bool = False
    disagreement_penalty_applied: bool = False
    low_confidence_skipped:       List[str] = field(default_factory=list)
    run_date:                     Optional[date] = None
    correlation_adjusted:         bool = False        # ── NEW (Section 23)

    def to_dict(self) -> Dict:
        return {
            "symbol":      self.symbol,
            "signal":      self.signal,
            "fused_score": round(self.fused_score, 2),
            "confidence":  round(self.confidence, 2),
            "regime":      self.regime,
            "entry_price": self.entry_price,
            "stop_loss":   self.stop_loss,
            "target_price": self.target_price,
            "reasons":     self.reasons,
            "correlation_adjusted": self.correlation_adjusted,
            **{f"{k}_score": round(v, 2) for k, v in self.strategy_scores.items()},
        }


def fuse(
    symbol:           str,
    strategy_results: List[StrategyResult],
    regime_result:    RegimeResult,
    run_date:         date,
    features:         Optional[Dict] = None,
    save_to_db:       bool = True,
) -> FusedSignal:
    base_weights = regime_result.fusion_weights
    regime  = regime_result.regime
    reasons = []
    skipped = []

    # ── 1. Confidence gate ────────────────────────────────────
    eligible = []
    for r in strategy_results:
        if r.confidence < CONFIDENCE_GATE:
            skipped.append(r.strategy_id)
        else:
            eligible.append(r)

    if skipped:
        reasons.append(
            f"Skipped {len(skipped)} low-confidence strategies: {', '.join(skipped)}"
        )

    if not eligible:
        return FusedSignal(
            symbol=symbol, signal="HOLD", fused_score=0.0, confidence=0.0,
            regime=regime, reasons=["All strategies below confidence gate"],
            run_date=run_date,
        )

    # ── 1b. Volatility-structure weight adjustment ────────────
    # Adjusts regime weights per-stock using 3 metrics:
    #   1. 30d realized vol   — low-vol stocks get less trend/breakout
    #   2. ADX                — weak trend stocks get less trend/breakout
    #   3. Return autocorr    — mean-reverting stocks get more reversion
    vol_weights, vol_notes = vol_adjust_weights(
        symbol=symbol,
        base_weights=base_weights,
        features=features or {},
    )
    if vol_notes:
        reasons.append("Vol-structure adj: " + " | ".join(vol_notes))

    # ── 1c. Section 23: Dynamic correlation penalty ───────────
    # Strategies that fired (passed confidence gate) are "active" for
    # the purpose of the static co-fire check and correlation lookup.
    active_strategy_ids = {r.strategy_id for r in eligible}
    weights, corr_notes = apply_correlation_penalty(
        stock=symbol,
        base_weights=vol_weights,     # use vol-adjusted weights as input
        active_strategies=active_strategy_ids,
        as_of_date=run_date,
    )
    correlation_adjusted = len(corr_notes) > 0
    if corr_notes:
        reasons.append("Correlation penalty (Section 23): " + "; ".join(corr_notes))

    # ── 2. Weighted fusion ────────────────────────────────────
    strategy_scores: Dict[str, float] = {}
    fused       = 0.0
    weight_used = 0.0

    for r in eligible:
        w = weights.get(r.strategy_id, 0.20)
        fused       += r.score * w
        weight_used += w
        strategy_scores[r.strategy_id] = r.score

    if 0 < weight_used < 1.0:
        fused = fused / weight_used

    # ── 3. Agreement bonus / disagreement penalty ─────────────
    all_scores       = [r.score for r in eligible]
    bull_count       = sum(1 for s in all_scores if s > 15)
    bear_count       = sum(1 for s in all_scores if s < -15)
    agreement_bonus  = False
    disagree_penalty = False

    if bull_count >= 4 or bear_count >= 4:
        direction       = 1 if bull_count >= bear_count else -1
        fused          += direction * AGREEMENT_BONUS
        agreement_bonus = True
        reasons.append(
            f"Agreement bonus +{AGREEMENT_BONUS}: "
            f"{max(bull_count, bear_count)}/5 strategies aligned"
        )
    elif bull_count >= 2 and bear_count >= 2:
        direction       = 1 if fused >= 0 else -1
        fused          -= direction * DISAGREEMENT_PENALTY
        disagree_penalty = True
        reasons.append(
            f"Disagreement penalty -{DISAGREEMENT_PENALTY}: "
            f"{bull_count} bull vs {bear_count} bear signals"
        )

    fused = max(-100.0, min(100.0, fused))

    # ── 4. Signal gate ────────────────────────────────────────
    signal     = "HOLD" if abs(fused) < SIGNAL_THRESHOLD else score_to_signal(fused)
    confidence = min(95.0, abs(fused))

    # ── 5. Consensus prices ───────────────────────────────────
    direction = 1 if fused >= 0 else -1
    aligned   = [r for r in eligible if r.score * direction > 0]

    entry  = _median([r.entry_price  for r in aligned if r.entry_price])
    stop   = _median([r.stop_loss    for r in aligned if r.stop_loss])
    target = _median([r.target_price for r in aligned if r.target_price])

    if features:
        close = features.get("close")
        if not entry:  entry  = close
        if direction > 0:   # BUY — stop below entry, target above
            if not stop:   stop   = features.get("atr_stop_15x")
            if not target or target <= entry:
                target = features.get("atr_target_2x")
                if target and target <= entry:
                    reasons.append(
                        f"BUY target validation: adjusted to ATR-based target above entry"
                    )
        else:              # SELL — intraday short: stop above entry, target below
            if not stop:   stop   = features.get("atr_stop_15x_sell")
            if not target or target >= entry:
                target = features.get("atr_target_2x_sell")
                if target and target >= entry:
                    reasons.append(
                        f"SELL target validation: adjusted to ATR-based target below entry"
                    )
            if not stop or stop <= entry:
                stop = features.get("atr_stop_15x_sell") or (close + 1.5 * (features.get("atr_14") or 0))
                if stop and stop <= entry:
                    reasons.append(
                        f"SELL stop validation: adjusted to ATR-based stop above entry"
                    )

    # ── 6. Top reasons from strategies ───────────────────────
    for r in eligible:
        if r.reasons:
            reasons.append(f"[{r.strategy_id}] {r.reasons[0]}")

    fs = FusedSignal(
        symbol=symbol, signal=signal,
        fused_score=round(fused, 2), confidence=round(confidence, 2),
        regime=regime, strategy_scores=strategy_scores,
        entry_price=_round(entry), stop_loss=_round(stop), target_price=_round(target),
        reasons=reasons,
        agreement_bonus_applied=agreement_bonus,
        disagreement_penalty_applied=disagree_penalty,
        low_confidence_skipped=skipped,
        run_date=run_date,
        correlation_adjusted=correlation_adjusted,
    )

    logger.info(
        f"{symbol} | {signal} | score={fused:+.1f} | conf={confidence:.0f}% | "
        f"regime={regime} | bull={bull_count} bear={bear_count} | "
        f"corr_adj={correlation_adjusted}"
    )

    if save_to_db:
        _save_to_db(fs)

    return fs


class FusionEngine:
    def __init__(self, regime_result: RegimeResult, run_date: date):
        self.regime_result = regime_result
        self.run_date      = run_date

    def run(
        self,
        all_strategy_results: Dict[str, List[StrategyResult]],
        all_features:         Optional[Dict[str, Dict]] = None,
        save_to_db:           bool = True,
    ) -> List[FusedSignal]:
        signals = []
        for symbol, results in all_strategy_results.items():
            features = (all_features or {}).get(symbol)
            try:
                fs = fuse(
                    symbol=symbol, strategy_results=results,
                    regime_result=self.regime_result, run_date=self.run_date,
                    features=features, save_to_db=save_to_db,
                )
                signals.append(fs)
            except Exception as e:
                logger.error(f"Fusion failed for {symbol}: {e}")

        signals.sort(key=lambda x: x.fused_score, reverse=True)
        return signals


def _save_to_db(fs: FusedSignal) -> None:
    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO daily_signals (
                    date, stock, signal, quant_score, confidence_pct,
                    trend_score, momentum_score, reversion_score,
                    breakout_score, volume_score,
                    entry_price_theoretical, stop_loss_theoretical,
                    exit_target_theoretical, regime, valid_until
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date, stock) DO UPDATE SET
                    signal=EXCLUDED.signal, quant_score=EXCLUDED.quant_score,
                    confidence_pct=EXCLUDED.confidence_pct,
                    trend_score=EXCLUDED.trend_score,
                    momentum_score=EXCLUDED.momentum_score,
                    reversion_score=EXCLUDED.reversion_score,
                    breakout_score=EXCLUDED.breakout_score,
                    volume_score=EXCLUDED.volume_score,
                    entry_price_theoretical=EXCLUDED.entry_price_theoretical,
                    stop_loss_theoretical=EXCLUDED.stop_loss_theoretical,
                    exit_target_theoretical=EXCLUDED.exit_target_theoretical,
                    regime=EXCLUDED.regime
                """,
                (
                    fs.run_date, fs.symbol, fs.signal,
                    fs.fused_score, fs.confidence,
                    fs.strategy_scores.get("trend",     0),
                    fs.strategy_scores.get("momentum",  0),
                    fs.strategy_scores.get("reversion", 0),
                    fs.strategy_scores.get("breakout",  0),
                    fs.strategy_scores.get("volume",    0),
                    fs.entry_price, fs.stop_loss, fs.target_price,
                    fs.regime, fs.run_date,
                ),
            )
    except Exception as e:
        logger.error(f"DB save failed for {fs.symbol}: {e}")


def _median(values: list) -> Optional[float]:
    v = [x for x in values if x is not None]
    if not v: return None
    v.sort()
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2


def _round(v) -> Optional[float]:
    return round(float(v), 2) if v is not None else None