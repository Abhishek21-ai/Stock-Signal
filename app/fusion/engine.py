"""
Signal Fusion Engine — Section 9
Aggregates outputs from all 5 strategy engines into a single ranked signal per stock.

Design spec (Section 9):
  1. Weighted score aggregation using regime-driven weights (Section 7.2)
  2. Confidence gate — skip strategies below MIN_CONFIDENCE threshold
  3. Agreement bonus — if 3+ strategies agree on direction, boost final score by 10%
  4. Disagreement penalty — if strategies strongly conflict, reduce score by 15%
  5. Deduplication penalty — if same stock signalled previous N days, reduce score
  6. Final signal classification via score thresholds (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)
  7. Persist fused signal to daily_signals table in DB
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

from app.db import get_sync_db
from app.logger import get_logger
from app.regime.detector import RegimeResult, REGIME_WEIGHTS
from app.strategies.base import StrategyResult, score_to_signal

logger = get_logger("fusion")

# ── Thresholds (Section 9) ────────────────────────────────────
MIN_CONFIDENCE       = 20.0   # strategies below this are excluded from fusion
AGREEMENT_BONUS      = 10.0   # score bonus when all strategies agree on direction
DISAGREEMENT_PENALTY = 15.0   # score reduction when strategies strongly conflict
DEDUP_LOOKBACK_DAYS  = 3      # days to look back for duplicate signals
DEDUP_PENALTY        = 12.0   # score penalty per recent duplicate

# Strategy ID → regime weight key mapping
STRATEGY_WEIGHT_KEY: Dict[str, str] = {
    "trend":     "trend",
    "momentum":  "momentum",
    "reversion": "reversion",
    "breakout":  "breakout",
    "volume":    "volume",
}


@dataclass
class FusedSignal:
    symbol: str
    run_date: date
    fused_score: float                   # -100 to +100
    signal: str                          # STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
    confidence: float                    # 0-100 (weighted avg of participating strategies)
    regime: str
    strategy_scores: Dict[str, float] = field(default_factory=dict)
    strategy_signals: Dict[str, str]  = field(default_factory=dict)
    fusion_weights_used: Dict[str, float] = field(default_factory=dict)
    agreement_bonus_applied: bool = False
    disagreement_penalty_applied: bool = False
    dedup_penalty: float = 0.0
    recent_signal_days: int = 0
    reasons: List[str] = field(default_factory=list)
    entry_price:  Optional[float] = None
    stop_loss:    Optional[float] = None
    target_price: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "symbol":                       self.symbol,
            "run_date":                     str(self.run_date),
            "fused_score":                  round(self.fused_score, 2),
            "signal":                       self.signal,
            "confidence":                   round(self.confidence, 2),
            "regime":                       self.regime,
            "strategy_scores":              {k: round(v, 2) for k, v in self.strategy_scores.items()},
            "strategy_signals":             self.strategy_signals,
            "fusion_weights_used":          self.fusion_weights_used,
            "agreement_bonus_applied":      self.agreement_bonus_applied,
            "disagreement_penalty_applied": self.disagreement_penalty_applied,
            "dedup_penalty":                round(self.dedup_penalty, 2),
            "recent_signal_days":           self.recent_signal_days,
            "entry_price":                  self.entry_price,
            "stop_loss":                    self.stop_loss,
            "target_price":                 self.target_price,
            "reasons":                      self.reasons,
        }


# ── Helper: recent duplicate check ───────────────────────────

def _get_recent_signal_count(symbol: str, run_date: date, lookback: int) -> int:
    """Count how many of the last N days this symbol already had a BUY/STRONG_BUY signal."""
    cutoff = run_date - timedelta(days=lookback)
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) as cnt
                FROM daily_signals
                WHERE stock = %s
                  AND date > %s
                  AND date < %s
                  AND signal IN ('BUY', 'STRONG_BUY')
                """,
                (symbol, cutoff, run_date),
            )
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0
    except Exception as e:
        logger.warning(f"Dedup check failed for {symbol}: {e}")
        return 0


def _save_signal(signal: FusedSignal) -> None:
    """Upsert fused signal into daily_signals table."""
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO daily_signals (
                    stock, date, signal, quant_score, confidence_pct,
                    regime,
                    entry_price_theoretical, stop_loss_theoretical, exit_target_theoretical,
                    trend_score, momentum_score, reversion_score, breakout_score, volume_score,
                    llm_override, valid_until
                ) VALUES (
                    %s, %s, %s::signal_type, %s, %s,
                    %s::regime_type,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    'NONE', %s
                )
                ON CONFLICT (date, stock) DO UPDATE SET
                    signal                  = EXCLUDED.signal,
                    quant_score             = EXCLUDED.quant_score,
                    confidence_pct          = EXCLUDED.confidence_pct,
                    regime                  = EXCLUDED.regime,
                    entry_price_theoretical = EXCLUDED.entry_price_theoretical,
                    stop_loss_theoretical   = EXCLUDED.stop_loss_theoretical,
                    exit_target_theoretical = EXCLUDED.exit_target_theoretical,
                    trend_score             = EXCLUDED.trend_score,
                    momentum_score          = EXCLUDED.momentum_score,
                    reversion_score         = EXCLUDED.reversion_score,
                    breakout_score          = EXCLUDED.breakout_score,
                    volume_score            = EXCLUDED.volume_score
                """,
                (
                    signal.symbol,
                    signal.run_date,
                    signal.signal,
                    round(signal.fused_score, 2),
                    round(signal.confidence, 2),
                    signal.regime,
                    signal.entry_price,
                    signal.stop_loss,
                    signal.target_price,
                    round(signal.strategy_scores.get("trend",     0.0), 2),
                    round(signal.strategy_scores.get("momentum",  0.0), 2),
                    round(signal.strategy_scores.get("reversion", 0.0), 2),
                    round(signal.strategy_scores.get("breakout",  0.0), 2),
                    round(signal.strategy_scores.get("volume",    0.0), 2),
                    signal.run_date,   # valid_until = same day; pipeline updates this later
                ),
            )
    except Exception as e:
        logger.error(f"Failed to save signal for {signal.symbol}: {e}")


# ── Core fusion logic ─────────────────────────────────────────

def fuse(
    symbol: str,
    strategy_results: List[StrategyResult],
    regime_result: RegimeResult,
    run_date: Optional[date] = None,
    save_to_db: bool = True,
) -> FusedSignal:
    """
    Fuse strategy results into a single signal for one symbol.

    Steps:
      1. Filter strategies by MIN_CONFIDENCE
      2. Weighted score aggregation (regime weights)
      3. Agreement bonus / disagreement penalty
      4. Deduplication penalty
      5. Final signal + confidence
      6. Optionally persist to DB
    """
    run_date = run_date or date.today()
    regime   = regime_result.regime
    weights  = regime_result.fusion_weights or dict(REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["UNCERTAIN"]))
    reasons  = []

    # ── 1. Filter by confidence ───────────────────────────────
    eligible = [r for r in strategy_results if r.confidence >= MIN_CONFIDENCE]
    skipped  = [r.strategy_id for r in strategy_results if r.confidence < MIN_CONFIDENCE]
    if skipped:
        reasons.append(f"Skipped low-confidence strategies: {skipped}")

    if not eligible:
        logger.warning(f"{symbol}: No strategies passed confidence gate — returning HOLD")
        return FusedSignal(
            symbol=symbol, run_date=run_date,
            fused_score=0.0, signal="HOLD", confidence=0.0,
            regime=regime, reasons=["All strategies below confidence gate"],
        )

    # ── 2. Weighted score aggregation ────────────────────────
    weighted_score  = 0.0
    total_weight    = 0.0
    weighted_conf   = 0.0
    strategy_scores: Dict[str, float] = {}
    strategy_signals: Dict[str, str]  = {}

    for r in eligible:
        wkey   = STRATEGY_WEIGHT_KEY.get(r.strategy_id, "trend")
        weight = weights.get(wkey, 1.0 / len(eligible))
        weighted_score += r.score * weight
        total_weight   += weight
        weighted_conf  += r.confidence * weight
        strategy_scores[r.strategy_id]  = r.score
        strategy_signals[r.strategy_id] = r.signal

    if total_weight > 0:
        fused_score = weighted_score / total_weight
        confidence  = weighted_conf  / total_weight
    else:
        fused_score = 0.0
        confidence  = 0.0

    reasons.append(
        f"Weighted fusion ({len(eligible)} strategies, regime={regime}): "
        f"raw score={fused_score:.1f}"
    )

    # ── 3a. Agreement bonus ───────────────────────────────────
    buy_side  = sum(1 for r in eligible if r.score > 0)
    sell_side = sum(1 for r in eligible if r.score < 0)
    n         = len(eligible)

    agreement_bonus_applied      = False
    disagreement_penalty_applied = False

    if buy_side >= 3 and buy_side == n:
        fused_score += AGREEMENT_BONUS
        agreement_bonus_applied = True
        reasons.append(f"Agreement bonus +{AGREEMENT_BONUS}: all {n} strategies bullish")

    elif sell_side >= 3 and sell_side == n:
        fused_score -= AGREEMENT_BONUS
        agreement_bonus_applied = True
        reasons.append(f"Agreement bonus -{AGREEMENT_BONUS}: all {n} strategies bearish")

    # ── 3b. Disagreement penalty ──────────────────────────────
    elif buy_side >= 2 and sell_side >= 2:
        direction = 1 if fused_score > 0 else -1
        fused_score -= direction * DISAGREEMENT_PENALTY
        disagreement_penalty_applied = True
        reasons.append(
            f"Disagreement penalty -{DISAGREEMENT_PENALTY}: "
            f"{buy_side} bullish vs {sell_side} bearish strategies"
        )

    fused_score = max(-100.0, min(100.0, fused_score))

    # ── 4. Deduplication penalty ──────────────────────────────
    recent_days   = _get_recent_signal_count(symbol, run_date, DEDUP_LOOKBACK_DAYS)
    dedup_penalty = 0.0
    if recent_days > 0 and fused_score > 0:
        dedup_penalty = DEDUP_PENALTY * recent_days
        fused_score   = max(0.0, fused_score - dedup_penalty)
        reasons.append(
            f"Dedup penalty -{dedup_penalty:.0f}: symbol signalled "
            f"{recent_days}x in last {DEDUP_LOOKBACK_DAYS} days"
        )

    # ── 5. Final signal label + price levels ─────────────────
    signal = score_to_signal(fused_score)

    best         = max(eligible, key=lambda r: r.confidence)
    entry_price  = best.entry_price
    stop_loss    = best.stop_loss
    target_price = best.target_price

    reasons.append(f"Final: score={fused_score:.1f} → {signal} (conf={confidence:.1f}%)")

    result = FusedSignal(
        symbol=symbol,
        run_date=run_date,
        fused_score=round(fused_score, 2),
        signal=signal,
        confidence=round(confidence, 2),
        regime=regime,
        strategy_scores=strategy_scores,
        strategy_signals=strategy_signals,
        fusion_weights_used=weights,
        agreement_bonus_applied=agreement_bonus_applied,
        disagreement_penalty_applied=disagreement_penalty_applied,
        dedup_penalty=dedup_penalty,
        recent_signal_days=recent_days,
        reasons=reasons,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
    )

    if save_to_db:
        _save_signal(result)
        logger.info(
            f"{symbol}: {signal} | score={fused_score:.1f} | "
            f"conf={confidence:.1f}% | regime={regime} | "
            f"dedup_penalty={dedup_penalty:.0f}"
        )

    return result


# ── Batch fusion ──────────────────────────────────────────────

class FusionEngine:
    """
    Called by pipeline.py Stage 5.
    Fuses strategy results for all stocks and returns ranked signal list.
    """

    def __init__(self, regime_result: RegimeResult, run_date: Optional[date] = None):
        self.regime_result = regime_result
        self.run_date      = run_date or date.today()

    def run(
        self,
        all_results: Dict[str, List[StrategyResult]],
        save_to_db: bool = True,
    ) -> List[FusedSignal]:
        """
        all_results: { symbol -> [StrategyResult, ...] }
        Returns list of FusedSignals sorted by fused_score descending.
        """
        fused = []
        for symbol, results in all_results.items():
            try:
                fs = fuse(
                    symbol=symbol,
                    strategy_results=results,
                    regime_result=self.regime_result,
                    run_date=self.run_date,
                    save_to_db=save_to_db,
                )
                fused.append(fs)
            except Exception as e:
                logger.error(f"Fusion failed for {symbol}: {e}")

        priority = {"STRONG_BUY": 0, "BUY": 1, "HOLD": 2, "SELL": 3, "STRONG_SELL": 4}
        fused.sort(key=lambda f: (priority.get(f.signal, 5), -f.fused_score))

        logger.info(
            f"Fusion complete: {len(fused)} stocks | "
            f"BUY={sum(1 for f in fused if 'BUY' in f.signal)} | "
            f"SELL={sum(1 for f in fused if 'SELL' in f.signal)} | "
            f"HOLD={sum(1 for f in fused if f.signal == 'HOLD')}"
        )
        return fused