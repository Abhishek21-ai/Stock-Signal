"""
Base strategy class — all 5 strategies inherit from this.
Each strategy returns a StrategyResult with a score (-100 to +100)
and sub-signals that feed the fusion layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StrategyResult:
    strategy_id: str
    score: float                        # -100 (strong sell) to +100 (strong buy)
    signal: str                         # BUY | SELL | HOLD
    confidence: float                   # 0–100
    reasons: list[str] = field(default_factory=list)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "strategy_id":  self.strategy_id,
            "score":        round(self.score, 2),
            "signal":       self.signal,
            "confidence":   round(self.confidence, 2),
            "reasons":      self.reasons,
            "entry_price":  self.entry_price,
            "stop_loss":    self.stop_loss,
            "target_price": self.target_price,
            **self.meta,
        }


def score_to_signal(score: float) -> str:
    if score >= 60:   return "STRONG_BUY"
    if score >= 25:   return "BUY"
    if score <= -60:  return "STRONG_SELL"
    if score <= -25:  return "SELL"
    return "HOLD"


class BaseStrategy:
    strategy_id: str = "base"

    def run(self, features: Dict, regime: str = "UNCERTAIN") -> StrategyResult:
        raise NotImplementedError
