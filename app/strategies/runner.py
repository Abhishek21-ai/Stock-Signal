"""
Strategy Runner — executes all 5 core strategies and returns results.
Called by pipeline.py Stage 4.
"""
from __future__ import annotations

from typing import Dict, List

from app.strategies.base import StrategyResult
from app.strategies.trend import TrendFollowingStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.reversion import MeanReversionStrategy
from app.strategies.breakout import BreakoutStrategy
from app.strategies.volume import VolumeProfileStrategy
# Removed RiskStrategy from direct generation import
from app.logger import get_logger

logger = get_logger("strategy_runner")

# Only keep true alpha-generating strategies here
ALL_STRATEGIES = [
    TrendFollowingStrategy(),
    MomentumStrategy(),
    MeanReversionStrategy(),
    BreakoutStrategy(),
    VolumeProfileStrategy(),
]


class StrategyRunner:
    def run(self, features: Dict, regime: str = "UNCERTAIN") -> List[StrategyResult]:
        results = []
        for strategy in ALL_STRATEGIES:
            try:
                result = strategy.run(features, regime=regime)
                # Ensure we only track actual trade signals
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(
                    f"Strategy {strategy.strategy_id} failed for "
                    f"{features.get('symbol', '?')}: {e}"
                )
        return results