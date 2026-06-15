"""
Overnight gap detection stub.
Section 19.2: check open vs prior close before market opens (09:15 IST).
"""
from __future__ import annotations

from app.logger import get_logger

logger = get_logger("gap_detector")


async def check_overnight_gaps() -> None:
    """
    For each ACTIVE trade in the trades table:
    - Fetch today's open price
    - Compare to prior close
    - If gap > 1.5x ATR: mark signal as GAP_OPEN
    - If gap-down below stop loss: suppress signal (STOP_INVALIDATED)
    - If gap-up above entry: flag CHASE_RISK, reduce confidence 25%
    Full implementation: app/data/gap_detector_impl.py (next build layer)
    """
    logger.info("Gap check stub — implementation in next layer")
