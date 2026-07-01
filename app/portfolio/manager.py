"""
Portfolio Manager — Section 21
Thin adapter that pipes production data signals through the shared PortfolioAcceptanceEngine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from app.db import get_sync_db
from app.fusion.engine import FusedSignal
from app.logger import get_logger
from app.portfolio.acceptance import (
    MAX_OPEN_POSITIONS,
    PORTFOLIO_VALUE,
    AcceptanceCandidate,
    PortfolioAcceptanceEngine,
    PortfolioLedger,
    PositionSize,
    _pairwise_correlation,
)
from config.watchlist import WATCHLIST_WITH_SECTORS

logger = get_logger("portfolio")


@dataclass
class RejectedSignal:
    symbol:  str
    signal:  str
    score:   float
    reason:  str


@dataclass
class PortfolioResult:
    run_date:          date
    accepted:          List[FusedSignal]       = field(default_factory=list)
    rejected:          List[RejectedSignal]    = field(default_factory=list)
    position_sizes:    Dict[str, PositionSize] = field(default_factory=dict)
    active_positions:  int  = 0
    slots_available:   int  = 0
    sector_exposure:   Dict[str, float] = field(default_factory=dict)
    total_risk_pct:    float = 0.0

    def summary(self) -> str:
        return (
            f"accepted={len(self.accepted)} | rejected={len(self.rejected)} | "
            f"active={self.active_positions} | slots={self.slots_available} | "
            f"total_risk={self.total_risk_pct:.1%}"
        )


def get_active_positions() -> List[Dict]:
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT stock, position_size_shares, position_value_inr,
                       stop_loss_realistic, entry_price_actual
                FROM trades
                WHERE status IN ('ACTIVE', 'PENDING')
                """
            )
            rows = cursor.fetchall()
            positions = []
            for row in rows:
                stock  = row["stock"]
                entry  = float(row["entry_price_actual"] or 0)
                stop   = float(row["stop_loss_realistic"] or 0)
                shares = int(row["position_size_shares"] or 0)
                positions.append({
                    "stock":              stock,
                    "sector":             WATCHLIST_WITH_SECTORS.get(stock, "Unknown"),
                    "position_value_inr": float(row["position_value_inr"] or 0),
                    "risk_amount_inr":    abs(entry - stop) * shares if entry and stop else 0,
                })
            return positions
    except Exception as e:
        logger.warning(f"Could not fetch active positions: {e} — assuming 0")
        return []


class PortfolioManager:
    def __init__(self, run_date: Optional[date] = None, features_map: Optional[Dict[str, Dict]] = None):
        self.run_date     = run_date or date.today()
        self.features_map = features_map or {}

    def run(self, signals: List[FusedSignal]) -> PortfolioResult:
        result = PortfolioResult(run_date=self.run_date)
        active = get_active_positions()
        
        result.active_positions = len(active)
        result.slots_available  = max(0, MAX_OPEN_POSITIONS - len(active))

        ledger = PortfolioLedger.from_active_positions(active, portfolio_value=PORTFOLIO_VALUE)
        engine = PortfolioAcceptanceEngine(ledger, correlation_fn=_pairwise_correlation)

        buy_signals   = [s for s in signals if "BUY" in s.signal]
        other_signals = [s for s in signals if "BUY" not in s.signal]

        if result.slots_available == 0:
            for s in buy_signals:
                result.rejected.append(RejectedSignal(s.symbol, s.signal, s.fused_score, "PORTFOLIO_FULL"))
            result.accepted = other_signals
            return result

        candidates: List[AcceptanceCandidate] = []
        signal_by_symbol: Dict[str, FusedSignal] = {}
        for signal in buy_signals:
            if not signal.entry_price or not signal.stop_loss:
                result.rejected.append(RejectedSignal(signal.symbol, signal.signal, signal.fused_score, "INSUFFICIENT_DATA"))
                continue
            
            features = self.features_map.get(signal.symbol)
            vol_sma20 = features.get("volume_sma_20", 0) if features else 0
            adv_est = float(vol_sma20) * signal.entry_price if vol_sma20 > 0 else None

            candidates.append(AcceptanceCandidate(
                symbol=signal.symbol, sector=WATCHLIST_WITH_SECTORS.get(signal.symbol, "Unknown"),
                fused_score=signal.fused_score, entry_date=self.run_date, entry_price=signal.entry_price,
                stop_loss=signal.stop_loss, adv_estimate=adv_est, ref=signal,
            ))
            signal_by_symbol[signal.symbol] = signal

        eval_result = engine.evaluate_batch(self.run_date, candidates)

        accepted_buy: List[FusedSignal] = []
        for ac in eval_result.accepted:
            signal = signal_by_symbol[ac.candidate.symbol]
            ps     = ac.position
            result.position_sizes[signal.symbol] = ps

            signal.position_size_shares = ps.final_shares
            signal.position_value_inr   = ps.position_value_inr
            signal.risk_amount_inr      = ps.risk_amount_inr

            if ac.penalties:
                signal.reasons.append(f"Portfolio penalties: {' | '.join(ac.penalties)}")
            signal.reasons.append(f"Position committed via: {ps.binding_constraint}")
            accepted_buy.append(signal)

        for rc in eval_result.rejected:
            result.rejected.append(RejectedSignal(rc.candidate.symbol, rc.candidate.ref.signal, rc.candidate.fused_score, rc.reason))

        result.accepted        = accepted_buy + other_signals
        result.total_risk_pct  = ledger.total_risk_pct()
        result.sector_exposure = {sec: round(ledger.sector_value(sec) / PORTFOLIO_VALUE, 4) for sec in {c.sector for c in candidates} if ledger.sector_value(sec) > 0}

        return result