"""
Portfolio Acceptance Engine — Section 21 + 23.2 (canonical implementation)
Consolidated engine to ensure absolute logic parity between backtesting and live trading.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from app.logger import get_logger
from config.settings import settings
from config.watchlist import WATCHLIST_WITH_SECTORS

logger = get_logger("portfolio.acceptance")

# ── Constants ────────────────────────────────────────────────────────
PORTFOLIO_VALUE      = settings.portfolio_value_inr
RISK_PER_TRADE_PCT   = settings.risk_per_trade_pct
MAX_OPEN_POSITIONS   = settings.max_open_positions
MAX_SINGLE_STOCK_PCT = settings.max_single_stock_pct
MAX_SECTOR_EXPOSURE  = settings.max_sector_exposure_pct
MAX_TOTAL_RISK_PCT   = 0.10
ADV_CAP_PCT          = 0.05

SAME_SECTOR_PENALTY  = 0.20

CROSS_STOCK_ROLLING_DAYS   = 60
CROSS_STOCK_CORR_THRESHOLD = 0.7
CROSS_STOCK_MAX_PENALTY    = 0.15
CROSS_STOCK_MIN_SAMPLES    = 20

MAX_SECTOR_SIGNALS_PER_DAY = 2
SECTOR_QUARANTINE_SESSIONS = 5
QUARANTINE_TRIGGER_STOPS   = 2   # consecutive STOP exits in a sector


@dataclass
class AcceptanceCandidate:
    symbol:         str
    sector:         str
    fused_score:    float
    entry_date:     date
    entry_price:    float
    stop_loss:      float
    exit_date:      Optional[date] = None   # backtest only
    exit_reason:    Optional[str]  = None   # backtest only
    adv_estimate:   Optional[float] = None  # ADV value in ₹
    ref:            object = None           # Passthrough mapping reference


@dataclass
class PositionSize:
    symbol:              str
    shares_risk_based:   int
    shares_stock_capped: int
    shares_adv_capped:   int
    final_shares:        int
    position_value_inr:  float
    risk_amount_inr:     float
    risk_pct_portfolio:  float
    binding_constraint:  str       # RISK | STOCK_CAP | ADV_CAP


@dataclass
class AcceptedCandidate:
    candidate:    AcceptanceCandidate
    position:     PositionSize
    penalties:    List[str]
    effective_risk_pct: float


@dataclass
class RejectedCandidate:
    candidate: AcceptanceCandidate
    reason:    str


@dataclass
class AcceptanceResult:
    accepted: List[AcceptedCandidate] = field(default_factory=list)
    rejected: List[RejectedCandidate] = field(default_factory=list)


def calculate_position_size(
    symbol:       str,
    entry_price:  float,
    stop_loss:    float,
    portfolio_value: float,
    adv_estimate: Optional[float] = None,
) -> PositionSize:
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        risk_per_share = entry_price * 0.03

    risk_inr_budget  = portfolio_value * RISK_PER_TRADE_PCT
    shares_risk      = max(1, int(risk_inr_budget / risk_per_share))

    max_value_stock  = portfolio_value * MAX_SINGLE_STOCK_PCT
    shares_stock_cap = max(1, int(max_value_stock / entry_price))

    adv_value        = adv_estimate if (adv_estimate and adv_estimate > 0) else 50_000_000.0
    max_order_value  = adv_value * ADV_CAP_PCT
    shares_adv_cap   = max(1, int(max_order_value / entry_price))

    final_shares = min(shares_risk, shares_stock_cap, shares_adv_cap)

    if final_shares == shares_adv_cap and shares_adv_cap < shares_risk:
        binding = "ADV_CAP"
    elif final_shares == shares_stock_cap and shares_stock_cap < shares_risk:
        binding = "STOCK_CAP"
    else:
        binding = "RISK"

    position_value = final_shares * entry_price
    risk_amount    = final_shares * risk_per_share
    risk_pct       = risk_amount / portfolio_value if portfolio_value else 0.0

    return PositionSize(
        symbol=symbol, shares_risk_based=shares_risk, shares_stock_capped=shares_stock_cap,
        shares_adv_capped=shares_adv_cap, final_shares=final_shares,
        position_value_inr=round(position_value, 2), risk_amount_inr=round(risk_amount, 2),
        risk_pct_portfolio=round(risk_pct, 4), binding_constraint=binding,
    )


def _cross_stock_penalty(r: float) -> float:
    r = abs(r)
    if r <= CROSS_STOCK_CORR_THRESHOLD:
        return 0.0
    r = min(r, 1.0)
    return CROSS_STOCK_MAX_PENALTY * (r - CROSS_STOCK_CORR_THRESHOLD) / (1.0 - CROSS_STOCK_CORR_THRESHOLD)


def _fetch_return_series(stock: str, as_of_date: date, window_days: int):
    from app.db import get_sync_db
    cutoff = as_of_date - timedelta(days=window_days)
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT date, close FROM market_data
                WHERE stock = %s AND date BETWEEN %s AND %s
                ORDER BY date ASC
                """,
                (stock, cutoff, as_of_date),
            )
            rows = cursor.fetchall()
    except Exception as e:
        logger.warning(f"Could not fetch return series for {stock}: {e}")
        return None

    if len(rows) < CROSS_STOCK_MIN_SAMPLES + 1:
        return None

    closes  = np.array([float(r["close"]) for r in rows])
    returns = np.diff(closes) / closes[:-1]
    return returns


def _pairwise_correlation(stock_a: str, stock_b: str, as_of_date: date) -> Optional[float]:
    ra = _fetch_return_series(stock_a, as_of_date, CROSS_STOCK_ROLLING_DAYS)
    rb = _fetch_return_series(stock_b, as_of_date, CROSS_STOCK_ROLLING_DAYS)
    if ra is None or rb is None:
        return None

    n = min(len(ra), len(rb))
    if n < CROSS_STOCK_MIN_SAMPLES:
        return None
    ra, rb = ra[-n:], rb[-n:]

    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(ra, rb)[0, 1]
    return None if np.isnan(r) else float(r)


CorrelationFn = Callable[[str, str, date], Optional[float]]


@dataclass
class _OpenPosition:
    symbol:              str
    sector:              str
    position_value_inr:  float
    risk_amount_inr:     float
    exit_date:           Optional[date]
    exit_reason:         Optional[str] = None


class PortfolioLedger:
    def __init__(self, portfolio_value: float = PORTFOLIO_VALUE):
        self.portfolio_value = portfolio_value
        self._open: List[_OpenPosition] = []
        self._sector_recent_exits: Dict[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=QUARANTINE_TRIGGER_STOPS)
        )
        self._sector_quarantine_until: Dict[str, date] = {}

    @classmethod
    def from_active_positions(cls, active_positions: List[Dict], portfolio_value: float = PORTFOLIO_VALUE) -> PortfolioLedger:
        ledger = cls(portfolio_value=portfolio_value)
        for pos in active_positions:
            ledger._open.append(_OpenPosition(
                symbol=pos["stock"], sector=pos["sector"],
                position_value_inr=pos["position_value_inr"],
                risk_amount_inr=pos["risk_amount_inr"], exit_date=None,
            ))
        return ledger

    def advance_day_backtest(self, today: date, quarantine_handler: Callable[[str, date], None]) -> None:
        still_open = []
        for pos in self._open:
            if pos.exit_date is not None and pos.exit_date <= today:
                if pos.exit_reason:
                    self._sector_recent_exits[pos.sector].append(pos.exit_reason)
                    recent = list(self._sector_recent_exits[pos.sector])
                    if len(recent) == QUARANTINE_TRIGGER_STOPS and all(r == "STOP" for r in recent):
                        quarantine_handler(pos.sector, today)
            else:
                still_open.append(pos)
        self._open = still_open

    def set_sector_quarantine(self, sector: str, until_date: date) -> None:
        self._sector_quarantine_until[sector] = until_date

    def is_quarantined(self, sector: str, today: date) -> bool:
        expiry = self._sector_quarantine_until.get(sector)
        return expiry is not None and today < expiry

    def open_count(self) -> int: return len(self._open)
    def sector_open_count(self, sector: str) -> int: return sum(1 for p in self._open if p.sector == sector)
    def sector_value(self, sector: str) -> float: return sum(p.position_value_inr for p in self._open if p.sector == sector)
    def total_risk_pct(self) -> float: return sum(p.risk_amount_inr for p in self._open) / self.portfolio_value if self.portfolio_value else 0.0
    def all_open_symbols(self) -> List[str]: return [p.symbol for p in self._open]

    def record_open(self, symbol: str, sector: str, position_value_inr: float, risk_amount_inr: float, exit_date: Optional[date] = None, exit_reason: Optional[str] = None) -> None:
        self._open.append(_OpenPosition(symbol=symbol, sector=sector, position_value_inr=position_value_inr, risk_amount_inr=risk_amount_inr, exit_date=exit_date, exit_reason=exit_reason))


class PortfolioAcceptanceEngine:
    def __init__(self, ledger: PortfolioLedger, correlation_fn: Optional[CorrelationFn] = None):
        self.ledger = ledger
        self.correlation_fn = correlation_fn

    def _sector_score_penalty(self, candidate: AcceptanceCandidate, sector_seen_today: Dict[str, int]) -> Tuple[float, List[str]]:
        penalties = []
        score = candidate.fused_score
        sector = candidate.sector
        if sector_seen_today.get(sector, 0) >= 1:
            score *= (1 - SAME_SECTOR_PENALTY)
            penalties.append(f"same-sector penalty -{SAME_SECTOR_PENALTY:.0%} ({sector})")
        if self.ledger.sector_open_count(sector) > 0:
            score *= (1 - SAME_SECTOR_PENALTY)
            penalties.append(f"active-sector penalty -{SAME_SECTOR_PENALTY:.0%} ({sector})")
        return score, penalties

    def _correlation_penalty(self, candidate: AcceptanceCandidate, accepted_so_far: List[str], as_of_date: date) -> Tuple[float, Optional[str]]:
        if self.correlation_fn is None:
            return 0.0, None
        already = accepted_so_far + self.ledger.all_open_symbols()
        worst_pen, worst_with = 0.0, None
        for other in already:
            if other == candidate.symbol: continue
            r = self.correlation_fn(candidate.symbol, other, as_of_date)
            if r is None: continue
            pen = _cross_stock_penalty(r)
            if pen > worst_pen: worst_pen, worst_with = pen, other
        return worst_pen, worst_with

    def evaluate_batch(self, today: date, candidates: List[AcceptanceCandidate]) -> AcceptanceResult:
        result = AcceptanceResult()
        if not candidates:
            return result

        # Pre-scoring and structural mathematical fix: Use raw percentage without multiplying by 100
        sector_seen: Dict[str, int] = {}
        scored: List[Tuple[float, AcceptanceCandidate, PositionSize, List[str]]] = []
        for c in candidates:
            ps = calculate_position_size(c.symbol, c.entry_price, c.stop_loss, self.ledger.portfolio_value, c.adv_estimate)
            
            # CRITICAL FIX: Base floor decimal mapping to protect small risks and protect correct ordinal ranks
            risk_contrib = max(ps.risk_pct_portfolio, 0.0001) 
            raw_adj = c.fused_score / risk_contrib
            
            adj_score, penalties = self._sector_score_penalty(c, sector_seen)
            final_score = raw_adj * (adj_score / c.fused_score if c.fused_score else 1.0)
            scored.append((final_score, c, ps, penalties))
            sector_seen[c.sector] = sector_seen.get(c.sector, 0) + 1

        scored.sort(key=lambda x: -x[0])

        accepted_symbols: List[str] = []
        running_risk = self.ledger.total_risk_pct()
        sector_value_running = {s: self.ledger.sector_value(s) for s in {c.sector for c in candidates}}
        sector_count_today: Dict[str, int] = defaultdict(int)
        accepted_count_today = 0

        for final_score, c, ps, penalties in scored:
            if self.ledger.open_count() + accepted_count_today >= MAX_OPEN_POSITIONS:
                result.rejected.append(RejectedCandidate(c, "PORTFOLIO_FULL"))
                continue

            if self.ledger.is_quarantined(c.sector, today):
                result.rejected.append(RejectedCandidate(c, f"SECTOR_QUARANTINE ({c.sector})"))
                continue

            # Unified Congestion Rule Check
            today_sector_count = self.ledger.sector_open_count(c.sector) + sector_count_today[c.sector]
            if today_sector_count >= MAX_SECTOR_SIGNALS_PER_DAY:
                result.rejected.append(RejectedCandidate(c, f"SECTOR_DAILY_CAP ({c.sector} already {today_sector_count} today)"))
                continue

            sector_val_after = sector_value_running.get(c.sector, 0) + ps.position_value_inr
            sector_pct_after = sector_val_after / self.ledger.portfolio_value
            if sector_pct_after > MAX_SECTOR_EXPOSURE:
                result.rejected.append(RejectedCandidate(c, f"SECTOR_CAP ({c.sector} would be {sector_pct_after:.0%})"))
                continue

            corr_pen, corr_with = self._correlation_penalty(c, accepted_symbols, today)
            effective_risk_pct = ps.risk_pct_portfolio * (1 + corr_pen)

            risk_after = running_risk + effective_risk_pct
            if risk_after > MAX_TOTAL_RISK_PCT:
                reason = f"TOTAL_RISK_CAP (would be {risk_after:.1%})"
                if corr_pen > 0: reason += f" [corr_adj +{corr_pen:.0%} vs {corr_with}]"
                result.rejected.append(RejectedCandidate(c, reason))
                continue

            # State Commitment
            running_risk += effective_risk_pct
            sector_value_running[c.sector] = sector_val_after
            sector_count_today[c.sector] += 1
            accepted_count_today += 1
            accepted_symbols.append(c.symbol)

            if corr_pen > 0:
                penalties = list(penalties) + [f"cross-stock correlation +{corr_pen:.0%} effective risk (vs {corr_with})"]

            result.accepted.append(AcceptedCandidate(c, ps, penalties, effective_risk_pct))
            self.ledger.record_open(c.symbol, c.sector, ps.position_value_inr, ps.risk_amount_inr, c.exit_date, c.exit_reason)

        return result