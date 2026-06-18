"""
Portfolio Manager — Section 21
Sits between Signal Fusion/LLM and Notifications.

Responsibilities:
  21.1 Position Sizing  — RiskBased → StockCap → LiquidityCap (min of 3)
  21.2 Portfolio Constraints — max positions, sector cap, total risk cap
  21.3 Signal Prioritization — rank by RiskAdjustedScore, apply penalties

Flow:
  FusedSignal[] → PortfolioManager.run() → PortfolioResult
    - Each signal gets position_size_shares + position_value_inr added
    - Signals that fail constraints are rejected with reason logged
    - Top N pass through to notifications
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from app.db import get_sync_db
from app.fusion.engine import FusedSignal
from app.logger import get_logger
from config.settings import settings
from config.watchlist import WATCHLIST_WITH_SECTORS

logger = get_logger("portfolio")

# ── Constants (from settings, with fallbacks) ─────────────────
PORTFOLIO_VALUE     = settings.portfolio_value_inr        # e.g. ₹10,00,000
RISK_PER_TRADE_PCT  = settings.risk_per_trade_pct         # 1.5% default
MAX_OPEN_POSITIONS  = settings.max_open_positions         # 8 default
MAX_SINGLE_STOCK_PCT= settings.max_single_stock_pct       # 15%
MAX_SECTOR_EXPOSURE = settings.max_sector_exposure_pct    # 30%
MAX_TOTAL_RISK_PCT  = 0.10    # 10% total portfolio at risk across all positions
MAX_SINGLE_RISK_PCT = 0.02    # 2% max risk per stock
ADV_CAP_PCT         = 0.05    # order ≤ 5% of avg daily volume value

# Prioritization penalties (Section 21.3)
SAME_SECTOR_PENALTY = 0.20    # 20% score penalty beyond first in sector
CORRELATION_PENALTY = 0.15    # 15% penalty for high correlation (§23 future)


@dataclass
class PositionSize:
    """Result of position sizing calculation for one signal."""
    symbol:              str
    shares_risk_based:   int      # raw from risk formula
    shares_stock_capped: int      # after stock % cap
    shares_adv_capped:   int      # after ADV liquidity cap
    final_shares:        int      # min of all three
    position_value_inr:  float
    risk_amount_inr:     float    # final_shares × (entry - stop)
    risk_pct_portfolio:  float    # risk_amount / portfolio
    binding_constraint:  str      # which cap was binding: RISK | STOCK_CAP | ADV_CAP
    adv_estimate:        float    # avg daily volume × price


@dataclass
class RejectedSignal:
    symbol:  str
    signal:  str
    score:   float
    reason:  str      # PORTFOLIO_FULL | SECTOR_CAP | TOTAL_RISK_CAP | INSUFFICIENT_DATA


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


# ── Active position reader ────────────────────────────────────

def get_active_positions() -> List[Dict]:
    """
    Fetch currently ACTIVE or PENDING trades from DB.
    Returns list of {stock, sector, position_value_inr, risk_amount_inr}
    """
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
                sector = WATCHLIST_WITH_SECTORS.get(stock, "Unknown")
                entry  = float(row["entry_price_actual"] or 0)
                stop   = float(row["stop_loss_realistic"] or 0)
                shares = int(row["position_size_shares"] or 0)
                risk   = abs(entry - stop) * shares if entry and stop else 0
                positions.append({
                    "stock":              stock,
                    "sector":             sector,
                    "position_value_inr": float(row["position_value_inr"] or 0),
                    "risk_amount_inr":    risk,
                })
            return positions
    except Exception as e:
        logger.warning(f"Could not fetch active positions: {e} — assuming 0")
        return []


# ── Position sizing ───────────────────────────────────────────

def _estimate_adv(features: Optional[Dict], entry_price: float) -> float:
    """
    Estimate Average Daily Value (₹) from feature vector.
    Falls back to a conservative default if not available.
    """
    if features:
        vol_sma20 = features.get("volume_sma_20", 0)
        if vol_sma20 and vol_sma20 > 0:
            return float(vol_sma20) * entry_price
    # Conservative fallback: assume ₹5Cr ADV (mid-cap NSE stock)
    return 50_000_000.0


def calculate_position_size(
    symbol:      str,
    entry_price: float,
    stop_loss:   float,
    features:    Optional[Dict] = None,
) -> PositionSize:
    """
    Section 21.1: Position size = min(risk_based, stock_capped, adv_capped).
    """
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        risk_per_share = entry_price * 0.03   # fallback: 3% stop

    # ── 1. Risk-based sizing ──────────────────────────────────
    risk_inr_budget  = PORTFOLIO_VALUE * RISK_PER_TRADE_PCT
    shares_risk      = max(1, int(risk_inr_budget / risk_per_share))

    # ── 2. Stock cap (max % of portfolio in one stock) ────────
    max_value_stock  = PORTFOLIO_VALUE * MAX_SINGLE_STOCK_PCT
    shares_stock_cap = max(1, int(max_value_stock / entry_price))

    # ── 3. ADV liquidity cap (max 5% of ADV) ─────────────────
    adv_value        = _estimate_adv(features, entry_price)
    max_order_value  = adv_value * ADV_CAP_PCT
    shares_adv_cap   = max(1, int(max_order_value / entry_price))

    # ── Final: minimum of all three ───────────────────────────
    final_shares = min(shares_risk, shares_stock_cap, shares_adv_cap)

    # Determine binding constraint
    if final_shares == shares_adv_cap and shares_adv_cap < shares_risk:
        binding = "ADV_CAP"
    elif final_shares == shares_stock_cap and shares_stock_cap < shares_risk:
        binding = "STOCK_CAP"
    else:
        binding = "RISK"

    position_value = final_shares * entry_price
    risk_amount    = final_shares * risk_per_share
    risk_pct       = risk_amount / PORTFOLIO_VALUE

    return PositionSize(
        symbol=symbol,
        shares_risk_based=shares_risk,
        shares_stock_capped=shares_stock_cap,
        shares_adv_capped=shares_adv_cap,
        final_shares=final_shares,
        position_value_inr=round(position_value, 2),
        risk_amount_inr=round(risk_amount, 2),
        risk_pct_portfolio=round(risk_pct, 4),
        binding_constraint=binding,
        adv_estimate=round(adv_value, 0),
    )


# ── Risk-adjusted score ───────────────────────────────────────

def risk_adjusted_score(
    signal:           FusedSignal,
    pos_size:         PositionSize,
    sector_seen:      Dict[str, int],
    active_sectors:   Dict[str, float],
) -> Tuple[float, List[str]]:
    """
    Section 21.3: RiskAdjustedScore = QuantScore / PositionRiskPct
    Apply same-sector penalty if sector already in portfolio.
    """
    base_score   = signal.fused_score
    risk_contrib = pos_size.risk_pct_portfolio * 100  # as percent
    if risk_contrib <= 0:
        risk_contrib = 1.0

    raw_adj = base_score / risk_contrib
    penalties = []

    # Same-sector penalty (beyond first in sector from today's signals)
    sector = WATCHLIST_WITH_SECTORS.get(signal.symbol, "Unknown")
    if sector_seen.get(sector, 0) >= 1:
        raw_adj    *= (1 - SAME_SECTOR_PENALTY)
        penalties.append(f"same-sector penalty -{SAME_SECTOR_PENALTY:.0%} ({sector})")

    # Sector already has active position → compound penalty
    if active_sectors.get(sector, 0) > 0:
        raw_adj    *= (1 - SAME_SECTOR_PENALTY)
        penalties.append(f"active-sector penalty -{SAME_SECTOR_PENALTY:.0%} ({sector})")

    return round(raw_adj, 4), penalties


# ── Main Portfolio Manager ────────────────────────────────────

class PortfolioManager:
    """
    Called by pipeline.py Stage 9 (replaces the simple cap logic).
    Applies full Section 21 logic.
    """

    def __init__(
        self,
        run_date:     Optional[date] = None,
        features_map: Optional[Dict[str, Dict]] = None,
    ):
        self.run_date     = run_date or date.today()
        self.features_map = features_map or {}

    def run(self, signals: List[FusedSignal]) -> PortfolioResult:
        result = PortfolioResult(run_date=self.run_date)

        # ── Load current active positions ──────────────────────
        active = get_active_positions()
        result.active_positions = len(active)

        # Sector exposure from active positions
        active_sector_value: Dict[str, float] = {}
        active_risk_total   = 0.0
        for pos in active:
            sec = pos["sector"]
            active_sector_value[sec] = (
                active_sector_value.get(sec, 0) + pos["position_value_inr"]
            )
            active_risk_total += pos["risk_amount_inr"]

        slots_available = MAX_OPEN_POSITIONS - len(active)
        result.slots_available = max(0, slots_available)
        result.total_risk_pct  = active_risk_total / PORTFOLIO_VALUE

        logger.info(
            f"Portfolio state: active={len(active)} | "
            f"slots={result.slots_available} | "
            f"total_risk={result.total_risk_pct:.1%}"
        )

        if result.slots_available == 0:
            logger.info("No slots available — rejecting all signals")
            for s in signals:
                if "BUY" in s.signal:
                    result.rejected.append(RejectedSignal(
                        symbol=s.symbol, signal=s.signal,
                        score=s.fused_score, reason="PORTFOLIO_FULL"
                    ))
            result.accepted = [s for s in signals if "BUY" not in s.signal]
            return result

        # ── Separate BUY candidates from rest ─────────────────
        buy_signals   = [s for s in signals if "BUY" in s.signal]
        other_signals = [s for s in signals if "BUY" not in s.signal]

        # ── Calculate position sizes ───────────────────────────
        position_sizes: Dict[str, PositionSize] = {}
        for signal in buy_signals:
            if not signal.entry_price or not signal.stop_loss:
                continue
            features = self.features_map.get(signal.symbol)
            ps = calculate_position_size(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                features=features,
            )
            position_sizes[signal.symbol] = ps

        # ── Risk-adjusted scoring + penalties ─────────────────
        sector_seen: Dict[str, int] = {}
        scored: List[Tuple[float, FusedSignal, List[str]]] = []

        for signal in buy_signals:
            ps = position_sizes.get(signal.symbol)
            if not ps:
                result.rejected.append(RejectedSignal(
                    symbol=signal.symbol, signal=signal.signal,
                    score=signal.fused_score, reason="INSUFFICIENT_DATA"
                ))
                continue
            adj_score, penalties = risk_adjusted_score(
                signal, ps, sector_seen, active_sector_value
            )
            scored.append((adj_score, signal, penalties))
            sec = WATCHLIST_WITH_SECTORS.get(signal.symbol, "Unknown")
            sector_seen[sec] = sector_seen.get(sec, 0) + 1

        # Sort by risk-adjusted score descending
        scored.sort(key=lambda x: -x[0])

        # ── Apply constraints and select top N ─────────────────
        accepted_buy:    List[FusedSignal] = []
        running_risk     = result.total_risk_pct
        sector_value_new: Dict[str, float] = dict(active_sector_value)

        for adj_score, signal, penalties in scored:
            if len(accepted_buy) >= result.slots_available:
                result.rejected.append(RejectedSignal(
                    symbol=signal.symbol, signal=signal.signal,
                    score=signal.fused_score, reason="PORTFOLIO_FULL"
                ))
                continue

            ps     = position_sizes[signal.symbol]
            sector = WATCHLIST_WITH_SECTORS.get(signal.symbol, "Unknown")

            # ── Sector cap check ──────────────────────────────
            sector_val_after = sector_value_new.get(sector, 0) + ps.position_value_inr
            sector_pct_after = sector_val_after / PORTFOLIO_VALUE
            if sector_pct_after > MAX_SECTOR_EXPOSURE:
                result.rejected.append(RejectedSignal(
                    symbol=signal.symbol, signal=signal.signal,
                    score=signal.fused_score,
                    reason=f"SECTOR_CAP ({sector} would be {sector_pct_after:.0%})"
                ))
                logger.info(
                    f"{signal.symbol}: rejected — sector cap "
                    f"({sector} {sector_pct_after:.0%} > {MAX_SECTOR_EXPOSURE:.0%})"
                )
                continue

            # ── Total risk cap check ──────────────────────────
            risk_after = running_risk + ps.risk_pct_portfolio
            if risk_after > MAX_TOTAL_RISK_PCT:
                result.rejected.append(RejectedSignal(
                    symbol=signal.symbol, signal=signal.signal,
                    score=signal.fused_score,
                    reason=f"TOTAL_RISK_CAP (would be {risk_after:.1%})"
                ))
                logger.info(
                    f"{signal.symbol}: rejected — total risk cap "
                    f"({risk_after:.1%} > {MAX_TOTAL_RISK_PCT:.0%})"
                )
                continue

            # ── Accepted ──────────────────────────────────────
            running_risk    += ps.risk_pct_portfolio
            sector_value_new[sector] = sector_val_after

            # Attach position sizing to signal
            signal.position_size_shares = ps.final_shares
            signal.position_value_inr   = ps.position_value_inr
            signal.risk_amount_inr      = ps.risk_amount_inr

            if penalties:
                signal.reasons.append(f"Portfolio penalties: {' | '.join(penalties)}")
            signal.reasons.append(
                f"Position: {ps.final_shares} shares × ₹{signal.entry_price:,.1f} "
                f"= ₹{ps.position_value_inr:,.0f} | "
                f"risk=₹{ps.risk_amount_inr:,.0f} ({ps.risk_pct_portfolio:.2%}) | "
                f"constraint={ps.binding_constraint}"
            )

            accepted_buy.append(signal)
            position_sizes[signal.symbol] = ps

            logger.info(
                f"{signal.symbol}: accepted | "
                f"shares={ps.final_shares} | "
                f"value=₹{ps.position_value_inr:,.0f} | "
                f"risk={ps.risk_pct_portfolio:.2%} | "
                f"constraint={ps.binding_constraint}"
            )

        result.accepted        = accepted_buy + other_signals
        result.position_sizes  = position_sizes
        result.total_risk_pct  = running_risk
        result.sector_exposure = {
            k: round(v / PORTFOLIO_VALUE, 4)
            for k, v in sector_value_new.items() if v > 0
        }

        logger.info(
            f"Portfolio complete: {result.summary()}"
        )
        return result

    def save_rejected(self, result: PortfolioResult) -> None:
        """Log rejected signals to daily_signals with rejection reason."""
        if not result.rejected:
            return
        try:
            with get_sync_db() as conn:
                cursor = conn.cursor()
                for r in result.rejected:
                    cursor.execute(
                        """
                        UPDATE daily_signals
                        SET llm_status = %s
                        WHERE stock = %s AND date = %s
                        """,
                        (f"REJECTED:{r.reason}", r.symbol, self.run_date),
                    )
        except Exception as e:
            logger.warning(f"Could not save rejection reasons: {e}")
