"""
Execution Realism Layer — Section 20
Converts theoretical signal prices to realistic execution prices
accounting for slippage, impact cost, and brokerage.

Tiers (Section 20.1):
  Nifty50   → slippage 0.05%
  NiftyNext → slippage 0.10%
  MidSmall  → slippage 0.20%

Brokerage model (Zerodha flat fee):
  Delivery: 0% brokerage, 0.1% STT on sell side
  Intraday: ₹20 per order or 0.03%, whichever lower
"""
from __future__ import annotations

from typing import Optional

from config.watchlist import NIFTY50_STOCKS, NIFTY_NEXT50_STOCKS
from config.settings import settings


# Slippage tiers
SLIPPAGE_TIERS = {
    "NIFTY50":    0.0005,   # 0.05%
    "NIFTY_NEXT": 0.0010,   # 0.10%
    "MID_SMALL":  0.0020,   # 0.20%
}

# Transaction costs (delivery — most signals are positional)
STT_SELL_RATE = 0.001       # 0.1% STT on sell
SEBI_TURNOVER = 0.0001      # ₹10 per crore
EXCHANGE_TXN  = 0.0000325   # NSE txn charge
STAMP_DUTY    = 0.00015     # 0.015% on buy
GST_RATE      = 0.18        # on brokerage + SEBI


def get_slippage_tier(symbol: str) -> str:
    if symbol in NIFTY50_STOCKS:
        return "NIFTY50"
    if symbol in NIFTY_NEXT50_STOCKS:
        return "NIFTY_NEXT"
    return "MID_SMALL"


def get_slippage_factor(symbol: str) -> float:
    tier = get_slippage_tier(symbol)
    return SLIPPAGE_TIERS[tier]


def realistic_entry(symbol: str, theoretical_entry: float) -> tuple[float, float]:
    """
    Returns (realistic_entry_price, slippage_pct).
    Buy side: entry is higher than theoretical due to ask spread + slippage.
    """
    slip = get_slippage_factor(symbol)
    realistic = theoretical_entry * (1 + slip)
    return round(realistic, 2), slip


def realistic_exit(symbol: str, theoretical_exit: float) -> tuple[float, float]:
    """
    Returns (realistic_exit_price, total_cost_pct).
    Sell side: exit is lower due to bid spread + STT + exchange charges.
    """
    slip = get_slippage_factor(symbol)
    total_sell_cost = slip + STT_SELL_RATE + EXCHANGE_TXN + SEBI_TURNOVER
    realistic = theoretical_exit * (1 - total_sell_cost)
    return round(realistic, 2), total_sell_cost


def realistic_stop(symbol: str, theoretical_stop: float) -> float:
    """
    Stop loss realistic execution — gap risk on stop trigger.
    Mid/small caps get an extra 0.1% gap buffer.
    """
    tier = get_slippage_tier(symbol)
    extra_gap = 0.001 if tier == "MID_SMALL" else 0
    slip = get_slippage_factor(symbol)
    return round(theoretical_stop * (1 - slip - extra_gap), 2)


def calculate_impact_cost(
    symbol: str,
    order_value_inr: float,
    avg_daily_volume: float,
    close_price: float,
) -> float:
    """
    Estimates market impact cost for large orders (Section 20.2).
    If order > 5% of ADV → significant impact cost.
    Returns impact cost as fraction (e.g. 0.002 = 0.2%).
    """
    adv_value = avg_daily_volume * close_price
    order_pct_of_adv = order_value_inr / adv_value if adv_value > 0 else 0

    if order_pct_of_adv > settings.adv_order_cap_pct:
        # Linear impact model: 1% impact per 10% of ADV consumed
        impact = order_pct_of_adv * 0.10
        return min(impact, 0.02)   # cap at 2%

    return 0.0


def apply_execution_realism(
    symbol: str,
    theoretical_entry: float,
    theoretical_target: float,
    theoretical_stop: float,
    position_value_inr: float,
    avg_daily_volume: float,
    close_price: float,
) -> dict:
    """
    Full execution realism pass for a single signal.
    Returns dict with realistic prices and cost breakdown.
    """
    entry_r, slip_entry = realistic_entry(symbol, theoretical_entry)
    target_r, sell_cost = realistic_exit(symbol, theoretical_target)
    stop_r = realistic_stop(symbol, theoretical_stop)

    impact = calculate_impact_cost(
        symbol, position_value_inr, avg_daily_volume, close_price
    )

    # Adjust target further for impact cost on exit
    target_r_final = round(target_r * (1 - impact), 2)

    # Realistic R:R ratio
    risk_theoretical = theoretical_entry - theoretical_stop
    risk_realistic = entry_r - stop_r
    reward_realistic = target_r_final - entry_r

    rr_theoretical = (
        (theoretical_target - theoretical_entry) / risk_theoretical
        if risk_theoretical > 0 else 0
    )
    rr_realistic = reward_realistic / risk_realistic if risk_realistic > 0 else 0

    return {
        "entry_price_realistic":  entry_r,
        "exit_target_realistic":  target_r_final,
        "stop_loss_realistic":    stop_r,
        "slippage_factor_pct":    slip_entry,
        "impact_cost_pct":        impact,
        "sell_cost_pct":          sell_cost,
        "rr_ratio_theoretical":   round(rr_theoretical, 2),
        "rr_ratio_realistic":     round(rr_realistic, 2),
        "liquidity_warning":      position_value_inr / (avg_daily_volume * close_price) > settings.adv_order_cap_pct
        if avg_daily_volume > 0 and close_price > 0 else False,
    }
