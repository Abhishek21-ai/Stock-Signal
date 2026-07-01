"""
System Stress Testing Harness — Institutional Validation Suite
"""

import os
os.environ["LOG_LEVEL"] = "WARNING"

import sys
import logging
import numpy as np
from datetime import date

# Silence logs
logging.getLogger().setLevel(logging.WARNING)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.WARNING)

# Ensure project root in path
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app.backtest.engine import (
    BacktestEngine,
    TradeRecord,
    calculate_metrics,
    STT_SELL_RATE,
    BROKERAGE_PER_TRADE,
)

from app.portfolio import acceptance
from config.settings import settings


##############################################################################
# Helpers
##############################################################################

def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f" 🔥 STRESS TEST: {title}")
    print("=" * 75)


##############################################################################
# Historical Slice Tests
##############################################################################

def run_historical_slice_test(
    all_trades,
    start_date,
    end_date,
    name,
):
    print_banner(name)

    slice_trades = [
        t
        for t in all_trades
        if start_date <= t.entry_date <= end_date
    ]

    if not slice_trades:
        print(
            f"⚠️ No trades found between "
            f"{start_date} and {end_date}."
        )
        return

    res = calculate_metrics(
        slice_trades,
        "combined",
        start_date,
        end_date,
    )

    print(f"  Total Trades Run:  {res.total_trades}")
    print(f"  Win Rate:          {res.win_rate_realistic}%")
    print(f"  Sharpe Ratio:      {res.sharpe_realistic}")
    print(f"  Max Drawdown:      {res.max_drawdown_pct}%")
    print(f"  Annualized Return: {res.annualized_return_realistic}%")
    print(
        f"  Acceptance Status: "
        f"{'✅ PASS' if res.meets_acceptance_criteria else '❌ FAIL'}"
    )


##############################################################################
# Sector Concentration
##############################################################################

def run_sector_concentration_stress(
    all_trades,
    start_date,
    end_date,
):
    print_banner("SECTOR CONCENTRATION SHOCK")

    counts = {}

    for t in all_trades:
        counts[t.sector] = counts.get(t.sector, 0) + 1

    dominant_sector = max(counts, key=counts.get)

    print(
        f"  Highest exposure vector detected: "
        f"'{dominant_sector}' "
        f"({counts[dominant_sector]} trades)"
    )

    shocked = []

    for t in all_trades:
        exit_price = t.exit_price
        return_pct = t.return_pct
        net_pnl = t.net_pnl

        if t.sector == dominant_sector:
            if "BUY" in t.signal:
                exit_price *= 0.95
            else:
                exit_price *= 1.05

            gross = (
                (exit_price - t.entry_price) * t.shares
                if "BUY" in t.signal
                else (t.entry_price - exit_price) * t.shares
            )

            net_pnl = (
                gross
                - exit_price * t.shares * STT_SELL_RATE
                - BROKERAGE_PER_TRADE * 2
            )

            return_pct = (
                (exit_price - t.entry_price)
                / t.entry_price
                * 100
            )

        shocked.append(
            TradeRecord(
                **{
                    **t.__dict__,
                    "exit_price": exit_price,
                    "net_pnl": net_pnl,
                    "return_pct": return_pct,
                }
            )
        )

    res = calculate_metrics(
        shocked,
        "combined",
        start_date,
        end_date,
    )

    print(f"  Shocked Sharpe (Realistic): {res.sharpe_realistic}")
    print(f"  Shocked Max Drawdown:       {res.max_drawdown_pct}%")


##############################################################################
# Execution Degradation
##############################################################################

def run_slippage_stress(
    all_trades,
    start_date,
    end_date,
):
    print_banner(
        "EXECUTION SLIPPAGE & SPREAD DETERIORATION "
        "(+50% Fee + 0.5% Slippage)"
    )

    shocked = []

    for t in all_trades:

        if "BUY" in t.signal:
            entry = t.entry_price * 1.005
            exit_ = t.exit_price * 0.995
        else:
            entry = t.entry_price * 0.995
            exit_ = t.exit_price * 1.005

        gross = (
            (exit_ - entry) * t.shares
            if "BUY" in t.signal
            else (entry - exit_) * t.shares
        )

        net = (
            gross
            - exit_ * t.shares * STT_SELL_RATE * 1.5
            - BROKERAGE_PER_TRADE * 3
        )

        ret = (
            (exit_ - entry)
            / entry
            * 100
            if "BUY" in t.signal
            else (entry - exit_)
            / entry
            * 100
        )

        shocked.append(
            TradeRecord(
                window_id=t.window_id,
                symbol=t.symbol,
                sector=t.sector,
                signal=t.signal,
                regime=t.regime,
                strategy_id=t.strategy_id,
                entry_date=t.entry_date,
                exit_date=t.exit_date,
                entry_price=entry,
                exit_price=exit_,
                stop_loss=t.stop_loss,
                target_price=t.target_price,
                shares=t.shares,
                gross_pnl=gross,
                net_pnl=net,
                return_pct=ret,
            )
        )

    res = calculate_metrics(
        shocked,
        "combined",
        start_date,
        end_date,
    )

    print(f"  Adjusted Sharpe (Realistic): {res.sharpe_realistic}")
    print(f"  Adjusted Return (Realistic): {res.annualized_return_realistic}%")
    print(f"  Max Adjusted Drawdown:      {res.max_drawdown_pct}%")


##############################################################################
# Monte Carlo
##############################################################################

def run_monte_carlo_reshuffle(
    all_trades,
    iterations=5000,
):
    print_banner(
        f"MONTE CARLO RISK-PROPORTIONAL RESHUFFLING "
        f"({iterations} Iterations)"
    )

    trade_returns = [
        t.return_pct
        for t in all_trades
        if t.return_pct is not None
    ]

    if len(trade_returns) < 20:
        print("⚠️ Not enough trades.")
        return

    risk_per_trade = settings.risk_per_trade_pct
    initial = settings.portfolio_value_inr

    drawdowns = []
    failures = 0

    for _ in range(iterations):

        sampled = np.random.choice(
            trade_returns,
            size=len(trade_returns),
            replace=True,
        )

        equity = initial
        curve = [equity]

        for r in sampled:
            equity *= (
                1
                + risk_per_trade
                * (r / 100.0)
            )
            curve.append(equity)

        curve = np.array(curve)

        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / peak * 100
        max_dd = abs(dd.min())

        drawdowns.append(max_dd)

        if max_dd > 20:
            failures += 1

    print(f"  Mean Simulated Drawdown:      {np.mean(drawdowns):.2f}%")
    print(f"  95th Percentile Drawdown:     {np.percentile(drawdowns,95):.2f}%")
    print(f"  99th Percentile Drawdown:     {np.percentile(drawdowns,99):.2f}%")
    print(
        f"  Probability of breaking "
        f"MaxDD (>20%): {(failures/iterations)*100:.2f}%"
    )


##############################################################################
# Parameter Sensitivity
##############################################################################

def run_parameter_sensitivity():
    print_banner("ROBUSTNESS MATRIX: PARAMETER SENSITIVITY")

    original_risk = settings.risk_per_trade_pct

    scenarios = [
        ("Tight Matrix", 0.010),
        ("Expanded Matrix", 0.020),
    ]

    for name, risk in scenarios:

        print(
            f" ⚙️ Testing configuration: "
            f"{name} ({risk:.1%} Risk)"
        )

        settings.risk_per_trade_pct = risk

        engine = BacktestEngine(
            verbose=False,
            save_to_db=False,
        )

        summary = engine.run()

        metrics = summary["aggregate_metrics"]

        print(
            f"    ↳ Result: "
            f"Sharpe={metrics.sharpe_realistic} | "
            f"MaxDD={metrics.max_drawdown_pct}% | "
            f"Return={metrics.annualized_return_realistic}%"
        )

    settings.risk_per_trade_pct = original_risk


##############################################################################
# Correlation Stress
##############################################################################

def run_correlation_stress():
    print_banner(
        "CONCENTRATION PROXIMITY: "
        "CROSS-STOCK CORRELATION SHOCK"
    )

    original = acceptance.CROSS_STOCK_MAX_PENALTY

    for penalty in [0.30, 0.50]:

        print(
            f" ⚙️ Simulating "
            f"CROSS_STOCK_MAX_PENALTY={penalty:.0%}"
        )

        acceptance.CROSS_STOCK_MAX_PENALTY = penalty

        engine = BacktestEngine(
            verbose=False,
            save_to_db=False,
        )

        summary = engine.run()

        metrics = summary["aggregate_metrics"]

        print(
            f"    ↳ Result at {penalty:.0%}: "
            f"Sharpe={metrics.sharpe_realistic} | "
            f"Ret={metrics.annualized_return_realistic}% | "
            f"Trades Count={metrics.total_trades}"
        )

    acceptance.CROSS_STOCK_MAX_PENALTY = original


##############################################################################
# Main
##############################################################################

def main():
    print("=" * 75)
    print(" STOCK SIGNAL PLATFORM — PRODUCTION INSTITUTIONAL STRESS TEST MATRIX")
    print("=" * 75)

    print("🔄 Initializing system baseline data harvest...")

    engine = BacktestEngine(
        verbose=False,
        save_to_db=False,
    )

    summary = engine.run()

    all_trades = summary["all_trades"]

    print(
        f"✅ Harvest successful. "
        f"Mapped {len(all_trades)} trades."
    )

    start = min(t.entry_date for t in all_trades)
    end = max(t.exit_date for t in all_trades)

    if start <= date(2020, 6, 30):
        run_historical_slice_test(
            all_trades,
            date(2020, 1, 1),
            date(2020, 6, 30),
            "2020 COVID CRASH MARGIN SKEW",
        )

    if start <= date(2022, 12, 31):
        run_historical_slice_test(
            all_trades,
            date(2022, 1, 1),
            date(2022, 12, 31),
            "2022 BEAR MARKET SKEW",
        )

    run_sector_concentration_stress(
        all_trades,
        start,
        end,
    )

    run_slippage_stress(
        all_trades,
        start,
        end,
    )

    run_monte_carlo_reshuffle(all_trades)

    run_parameter_sensitivity()

    run_correlation_stress()


if __name__ == "__main__":
    main()